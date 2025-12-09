"""
DuckDB-based data storage for efficient queries.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from loguru import logger


class DuckDBStore:
    """DuckDB-based storage for efficient analytical queries."""

    def __init__(self, db_path: str | Path | None = None):
        """
        Initialize DuckDB store.

        Args:
            db_path: Path to DuckDB file. If None, uses in-memory database.
        """
        self.db_path = Path(db_path) if db_path else None

        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = duckdb.connect(str(self.db_path))
        else:
            self.conn = duckdb.connect()

        self.logger = logger.bind(module="duckdb_store")
        self._init_tables()

    def _init_tables(self) -> None:
        """Initialize database tables."""
        # OHLCV table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                timestamp TIMESTAMP WITH TIME ZONE,
                symbol VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                quote_volume DOUBLE,
                trades BIGINT,
                taker_buy_volume DOUBLE,
                taker_buy_quote_volume DOUBLE,
                PRIMARY KEY (timestamp, symbol)
            )
        """)

        # Order book metrics table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS orderbook_metrics (
                timestamp TIMESTAMP WITH TIME ZONE,
                symbol VARCHAR,
                best_bid_price DOUBLE,
                best_ask_price DOUBLE,
                mid_price DOUBLE,
                spread DOUBLE,
                spread_bps DOUBLE,
                imbalance DOUBLE,
                bid_depth DOUBLE,
                ask_depth DOUBLE,
                PRIMARY KEY (timestamp, symbol)
            )
        """)

        # Futures metrics table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS futures_metrics (
                timestamp TIMESTAMP WITH TIME ZONE,
                symbol VARCHAR,
                funding_rate DOUBLE,
                open_interest DOUBLE,
                open_interest_value DOUBLE,
                long_short_ratio DOUBLE,
                top_trader_ls_ratio DOUBLE,
                taker_buy_sell_ratio DOUBLE,
                PRIMARY KEY (timestamp, symbol)
            )
        """)

        # Features table (wide format)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                timestamp TIMESTAMP WITH TIME ZONE,
                symbol VARCHAR,
                PRIMARY KEY (timestamp, symbol)
            )
        """)

        # Sentiment table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment (
                timestamp TIMESTAMP WITH TIME ZONE,
                source VARCHAR,
                symbol VARCHAR,
                title VARCHAR,
                sentiment_score DOUBLE,
                engagement_score DOUBLE
            )
        """)

    def insert_ohlcv(
        self,
        df: pl.DataFrame,
        upsert: bool = True,
    ) -> int:
        """
        Insert OHLCV data.

        Args:
            df: DataFrame with OHLCV data
            upsert: If True, update existing rows

        Returns:
            Number of rows inserted
        """
        if df.is_empty():
            return 0

        # Convert to pandas for DuckDB
        pdf = df.to_pandas()

        if upsert:
            self.conn.execute("""
                INSERT OR REPLACE INTO ohlcv
                SELECT * FROM pdf
            """)
        else:
            self.conn.execute("""
                INSERT INTO ohlcv
                SELECT * FROM pdf
            """)

        return len(df)

    def insert_futures_metrics(
        self,
        df: pl.DataFrame,
        upsert: bool = True,
    ) -> int:
        """Insert futures metrics data."""
        if df.is_empty():
            return 0

        pdf = df.to_pandas()

        if upsert:
            self.conn.execute("""
                INSERT OR REPLACE INTO futures_metrics
                SELECT * FROM pdf
            """)
        else:
            self.conn.execute("""
                INSERT INTO futures_metrics
                SELECT * FROM pdf
            """)

        return len(df)

    def query_ohlcv(
        self,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Query OHLCV data.

        Args:
            symbol: Filter by symbol
            start: Start datetime
            end: End datetime
            columns: Columns to select

        Returns:
            Polars DataFrame with results
        """
        col_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {col_str} FROM ohlcv WHERE 1=1"

        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)

        query += " ORDER BY timestamp"

        result = self.conn.execute(query, params).fetchdf()
        return pl.from_pandas(result)

    def query_with_indicators(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        sma_windows: list[int] | None = None,
        ema_windows: list[int] | None = None,
    ) -> pl.DataFrame:
        """
        Query OHLCV with computed indicators using SQL window functions.

        Args:
            symbol: Trading symbol
            start: Start datetime
            end: End datetime
            sma_windows: SMA window sizes
            ema_windows: EMA window sizes (approximated)

        Returns:
            DataFrame with OHLCV and indicators
        """
        sma_cols = ""
        if sma_windows:
            for w in sma_windows:
                sma_cols += f"""
                    AVG(close) OVER (
                        ORDER BY timestamp
                        ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW
                    ) AS sma_{w},
                """

        # Returns calculation
        returns_col = """
            (close / LAG(close) OVER (ORDER BY timestamp) - 1) AS returns,
            LN(close / LAG(close) OVER (ORDER BY timestamp)) AS log_returns,
        """

        # Volatility (rolling std of returns)
        vol_col = """
            STDDEV(LN(close / LAG(close) OVER (ORDER BY timestamp)))
            OVER (ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
            AS volatility_20,
        """

        query = f"""
            SELECT
                timestamp,
                symbol,
                open,
                high,
                low,
                close,
                volume,
                {sma_cols}
                {returns_col}
                {vol_col}
                (high - low) / close AS range_pct
            FROM ohlcv
            WHERE symbol = ?
        """

        params = [symbol]
        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)

        query += " ORDER BY timestamp"

        result = self.conn.execute(query, params).fetchdf()
        return pl.from_pandas(result)

    def query_merged(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """
        Query merged data from all tables.

        Args:
            symbol: Trading symbol
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with merged data
        """
        query = """
            SELECT
                o.*,
                f.funding_rate,
                f.open_interest,
                f.long_short_ratio,
                f.taker_buy_sell_ratio,
                ob.mid_price AS ob_mid_price,
                ob.spread_bps,
                ob.imbalance AS ob_imbalance
            FROM ohlcv o
            LEFT JOIN futures_metrics f
                ON o.timestamp = f.timestamp AND o.symbol = f.symbol
            LEFT JOIN orderbook_metrics ob
                ON o.timestamp = ob.timestamp AND o.symbol = ob.symbol
            WHERE o.symbol = ?
        """

        params = [symbol]
        if start:
            query += " AND o.timestamp >= ?"
            params.append(start)
        if end:
            query += " AND o.timestamp <= ?"
            params.append(end)

        query += " ORDER BY o.timestamp"

        result = self.conn.execute(query, params).fetchdf()
        return pl.from_pandas(result)

    def get_date_range(self, table: str, symbol: str | None = None) -> dict:
        """Get min/max dates for a table."""
        query = f"SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM {table}"
        params = []

        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)

        result = self.conn.execute(query, params).fetchone()
        return {
            "min_date": result[0] if result else None,
            "max_date": result[1] if result else None,
        }

    def get_row_count(self, table: str, symbol: str | None = None) -> int:
        """Get row count for a table."""
        query = f"SELECT COUNT(*) FROM {table}"
        params = []

        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)

        result = self.conn.execute(query, params).fetchone()
        return result[0] if result else 0

    def execute(self, query: str, params: list | None = None) -> pl.DataFrame:
        """Execute arbitrary SQL query."""
        result = self.conn.execute(query, params or []).fetchdf()
        return pl.from_pandas(result)

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
