"""
Data loaders for existing parquet files.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger


def symbol_to_filename(symbol: str, quote: str = "USDT") -> str:
    """
    Convert symbol to filename format.

    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT', 'ADAUSDT')
        quote: Quote currency (default: 'USDT')

    Returns:
        Filename (e.g., 'ohlcv_ETH_USDT.parquet')
    """
    base = symbol.replace(quote, "")
    return f"ohlcv_{base}_{quote}.parquet"


def filename_to_symbol(filename: str) -> str:
    """
    Convert filename to symbol format.

    Args:
        filename: File name (e.g., 'ohlcv_ETH_USDT.parquet')

    Returns:
        Symbol (e.g., 'ETHUSDT')
    """
    # Remove extension and prefix
    name = Path(filename).stem  # ohlcv_ETH_USDT
    parts = name.split("_")  # ['ohlcv', 'ETH', 'USDT']
    if len(parts) >= 3:
        return parts[1] + parts[2]  # ETHUSDT
    return name


class ExistingDataLoader:
    """Load existing OHLCV parquet data."""

    def __init__(self, raw_path: str | Path):
        """
        Initialize loader.

        Args:
            raw_path: Path to raw data directory
        """
        self.raw_path = Path(raw_path)
        self.logger = logger.bind(module="data_loader")

        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data path does not exist: {self.raw_path}")

    def list_available_symbols(self) -> list[str]:
        """List all available symbols from existing files."""
        symbols = []
        for file in self.raw_path.glob("ohlcv_*.parquet"):
            symbol = filename_to_symbol(file.name)
            symbols.append(symbol)
        return sorted(symbols)

    def load_ohlcv(
        self,
        symbol: str,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Load OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            start: Start datetime filter
            end: End datetime filter
            columns: Columns to load (None for all)

        Returns:
            DataFrame with OHLCV data
        """
        filename = symbol_to_filename(symbol)
        file_path = self.raw_path / filename

        if not file_path.exists():
            self.logger.warning(f"File not found: {file_path}")
            return pl.DataFrame()

        self.logger.info(f"Loading {file_path}")

        # Use lazy loading for efficiency
        lf = pl.scan_parquet(file_path)

        # Select columns if specified
        if columns:
            available_cols = pl.read_parquet(file_path, n_rows=0).columns
            cols_to_load = [c for c in columns if c in available_cols]
            if cols_to_load:
                lf = lf.select(cols_to_load)

        # Apply time filters if timestamp column exists
        schema = pl.read_parquet(file_path, n_rows=0).schema
        time_col = self._detect_time_column(schema)

        if time_col:
            if start:
                if isinstance(start, str):
                    start = datetime.fromisoformat(start)
                lf = lf.filter(pl.col(time_col) >= start)
            if end:
                if isinstance(end, str):
                    end = datetime.fromisoformat(end)
                lf = lf.filter(pl.col(time_col) <= end)

        df = lf.collect()
        self.logger.info(f"Loaded {len(df)} rows for {symbol}")

        return df

    def load_multiple(
        self,
        symbols: list[str],
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> dict[str, pl.DataFrame]:
        """
        Load OHLCV data for multiple symbols.

        Args:
            symbols: List of trading symbols
            start: Start datetime filter
            end: End datetime filter

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}
        for symbol in symbols:
            df = self.load_ohlcv(symbol, start, end)
            if not df.is_empty():
                data[symbol] = df
        return data

    def get_file_info(self, symbol: str) -> dict[str, Any]:
        """Get information about a symbol's data file."""
        filename = symbol_to_filename(symbol)
        file_path = self.raw_path / filename

        if not file_path.exists():
            return {"exists": False, "symbol": symbol}

        # Get file stats
        stats = file_path.stat()

        # Read schema and row count
        schema = pl.read_parquet(file_path, n_rows=0).schema
        row_count = pl.scan_parquet(file_path).select(pl.len()).collect().item()

        # Detect time column and get date range
        time_col = self._detect_time_column(schema)
        date_range = {}
        if time_col:
            lf = pl.scan_parquet(file_path)
            date_range = lf.select([
                pl.col(time_col).min().alias("start"),
                pl.col(time_col).max().alias("end"),
            ]).collect().to_dicts()[0]

        return {
            "exists": True,
            "symbol": symbol,
            "file_path": str(file_path),
            "file_size_mb": stats.st_size / (1024 * 1024),
            "row_count": row_count,
            "columns": list(schema.keys()),
            "schema": {k: str(v) for k, v in schema.items()},
            "time_column": time_col,
            "date_range": date_range,
        }

    def _detect_time_column(self, schema: dict) -> str | None:
        """Detect the timestamp column name."""
        time_columns = ["timestamp", "datetime", "time", "date", "open_time", "close_time"]
        for col in time_columns:
            if col in schema:
                return col
        # Check for datetime types
        for col, dtype in schema.items():
            if "datetime" in str(dtype).lower() or "date" in str(dtype).lower():
                return col
        return None

    def validate_data(self, symbol: str) -> dict[str, Any]:
        """Validate data integrity for a symbol."""
        df = self.load_ohlcv(symbol)

        if df.is_empty():
            return {"valid": False, "error": "No data loaded"}

        issues = []

        # Check for required OHLCV columns
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in [col.lower() for col in df.columns]]
        if missing:
            issues.append(f"Missing columns: {missing}")

        # Check for nulls
        null_counts = df.null_count().to_dicts()[0]
        nulls = {k: v for k, v in null_counts.items() if v > 0}
        if nulls:
            issues.append(f"Null values found: {nulls}")

        # Check for duplicates (if timestamp exists)
        time_col = self._detect_time_column(df.schema)
        if time_col:
            dupes = df.select(pl.col(time_col)).filter(
                pl.col(time_col).is_duplicated()
            ).height
            if dupes > 0:
                issues.append(f"Duplicate timestamps: {dupes}")

        # Check OHLC consistency
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            invalid_ohlc = df.filter(
                (pl.col("high") < pl.col("low")) |
                (pl.col("high") < pl.col("open")) |
                (pl.col("high") < pl.col("close")) |
                (pl.col("low") > pl.col("open")) |
                (pl.col("low") > pl.col("close"))
            ).height
            if invalid_ohlc > 0:
                issues.append(f"Invalid OHLC rows: {invalid_ohlc}")

        return {
            "valid": len(issues) == 0,
            "row_count": len(df),
            "issues": issues,
        }


def load_all_available_data(
    raw_path: str | Path,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
) -> dict[str, pl.DataFrame]:
    """
    Convenience function to load all available data.

    Args:
        raw_path: Path to raw data directory
        start: Start datetime filter
        end: End datetime filter

    Returns:
        Dictionary mapping symbol to DataFrame
    """
    loader = ExistingDataLoader(raw_path)
    symbols = loader.list_available_symbols()
    return loader.load_multiple(symbols, start, end)
