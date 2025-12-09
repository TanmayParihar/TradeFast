"""
Binance OHLCV data collector.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

import aiohttp
import polars as pl
from loguru import logger

from src.data.ingestion.base import DataSource, RateLimiter, RetryHandler


class BinanceOHLCVCollector(DataSource):
    """Collector for Binance Futures OHLCV data."""

    BASE_URL = "https://fapi.binance.com"
    KLINES_ENDPOINT = "/fapi/v1/klines"
    MAX_LIMIT = 1500  # Maximum candles per request

    COLUMNS = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    def __init__(self, testnet: bool = False):
        super().__init__("binance_ohlcv")
        self.base_url = (
            "https://testnet.binancefuture.com" if testnet else self.BASE_URL
        )
        self.rate_limiter = RateLimiter(calls_per_minute=1200)
        self.retry_handler = RetryHandler(max_retries=3)

    async def fetch(
        self,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        interval: str = "1m",
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        Fetch OHLCV data from Binance Futures.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start: Start datetime (UTC)
            end: End datetime (UTC)
            interval: Candle interval ('1m', '5m', '1h', etc.)

        Returns:
            Polars DataFrame with OHLCV data
        """
        if symbol is None:
            raise ValueError("Symbol is required")

        start_ts = int(start.timestamp() * 1000) if start else None
        end_ts = int(end.timestamp() * 1000) if end else None

        all_data: list[list] = []

        async with aiohttp.ClientSession() as session:
            current_start = start_ts

            while True:
                await self.rate_limiter.acquire()

                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "limit": self.MAX_LIMIT,
                }

                if current_start:
                    params["startTime"] = current_start
                if end_ts:
                    params["endTime"] = end_ts

                data = await self.retry_handler.execute(
                    self._fetch_klines, session, params
                )

                if not data:
                    break

                all_data.extend(data)
                self.logger.debug(
                    f"Fetched {len(data)} candles for {symbol}, "
                    f"total: {len(all_data)}"
                )

                # Move to next batch
                last_close_time = data[-1][6]
                current_start = last_close_time + 1

                if end_ts and current_start >= end_ts:
                    break

                if len(data) < self.MAX_LIMIT:
                    break

                await asyncio.sleep(0.1)  # Small delay between requests

        if not all_data:
            return pl.DataFrame()

        df = pl.DataFrame(all_data, schema=self.COLUMNS)
        df = self._process_dataframe(df, symbol)

        return df

    async def _fetch_klines(
        self, session: aiohttp.ClientSession, params: dict
    ) -> list:
        """Fetch klines from API."""
        url = f"{self.base_url}{self.KLINES_ENDPOINT}"

        async with session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()

    def _process_dataframe(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """Process raw klines data into proper format."""
        return (
            df.with_columns([
                pl.col("open_time").cast(pl.Int64),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
                pl.col("quote_volume").cast(pl.Float64),
                pl.col("trades").cast(pl.Int64),
                pl.col("taker_buy_volume").cast(pl.Float64),
                pl.col("taker_buy_quote_volume").cast(pl.Float64),
            ])
            .with_columns([
                (pl.col("open_time") * 1000)
                .cast(pl.Datetime("us"))
                .dt.replace_time_zone("UTC")
                .alias("timestamp"),
                pl.lit(symbol).alias("symbol"),
            ])
            .select([
                "timestamp",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "trades",
                "taker_buy_volume",
                "taker_buy_quote_volume",
            ])
            .sort("timestamp")
            .unique(subset=["timestamp", "symbol"], keep="last")
        )

    def validate(self, data: pl.DataFrame) -> bool:
        """Validate OHLCV data."""
        if data.is_empty():
            return False

        required_cols = [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        if not all(col in data.columns for col in required_cols):
            return False

        # Check for valid OHLC relationships
        invalid_ohlc = data.filter(
            (pl.col("high") < pl.col("low"))
            | (pl.col("high") < pl.col("open"))
            | (pl.col("high") < pl.col("close"))
            | (pl.col("low") > pl.col("open"))
            | (pl.col("low") > pl.col("close"))
        )

        if not invalid_ohlc.is_empty():
            logger.warning(f"Found {len(invalid_ohlc)} invalid OHLC rows")
            return False

        return True

    async def fetch_multiple(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        interval: str = "1m",
    ) -> dict[str, pl.DataFrame]:
        """Fetch OHLCV data for multiple symbols."""
        results = {}

        for symbol in symbols:
            self.logger.info(f"Fetching {symbol} from {start} to {end}")
            df = await self.fetch(symbol, start, end, interval)

            if self.validate(df):
                results[symbol] = df
                self.logger.info(f"Fetched {len(df)} rows for {symbol}")
            else:
                self.logger.warning(f"Invalid data for {symbol}")

        return results
