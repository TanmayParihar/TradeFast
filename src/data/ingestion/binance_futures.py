"""
Binance Futures data collector for on-chain metrics.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

import aiohttp
import polars as pl

from src.data.ingestion.base import DataSource, RateLimiter, RetryHandler


class BinanceFuturesCollector(DataSource):
    """Collector for Binance Futures metrics (funding, OI, long/short ratio)."""

    BASE_URL = "https://fapi.binance.com"

    ENDPOINTS = {
        "funding_rate": "/fapi/v1/fundingRate",
        "open_interest": "/fapi/v1/openInterest",
        "open_interest_hist": "/futures/data/openInterestHist",
        "long_short_ratio": "/futures/data/globalLongShortAccountRatio",
        "top_trader_ls_ratio": "/futures/data/topLongShortPositionRatio",
        "top_trader_ls_account": "/futures/data/topLongShortAccountRatio",
        "taker_buy_sell": "/futures/data/takerlongshortRatio",
    }

    def __init__(self, testnet: bool = False):
        super().__init__("binance_futures")
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
        metric: str = "all",
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        Fetch futures metrics.

        Args:
            symbol: Trading symbol
            start: Start datetime
            end: End datetime
            metric: Metric to fetch ('funding_rate', 'open_interest', etc.)

        Returns:
            DataFrame with requested metrics
        """
        if symbol is None:
            raise ValueError("Symbol is required")

        if metric == "all":
            return await self._fetch_all_metrics(symbol, start, end)

        if metric not in self.ENDPOINTS:
            raise ValueError(f"Unknown metric: {metric}")

        async with aiohttp.ClientSession() as session:
            data = await self._fetch_metric(session, symbol, metric, start, end)
            return self._process_metric(data, symbol, metric)

    async def _fetch_all_metrics(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Fetch all available metrics."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            metrics = [
                "funding_rate",
                "open_interest_hist",
                "long_short_ratio",
                "top_trader_ls_ratio",
                "taker_buy_sell",
            ]

            for metric in metrics:
                tasks.append(self._fetch_metric(session, symbol, metric, start, end))

            results = await asyncio.gather(*tasks, return_exceptions=True)

        dfs = []
        for metric, result in zip(metrics, results):
            if isinstance(result, Exception):
                self.logger.warning(f"Failed to fetch {metric}: {result}")
                continue

            df = self._process_metric(result, symbol, metric)
            if not df.is_empty():
                dfs.append(df)

        if not dfs:
            return pl.DataFrame()

        # Merge all metrics on timestamp
        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(
                df.drop("symbol"),
                on="timestamp",
                how="outer",
            )

        return result.sort("timestamp")

    async def _fetch_metric(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        metric: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list:
        """Fetch a single metric."""
        await self.rate_limiter.acquire()

        url = f"{self.base_url}{self.ENDPOINTS[metric]}"
        params: dict[str, Any] = {"symbol": symbol, "limit": 500}

        if metric in ["long_short_ratio", "top_trader_ls_ratio", "taker_buy_sell"]:
            params["period"] = "1h"

        if start:
            params["startTime"] = int(start.timestamp() * 1000)
        if end:
            # Cap end time to current time (API rejects future dates)
            now = datetime.now(timezone.utc)
            end_capped = min(end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end, now)
            params["endTime"] = int(end_capped.timestamp() * 1000)

        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()

        # Handle single object response (current OI)
        if isinstance(data, dict):
            data = [data]

        return data

    def _process_metric(
        self, data: list, symbol: str, metric: str
    ) -> pl.DataFrame:
        """Process metric data into DataFrame."""
        if not data:
            return pl.DataFrame()

        rows = []
        for item in data:
            row = {"symbol": symbol}

            # Parse timestamp
            if "fundingTime" in item:
                row["timestamp"] = datetime.fromtimestamp(
                    item["fundingTime"] / 1000, tz=timezone.utc
                )
            elif "timestamp" in item:
                row["timestamp"] = datetime.fromtimestamp(
                    item["timestamp"] / 1000, tz=timezone.utc
                )
            elif "time" in item:
                row["timestamp"] = datetime.fromtimestamp(
                    item["time"] / 1000, tz=timezone.utc
                )
            else:
                row["timestamp"] = datetime.now(timezone.utc)

            # Parse metric-specific fields
            if metric == "funding_rate":
                row["funding_rate"] = float(item.get("fundingRate", 0))
                row["mark_price"] = float(item.get("markPrice", 0))

            elif metric == "open_interest":
                row["open_interest"] = float(item.get("openInterest", 0))

            elif metric == "open_interest_hist":
                row["open_interest"] = float(
                    item.get("sumOpenInterest", item.get("openInterest", 0))
                )
                row["open_interest_value"] = float(
                    item.get("sumOpenInterestValue", 0)
                )

            elif metric == "long_short_ratio":
                row["long_short_ratio"] = float(item.get("longShortRatio", 0))
                row["long_account"] = float(item.get("longAccount", 0))
                row["short_account"] = float(item.get("shortAccount", 0))

            elif metric == "top_trader_ls_ratio":
                row["top_trader_ls_ratio"] = float(item.get("longShortRatio", 0))
                row["top_long_account"] = float(item.get("longAccount", 0))
                row["top_short_account"] = float(item.get("shortAccount", 0))

            elif metric == "taker_buy_sell":
                row["taker_buy_sell_ratio"] = float(item.get("buySellRatio", 0))
                row["buy_vol"] = float(item.get("buyVol", 0))
                row["sell_vol"] = float(item.get("sellVol", 0))

            rows.append(row)

        return pl.DataFrame(rows)

    def validate(self, data: pl.DataFrame) -> bool:
        """Validate futures metrics data."""
        if data.is_empty():
            return False

        required_cols = ["timestamp", "symbol"]
        return all(col in data.columns for col in required_cols)

    async def get_current_funding_rate(self, symbol: str) -> float:
        """Get current funding rate for symbol."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/fapi/v1/premiumIndex"
            params = {"symbol": symbol}

            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

            return float(data.get("lastFundingRate", 0))

    async def get_current_open_interest(self, symbol: str) -> dict:
        """Get current open interest for symbol."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/fapi/v1/openInterest"
            params = {"symbol": symbol}

            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

            return {
                "symbol": symbol,
                "open_interest": float(data.get("openInterest", 0)),
                "timestamp": datetime.now(timezone.utc),
            }

    async def fetch_multiple(
        self,
        symbols: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, pl.DataFrame]:
        """Fetch metrics for multiple symbols."""
        results = {}

        for symbol in symbols:
            self.logger.info(f"Fetching futures metrics for {symbol}")
            df = await self.fetch(symbol, start, end, metric="all")

            if self.validate(df):
                results[symbol] = df
                self.logger.info(f"Fetched {len(df)} rows for {symbol}")
            else:
                self.logger.warning(f"Invalid data for {symbol}")

        return results
