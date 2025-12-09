"""
Fear & Greed Index collector.
"""

from datetime import datetime, timezone
from typing import Any

import aiohttp
import polars as pl

from src.data.ingestion.base import DataSource


class FearGreedCollector(DataSource):
    """Collector for Alternative.me Fear & Greed Index."""

    BASE_URL = "https://api.alternative.me/fng"

    def __init__(self):
        super().__init__("fear_greed")

    async def fetch(
        self,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 365,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        Fetch Fear & Greed Index data.

        Args:
            symbol: Not used (index is for overall crypto market)
            start: Start datetime (filtered locally)
            end: End datetime (filtered locally)
            limit: Number of days to fetch (max 365)

        Returns:
            DataFrame with Fear & Greed data
        """
        async with aiohttp.ClientSession() as session:
            params = {"limit": min(limit, 365), "format": "json"}

            async with session.get(self.BASE_URL, params=params) as response:
                response.raise_for_status()
                data = await response.json()

        if "data" not in data:
            self.logger.error("No data in response")
            return pl.DataFrame()

        df = self._process_data(data["data"])

        # Filter by date range
        if start:
            df = df.filter(pl.col("timestamp") >= start)
        if end:
            df = df.filter(pl.col("timestamp") <= end)

        return df

    def _process_data(self, data: list[dict]) -> pl.DataFrame:
        """Process raw Fear & Greed data."""
        rows = []

        for item in data:
            timestamp = datetime.fromtimestamp(
                int(item.get("timestamp", 0)), tz=timezone.utc
            )
            value = int(item.get("value", 50))
            classification = item.get("value_classification", "Neutral")

            rows.append({
                "timestamp": timestamp,
                "fear_greed_value": value,
                "fear_greed_classification": classification,
                "fear_greed_normalized": value / 100,  # 0-1 scale
            })

        return pl.DataFrame(rows).sort("timestamp")

    def validate(self, data: pl.DataFrame) -> bool:
        """Validate Fear & Greed data."""
        if data.is_empty():
            return False

        required_cols = ["timestamp", "fear_greed_value"]
        if not all(col in data.columns for col in required_cols):
            return False

        # Check value range
        invalid = data.filter(
            (pl.col("fear_greed_value") < 0) | (pl.col("fear_greed_value") > 100)
        )

        return invalid.is_empty()

    @staticmethod
    def classify_value(value: int) -> str:
        """Classify Fear & Greed value."""
        if value <= 20:
            return "Extreme Fear"
        elif value <= 40:
            return "Fear"
        elif value <= 60:
            return "Neutral"
        elif value <= 80:
            return "Greed"
        else:
            return "Extreme Greed"

    def resample_to_minute(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Resample daily Fear & Greed data to minute frequency.

        Forward fills the daily value across all minutes of the day.
        """
        if df.is_empty():
            return df

        # Create minute range for each day
        min_date = df["timestamp"].min()
        max_date = df["timestamp"].max()

        # Generate minute timestamps
        minute_range = pl.datetime_range(
            min_date,
            max_date,
            interval="1m",
            eager=True,
        )

        minute_df = pl.DataFrame({"timestamp": minute_range})

        # Add date column for joining
        minute_df = minute_df.with_columns([
            pl.col("timestamp").dt.date().alias("date")
        ])

        df = df.with_columns([
            pl.col("timestamp").dt.date().alias("date")
        ])

        # Join and forward fill
        result = minute_df.join(
            df.select(["date", "fear_greed_value", "fear_greed_normalized"]),
            on="date",
            how="left",
        )

        return result.select([
            "timestamp",
            "fear_greed_value",
            "fear_greed_normalized",
        ]).fill_null(strategy="forward")
