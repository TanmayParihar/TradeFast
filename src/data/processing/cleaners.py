"""
Data cleaning utilities.
"""

from typing import Any

import numpy as np
import polars as pl
from loguru import logger


class DataCleaner:
    """Clean and preprocess market data."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logger.bind(module="cleaner")

    def clean_ohlcv(
        self,
        df: pl.DataFrame,
        fill_gaps: bool = True,
        remove_outliers: bool = True,
        outlier_std: float = 5.0,
    ) -> pl.DataFrame:
        """
        Clean OHLCV data.

        Args:
            df: Raw OHLCV DataFrame
            fill_gaps: Whether to fill missing timestamps
            remove_outliers: Whether to remove price outliers
            outlier_std: Number of standard deviations for outlier detection

        Returns:
            Cleaned DataFrame
        """
        if df.is_empty():
            return df

        df = self._ensure_sorted(df)
        df = self._fix_ohlc_relationships(df)

        if remove_outliers:
            df = self._remove_price_outliers(df, outlier_std)

        if fill_gaps:
            df = self._fill_time_gaps(df)

        df = self._handle_missing_values(df)
        df = self._remove_duplicates(df)

        return df

    def _ensure_sorted(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure data is sorted by timestamp."""
        return df.sort("timestamp")

    def _fix_ohlc_relationships(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fix invalid OHLC relationships."""
        return df.with_columns([
            # High should be >= Open, Close, Low
            pl.max_horizontal("open", "high", "close").alias("high"),
            # Low should be <= Open, Close, High
            pl.min_horizontal("open", "low", "close").alias("low"),
        ])

    def _remove_price_outliers(
        self, df: pl.DataFrame, n_std: float
    ) -> pl.DataFrame:
        """Remove price outliers based on rolling statistics."""
        # Calculate rolling mean and std
        window = 60  # 1 hour for 1-min data

        df = df.with_columns([
            pl.col("close").rolling_mean(window).alias("_rolling_mean"),
            pl.col("close").rolling_std(window).alias("_rolling_std"),
        ])

        # Mark outliers
        df = df.with_columns([
            (
                (pl.col("close") > pl.col("_rolling_mean") + n_std * pl.col("_rolling_std"))
                | (pl.col("close") < pl.col("_rolling_mean") - n_std * pl.col("_rolling_std"))
            ).alias("_is_outlier")
        ])

        # Count outliers
        n_outliers = df.filter(pl.col("_is_outlier")).height
        if n_outliers > 0:
            self.logger.info(f"Removing {n_outliers} price outliers")

        # Replace outliers with rolling mean
        df = df.with_columns([
            pl.when(pl.col("_is_outlier"))
            .then(pl.col("_rolling_mean"))
            .otherwise(pl.col("close"))
            .alias("close"),
            pl.when(pl.col("_is_outlier"))
            .then(pl.col("_rolling_mean"))
            .otherwise(pl.col("open"))
            .alias("open"),
            pl.when(pl.col("_is_outlier"))
            .then(pl.col("_rolling_mean"))
            .otherwise(pl.col("high"))
            .alias("high"),
            pl.when(pl.col("_is_outlier"))
            .then(pl.col("_rolling_mean"))
            .otherwise(pl.col("low"))
            .alias("low"),
        ])

        # Drop temporary columns
        return df.drop(["_rolling_mean", "_rolling_std", "_is_outlier"])

    def _fill_time_gaps(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill gaps in time series data."""
        if df.is_empty():
            return df

        # Get time range
        min_time = df["timestamp"].min()
        max_time = df["timestamp"].max()

        # Create complete time range (1-minute intervals)
        complete_range = pl.datetime_range(
            min_time,
            max_time,
            interval="1m",
            eager=True,
        )

        complete_df = pl.DataFrame({"timestamp": complete_range})

        # Join with original data
        result = complete_df.join(df, on="timestamp", how="left")

        # Forward fill prices, set volume to 0 for gaps
        result = result.with_columns([
            pl.col("symbol").fill_null(strategy="forward"),
            pl.col("open").fill_null(strategy="forward"),
            pl.col("high").fill_null(strategy="forward"),
            pl.col("low").fill_null(strategy="forward"),
            pl.col("close").fill_null(strategy="forward"),
            pl.col("volume").fill_null(0),
        ])

        # For gaps, set OHLC to previous close
        result = result.with_columns([
            pl.when(pl.col("open").is_null())
            .then(pl.col("close").shift(1))
            .otherwise(pl.col("open"))
            .alias("open"),
        ])

        n_gaps = complete_df.height - df.height
        if n_gaps > 0:
            self.logger.info(f"Filled {n_gaps} time gaps")

        return result

    def _handle_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle missing values in the dataset."""
        # Forward fill most columns
        numeric_cols = [
            col for col in df.columns
            if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]

        for col in numeric_cols:
            df = df.with_columns([
                pl.col(col).fill_null(strategy="forward").fill_null(0)
            ])

        return df

    def _remove_duplicates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove duplicate timestamps."""
        before = df.height
        df = df.unique(subset=["timestamp"], keep="last")
        after = df.height

        if before > after:
            self.logger.info(f"Removed {before - after} duplicate rows")

        return df

    def clean_orderbook(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean order book data."""
        if df.is_empty():
            return df

        # Remove invalid prices/quantities
        df = df.filter(
            (pl.col("price") > 0) & (pl.col("quantity") > 0)
        )

        return df.sort(["timestamp", "side", "level"])

    def clean_sentiment(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean sentiment data."""
        if df.is_empty():
            return df

        # Remove empty titles
        df = df.filter(pl.col("title").str.len_chars() > 0)

        return df.sort("timestamp")

    def normalize_returns(
        self, df: pl.DataFrame, method: str = "log"
    ) -> pl.DataFrame:
        """Add normalized returns column."""
        if method == "log":
            return df.with_columns([
                (pl.col("close") / pl.col("close").shift(1)).log().alias("returns")
            ])
        else:
            return df.with_columns([
                (pl.col("close") / pl.col("close").shift(1) - 1).alias("returns")
            ])


class VolumeCleaner:
    """Clean and process volume data."""

    def __init__(self):
        self.logger = logger.bind(module="volume_cleaner")

    def clean_volume(
        self,
        df: pl.DataFrame,
        remove_zero: bool = False,
        cap_outliers: bool = True,
        outlier_quantile: float = 0.999,
    ) -> pl.DataFrame:
        """Clean volume data."""
        if df.is_empty():
            return df

        # Handle negative volumes
        df = df.with_columns([
            pl.col("volume").abs()
        ])

        if remove_zero:
            df = df.filter(pl.col("volume") > 0)

        if cap_outliers:
            cap_value = df["volume"].quantile(outlier_quantile)
            df = df.with_columns([
                pl.when(pl.col("volume") > cap_value)
                .then(cap_value)
                .otherwise(pl.col("volume"))
                .alias("volume")
            ])

        return df

    def compute_dollar_volume(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add dollar volume column."""
        return df.with_columns([
            (pl.col("volume") * (pl.col("high") + pl.col("low")) / 2)
            .alias("dollar_volume")
        ])
