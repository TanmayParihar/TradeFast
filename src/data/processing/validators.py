"""
Data validation utilities.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import polars as pl
from loguru import logger


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    stats: dict[str, Any]

    def __bool__(self) -> bool:
        return self.is_valid


class DataValidator:
    """Validate market data quality."""

    def __init__(self):
        self.logger = logger.bind(module="validator")

    def validate_ohlcv(
        self,
        df: pl.DataFrame,
        symbol: str | None = None,
        min_rows: int = 100,
        max_gap_minutes: int = 60,
    ) -> ValidationResult:
        """
        Validate OHLCV data quality.

        Args:
            df: OHLCV DataFrame
            symbol: Expected symbol
            min_rows: Minimum required rows
            max_gap_minutes: Maximum allowed gap between timestamps

        Returns:
            ValidationResult with details
        """
        errors: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {}

        if df.is_empty():
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings, stats)

        # Check required columns
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")
            return ValidationResult(False, errors, warnings, stats)

        # Check minimum rows
        stats["row_count"] = df.height
        if df.height < min_rows:
            errors.append(f"Insufficient rows: {df.height} < {min_rows}")

        # Check symbol if specified
        if symbol and "symbol" in df.columns:
            unique_symbols = df["symbol"].unique().to_list()
            if symbol not in unique_symbols:
                errors.append(f"Symbol {symbol} not found in data")
            if len(unique_symbols) > 1:
                warnings.append(f"Multiple symbols found: {unique_symbols}")

        # Check OHLC relationships
        invalid_ohlc = df.filter(
            (pl.col("high") < pl.col("low"))
            | (pl.col("high") < pl.col("open"))
            | (pl.col("high") < pl.col("close"))
            | (pl.col("low") > pl.col("open"))
            | (pl.col("low") > pl.col("close"))
        )
        stats["invalid_ohlc_count"] = invalid_ohlc.height
        if invalid_ohlc.height > 0:
            warnings.append(f"Found {invalid_ohlc.height} invalid OHLC relationships")

        # Check for negative values
        negative_price = df.filter(
            (pl.col("open") < 0)
            | (pl.col("high") < 0)
            | (pl.col("low") < 0)
            | (pl.col("close") < 0)
        )
        if negative_price.height > 0:
            errors.append(f"Found {negative_price.height} negative prices")

        negative_volume = df.filter(pl.col("volume") < 0)
        if negative_volume.height > 0:
            warnings.append(f"Found {negative_volume.height} negative volumes")

        # Check for null values
        null_counts = {
            col: df[col].null_count()
            for col in required
        }
        stats["null_counts"] = null_counts
        total_nulls = sum(null_counts.values())
        if total_nulls > 0:
            warnings.append(f"Found {total_nulls} null values")

        # Check time gaps
        if df.height > 1:
            df_sorted = df.sort("timestamp")
            time_diffs = df_sorted.select([
                (pl.col("timestamp").diff().dt.total_minutes()).alias("gap_minutes")
            ])

            max_gap = time_diffs["gap_minutes"].max()
            stats["max_gap_minutes"] = max_gap

            if max_gap and max_gap > max_gap_minutes:
                warnings.append(f"Maximum time gap: {max_gap} minutes")

            # Count gaps
            gap_count = time_diffs.filter(pl.col("gap_minutes") > 1).height
            stats["gap_count"] = gap_count
            if gap_count > 0:
                warnings.append(f"Found {gap_count} time gaps > 1 minute")

        # Check date range
        stats["start_date"] = df["timestamp"].min()
        stats["end_date"] = df["timestamp"].max()

        # Check for duplicates
        duplicates = df.height - df.unique(subset=["timestamp"]).height
        stats["duplicate_count"] = duplicates
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate timestamps")

        # Price statistics
        stats["price_stats"] = {
            "min": df["close"].min(),
            "max": df["close"].max(),
            "mean": df["close"].mean(),
            "std": df["close"].std(),
        }

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, stats)

    def validate_orderbook(
        self,
        df: pl.DataFrame,
        min_levels: int = 5,
    ) -> ValidationResult:
        """Validate order book data."""
        errors: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {}

        if df.is_empty():
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings, stats)

        required = ["timestamp", "symbol", "side", "price", "quantity"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")
            return ValidationResult(False, errors, warnings, stats)

        # Check sides
        valid_sides = ["bid", "ask"]
        invalid_sides = df.filter(~pl.col("side").is_in(valid_sides))
        if invalid_sides.height > 0:
            errors.append("Invalid side values found")

        # Check for negative values
        if df.filter(pl.col("price") <= 0).height > 0:
            errors.append("Non-positive prices found")
        if df.filter(pl.col("quantity") < 0).height > 0:
            warnings.append("Negative quantities found")

        # Check level count
        level_counts = df.group_by(["timestamp", "symbol", "side"]).count()
        min_level_count = level_counts["count"].min()
        stats["min_levels"] = min_level_count

        if min_level_count and min_level_count < min_levels:
            warnings.append(f"Some snapshots have only {min_level_count} levels")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, stats)

    def validate_futures_metrics(self, df: pl.DataFrame) -> ValidationResult:
        """Validate futures metrics data."""
        errors: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {}

        if df.is_empty():
            warnings.append("DataFrame is empty")
            return ValidationResult(True, errors, warnings, stats)

        required = ["timestamp", "symbol"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")
            return ValidationResult(False, errors, warnings, stats)

        # Check funding rate range (typically -0.75% to 0.75%)
        if "funding_rate" in df.columns:
            extreme_funding = df.filter(
                (pl.col("funding_rate").abs() > 0.01)
            )
            if extreme_funding.height > 0:
                warnings.append(
                    f"Found {extreme_funding.height} extreme funding rates (>1%)"
                )

        # Check long/short ratio range (typically 0.5 to 2.0)
        if "long_short_ratio" in df.columns:
            stats["ls_ratio_range"] = {
                "min": df["long_short_ratio"].min(),
                "max": df["long_short_ratio"].max(),
            }

        stats["row_count"] = df.height

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, stats)

    def validate_sentiment(self, df: pl.DataFrame) -> ValidationResult:
        """Validate sentiment data."""
        errors: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {}

        if df.is_empty():
            warnings.append("DataFrame is empty (no sentiment data)")
            return ValidationResult(True, errors, warnings, stats)

        required = ["timestamp"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")

        stats["row_count"] = df.height

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, stats)

    def validate_features(
        self,
        df: pl.DataFrame,
        expected_features: list[str] | None = None,
    ) -> ValidationResult:
        """Validate feature DataFrame."""
        errors: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {}

        if df.is_empty():
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings, stats)

        # Check for expected features
        if expected_features:
            missing = [f for f in expected_features if f not in df.columns]
            if missing:
                errors.append(f"Missing features: {missing}")

        # Check for infinite values
        numeric_cols = [
            col for col in df.columns
            if df[col].dtype in [pl.Float64, pl.Float32]
        ]

        inf_counts = {}
        for col in numeric_cols:
            inf_count = df.filter(pl.col(col).is_infinite()).height
            if inf_count > 0:
                inf_counts[col] = inf_count

        if inf_counts:
            warnings.append(f"Infinite values found: {inf_counts}")

        # Check for excessive nulls
        null_pcts = {}
        for col in df.columns:
            null_pct = df[col].null_count() / df.height
            if null_pct > 0.1:  # More than 10% null
                null_pcts[col] = f"{null_pct:.1%}"

        if null_pcts:
            warnings.append(f"High null percentages: {null_pcts}")

        stats["feature_count"] = len(df.columns)
        stats["row_count"] = df.height

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, stats)
