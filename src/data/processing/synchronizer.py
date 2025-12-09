"""
Data synchronization for multi-modal data fusion.
"""

from datetime import datetime, timedelta
from typing import Any

import polars as pl
from loguru import logger


class DataSynchronizer:
    """Synchronize multi-modal data to common timestamps."""

    def __init__(self, base_interval: str = "1m"):
        """
        Initialize synchronizer.

        Args:
            base_interval: Base time interval for alignment ('1m', '5m', '1h')
        """
        self.base_interval = base_interval
        self.logger = logger.bind(module="synchronizer")

    def synchronize(
        self,
        ohlcv: pl.DataFrame,
        orderbook: pl.DataFrame | None = None,
        futures_metrics: pl.DataFrame | None = None,
        sentiment: pl.DataFrame | None = None,
        fear_greed: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """
        Synchronize all data sources to OHLCV timestamps.

        Args:
            ohlcv: Base OHLCV data (required)
            orderbook: Order book metrics
            futures_metrics: Funding rate, OI, etc.
            sentiment: News/Reddit sentiment
            fear_greed: Fear & Greed Index

        Returns:
            Synchronized DataFrame with all features
        """
        if ohlcv.is_empty():
            raise ValueError("OHLCV data is required")

        result = ohlcv.sort("timestamp")
        self.logger.info(f"Starting synchronization with {result.height} OHLCV rows")

        # Merge order book data
        if orderbook is not None and not orderbook.is_empty():
            result = self._merge_orderbook(result, orderbook)
            self.logger.debug("Merged order book data")

        # Merge futures metrics
        if futures_metrics is not None and not futures_metrics.is_empty():
            result = self._merge_futures_metrics(result, futures_metrics)
            self.logger.debug("Merged futures metrics")

        # Merge sentiment data
        if sentiment is not None and not sentiment.is_empty():
            result = self._merge_sentiment(result, sentiment)
            self.logger.debug("Merged sentiment data")

        # Merge Fear & Greed Index
        if fear_greed is not None and not fear_greed.is_empty():
            result = self._merge_fear_greed(result, fear_greed)
            self.logger.debug("Merged Fear & Greed data")

        # Forward fill sparse data
        result = self._forward_fill(result)

        self.logger.info(
            f"Synchronization complete: {result.height} rows, "
            f"{len(result.columns)} columns"
        )

        return result

    def _merge_orderbook(
        self, base: pl.DataFrame, orderbook: pl.DataFrame
    ) -> pl.DataFrame:
        """Merge order book metrics."""
        # Aggregate order book to minute level if needed
        ob_cols = [
            "mid_price",
            "spread",
            "bid_ask_spread_bps",
            "imbalance",
            "bid_volume",
            "ask_volume",
            "depth_imbalance",
            "microprice",
        ]

        available_cols = [c for c in ob_cols if c in orderbook.columns]

        if not available_cols:
            return base

        # Round timestamps to minute
        orderbook = orderbook.with_columns([
            pl.col("timestamp").dt.truncate("1m").alias("timestamp_minute")
        ])

        # Take last value per minute
        ob_agg = (
            orderbook.group_by("timestamp_minute")
            .agg([pl.col(c).last().alias(f"ob_{c}") for c in available_cols])
            .rename({"timestamp_minute": "timestamp"})
        )

        return base.join(ob_agg, on="timestamp", how="left")

    def _merge_futures_metrics(
        self, base: pl.DataFrame, futures: pl.DataFrame
    ) -> pl.DataFrame:
        """Merge futures metrics (funding, OI, etc.)."""
        metric_cols = [
            "funding_rate",
            "mark_price",
            "open_interest",
            "open_interest_value",
            "long_short_ratio",
            "top_trader_ls_ratio",
            "taker_buy_sell_ratio",
        ]

        available_cols = [c for c in metric_cols if c in futures.columns]

        if not available_cols:
            return base

        # Round timestamps to minute
        futures = futures.with_columns([
            pl.col("timestamp").dt.truncate("1m").alias("timestamp_minute")
        ])

        # Use asof join for irregular data
        futures_agg = (
            futures.group_by("timestamp_minute")
            .agg([pl.col(c).last() for c in available_cols])
            .rename({"timestamp_minute": "timestamp"})
            .sort("timestamp")
        )

        return base.join_asof(
            futures_agg,
            on="timestamp",
            strategy="backward",
        )

    def _merge_sentiment(
        self, base: pl.DataFrame, sentiment: pl.DataFrame
    ) -> pl.DataFrame:
        """Merge sentiment data."""
        # Aggregate sentiment to hourly level, then forward fill
        sentiment = sentiment.with_columns([
            pl.col("timestamp").dt.truncate("1h").alias("timestamp_hour")
        ])

        # Compute aggregate sentiment metrics
        sentiment_agg = sentiment.group_by("timestamp_hour").agg([
            pl.count().alias("news_count"),
            pl.col("sentiment_score").mean().alias("sentiment_score_avg")
            if "sentiment_score" in sentiment.columns
            else pl.lit(0.0).alias("sentiment_score_avg"),
        ])

        # Expand to minute level
        base = base.with_columns([
            pl.col("timestamp").dt.truncate("1h").alias("timestamp_hour")
        ])

        result = base.join(
            sentiment_agg.rename({"timestamp_hour": "_hour"}),
            left_on="timestamp_hour",
            right_on="_hour",
            how="left",
        ).drop("timestamp_hour")

        return result

    def _merge_fear_greed(
        self, base: pl.DataFrame, fear_greed: pl.DataFrame
    ) -> pl.DataFrame:
        """Merge Fear & Greed Index."""
        fg_cols = ["fear_greed_value", "fear_greed_normalized"]
        available_cols = [c for c in fg_cols if c in fear_greed.columns]

        if not available_cols:
            return base

        # Fear & Greed is daily, expand to minute
        fear_greed = fear_greed.with_columns([
            pl.col("timestamp").dt.date().alias("date")
        ])

        base = base.with_columns([
            pl.col("timestamp").dt.date().alias("date")
        ])

        fg_daily = (
            fear_greed.group_by("date")
            .agg([pl.col(c).first() for c in available_cols])
        )

        result = base.join(fg_daily, on="date", how="left").drop("date")

        return result

    def _forward_fill(self, df: pl.DataFrame) -> pl.DataFrame:
        """Forward fill sparse columns."""
        sparse_prefixes = ["ob_", "funding", "open_interest", "fear_greed", "sentiment"]

        for col in df.columns:
            if any(col.startswith(p) or p in col for p in sparse_prefixes):
                df = df.with_columns([
                    pl.col(col).fill_null(strategy="forward")
                ])

        return df

    def resample(
        self,
        df: pl.DataFrame,
        interval: str,
        agg_method: str = "ohlc",
    ) -> pl.DataFrame:
        """
        Resample data to different timeframe.

        Args:
            df: Input DataFrame with 'timestamp' column
            interval: Target interval ('5m', '15m', '1h', '4h', '1d')
            agg_method: Aggregation method ('ohlc', 'last', 'mean')

        Returns:
            Resampled DataFrame
        """
        if agg_method == "ohlc":
            return (
                df.group_by_dynamic("timestamp", every=interval)
                .agg([
                    pl.col("symbol").first(),
                    pl.col("open").first(),
                    pl.col("high").max(),
                    pl.col("low").min(),
                    pl.col("close").last(),
                    pl.col("volume").sum(),
                    # Other columns: take last value
                    *[
                        pl.col(c).last()
                        for c in df.columns
                        if c not in ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
                    ],
                ])
                .sort("timestamp")
            )

        elif agg_method == "last":
            return (
                df.group_by_dynamic("timestamp", every=interval)
                .agg([pl.col(c).last() for c in df.columns if c != "timestamp"])
                .sort("timestamp")
            )

        else:  # mean
            return (
                df.group_by_dynamic("timestamp", every=interval)
                .agg([
                    pl.col(c).mean() if df[c].dtype in [pl.Float64, pl.Float32] else pl.col(c).last()
                    for c in df.columns
                    if c != "timestamp"
                ])
                .sort("timestamp")
            )

    def align_multi_symbol(
        self,
        data_dict: dict[str, pl.DataFrame],
        fill_method: str = "forward",
    ) -> pl.DataFrame:
        """
        Align data from multiple symbols to common timestamps.

        Args:
            data_dict: Dictionary of symbol -> DataFrame
            fill_method: How to handle missing data ('forward', 'null')

        Returns:
            Wide-format DataFrame with all symbols
        """
        if not data_dict:
            return pl.DataFrame()

        # Get union of all timestamps
        all_timestamps = set()
        for df in data_dict.values():
            all_timestamps.update(df["timestamp"].to_list())

        all_timestamps = sorted(all_timestamps)
        base = pl.DataFrame({"timestamp": all_timestamps})

        # Merge each symbol
        for symbol, df in data_dict.items():
            # Rename columns to include symbol
            renamed = {}
            for col in df.columns:
                if col != "timestamp":
                    renamed[col] = f"{symbol}_{col}"

            df_renamed = df.rename(renamed)
            base = base.join(df_renamed, on="timestamp", how="left")

        if fill_method == "forward":
            for col in base.columns:
                if col != "timestamp":
                    base = base.with_columns([
                        pl.col(col).fill_null(strategy="forward")
                    ])

        return base.sort("timestamp")
