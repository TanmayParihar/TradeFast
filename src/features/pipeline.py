"""
Feature engineering pipeline for combining all feature sources.
"""

from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from src.features.technical import TechnicalFeatures
from src.features.orderbook_features import OrderBookFeatures
from src.features.onchain_features import OnChainFeatures
from src.features.sentiment_features import SentimentFeatures


class FeaturePipeline:
    """Pipeline for generating all features from multi-modal data."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logger.bind(module="feature_pipeline")

        # Initialize feature generators
        self.technical = TechnicalFeatures(config.get("technical", {}))
        self.orderbook = OrderBookFeatures(config.get("orderbook", {}))
        self.onchain = OnChainFeatures(config.get("onchain", {}))
        self.sentiment = SentimentFeatures(
            config.get("sentiment", {}),
            use_gpu=config.get("use_gpu", True),
        )

    def run(
        self,
        ohlcv_df: pl.DataFrame,
        orderbook_df: pl.DataFrame | None = None,
        futures_df: pl.DataFrame | None = None,
        news_df: pl.DataFrame | None = None,
        fear_greed_df: pl.DataFrame | None = None,
        reddit_df: pl.DataFrame | None = None,
        analyze_sentiment: bool = False,
    ) -> pl.DataFrame:
        """
        Run the full feature engineering pipeline.

        Args:
            ohlcv_df: Base OHLCV data
            orderbook_df: Order book metrics
            futures_df: Futures metrics (funding, OI, etc.)
            news_df: News data
            fear_greed_df: Fear & Greed Index
            reddit_df: Reddit posts
            analyze_sentiment: Whether to run NLP sentiment analysis

        Returns:
            DataFrame with all features
        """
        self.logger.info("Starting feature pipeline...")

        if ohlcv_df.is_empty():
            raise ValueError("OHLCV data is required")

        # Start with OHLCV
        df = ohlcv_df.sort("timestamp")
        initial_cols = len(df.columns)

        # 1. Technical features
        self.logger.info("Adding technical features...")
        df = self.technical.add_all_features(df)
        self.logger.info(f"Technical features added: {len(df.columns) - initial_cols}")

        # 2. Order book features
        if orderbook_df is not None and not orderbook_df.is_empty():
            self.logger.info("Adding order book features...")
            before = len(df.columns)
            df = self.orderbook.add_orderbook_features_to_df(df, orderbook_df)
            self.logger.info(f"Order book features added: {len(df.columns) - before}")

        # 3. On-chain/futures features
        if futures_df is not None and not futures_df.is_empty():
            self.logger.info("Merging futures metrics...")
            before = len(df.columns)
            df = self._merge_futures_data(df, futures_df)
            self.logger.info(f"Futures columns merged: {len(df.columns) - before}")

        # Add on-chain derived features
        self.logger.info("Adding on-chain features...")
        before = len(df.columns)
        df = self.onchain.add_all_features(df)
        self.logger.info(f"On-chain features added: {len(df.columns) - before}")

        # 4. Sentiment features
        if fear_greed_df is not None and not fear_greed_df.is_empty():
            self.logger.info("Adding Fear & Greed features...")
            before = len(df.columns)
            df = self.sentiment.add_fear_greed_features(df, fear_greed_df)
            self.logger.info(f"F&G features added: {len(df.columns) - before}")

        if news_df is not None and not news_df.is_empty():
            self.logger.info("Adding news features...")
            before = len(df.columns)
            df = self.sentiment.add_news_features(df, news_df, analyze_text=analyze_sentiment)
            self.logger.info(f"News features added: {len(df.columns) - before}")

        if reddit_df is not None and not reddit_df.is_empty():
            self.logger.info("Adding Reddit features...")
            before = len(df.columns)
            df = self.sentiment.add_reddit_features(df, reddit_df, analyze_text=analyze_sentiment)
            self.logger.info(f"Reddit features added: {len(df.columns) - before}")

        # Create sentiment composite
        df = self.sentiment.create_sentiment_composite(df)

        # 5. Cross-modal features
        self.logger.info("Adding cross-modal features...")
        before = len(df.columns)
        df = self._add_cross_modal_features(df)
        self.logger.info(f"Cross-modal features added: {len(df.columns) - before}")

        # 6. Clean up
        df = self._finalize_features(df)

        self.logger.info(
            f"Feature pipeline complete. "
            f"Total features: {len(df.columns)}, "
            f"Rows: {len(df)}"
        )

        return df

    def _merge_futures_data(
        self, df: pl.DataFrame, futures_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Merge futures metrics with OHLCV data."""
        # Round timestamps
        futures_df = futures_df.with_columns([
            pl.col("timestamp").dt.truncate("1m").alias("timestamp_minute")
        ])

        # Get available columns
        futures_cols = [
            c for c in futures_df.columns
            if c not in ["timestamp", "symbol", "timestamp_minute"]
        ]

        if not futures_cols:
            return df

        # Aggregate to minute (take last value)
        futures_agg = (
            futures_df.group_by("timestamp_minute")
            .agg([pl.col(c).last() for c in futures_cols])
            .rename({"timestamp_minute": "timestamp"})
            .sort("timestamp")
        )

        # Asof join (forward fill)
        return df.join_asof(
            futures_agg,
            on="timestamp",
            strategy="backward",
        )

    def _add_cross_modal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add features that combine multiple data sources."""
        # Price-Sentiment divergence
        if "returns_60" in df.columns and "sentiment_composite" in df.columns:
            df = df.with_columns([
                (
                    pl.col("returns_60").sign().cast(pl.Float64)
                    - pl.col("sentiment_composite").sign().cast(pl.Float64)
                ).alias("price_sentiment_divergence")
            ])

        # Volatility-adjusted sentiment
        if "volatility_20" in df.columns and "sentiment_composite" in df.columns:
            df = df.with_columns([
                (
                    pl.col("sentiment_composite")
                    / (pl.col("volatility_20") * 100 + 1)
                ).alias("sentiment_vol_adjusted")
            ])

        # OI-Volume relationship
        if "oi_change" in df.columns and "volume_ratio" in df.columns:
            df = df.with_columns([
                (pl.col("oi_change") * pl.col("volume_ratio")).alias("oi_volume_interaction")
            ])

        # Funding-Returns interaction
        if "funding_rate" in df.columns and "returns" in df.columns:
            df = df.with_columns([
                (pl.col("funding_rate") * pl.col("returns") * 10000).alias("funding_returns_interaction")
            ])

        # Order book-Price momentum interaction
        if "ob_imbalance" in df.columns and "roc_10" in df.columns:
            df = df.with_columns([
                (pl.col("ob_imbalance") * pl.col("roc_10")).alias("ob_momentum_interaction")
            ])

        return df

    def _finalize_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean up and finalize features."""
        # Handle infinities
        numeric_cols = [
            col for col in df.columns
            if df[col].dtype in [pl.Float64, pl.Float32]
        ]

        for col in numeric_cols:
            df = df.with_columns([
                pl.when(pl.col(col).is_infinite())
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            ])

        # Forward fill remaining nulls
        for col in numeric_cols:
            null_count = df[col].null_count()
            if null_count > 0:
                df = df.with_columns([
                    pl.col(col).fill_null(strategy="forward").fill_null(0)
                ])

        return df

    def get_feature_names(self, category: str | None = None) -> list[str]:
        """Get list of feature names by category."""
        technical_features = [
            "returns", "log_returns", "returns_5", "returns_15", "returns_60", "returns_240",
            "rsi_14", "rsi_7", "rsi_21", "macd", "macd_signal", "macd_hist",
            "stoch_k", "stoch_d", "bb_position", "bb_width", "atr", "atr_pct",
            "volatility_20", "volatility_60", "sma_10", "sma_20", "sma_50", "sma_200",
            "ema_10", "ema_20", "ema_50", "adx", "plus_di", "minus_di",
            "volume_ratio", "obv", "vwap_20",
        ]

        orderbook_features = [
            "ob_bid_ask_spread", "ob_bid_ask_spread_bps", "ob_imbalance",
            "ob_microprice", "ob_depth_imbalance",
        ]

        onchain_features = [
            "funding_rate", "funding_rate_zscore", "funding_regime",
            "oi_change", "oi_change_1h", "oi_change_24h", "oi_ma_deviation",
            "long_short_ratio", "ls_ratio_zscore", "taker_buy_sell_ratio",
        ]

        sentiment_features = [
            "fear_greed_value", "fear_greed_normalized", "fg_regime",
            "news_sentiment_avg", "news_count", "reddit_engagement",
            "sentiment_composite", "sentiment_regime",
        ]

        temporal_features = [
            "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
            "month_sin", "month_cos",
        ]

        categories = {
            "technical": technical_features,
            "orderbook": orderbook_features,
            "onchain": onchain_features,
            "sentiment": sentiment_features,
            "temporal": temporal_features,
        }

        if category:
            return categories.get(category, [])

        # Return all features
        all_features = []
        for features in categories.values():
            all_features.extend(features)
        return all_features

    def save_features(
        self,
        df: pl.DataFrame,
        path: str | Path,
        partition_by: str | None = None,
    ) -> None:
        """Save features to Parquet."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if partition_by and partition_by in df.columns:
            df.write_parquet(path, partition_by=partition_by)
        else:
            df.write_parquet(path)

        self.logger.info(f"Features saved to {path}")

    def load_features(
        self,
        path: str | Path,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Load features from Parquet."""
        return pl.read_parquet(path, columns=columns)
