"""
Feature registry for tracking and managing features.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import polars as pl
from loguru import logger


class FeatureType(Enum):
    """Types of features."""
    TECHNICAL = "technical"
    ORDERBOOK = "orderbook"
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"
    TEMPORAL = "temporal"
    CROSS_MODAL = "cross_modal"
    TARGET = "target"


@dataclass
class FeatureMetadata:
    """Metadata for a feature."""
    name: str
    type: FeatureType
    description: str
    compute_fn: Callable | None = None
    dependencies: list[str] = field(default_factory=list)
    is_target: bool = False
    lookback: int = 0
    fillna_strategy: str = "forward"
    normalize: bool = False
    clip_outliers: bool = False
    outlier_std: float = 5.0


class FeatureRegistry:
    """Registry for tracking and managing features."""

    def __init__(self):
        self._features: dict[str, FeatureMetadata] = {}
        self._target_features: list[str] = []
        self.logger = logger.bind(module="feature_registry")

        # Register default features
        self._register_default_features()

    def register(
        self,
        name: str,
        feature_type: FeatureType,
        description: str,
        compute_fn: Callable | None = None,
        dependencies: list[str] | None = None,
        is_target: bool = False,
        lookback: int = 0,
        fillna_strategy: str = "forward",
        normalize: bool = False,
        clip_outliers: bool = False,
        outlier_std: float = 5.0,
    ) -> None:
        """Register a new feature."""
        metadata = FeatureMetadata(
            name=name,
            type=feature_type,
            description=description,
            compute_fn=compute_fn,
            dependencies=dependencies or [],
            is_target=is_target,
            lookback=lookback,
            fillna_strategy=fillna_strategy,
            normalize=normalize,
            clip_outliers=clip_outliers,
            outlier_std=outlier_std,
        )

        self._features[name] = metadata

        if is_target:
            self._target_features.append(name)

        self.logger.debug(f"Registered feature: {name}")

    def get(self, name: str) -> FeatureMetadata | None:
        """Get feature metadata."""
        return self._features.get(name)

    def get_by_type(self, feature_type: FeatureType) -> list[FeatureMetadata]:
        """Get all features of a specific type."""
        return [f for f in self._features.values() if f.type == feature_type]

    def get_all_names(self) -> list[str]:
        """Get all feature names."""
        return list(self._features.keys())

    def get_input_features(self) -> list[str]:
        """Get all non-target feature names."""
        return [name for name, f in self._features.items() if not f.is_target]

    def get_target_features(self) -> list[str]:
        """Get target feature names."""
        return self._target_features

    def get_features_with_lookback(self, min_lookback: int = 1) -> list[str]:
        """Get features that require lookback data."""
        return [
            name for name, f in self._features.items()
            if f.lookback >= min_lookback
        ]

    def validate_features(self, df: pl.DataFrame) -> dict[str, bool]:
        """Validate that DataFrame contains expected features."""
        results = {}
        for name in self._features:
            results[name] = name in df.columns
        return results

    def get_missing_features(self, df: pl.DataFrame) -> list[str]:
        """Get list of missing features."""
        return [name for name in self._features if name not in df.columns]

    def compute_feature(
        self,
        name: str,
        df: pl.DataFrame,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Compute a single feature if compute function exists."""
        metadata = self._features.get(name)
        if metadata is None:
            raise ValueError(f"Feature '{name}' not found in registry")

        if metadata.compute_fn is None:
            raise ValueError(f"Feature '{name}' has no compute function")

        # Check dependencies
        missing_deps = [d for d in metadata.dependencies if d not in df.columns]
        if missing_deps:
            raise ValueError(f"Missing dependencies for '{name}': {missing_deps}")

        return metadata.compute_fn(df, **kwargs)

    def normalize_features(
        self,
        df: pl.DataFrame,
        method: str = "zscore",
        exclude: list[str] | None = None,
    ) -> pl.DataFrame:
        """Normalize features that have normalize=True."""
        exclude = exclude or []

        for name, metadata in self._features.items():
            if name in exclude or not metadata.normalize:
                continue
            if name not in df.columns:
                continue

            if method == "zscore":
                df = df.with_columns([
                    (
                        (pl.col(name) - pl.col(name).mean())
                        / (pl.col(name).std() + 1e-10)
                    ).alias(name)
                ])
            elif method == "minmax":
                df = df.with_columns([
                    (
                        (pl.col(name) - pl.col(name).min())
                        / (pl.col(name).max() - pl.col(name).min() + 1e-10)
                    ).alias(name)
                ])

        return df

    def clip_outliers(
        self,
        df: pl.DataFrame,
        exclude: list[str] | None = None,
    ) -> pl.DataFrame:
        """Clip outliers for features that have clip_outliers=True."""
        exclude = exclude or []

        for name, metadata in self._features.items():
            if name in exclude or not metadata.clip_outliers:
                continue
            if name not in df.columns:
                continue

            n_std = metadata.outlier_std
            df = df.with_columns([
                pl.col(name).clip(
                    pl.col(name).mean() - n_std * pl.col(name).std(),
                    pl.col(name).mean() + n_std * pl.col(name).std(),
                )
            ])

        return df

    def _register_default_features(self) -> None:
        """Register default features."""
        # Technical features
        technical_features = [
            ("returns", "Simple returns"),
            ("log_returns", "Log returns"),
            ("rsi_14", "14-period RSI"),
            ("macd", "MACD line"),
            ("macd_signal", "MACD signal line"),
            ("bb_position", "Bollinger Band position"),
            ("atr", "Average True Range"),
            ("volatility_20", "20-period volatility"),
            ("sma_20", "20-period SMA"),
            ("sma_50", "50-period SMA"),
            ("adx", "Average Directional Index"),
            ("volume_ratio", "Volume relative to SMA"),
        ]

        for name, desc in technical_features:
            self.register(name, FeatureType.TECHNICAL, desc)

        # On-chain features
        onchain_features = [
            ("funding_rate", "Funding rate"),
            ("oi_change", "Open interest change"),
            ("long_short_ratio", "Long/short ratio"),
            ("taker_buy_sell_ratio", "Taker buy/sell ratio"),
        ]

        for name, desc in onchain_features:
            self.register(name, FeatureType.ONCHAIN, desc)

        # Sentiment features
        sentiment_features = [
            ("fear_greed_value", "Fear & Greed Index"),
            ("sentiment_composite", "Composite sentiment score"),
        ]

        for name, desc in sentiment_features:
            self.register(name, FeatureType.SENTIMENT, desc)

        # Temporal features
        temporal_features = [
            ("hour_sin", "Hour of day (sin)"),
            ("hour_cos", "Hour of day (cos)"),
            ("day_of_week_sin", "Day of week (sin)"),
            ("day_of_week_cos", "Day of week (cos)"),
        ]

        for name, desc in temporal_features:
            self.register(name, FeatureType.TEMPORAL, desc)

    def to_dict(self) -> dict[str, dict]:
        """Export registry to dictionary."""
        return {
            name: {
                "type": f.type.value,
                "description": f.description,
                "dependencies": f.dependencies,
                "is_target": f.is_target,
                "lookback": f.lookback,
            }
            for name, f in self._features.items()
        }

    def __len__(self) -> int:
        return len(self._features)

    def __contains__(self, name: str) -> bool:
        return name in self._features
