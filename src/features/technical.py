"""
Technical indicator features using Polars.
"""

from typing import Any

import numpy as np
import polars as pl
from loguru import logger


class TechnicalFeatures:
    """Technical indicator feature generator using Polars."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logger.bind(module="technical_features")

    def add_all_features(
        self,
        df: pl.DataFrame,
        include_volume: bool = True,
        include_momentum: bool = True,
        include_volatility: bool = True,
        include_trend: bool = True,
    ) -> pl.DataFrame:
        """Add all technical features to DataFrame."""
        self.logger.info("Computing technical features...")

        # Basic returns
        df = self.add_returns(df)

        if include_momentum:
            df = self.add_momentum_features(df)

        if include_volatility:
            df = self.add_volatility_features(df)

        if include_trend:
            df = self.add_trend_features(df)

        if include_volume:
            df = self.add_volume_features(df)

        # Temporal features
        df = self.add_temporal_features(df)

        self.logger.info(f"Added technical features. Total columns: {len(df.columns)}")
        return df

    def add_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add return-based features."""
        return df.with_columns([
            # Simple returns
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("returns"),
            # Log returns
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_returns"),
            # Multi-period returns
            (pl.col("close") / pl.col("close").shift(5) - 1).alias("returns_5"),
            (pl.col("close") / pl.col("close").shift(15) - 1).alias("returns_15"),
            (pl.col("close") / pl.col("close").shift(60) - 1).alias("returns_60"),
            (pl.col("close") / pl.col("close").shift(240) - 1).alias("returns_240"),
            # Intrabar return
            ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("intrabar_return"),
            # High-low range
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_pct"),
        ])

    def add_momentum_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add momentum indicators."""
        # RSI
        df = self._add_rsi(df, periods=[14, 7, 21])

        # MACD
        df = self._add_macd(df)

        # Stochastic
        df = self._add_stochastic(df)

        # Rate of Change
        df = df.with_columns([
            (pl.col("close") / pl.col("close").shift(10) - 1).alias("roc_10"),
            (pl.col("close") / pl.col("close").shift(20) - 1).alias("roc_20"),
        ])

        # Momentum
        df = df.with_columns([
            (pl.col("close") - pl.col("close").shift(10)).alias("momentum_10"),
            (pl.col("close") - pl.col("close").shift(20)).alias("momentum_20"),
        ])

        return df

    def _add_rsi(
        self, df: pl.DataFrame, periods: list[int] | None = None
    ) -> pl.DataFrame:
        """Add RSI indicator."""
        periods = periods or [14]

        for period in periods:
            # Calculate price changes
            delta = pl.col("close") - pl.col("close").shift(1)

            # Separate gains and losses
            gain = pl.when(delta > 0).then(delta).otherwise(0)
            loss = pl.when(delta < 0).then(-delta).otherwise(0)

            df = df.with_columns([
                gain.alias("_gain"),
                loss.alias("_loss"),
            ])

            # Calculate average gain/loss using EMA
            df = df.with_columns([
                pl.col("_gain").ewm_mean(span=period).alias("_avg_gain"),
                pl.col("_loss").ewm_mean(span=period).alias("_avg_loss"),
            ])

            # Calculate RSI
            df = df.with_columns([
                (100 - 100 / (1 + pl.col("_avg_gain") / (pl.col("_avg_loss") + 1e-10)))
                .alias(f"rsi_{period}")
            ])

            df = df.drop(["_gain", "_loss", "_avg_gain", "_avg_loss"])

        return df

    def _add_macd(
        self,
        df: pl.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pl.DataFrame:
        """Add MACD indicator."""
        df = df.with_columns([
            pl.col("close").ewm_mean(span=fast).alias("_ema_fast"),
            pl.col("close").ewm_mean(span=slow).alias("_ema_slow"),
        ])

        df = df.with_columns([
            (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd"),
        ])

        df = df.with_columns([
            pl.col("macd").ewm_mean(span=signal).alias("macd_signal"),
        ])

        df = df.with_columns([
            (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist"),
        ])

        return df.drop(["_ema_fast", "_ema_slow"])

    def _add_stochastic(
        self, df: pl.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> pl.DataFrame:
        """Add Stochastic oscillator."""
        df = df.with_columns([
            pl.col("high").rolling_max(k_period).alias("_highest_high"),
            pl.col("low").rolling_min(k_period).alias("_lowest_low"),
        ])

        df = df.with_columns([
            (
                (pl.col("close") - pl.col("_lowest_low"))
                / (pl.col("_highest_high") - pl.col("_lowest_low") + 1e-10)
                * 100
            ).alias("stoch_k")
        ])

        df = df.with_columns([
            pl.col("stoch_k").rolling_mean(d_period).alias("stoch_d"),
        ])

        return df.drop(["_highest_high", "_lowest_low"])

    def add_volatility_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add volatility indicators."""
        # Bollinger Bands
        df = self._add_bollinger_bands(df)

        # ATR
        df = self._add_atr(df)

        # Rolling volatility
        df = df.with_columns([
            pl.col("log_returns").rolling_std(20).alias("volatility_20"),
            pl.col("log_returns").rolling_std(60).alias("volatility_60"),
            pl.col("log_returns").rolling_std(240).alias("volatility_240"),
        ])

        # Volatility ratio
        df = df.with_columns([
            (pl.col("volatility_20") / (pl.col("volatility_60") + 1e-10))
            .alias("vol_ratio_20_60"),
        ])

        # Parkinson volatility (using high-low)
        df = df.with_columns([
            (
                (pl.col("high") / pl.col("low")).log().pow(2)
                / (4 * np.log(2))
            )
            .rolling_mean(20)
            .sqrt()
            .alias("parkinson_vol_20")
        ])

        return df

    def _add_bollinger_bands(
        self, df: pl.DataFrame, period: int = 20, std: float = 2.0
    ) -> pl.DataFrame:
        """Add Bollinger Bands."""
        df = df.with_columns([
            pl.col("close").rolling_mean(period).alias("bb_mid"),
            pl.col("close").rolling_std(period).alias("_bb_std"),
        ])

        df = df.with_columns([
            (pl.col("bb_mid") + std * pl.col("_bb_std")).alias("bb_upper"),
            (pl.col("bb_mid") - std * pl.col("_bb_std")).alias("bb_lower"),
        ])

        # BB position (where price is within bands)
        df = df.with_columns([
            (
                (pl.col("close") - pl.col("bb_lower"))
                / (pl.col("bb_upper") - pl.col("bb_lower") + 1e-10)
            ).alias("bb_position"),
            # BB width
            (
                (pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_mid")
            ).alias("bb_width"),
        ])

        return df.drop("_bb_std")

    def _add_atr(self, df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Add Average True Range."""
        df = df.with_columns([
            (pl.col("high") - pl.col("low")).alias("_tr1"),
            (pl.col("high") - pl.col("close").shift(1)).abs().alias("_tr2"),
            (pl.col("low") - pl.col("close").shift(1)).abs().alias("_tr3"),
        ])

        df = df.with_columns([
            pl.max_horizontal("_tr1", "_tr2", "_tr3").alias("true_range")
        ])

        df = df.with_columns([
            pl.col("true_range").ewm_mean(span=period).alias("atr"),
        ])

        # Normalized ATR
        df = df.with_columns([
            (pl.col("atr") / pl.col("close")).alias("atr_pct"),
        ])

        return df.drop(["_tr1", "_tr2", "_tr3"])

    def add_trend_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add trend indicators."""
        # SMAs
        for period in [10, 20, 50, 100, 200]:
            df = df.with_columns([
                pl.col("close").rolling_mean(period).alias(f"sma_{period}")
            ])

        # EMAs
        for period in [10, 20, 50]:
            df = df.with_columns([
                pl.col("close").ewm_mean(span=period).alias(f"ema_{period}")
            ])

        # Price relative to MAs
        df = df.with_columns([
            (pl.col("close") / pl.col("sma_20") - 1).alias("price_sma20_dist"),
            (pl.col("close") / pl.col("sma_50") - 1).alias("price_sma50_dist"),
            (pl.col("close") / pl.col("sma_200") - 1).alias("price_sma200_dist"),
        ])

        # MA slopes
        df = df.with_columns([
            (pl.col("sma_20") / pl.col("sma_20").shift(5) - 1).alias("sma20_slope"),
            (pl.col("sma_50") / pl.col("sma_50").shift(5) - 1).alias("sma50_slope"),
        ])

        # MA crossovers
        df = df.with_columns([
            (pl.col("sma_10") > pl.col("sma_20")).cast(pl.Int8).alias("sma_10_20_cross"),
            (pl.col("sma_20") > pl.col("sma_50")).cast(pl.Int8).alias("sma_20_50_cross"),
            (pl.col("sma_50") > pl.col("sma_200")).cast(pl.Int8).alias("sma_50_200_cross"),
        ])

        # ADX (simplified)
        df = self._add_adx(df)

        return df

    def _add_adx(self, df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Add ADX (Average Directional Index)."""
        df = df.with_columns([
            (pl.col("high") - pl.col("high").shift(1)).alias("_up_move"),
            (pl.col("low").shift(1) - pl.col("low")).alias("_down_move"),
        ])

        # +DM and -DM
        df = df.with_columns([
            pl.when(
                (pl.col("_up_move") > pl.col("_down_move")) & (pl.col("_up_move") > 0)
            )
            .then(pl.col("_up_move"))
            .otherwise(0)
            .alias("_plus_dm"),
            pl.when(
                (pl.col("_down_move") > pl.col("_up_move")) & (pl.col("_down_move") > 0)
            )
            .then(pl.col("_down_move"))
            .otherwise(0)
            .alias("_minus_dm"),
        ])

        # Smoothed +DI and -DI
        df = df.with_columns([
            (
                pl.col("_plus_dm").ewm_mean(span=period)
                / (pl.col("atr") + 1e-10)
                * 100
            ).alias("plus_di"),
            (
                pl.col("_minus_dm").ewm_mean(span=period)
                / (pl.col("atr") + 1e-10)
                * 100
            ).alias("minus_di"),
        ])

        # DX and ADX
        df = df.with_columns([
            (
                (pl.col("plus_di") - pl.col("minus_di")).abs()
                / (pl.col("plus_di") + pl.col("minus_di") + 1e-10)
                * 100
            ).alias("_dx")
        ])

        df = df.with_columns([
            pl.col("_dx").ewm_mean(span=period).alias("adx")
        ])

        return df.drop(["_up_move", "_down_move", "_plus_dm", "_minus_dm", "_dx"])

    def add_volume_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add volume-based features."""
        # Volume SMAs
        df = df.with_columns([
            pl.col("volume").rolling_mean(20).alias("volume_sma_20"),
            pl.col("volume").rolling_mean(50).alias("volume_sma_50"),
        ])

        # Volume ratio
        df = df.with_columns([
            (pl.col("volume") / (pl.col("volume_sma_20") + 1e-10)).alias("volume_ratio"),
        ])

        # OBV (On-Balance Volume)
        df = df.with_columns([
            pl.when(pl.col("close") > pl.col("close").shift(1))
            .then(pl.col("volume"))
            .when(pl.col("close") < pl.col("close").shift(1))
            .then(-pl.col("volume"))
            .otherwise(0)
            .alias("_obv_change")
        ])

        df = df.with_columns([
            pl.col("_obv_change").cum_sum().alias("obv")
        ])

        # OBV momentum
        df = df.with_columns([
            (pl.col("obv") - pl.col("obv").shift(20)).alias("obv_momentum")
        ])

        # VWAP (rolling)
        df = df.with_columns([
            (
                (pl.col("close") * pl.col("volume")).rolling_sum(20)
                / (pl.col("volume").rolling_sum(20) + 1e-10)
            ).alias("vwap_20")
        ])

        # Price-Volume trend
        df = df.with_columns([
            (pl.col("returns") * pl.col("volume")).rolling_sum(20).alias("pv_trend")
        ])

        # Money Flow Index components
        df = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical_price"),
        ])

        df = df.with_columns([
            (pl.col("typical_price") * pl.col("volume")).alias("raw_money_flow"),
        ])

        return df.drop("_obv_change")

    def add_temporal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add time-based cyclical features."""
        if "timestamp" not in df.columns:
            return df

        # Extract time components
        df = df.with_columns([
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.minute().alias("minute"),
            pl.col("timestamp").dt.weekday().alias("day_of_week"),
            pl.col("timestamp").dt.month().alias("month"),
        ])

        # Cyclical encoding
        df = df.with_columns([
            (pl.col("hour") * 2 * np.pi / 24).sin().alias("hour_sin"),
            (pl.col("hour") * 2 * np.pi / 24).cos().alias("hour_cos"),
            (pl.col("minute") * 2 * np.pi / 60).sin().alias("minute_sin"),
            (pl.col("minute") * 2 * np.pi / 60).cos().alias("minute_cos"),
            (pl.col("day_of_week") * 2 * np.pi / 7).sin().alias("day_of_week_sin"),
            (pl.col("day_of_week") * 2 * np.pi / 7).cos().alias("day_of_week_cos"),
            (pl.col("month") * 2 * np.pi / 12).sin().alias("month_sin"),
            (pl.col("month") * 2 * np.pi / 12).cos().alias("month_cos"),
        ])

        return df
