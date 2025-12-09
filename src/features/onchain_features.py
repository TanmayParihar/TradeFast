"""
On-chain and futures metrics feature engineering.
"""

from typing import Any

import numpy as np
import polars as pl
from loguru import logger


class OnChainFeatures:
    """Generate features from on-chain and futures metrics."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logger.bind(module="onchain_features")

    def add_all_features(
        self,
        df: pl.DataFrame,
        include_funding: bool = True,
        include_oi: bool = True,
        include_sentiment: bool = True,
    ) -> pl.DataFrame:
        """Add all on-chain/futures features."""
        self.logger.info("Computing on-chain features...")

        if include_funding and "funding_rate" in df.columns:
            df = self.add_funding_features(df)

        if include_oi and "open_interest" in df.columns:
            df = self.add_open_interest_features(df)

        if include_sentiment:
            df = self.add_sentiment_ratio_features(df)

        self.logger.info(f"Added on-chain features. Total columns: {len(df.columns)}")
        return df

    def add_funding_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add funding rate-based features."""
        if "funding_rate" not in df.columns:
            return df

        df = df.with_columns([
            # Forward fill funding rate (updates every 8 hours)
            pl.col("funding_rate").fill_null(strategy="forward"),
        ])

        df = df.with_columns([
            # Annualized funding rate (3 fundings per day * 365)
            (pl.col("funding_rate") * 3 * 365).alias("funding_rate_annual"),

            # Funding rate change
            (pl.col("funding_rate") - pl.col("funding_rate").shift(1))
            .alias("funding_rate_change"),

            # Rolling average funding rate
            pl.col("funding_rate").rolling_mean(480).alias("funding_rate_ma_8h"),
            pl.col("funding_rate").rolling_mean(1440).alias("funding_rate_ma_24h"),

            # Cumulative funding (cost of holding position)
            pl.col("funding_rate").cum_sum().alias("cumulative_funding"),
        ])

        # Funding rate z-score
        df = df.with_columns([
            (
                (pl.col("funding_rate") - pl.col("funding_rate").rolling_mean(1440))
                / (pl.col("funding_rate").rolling_std(1440) + 1e-10)
            ).alias("funding_rate_zscore")
        ])

        # Funding regime (high/low/neutral)
        df = df.with_columns([
            pl.when(pl.col("funding_rate") > 0.0001)
            .then(1)
            .when(pl.col("funding_rate") < -0.0001)
            .then(-1)
            .otherwise(0)
            .alias("funding_regime")
        ])

        return df

    def add_open_interest_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add open interest-based features."""
        if "open_interest" not in df.columns:
            return df

        df = df.with_columns([
            pl.col("open_interest").fill_null(strategy="forward"),
        ])

        df = df.with_columns([
            # OI change
            (pl.col("open_interest") / pl.col("open_interest").shift(1) - 1)
            .alias("oi_change"),

            # OI momentum (change over various periods)
            (pl.col("open_interest") / pl.col("open_interest").shift(60) - 1)
            .alias("oi_change_1h"),
            (pl.col("open_interest") / pl.col("open_interest").shift(240) - 1)
            .alias("oi_change_4h"),
            (pl.col("open_interest") / pl.col("open_interest").shift(1440) - 1)
            .alias("oi_change_24h"),

            # OI moving averages
            pl.col("open_interest").rolling_mean(60).alias("oi_ma_1h"),
            pl.col("open_interest").rolling_mean(480).alias("oi_ma_8h"),
        ])

        # OI relative to MA
        df = df.with_columns([
            (pl.col("open_interest") / pl.col("oi_ma_8h") - 1).alias("oi_ma_deviation"),
        ])

        # OI-Price divergence (OI up, price down = bearish divergence)
        if "close" in df.columns:
            df = df.with_columns([
                (pl.col("oi_change_1h") - pl.col("returns_60")).alias("oi_price_divergence_1h"),
            ])

        # OI velocity (rate of change of change)
        df = df.with_columns([
            (pl.col("oi_change") - pl.col("oi_change").shift(1)).alias("oi_acceleration"),
        ])

        return df

    def add_sentiment_ratio_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add long/short ratio and taker buy/sell features."""
        # Long/Short Ratio features
        if "long_short_ratio" in df.columns:
            df = df.with_columns([
                pl.col("long_short_ratio").fill_null(strategy="forward"),
            ])

            df = df.with_columns([
                # LS ratio change
                (pl.col("long_short_ratio") - pl.col("long_short_ratio").shift(1))
                .alias("ls_ratio_change"),

                # LS ratio moving averages
                pl.col("long_short_ratio").rolling_mean(60).alias("ls_ratio_ma_1h"),
                pl.col("long_short_ratio").rolling_mean(480).alias("ls_ratio_ma_8h"),

                # Extreme positioning (crowd usually wrong at extremes)
                pl.when(pl.col("long_short_ratio") > 2.0)
                .then(1)  # Extremely long
                .when(pl.col("long_short_ratio") < 0.5)
                .then(-1)  # Extremely short
                .otherwise(0)
                .alias("ls_extreme_positioning"),
            ])

            # LS ratio z-score
            df = df.with_columns([
                (
                    (pl.col("long_short_ratio") - pl.col("ls_ratio_ma_8h"))
                    / (pl.col("long_short_ratio").rolling_std(480) + 1e-10)
                ).alias("ls_ratio_zscore")
            ])

        # Top Trader LS Ratio
        if "top_trader_ls_ratio" in df.columns:
            df = df.with_columns([
                pl.col("top_trader_ls_ratio").fill_null(strategy="forward"),
            ])

            df = df.with_columns([
                (pl.col("top_trader_ls_ratio") - pl.col("top_trader_ls_ratio").shift(1))
                .alias("top_trader_ls_change"),
            ])

            # Divergence between retail and top traders
            if "long_short_ratio" in df.columns:
                df = df.with_columns([
                    (pl.col("top_trader_ls_ratio") - pl.col("long_short_ratio"))
                    .alias("smart_money_divergence")
                ])

        # Taker Buy/Sell Ratio
        if "taker_buy_sell_ratio" in df.columns:
            df = df.with_columns([
                pl.col("taker_buy_sell_ratio").fill_null(strategy="forward"),
            ])

            df = df.with_columns([
                # Taker ratio change
                (pl.col("taker_buy_sell_ratio") - pl.col("taker_buy_sell_ratio").shift(1))
                .alias("taker_ratio_change"),

                # Taker ratio MA
                pl.col("taker_buy_sell_ratio").rolling_mean(60).alias("taker_ratio_ma_1h"),

                # Aggressive buying/selling indicator
                pl.when(pl.col("taker_buy_sell_ratio") > 1.5)
                .then(1)
                .when(pl.col("taker_buy_sell_ratio") < 0.67)
                .then(-1)
                .otherwise(0)
                .alias("aggressive_flow"),
            ])

        return df

    def add_composite_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add composite features combining multiple metrics."""
        components = []

        # Check which components are available
        if "funding_rate_zscore" in df.columns:
            components.append("funding_rate_zscore")
        if "ls_ratio_zscore" in df.columns:
            components.append("ls_ratio_zscore")
        if "oi_ma_deviation" in df.columns:
            components.append("oi_ma_deviation")

        if len(components) >= 2:
            # Sentiment composite (average of z-scores)
            df = df.with_columns([
                pl.mean_horizontal(*[pl.col(c) for c in components])
                .alias("sentiment_composite")
            ])

            # Sentiment regime
            df = df.with_columns([
                pl.when(pl.col("sentiment_composite") > 1.0)
                .then(1)  # Bullish extreme
                .when(pl.col("sentiment_composite") < -1.0)
                .then(-1)  # Bearish extreme
                .otherwise(0)
                .alias("sentiment_regime")
            ])

        return df

    def add_liquidation_features(
        self,
        df: pl.DataFrame,
        liquidation_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Add liquidation-based features if data available."""
        if liquidation_df is None or liquidation_df.is_empty():
            return df

        # Aggregate liquidations to minute level
        liq_agg = (
            liquidation_df.with_columns([
                pl.col("timestamp").dt.truncate("1m").alias("timestamp_minute")
            ])
            .group_by("timestamp_minute")
            .agg([
                pl.col("quantity").filter(pl.col("side") == "BUY").sum().alias("long_liquidations"),
                pl.col("quantity").filter(pl.col("side") == "SELL").sum().alias("short_liquidations"),
            ])
            .rename({"timestamp_minute": "timestamp"})
        )

        # Merge with main dataframe
        df = df.join(liq_agg, on="timestamp", how="left")

        # Fill nulls with 0 (no liquidations)
        df = df.with_columns([
            pl.col("long_liquidations").fill_null(0),
            pl.col("short_liquidations").fill_null(0),
        ])

        # Liquidation features
        df = df.with_columns([
            (pl.col("long_liquidations") + pl.col("short_liquidations"))
            .alias("total_liquidations"),

            (pl.col("long_liquidations") - pl.col("short_liquidations"))
            / (pl.col("long_liquidations") + pl.col("short_liquidations") + 1e-10)
            .alias("liquidation_imbalance"),

            # Rolling liquidation volume
            pl.col("long_liquidations").rolling_sum(60).alias("long_liq_1h"),
            pl.col("short_liquidations").rolling_sum(60).alias("short_liq_1h"),
        ])

        return df
