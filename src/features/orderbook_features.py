"""
Order book feature engineering.
"""

from typing import Any

import numpy as np
import polars as pl
from loguru import logger


class OrderBookFeatures:
    """Generate features from order book data."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logger.bind(module="orderbook_features")

    def compute_features(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        mid_price: float,
    ) -> dict[str, float]:
        """
        Compute order book features from bid/ask levels.

        Args:
            bids: List of (price, quantity) tuples, sorted descending
            asks: List of (price, quantity) tuples, sorted ascending
            mid_price: Current mid price

        Returns:
            Dictionary of feature names and values
        """
        if not bids or not asks:
            return self._empty_features()

        features = {}

        # Basic spread features
        best_bid_p, best_bid_q = bids[0]
        best_ask_p, best_ask_q = asks[0]

        features["bid_ask_spread"] = best_ask_p - best_bid_p
        features["bid_ask_spread_bps"] = (best_ask_p - best_bid_p) / mid_price * 10000
        features["mid_price"] = mid_price

        # Microprice (volume-weighted mid)
        features["microprice"] = (
            (best_bid_p * best_ask_q + best_ask_p * best_bid_q)
            / (best_bid_q + best_ask_q)
        )

        # Volume imbalance at best level
        features["imbalance_level_0"] = (
            (best_bid_q - best_ask_q) / (best_bid_q + best_ask_q)
        )

        # Multi-level features
        features.update(self._compute_depth_features(bids, asks, mid_price))
        features.update(self._compute_imbalance_features(bids, asks))
        features.update(self._compute_pressure_features(bids, asks, mid_price))

        return features

    def _compute_depth_features(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        mid_price: float,
    ) -> dict[str, float]:
        """Compute depth-based features."""
        features = {}

        # Total depth within price levels
        for pct in [0.1, 0.5, 1.0]:  # 0.1%, 0.5%, 1% from mid
            bid_depth = sum(
                qty for p, qty in bids if p >= mid_price * (1 - pct / 100)
            )
            ask_depth = sum(
                qty for p, qty in asks if p <= mid_price * (1 + pct / 100)
            )

            suffix = str(pct).replace(".", "")
            features[f"bid_depth_{suffix}pct"] = bid_depth
            features[f"ask_depth_{suffix}pct"] = ask_depth
            features[f"depth_imbalance_{suffix}pct"] = (
                (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-10)
            )

        # Dollar depth (price * quantity)
        bid_dollar = sum(p * q for p, q in bids[:20])
        ask_dollar = sum(p * q for p, q in asks[:20])
        features["bid_dollar_depth"] = bid_dollar
        features["ask_dollar_depth"] = ask_dollar
        features["dollar_depth_ratio"] = bid_dollar / (ask_dollar + 1e-10)

        return features

    def _compute_imbalance_features(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> dict[str, float]:
        """Compute volume imbalance features."""
        features = {}

        # Cumulative imbalance at different levels
        for n_levels in [1, 5, 10, 20]:
            bid_vol = sum(q for _, q in bids[:n_levels])
            ask_vol = sum(q for _, q in asks[:n_levels])

            features[f"imbalance_{n_levels}"] = (
                (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)
            )

        # Weighted imbalance (closer levels weighted more)
        bid_weighted = sum(q / (i + 1) for i, (_, q) in enumerate(bids[:10]))
        ask_weighted = sum(q / (i + 1) for i, (_, q) in enumerate(asks[:10]))
        features["weighted_imbalance"] = (
            (bid_weighted - ask_weighted) / (bid_weighted + ask_weighted + 1e-10)
        )

        return features

    def _compute_pressure_features(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        mid_price: float,
    ) -> dict[str, float]:
        """Compute buying/selling pressure features."""
        features = {}

        # Volume-weighted average price distance from mid
        bid_vwap_dist = sum(
            (mid_price - p) * q for p, q in bids[:10]
        ) / (sum(q for _, q in bids[:10]) + 1e-10)

        ask_vwap_dist = sum(
            (p - mid_price) * q for p, q in asks[:10]
        ) / (sum(q for _, q in asks[:10]) + 1e-10)

        features["bid_vwap_distance"] = bid_vwap_dist
        features["ask_vwap_distance"] = ask_vwap_dist
        features["vwap_distance_ratio"] = bid_vwap_dist / (ask_vwap_dist + 1e-10)

        # Slope of order book (how quickly depth increases)
        if len(bids) >= 10:
            bid_prices = [p for p, _ in bids[:10]]
            bid_qtys = [q for _, q in bids[:10]]
            bid_slope = np.polyfit(range(10), bid_qtys, 1)[0] if len(bid_qtys) == 10 else 0
            features["bid_slope"] = bid_slope

        if len(asks) >= 10:
            ask_prices = [p for p, _ in asks[:10]]
            ask_qtys = [q for _, q in asks[:10]]
            ask_slope = np.polyfit(range(10), ask_qtys, 1)[0] if len(ask_qtys) == 10 else 0
            features["ask_slope"] = ask_slope

        return features

    def _empty_features(self) -> dict[str, float]:
        """Return empty feature dictionary."""
        return {
            "bid_ask_spread": 0.0,
            "bid_ask_spread_bps": 0.0,
            "mid_price": 0.0,
            "microprice": 0.0,
            "imbalance_level_0": 0.0,
        }

    def add_orderbook_features_to_df(
        self,
        ohlcv_df: pl.DataFrame,
        orderbook_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Merge order book features into OHLCV DataFrame.

        Args:
            ohlcv_df: OHLCV DataFrame with timestamp
            orderbook_df: Order book metrics DataFrame

        Returns:
            Merged DataFrame
        """
        if orderbook_df.is_empty():
            self.logger.warning("Order book DataFrame is empty")
            return ohlcv_df

        # Select relevant columns
        ob_cols = [
            c for c in orderbook_df.columns
            if c not in ["timestamp", "symbol"] or c == "timestamp"
        ]

        # Round timestamps to minute for joining
        orderbook_df = orderbook_df.with_columns([
            pl.col("timestamp").dt.truncate("1m").alias("timestamp_minute")
        ])

        # Aggregate to minute level (take last value)
        ob_agg = (
            orderbook_df.group_by("timestamp_minute")
            .agg([pl.col(c).last() for c in ob_cols if c != "timestamp"])
            .rename({"timestamp_minute": "timestamp"})
        )

        # Join with OHLCV
        result = ohlcv_df.join(ob_agg, on="timestamp", how="left", suffix="_ob")

        # Forward fill order book features
        for col in result.columns:
            if "_ob" in col or col in ob_cols:
                result = result.with_columns([
                    pl.col(col).fill_null(strategy="forward")
                ])

        return result

    def compute_flow_features(
        self,
        trades_df: pl.DataFrame,
        window: int = 60,
    ) -> pl.DataFrame:
        """
        Compute order flow features from trade data.

        Args:
            trades_df: DataFrame with trade data
            window: Rolling window in rows

        Returns:
            DataFrame with flow features
        """
        if trades_df.is_empty():
            return trades_df

        # Buy/sell volume
        trades_df = trades_df.with_columns([
            pl.when(pl.col("is_buyer_maker") == False)
            .then(pl.col("quantity"))
            .otherwise(0)
            .alias("buy_volume"),
            pl.when(pl.col("is_buyer_maker") == True)
            .then(pl.col("quantity"))
            .otherwise(0)
            .alias("sell_volume"),
        ])

        # Rolling aggregations
        trades_df = trades_df.with_columns([
            pl.col("buy_volume").rolling_sum(window).alias(f"buy_volume_{window}"),
            pl.col("sell_volume").rolling_sum(window).alias(f"sell_volume_{window}"),
        ])

        # Order flow imbalance
        trades_df = trades_df.with_columns([
            (
                (pl.col(f"buy_volume_{window}") - pl.col(f"sell_volume_{window}"))
                / (pl.col(f"buy_volume_{window}") + pl.col(f"sell_volume_{window}") + 1e-10)
            ).alias(f"order_flow_imbalance_{window}")
        ])

        # VPIN (Volume-synchronized Probability of Informed Trading) approximation
        trades_df = trades_df.with_columns([
            (
                pl.col(f"buy_volume_{window}").abs()
                - pl.col(f"sell_volume_{window}").abs()
            ).abs().alias("abs_imbalance")
        ])

        trades_df = trades_df.with_columns([
            (
                pl.col("abs_imbalance")
                / (pl.col(f"buy_volume_{window}") + pl.col(f"sell_volume_{window}") + 1e-10)
            ).alias(f"vpin_proxy_{window}")
        ])

        return trades_df
