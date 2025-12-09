"""
Triple Barrier Method for labeling trading signals.

Based on Marcos Lopez de Prado's methodology.
"""

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger


def get_daily_volatility(
    close: pd.Series | np.ndarray,
    span: int = 1440,
) -> pd.Series:
    """
    Calculate rolling volatility for position sizing.

    Args:
        close: Close prices
        span: Rolling window (1440 = 1 day for 1-min data)

    Returns:
        Rolling standard deviation of returns
    """
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    returns = close.pct_change()
    return returns.ewm(span=span).std()


class TripleBarrier:
    """
    Triple Barrier Method for generating trade labels.

    Creates labels based on which barrier is hit first:
    - Upper barrier (take profit): Label = 1
    - Lower barrier (stop loss): Label = -1
    - Vertical barrier (max holding): Label = 0
    """

    def __init__(
        self,
        pt_mult: float = 2.0,
        sl_mult: float = 1.0,
        max_holding: int = 240,
        vol_span: int = 1440,
        min_vol: float = 0.0001,
    ):
        """
        Initialize Triple Barrier.

        Args:
            pt_mult: Take-profit multiplier (times volatility)
            sl_mult: Stop-loss multiplier (times volatility)
            max_holding: Maximum holding period in bars
            vol_span: Volatility calculation span
            min_vol: Minimum volatility threshold
        """
        self.pt_mult = pt_mult
        self.sl_mult = sl_mult
        self.max_holding = max_holding
        self.vol_span = vol_span
        self.min_vol = min_vol
        self.logger = logger.bind(module="triple_barrier")

    def get_labels(
        self,
        df: pd.DataFrame | pl.DataFrame,
        side: np.ndarray | None = None,
    ) -> pd.Series:
        """
        Generate triple barrier labels.

        Args:
            df: DataFrame with 'close', 'high', 'low' columns
            side: Optional array of trade sides (1 for long, -1 for short)
                 If None, assumes long-only

        Returns:
            Series with labels: 1 (profit), -1 (loss), 0 (timeout)
        """
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # Calculate volatility
        vol = get_daily_volatility(df["close"], span=self.vol_span).values

        n = len(close)
        labels = np.zeros(n)
        returns = np.zeros(n)
        durations = np.zeros(n)

        for i in range(n - self.max_holding):
            if np.isnan(vol[i]) or vol[i] < self.min_vol:
                continue

            entry = close[i]
            current_side = side[i] if side is not None else 1

            # Calculate barriers
            pt_level = entry * (1 + self.pt_mult * vol[i] * current_side)
            sl_level = entry * (1 - self.sl_mult * vol[i] * current_side)

            # Scan forward to find which barrier is hit first
            for j in range(1, self.max_holding + 1):
                idx = i + j
                if idx >= n:
                    break

                if current_side == 1:  # Long position
                    # Check take profit (high price >= target)
                    if high[idx] >= pt_level:
                        labels[i] = 1
                        returns[i] = (pt_level - entry) / entry
                        durations[i] = j
                        break

                    # Check stop loss (low price <= stop)
                    if low[idx] <= sl_level:
                        labels[i] = -1
                        returns[i] = (sl_level - entry) / entry
                        durations[i] = j
                        break

                else:  # Short position
                    # Check take profit (low price <= target)
                    if low[idx] <= pt_level:
                        labels[i] = 1
                        returns[i] = (entry - pt_level) / entry
                        durations[i] = j
                        break

                    # Check stop loss (high price >= stop)
                    if high[idx] >= sl_level:
                        labels[i] = -1
                        returns[i] = (entry - sl_level) / entry
                        durations[i] = j
                        break

            # If no barrier hit, check final return
            if labels[i] == 0 and i + self.max_holding < n:
                final_price = close[i + self.max_holding]
                final_return = (final_price - entry) / entry * current_side
                returns[i] = final_return
                durations[i] = self.max_holding

        result = pd.DataFrame({
            "label": labels,
            "return": returns,
            "duration": durations,
        }, index=df.index)

        return result

    def get_meta_labels(
        self,
        df: pd.DataFrame | pl.DataFrame,
        primary_signals: np.ndarray,
    ) -> pd.DataFrame:
        """
        Generate meta-labels for trade quality.

        Args:
            df: Price DataFrame
            primary_signals: Predictions from primary model (1 for buy, -1 for sell, 0 for hold)

        Returns:
            DataFrame with meta-labels (1 if trade was profitable, 0 otherwise)
        """
        # Get triple barrier labels with trade sides
        side = np.where(primary_signals == 1, 1, np.where(primary_signals == -1, -1, 0))

        results = self.get_labels(df, side=side)

        # Meta-label: was the trade profitable?
        meta_label = (results["label"] == 1).astype(int)

        # Only label trades (where primary signal != 0)
        meta_label = np.where(primary_signals != 0, meta_label, np.nan)

        results["meta_label"] = meta_label
        results["primary_signal"] = primary_signals

        return results

    def get_class_labels(
        self,
        df: pd.DataFrame | pl.DataFrame,
        n_classes: int = 3,
    ) -> np.ndarray:
        """
        Get multi-class labels for classification.

        Args:
            df: Price DataFrame
            n_classes: Number of classes (3 = Buy/Hold/Sell)

        Returns:
            Array of class labels
        """
        results = self.get_labels(df)
        labels = results["label"].values

        if n_classes == 3:
            # 0 = Buy, 1 = Hold, 2 = Sell
            class_labels = np.where(labels == 1, 0,
                                   np.where(labels == -1, 2, 1))
        else:
            # Binary: 0 = Not profitable, 1 = Profitable
            class_labels = (labels == 1).astype(int)

        return class_labels


# Recommended barrier configurations per asset
BARRIER_CONFIGS = {
    "BTCUSDT": {"pt_mult": 1.5, "sl_mult": 1.0, "max_holding": 480},
    "ETHUSDT": {"pt_mult": 1.5, "sl_mult": 1.0, "max_holding": 480},
    "BNBUSDT": {"pt_mult": 2.0, "sl_mult": 1.0, "max_holding": 360},
    "SOLUSDT": {"pt_mult": 2.0, "sl_mult": 1.0, "max_holding": 240},
    "default": {"pt_mult": 2.0, "sl_mult": 1.0, "max_holding": 240},
}


def get_barrier_config(symbol: str) -> dict:
    """Get recommended barrier configuration for a symbol."""
    return BARRIER_CONFIGS.get(symbol, BARRIER_CONFIGS["default"])
