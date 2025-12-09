"""Signal generation from model predictions."""

from typing import Any
import numpy as np
import polars as pl
from loguru import logger
from src.strategy.base import Signal


class SignalGenerator:
    """Generate trading signals from model predictions."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logger.bind(module="signal_generator")
        self.long_threshold = config.get("long_threshold", 0.55)
        self.short_threshold = config.get("short_threshold", 0.45)
        self.min_confidence = config.get("min_confidence", 0.6)

    def generate(
        self,
        predictions: np.ndarray,
        timestamps: np.ndarray,
        symbol: str,
        prices: np.ndarray | None = None,
        volatility: np.ndarray | None = None,
    ) -> list[Signal]:
        """Generate signals from predictions."""
        signals = []

        for i, (pred, ts) in enumerate(zip(predictions, timestamps)):
            if pred.ndim == 0:
                continue

            # Get probabilities
            if len(pred) == 3:
                buy_prob, hold_prob, sell_prob = pred
            else:
                buy_prob = pred[0] if len(pred) > 0 else 0.5
                sell_prob = 1 - buy_prob
                hold_prob = 0

            # Determine direction
            max_prob = max(buy_prob, sell_prob, hold_prob)

            if buy_prob > self.long_threshold and buy_prob == max_prob:
                direction = 1
                confidence = buy_prob
            elif sell_prob > (1 - self.short_threshold) and sell_prob == max_prob:
                direction = -1
                confidence = sell_prob
            else:
                direction = 0
                confidence = hold_prob

            # Skip low confidence signals
            if confidence < self.min_confidence:
                direction = 0

            if direction != 0:
                signal = Signal(
                    timestamp=ts,
                    symbol=symbol,
                    direction=direction,
                    confidence=confidence,
                    size=0.0,  # Will be set by position sizer
                    metadata={"buy_prob": buy_prob, "sell_prob": sell_prob},
                )

                # Add stop/take profit if price/vol available
                if prices is not None and volatility is not None and i < len(prices):
                    price = prices[i]
                    vol = volatility[i] if i < len(volatility) else 0.02
                    signal.stop_loss = price * (1 - 2 * vol * direction)
                    signal.take_profit = price * (1 + 3 * vol * direction)

                signals.append(signal)

        return signals
