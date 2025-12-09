"""Position sizing methods."""

import numpy as np
from typing import Any
from loguru import logger


class PositionSizer:
    """Position sizing calculator."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.method = config.get("method", "kelly")
        self.max_position_pct = config.get("max_position_pct", 0.1)
        self.min_position_pct = config.get("min_position_pct", 0.01)

    def calculate(
        self,
        confidence: float,
        portfolio_value: float,
        volatility: float = 0.02,
        win_rate: float | None = None,
    ) -> float:
        """Calculate position size."""
        if self.method == "kelly":
            size = KellyCriterion.calculate(
                confidence,
                odds=1.0,
                fraction=self.config.get("kelly_fraction", 0.5),
            )
        elif self.method == "fixed":
            size = self.config.get("fixed_size", 0.05)
        elif self.method == "volatility":
            target_risk = self.config.get("target_risk", 0.02)
            size = target_risk / (volatility + 1e-10)
        else:
            size = 0.05

        # Apply limits
        size = max(self.min_position_pct, min(size, self.max_position_pct))
        return size * portfolio_value


class KellyCriterion:
    """Kelly Criterion for optimal position sizing."""

    @staticmethod
    def calculate(
        win_probability: float,
        odds: float = 1.0,
        fraction: float = 0.5,
    ) -> float:
        """Calculate Kelly fraction."""
        if win_probability <= 0 or win_probability >= 1:
            return 0.0

        q = 1 - win_probability
        kelly = (win_probability * odds - q) / odds

        if kelly <= 0:
            return 0.0

        return kelly * fraction
