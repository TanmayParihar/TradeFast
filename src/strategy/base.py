"""Base strategy class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import numpy as np
import polars as pl
from loguru import logger


@dataclass
class Signal:
    """Trading signal."""
    timestamp: Any
    symbol: str
    direction: int  # 1 = long, -1 = short, 0 = neutral
    confidence: float
    size: float
    stop_loss: float | None = None
    take_profit: float | None = None
    metadata: dict = None


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logger.bind(module="strategy")

    @abstractmethod
    def generate_signals(self, data: pl.DataFrame) -> list[Signal]:
        """Generate trading signals from data."""
        pass

    @abstractmethod
    def calculate_position_size(self, signal: Signal, portfolio_value: float) -> float:
        """Calculate position size for a signal."""
        pass
