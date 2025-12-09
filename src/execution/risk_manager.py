"""Risk management system."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
import numpy as np
from loguru import logger


@dataclass
class Position:
    """Trading position."""
    symbol: str
    side: int  # 1 = long, -1 = short
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime


class RiskManager:
    """Risk management and position monitoring."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logger.bind(module="risk_manager")

        self.max_drawdown = config.get("max_drawdown", 0.15)
        self.daily_var_limit = config.get("daily_var_limit", 0.02)
        self.max_positions = config.get("max_positions", 5)
        self.position_timeout = config.get("position_timeout", 480)  # minutes

        self.positions: dict[str, Position] = {}
        self.peak_equity = 0.0
        self.daily_pnl = 0.0

    def check_new_trade(
        self,
        symbol: str,
        size: float,
        portfolio_value: float,
    ) -> tuple[bool, str]:
        """Check if a new trade is allowed."""
        # Check max positions
        if len(self.positions) >= self.max_positions and symbol not in self.positions:
            return False, "Maximum positions reached"

        # Check drawdown
        if portfolio_value < self.peak_equity * (1 - self.max_drawdown):
            return False, "Maximum drawdown exceeded"

        # Check daily VAR
        if abs(self.daily_pnl / portfolio_value) > self.daily_var_limit:
            return False, "Daily VAR limit exceeded"

        return True, "OK"

    def update_position(
        self,
        symbol: str,
        side: int,
        quantity: float,
        entry_price: float,
        current_price: float,
    ) -> None:
        """Update or create position."""
        pnl = (current_price - entry_price) * quantity * side

        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=pnl,
            entry_time=datetime.utcnow(),
        )

    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close a position and return realized PnL."""
        if symbol not in self.positions:
            return 0.0

        pos = self.positions[symbol]
        realized_pnl = (exit_price - pos.entry_price) * pos.quantity * pos.side

        del self.positions[symbol]
        self.daily_pnl += realized_pnl

        return realized_pnl

    def get_portfolio_exposure(self) -> float:
        """Get total portfolio exposure."""
        return sum(abs(p.quantity * p.current_price) for p in self.positions.values())

    def check_position_timeouts(self) -> list[str]:
        """Check for positions that exceeded timeout."""
        now = datetime.utcnow()
        timeout = timedelta(minutes=self.position_timeout)

        timed_out = []
        for symbol, pos in self.positions.items():
            if now - pos.entry_time > timeout:
                timed_out.append(symbol)

        return timed_out

    def update_equity(self, equity: float) -> None:
        """Update peak equity for drawdown calculation."""
        self.peak_equity = max(self.peak_equity, equity)

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self.daily_pnl = 0.0
