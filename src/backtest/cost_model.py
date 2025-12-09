"""Transaction cost model for Binance Futures."""

import numpy as np


BINANCE_FEES = {
    "Regular": {"maker": 0.0002, "taker": 0.0005},
    "VIP1": {"maker": 0.00016, "taker": 0.0004},
    "VIP2": {"maker": 0.00014, "taker": 0.00035},
}


class BinanceCostModel:
    """Transaction cost model for Binance Futures."""

    def __init__(self, vip_level: str = "Regular", use_bnb: bool = True):
        self.fees = BINANCE_FEES[vip_level].copy()
        if use_bnb:
            self.fees = {k: v * 0.9 for k, v in self.fees.items()}

    def slippage(self, order_usd: float, spread_bps: float = 1.0) -> float:
        """Calculate slippage using square-root market impact model."""
        base = spread_bps / 20000
        impact = 0.1 * np.sqrt(order_usd / 100000) / 10000
        return base + impact

    def funding_cost(self, position_value: float, hours: float, rate: float = 0.0001) -> float:
        """Calculate funding cost (funding every 8 hours)."""
        return position_value * rate * (hours / 8)

    def total_cost(
        self,
        price: float,
        size: float,
        hours: float = 0,
        maker: bool = False,
    ) -> dict:
        """Calculate total transaction costs."""
        value = price * size
        fee = self.fees["maker" if maker else "taker"]
        slip = self.slippage(value)

        return {
            "fees": value * fee * 2,  # Entry + exit
            "slippage": slip * value * 2,
            "funding": self.funding_cost(value, hours),
            "total": value * fee * 2 + slip * value * 2 + self.funding_cost(value, hours),
            "total_bps": (fee * 2 + slip * 2) * 10000,
        }
