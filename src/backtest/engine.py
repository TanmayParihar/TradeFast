"""Backtesting engine."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from src.backtest.cost_model import BinanceCostModel
from src.backtest.metrics import calculate_metrics


@dataclass
class BacktestResult:
    """Backtesting results."""
    returns: pd.Series
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict


class BacktestEngine:
    """Vectorized backtesting engine."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logger.bind(module="backtest")
        self.initial_capital = config.get("initial_capital", 100000)
        self.cost_model = BinanceCostModel(
            vip_level=config.get("vip_level", "Regular"),
            use_bnb=config.get("use_bnb", True),
        )

    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        position_sizes: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> BacktestResult:
        """
        Run vectorized backtest.

        Args:
            prices: Price array
            signals: Signal array (-1, 0, 1)
            position_sizes: Optional position sizes
            timestamps: Optional timestamps

        Returns:
            BacktestResult with metrics
        """
        n = len(prices)
        if position_sizes is None:
            position_sizes = np.ones(n) * 0.1

        # Calculate returns
        price_returns = np.diff(prices) / prices[:-1]
        price_returns = np.insert(price_returns, 0, 0)

        # Position changes
        positions = signals * position_sizes
        position_changes = np.diff(positions)
        position_changes = np.insert(position_changes, 0, positions[0])

        # Calculate costs
        costs = np.zeros(n)
        for i in range(n):
            if position_changes[i] != 0:
                trade_value = abs(position_changes[i]) * prices[i] * self.initial_capital
                cost = self.cost_model.total_cost(prices[i], abs(position_changes[i]) * self.initial_capital)
                costs[i] = cost["total"] / self.initial_capital

        # Strategy returns
        strategy_returns = positions[:-1] * price_returns[1:] - costs[1:]
        strategy_returns = np.insert(strategy_returns, 0, 0)

        # Equity curve
        equity = self.initial_capital * (1 + strategy_returns).cumprod()

        # Create trades DataFrame
        trade_indices = np.where(position_changes != 0)[0]
        trades_data = {
            "index": trade_indices,
            "price": prices[trade_indices],
            "signal": signals[trade_indices],
            "size": position_sizes[trade_indices],
        }
        if timestamps is not None:
            trades_data["timestamp"] = timestamps[trade_indices]
        trades_df = pd.DataFrame(trades_data)

        # Calculate metrics
        returns_series = pd.Series(strategy_returns)
        equity_series = pd.Series(equity)
        metrics = calculate_metrics(returns_series)

        self.logger.info(f"Backtest complete. Sharpe: {metrics['sharpe']:.2f}, Max DD: {metrics['max_drawdown']:.2%}")

        return BacktestResult(
            returns=returns_series,
            equity=equity_series,
            trades=trades_df,
            metrics=metrics,
        )


def walk_forward_optimize(
    df: pd.DataFrame,
    strategy_func,
    param_grid: dict,
    train_bars: int = 525600,
    test_bars: int = 131400,
    n_splits: int = 8,
    target: str = "sharpe",
) -> pd.DataFrame:
    """Walk-forward optimization."""
    from itertools import product

    results = []
    step = test_bars
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    for i in range(n_splits):
        start = i * step
        train_end = start + train_bars
        test_end = train_end + test_bars

        if test_end > len(df):
            break

        train_data = df.iloc[start:train_end]
        test_data = df.iloc[train_end:test_end]

        best_score = -np.inf
        best_params = None

        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            metrics = strategy_func(train_data, **param_dict)
            score = metrics.get(target, 0)

            if score > best_score:
                best_score = score
                best_params = param_dict

        oos_metrics = strategy_func(test_data, **best_params)

        results.append({
            "split": i,
            "params": best_params,
            "is_score": best_score,
            "oos_score": oos_metrics.get(target, 0),
            **{f"oos_{k}": v for k, v in oos_metrics.items()},
        })

    return pd.DataFrame(results)
