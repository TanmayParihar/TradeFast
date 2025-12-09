"""Performance metrics for backtesting."""

import numpy as np
import pandas as pd

MINUTES_PER_YEAR = 525600


def sharpe_ratio(returns: pd.Series | np.ndarray, rf: float = 0) -> float:
    """Calculate annualized Sharpe ratio for minute data."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    if returns.std() == 0:
        return np.nan
    excess = returns.mean() - rf / MINUTES_PER_YEAR
    return excess / returns.std() * np.sqrt(MINUTES_PER_YEAR)


def sortino_ratio(returns: pd.Series | np.ndarray, rf: float = 0) -> float:
    """Calculate Sortino ratio."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    downside = returns[returns < 0].std()
    if downside == 0:
        return np.nan
    excess = returns.mean() - rf / MINUTES_PER_YEAR
    return excess / downside * np.sqrt(MINUTES_PER_YEAR)


def max_drawdown(equity: pd.Series | np.ndarray) -> float:
    """Calculate maximum drawdown."""
    if isinstance(equity, np.ndarray):
        equity = pd.Series(equity)
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return dd.min()


def calmar_ratio(returns: pd.Series | np.ndarray) -> float:
    """Calculate Calmar ratio."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    equity = (1 + returns).cumprod()
    ann_ret = (equity.iloc[-1] ** (MINUTES_PER_YEAR / len(returns))) - 1
    mdd = abs(max_drawdown(equity))
    return ann_ret / mdd if mdd > 0 else np.nan


def win_rate(returns: pd.Series | np.ndarray) -> float:
    """Calculate win rate."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    return (returns > 0).mean()


def profit_factor(returns: pd.Series | np.ndarray) -> float:
    """Calculate profit factor."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return gains / losses if losses > 0 else np.nan


def calculate_metrics(returns: pd.Series | np.ndarray, n_trials: int = 100) -> dict:
    """Calculate all performance metrics."""
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    equity = (1 + returns).cumprod()

    return {
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "calmar": calmar_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "total_return": equity.iloc[-1] - 1,
        "n_trades": (returns != 0).sum(),
        "avg_return": returns[returns != 0].mean() if (returns != 0).any() else 0,
    }
