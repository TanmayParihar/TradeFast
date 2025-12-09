"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import polars as pl


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data."""
    n = 1000
    np.random.seed(42)

    close = 100 * np.cumprod(1 + np.random.randn(n) * 0.01)
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_ = close * (1 + np.random.randn(n) * 0.002)
    volume = np.random.rand(n) * 1000000

    return pl.DataFrame({
        "timestamp": pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1) + pl.duration(minutes=n-1),
            interval="1m",
            eager=True,
        ),
        "symbol": "BTCUSDT",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def sample_features():
    """Generate sample feature data."""
    n = 1000
    np.random.seed(42)

    return np.random.randn(n, 50)


@pytest.fixture
def sample_labels():
    """Generate sample labels."""
    n = 1000
    np.random.seed(42)
    return np.random.randint(0, 3, n)
