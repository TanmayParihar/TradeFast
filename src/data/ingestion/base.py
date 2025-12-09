"""
Abstract base class for data sources.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd
import polars as pl
from loguru import logger


class DataSource(ABC):
    """Abstract base class for all data sources."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logger.bind(source=name)

    @abstractmethod
    async def fetch(
        self,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        Fetch data from the source.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start: Start datetime
            end: End datetime
            **kwargs: Additional parameters

        Returns:
            Polars DataFrame with fetched data
        """
        pass

    @abstractmethod
    def validate(self, data: pl.DataFrame) -> bool:
        """
        Validate fetched data.

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid
        """
        pass

    def to_pandas(self, data: pl.DataFrame) -> pd.DataFrame:
        """Convert Polars DataFrame to Pandas."""
        return data.to_pandas()

    def to_polars(self, data: pd.DataFrame) -> pl.DataFrame:
        """Convert Pandas DataFrame to Polars."""
        return pl.from_pandas(data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls: list[float] = []

    async def acquire(self) -> None:
        """Wait if rate limit would be exceeded."""
        import asyncio
        import time

        now = time.time()
        minute_ago = now - 60

        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if t > minute_ago]

        if len(self.calls) >= self.calls_per_minute:
            sleep_time = self.calls[0] - minute_ago
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.calls.append(time.time())


class RetryHandler:
    """Retry handler with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    async def execute(self, func, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        import asyncio

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.exponential_base**attempt),
                        self.max_delay,
                    )
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

        raise last_exception  # type: ignore
