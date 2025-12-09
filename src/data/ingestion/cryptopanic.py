"""
CryptoPanic news collector.
"""

import os
from datetime import datetime, timezone
from typing import Any

import aiohttp
import polars as pl

from src.data.ingestion.base import DataSource, RateLimiter


class CryptoPanicCollector(DataSource):
    """Collector for CryptoPanic news API."""

    BASE_URL = "https://cryptopanic.com/api/v1"

    # Map symbols to CryptoPanic currencies
    SYMBOL_MAP = {
        "BTCUSDT": "BTC",
        "ETHUSDT": "ETH",
        "BNBUSDT": "BNB",
        "SOLUSDT": "SOL",
        "ADAUSDT": "ADA",
        "AVAXUSDT": "AVAX",
        "DOGEUSDT": "DOGE",
        "DOTUSDT": "DOT",
        "MATICUSDT": "MATIC",
        "XRPUSDT": "XRP",
    }

    def __init__(self, api_token: str | None = None):
        super().__init__("cryptopanic")
        self.api_token = api_token or os.getenv("CRYPTOPANIC_TOKEN")
        self.rate_limiter = RateLimiter(calls_per_minute=10)  # Free tier limit

        if not self.api_token:
            self.logger.warning(
                "No CryptoPanic API token provided. Set CRYPTOPANIC_TOKEN env var."
            )

    async def fetch(
        self,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        filter_type: str = "hot",
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        Fetch news from CryptoPanic.

        Args:
            symbol: Trading symbol to filter (e.g., 'BTCUSDT')
            start: Start datetime (not supported by API, filtered locally)
            end: End datetime (not supported by API, filtered locally)
            filter_type: 'hot', 'rising', 'bullish', 'bearish', 'important', 'lol'

        Returns:
            DataFrame with news data
        """
        if not self.api_token:
            self.logger.error("No API token available")
            return pl.DataFrame()

        await self.rate_limiter.acquire()

        params: dict[str, Any] = {
            "auth_token": self.api_token,
            "filter": filter_type,
            "public": "true",
        }

        if symbol:
            currency = self.SYMBOL_MAP.get(symbol, symbol.replace("USDT", ""))
            params["currencies"] = currency

        all_posts: list[dict] = []

        async with aiohttp.ClientSession() as session:
            url = f"{self.BASE_URL}/posts/"

            # Fetch multiple pages
            for page in range(1, 6):  # Max 5 pages
                params["page"] = page

                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 403:
                            self.logger.error("API rate limit exceeded")
                            break

                        response.raise_for_status()
                        data = await response.json()

                    posts = data.get("results", [])
                    if not posts:
                        break

                    all_posts.extend(posts)

                    if not data.get("next"):
                        break

                    await self.rate_limiter.acquire()

                except aiohttp.ClientError as e:
                    self.logger.error(f"Error fetching news: {e}")
                    break

        if not all_posts:
            return pl.DataFrame()

        df = self._process_posts(all_posts)

        # Filter by date range
        if start:
            df = df.filter(pl.col("timestamp") >= start)
        if end:
            df = df.filter(pl.col("timestamp") <= end)

        return df

    def _process_posts(self, posts: list[dict]) -> pl.DataFrame:
        """Process raw posts into DataFrame."""
        rows = []

        for post in posts:
            # Parse timestamp
            created_at = post.get("created_at", "")
            if created_at:
                timestamp = datetime.fromisoformat(
                    created_at.replace("Z", "+00:00")
                )
            else:
                timestamp = datetime.now(timezone.utc)

            # Extract currencies
            currencies = [c.get("code", "") for c in post.get("currencies", [])]

            # Get votes
            votes = post.get("votes", {})

            rows.append({
                "timestamp": timestamp,
                "id": post.get("id"),
                "title": post.get("title", ""),
                "kind": post.get("kind", "news"),
                "domain": post.get("domain", ""),
                "source": post.get("source", {}).get("title", ""),
                "url": post.get("url", ""),
                "currencies": currencies,
                "votes_positive": votes.get("positive", 0),
                "votes_negative": votes.get("negative", 0),
                "votes_important": votes.get("important", 0),
                "votes_liked": votes.get("liked", 0),
                "votes_disliked": votes.get("disliked", 0),
                "votes_lol": votes.get("lol", 0),
                "votes_toxic": votes.get("toxic", 0),
                "votes_saved": votes.get("saved", 0),
            })

        return pl.DataFrame(rows).sort("timestamp", descending=True)

    def validate(self, data: pl.DataFrame) -> bool:
        """Validate news data."""
        if data.is_empty():
            return True  # Empty is valid (no news)

        required_cols = ["timestamp", "title"]
        return all(col in data.columns for col in required_cols)

    def compute_sentiment_score(self, row: dict) -> float:
        """Compute sentiment score from votes."""
        positive = row.get("votes_positive", 0) + row.get("votes_liked", 0)
        negative = row.get("votes_negative", 0) + row.get("votes_disliked", 0)
        toxic = row.get("votes_toxic", 0)

        total = positive + negative + toxic + 1  # Add 1 to avoid division by zero

        return (positive - negative - toxic * 2) / total

    def add_sentiment_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add sentiment scores to DataFrame."""
        return df.with_columns([
            (
                (pl.col("votes_positive") + pl.col("votes_liked"))
                - (pl.col("votes_negative") + pl.col("votes_disliked"))
                - (pl.col("votes_toxic") * 2)
            )
            / (
                pl.col("votes_positive")
                + pl.col("votes_liked")
                + pl.col("votes_negative")
                + pl.col("votes_disliked")
                + pl.col("votes_toxic")
                + 1
            ).alias("sentiment_score")
        ])
