"""
Reddit sentiment collector using PRAW.
"""

import os
from datetime import datetime, timezone
from typing import Any

import polars as pl

from src.data.ingestion.base import DataSource, RateLimiter


class RedditCollector(DataSource):
    """Collector for Reddit sentiment data."""

    SUBREDDITS = [
        "cryptocurrency",
        "bitcoin",
        "ethereum",
        "CryptoMarkets",
        "altcoin",
    ]

    # Map symbols to search terms
    SYMBOL_SEARCH = {
        "BTCUSDT": ["bitcoin", "btc"],
        "ETHUSDT": ["ethereum", "eth"],
        "BNBUSDT": ["binance", "bnb"],
        "SOLUSDT": ["solana", "sol"],
        "ADAUSDT": ["cardano", "ada"],
        "AVAXUSDT": ["avalanche", "avax"],
        "DOGEUSDT": ["dogecoin", "doge"],
        "DOTUSDT": ["polkadot", "dot"],
        "MATICUSDT": ["polygon", "matic"],
        "XRPUSDT": ["ripple", "xrp"],
    }

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
    ):
        super().__init__("reddit")
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.getenv(
            "REDDIT_USER_AGENT", "crypto_bot:v1.0.0"
        )
        self.rate_limiter = RateLimiter(calls_per_minute=60)
        self._reddit = None

    def _get_reddit(self):
        """Lazy initialization of Reddit client."""
        if self._reddit is None:
            try:
                import praw

                self._reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                )
            except ImportError:
                self.logger.error("praw not installed. Run: pip install praw")
                raise
            except Exception as e:
                self.logger.error(f"Failed to initialize Reddit client: {e}")
                raise

        return self._reddit

    async def fetch(
        self,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
        subreddits: list[str] | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        Fetch Reddit posts and comments.

        Args:
            symbol: Trading symbol to search for
            start: Start datetime
            end: End datetime
            limit: Maximum posts per subreddit
            subreddits: List of subreddits to search

        Returns:
            DataFrame with Reddit data
        """
        # Note: PRAW is synchronous, so we run in thread pool
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._fetch_sync,
            symbol,
            start,
            end,
            limit,
            subreddits or self.SUBREDDITS,
        )

    def _fetch_sync(
        self,
        symbol: str | None,
        start: datetime | None,
        end: datetime | None,
        limit: int,
        subreddits: list[str],
    ) -> pl.DataFrame:
        """Synchronous fetch implementation."""
        if not self.client_id or not self.client_secret:
            self.logger.warning("Reddit credentials not configured")
            return pl.DataFrame()

        try:
            reddit = self._get_reddit()
        except Exception:
            return pl.DataFrame()

        posts: list[dict] = []
        search_terms = []

        if symbol:
            search_terms = self.SYMBOL_SEARCH.get(symbol, [symbol.replace("USDT", "")])

        for subreddit_name in subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)

                # Get hot posts
                for post in subreddit.hot(limit=limit):
                    post_data = self._extract_post_data(post, subreddit_name)

                    # Filter by symbol if specified
                    if search_terms:
                        title_lower = post.title.lower()
                        if not any(term.lower() in title_lower for term in search_terms):
                            continue

                    # Filter by date
                    post_time = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                    if start and post_time < start:
                        continue
                    if end and post_time > end:
                        continue

                    posts.append(post_data)

            except Exception as e:
                self.logger.warning(f"Error fetching from r/{subreddit_name}: {e}")
                continue

        if not posts:
            return pl.DataFrame()

        return pl.DataFrame(posts).sort("timestamp", descending=True)

    def _extract_post_data(self, post, subreddit_name: str) -> dict:
        """Extract relevant data from a Reddit post."""
        return {
            "timestamp": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
            "id": post.id,
            "subreddit": subreddit_name,
            "title": post.title,
            "selftext": post.selftext[:1000] if post.selftext else "",
            "score": post.score,
            "upvote_ratio": post.upvote_ratio,
            "num_comments": post.num_comments,
            "author": str(post.author) if post.author else "[deleted]",
            "url": post.url,
            "is_self": post.is_self,
            "over_18": post.over_18,
            "stickied": post.stickied,
        }

    def validate(self, data: pl.DataFrame) -> bool:
        """Validate Reddit data."""
        if data.is_empty():
            return True  # Empty is valid (no posts)

        required_cols = ["timestamp", "title", "score"]
        return all(col in data.columns for col in required_cols)

    def compute_engagement_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add engagement score to DataFrame."""
        return df.with_columns([
            (
                pl.col("score") * pl.col("upvote_ratio")
                + pl.col("num_comments") * 0.5
            ).alias("engagement_score")
        ])

    def aggregate_sentiment(
        self, df: pl.DataFrame, interval: str = "1h"
    ) -> pl.DataFrame:
        """Aggregate Reddit metrics over time intervals."""
        if df.is_empty():
            return df

        return (
            df.group_by_dynamic("timestamp", every=interval)
            .agg([
                pl.count().alias("post_count"),
                pl.col("score").mean().alias("avg_score"),
                pl.col("score").sum().alias("total_score"),
                pl.col("upvote_ratio").mean().alias("avg_upvote_ratio"),
                pl.col("num_comments").sum().alias("total_comments"),
                pl.col("num_comments").mean().alias("avg_comments"),
            ])
            .sort("timestamp")
        )
