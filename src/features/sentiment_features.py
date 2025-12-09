"""
Sentiment and NLP-based feature engineering.
"""

import os
from typing import Any

import numpy as np
import polars as pl
import torch
from loguru import logger


class SentimentFeatures:
    """Generate features from news and social media sentiment."""

    def __init__(self, config: dict[str, Any] | None = None, use_gpu: bool = True):
        self.config = config or {}
        self.logger = logger.bind(module="sentiment_features")
        self.use_gpu = use_gpu and torch.cuda.is_available()

        self._finbert_model = None
        self._finbert_tokenizer = None

    def _load_finbert(self):
        """Lazy load FinBERT model."""
        if self._finbert_model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_name = "ProsusAI/finbert"
            self.logger.info(f"Loading FinBERT model: {model_name}")

            self._finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._finbert_model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )

            if self.use_gpu:
                self._finbert_model = self._finbert_model.cuda().half()  # FP16
            else:
                self._finbert_model = self._finbert_model.eval()

        return self._finbert_model, self._finbert_tokenizer

    def analyze_text(self, text: str) -> dict[str, float]:
        """
        Analyze sentiment of a single text using FinBERT.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with positive, negative, neutral scores
        """
        model, tokenizer = self._load_finbert()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        if self.use_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]

        return {
            "positive": probs[0].item(),
            "negative": probs[1].item(),
            "neutral": probs[2].item(),
            "sentiment_score": probs[0].item() - probs[1].item(),  # -1 to 1
        }

    def analyze_batch(
        self, texts: list[str], batch_size: int = 16
    ) -> list[dict[str, float]]:
        """
        Analyze sentiment of multiple texts in batches.

        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing

        Returns:
            List of sentiment dictionaries
        """
        model, tokenizer = self._load_finbert()
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            if self.use_gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1)

            for j in range(len(batch_texts)):
                results.append({
                    "positive": probs[j, 0].item(),
                    "negative": probs[j, 1].item(),
                    "neutral": probs[j, 2].item(),
                    "sentiment_score": probs[j, 0].item() - probs[j, 1].item(),
                })

        return results

    def add_news_features(
        self,
        ohlcv_df: pl.DataFrame,
        news_df: pl.DataFrame,
        analyze_text: bool = True,
    ) -> pl.DataFrame:
        """
        Add news sentiment features to OHLCV DataFrame.

        Args:
            ohlcv_df: OHLCV DataFrame with timestamp
            news_df: News DataFrame with timestamp, title, text
            analyze_text: Whether to run FinBERT analysis

        Returns:
            DataFrame with news features
        """
        if news_df.is_empty():
            self.logger.warning("News DataFrame is empty")
            return ohlcv_df

        # Analyze sentiment if requested
        if analyze_text and "title" in news_df.columns:
            if "sentiment_score" not in news_df.columns:
                titles = news_df["title"].to_list()
                sentiments = self.analyze_batch(titles)

                news_df = news_df.with_columns([
                    pl.Series([s["sentiment_score"] for s in sentiments])
                    .alias("sentiment_score"),
                    pl.Series([s["positive"] for s in sentiments])
                    .alias("sentiment_positive"),
                    pl.Series([s["negative"] for s in sentiments])
                    .alias("sentiment_negative"),
                ])

        # Aggregate to hourly level
        news_df = news_df.with_columns([
            pl.col("timestamp").dt.truncate("1h").alias("timestamp_hour")
        ])

        news_agg = news_df.group_by("timestamp_hour").agg([
            pl.count().alias("news_count"),
            pl.col("sentiment_score").mean().alias("news_sentiment_avg"),
            pl.col("sentiment_score").std().alias("news_sentiment_std"),
            pl.col("sentiment_score").max().alias("news_sentiment_max"),
            pl.col("sentiment_score").min().alias("news_sentiment_min"),
            pl.col("sentiment_positive").mean().alias("news_positive_avg"),
            pl.col("sentiment_negative").mean().alias("news_negative_avg"),
        ])

        # Join with OHLCV (expand to minute level)
        ohlcv_df = ohlcv_df.with_columns([
            pl.col("timestamp").dt.truncate("1h").alias("timestamp_hour")
        ])

        result = ohlcv_df.join(
            news_agg,
            on="timestamp_hour",
            how="left",
        ).drop("timestamp_hour")

        # Forward fill and add derived features
        for col in ["news_count", "news_sentiment_avg", "news_positive_avg", "news_negative_avg"]:
            if col in result.columns:
                result = result.with_columns([
                    pl.col(col).fill_null(0 if col == "news_count" else 0.0)
                ])

        # Add momentum features
        if "news_sentiment_avg" in result.columns:
            result = result.with_columns([
                pl.col("news_sentiment_avg").rolling_mean(60).alias("news_sentiment_ma_1h"),
                (pl.col("news_sentiment_avg") - pl.col("news_sentiment_avg").shift(60))
                .alias("news_sentiment_momentum"),
            ])

        return result

    def add_fear_greed_features(
        self,
        ohlcv_df: pl.DataFrame,
        fear_greed_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Add Fear & Greed Index features.

        Args:
            ohlcv_df: OHLCV DataFrame
            fear_greed_df: Fear & Greed DataFrame

        Returns:
            DataFrame with F&G features
        """
        if fear_greed_df.is_empty():
            return ohlcv_df

        # F&G is daily, expand to minute level
        fear_greed_df = fear_greed_df.with_columns([
            pl.col("timestamp").dt.date().alias("date")
        ])

        ohlcv_df = ohlcv_df.with_columns([
            pl.col("timestamp").dt.date().alias("date")
        ])

        fg_daily = fear_greed_df.group_by("date").agg([
            pl.col("fear_greed_value").first(),
            pl.col("fear_greed_normalized").first(),
        ])

        result = ohlcv_df.join(fg_daily, on="date", how="left")

        # Forward fill
        result = result.with_columns([
            pl.col("fear_greed_value").fill_null(strategy="forward"),
            pl.col("fear_greed_normalized").fill_null(strategy="forward"),
        ])

        # Add derived features
        result = result.with_columns([
            # F&G change
            (pl.col("fear_greed_value") - pl.col("fear_greed_value").shift(1440))
            .alias("fg_change_24h"),

            # F&G regime
            pl.when(pl.col("fear_greed_value") <= 25)
            .then(-2)  # Extreme Fear
            .when(pl.col("fear_greed_value") <= 45)
            .then(-1)  # Fear
            .when(pl.col("fear_greed_value") <= 55)
            .then(0)   # Neutral
            .when(pl.col("fear_greed_value") <= 75)
            .then(1)   # Greed
            .otherwise(2)  # Extreme Greed
            .alias("fg_regime"),

            # Extreme levels (contrarian signals)
            pl.when(pl.col("fear_greed_value") <= 20)
            .then(1)  # Buy signal
            .when(pl.col("fear_greed_value") >= 80)
            .then(-1)  # Sell signal
            .otherwise(0)
            .alias("fg_extreme_signal"),
        ])

        return result.drop("date")

    def add_reddit_features(
        self,
        ohlcv_df: pl.DataFrame,
        reddit_df: pl.DataFrame,
        analyze_text: bool = False,
    ) -> pl.DataFrame:
        """
        Add Reddit sentiment features.

        Args:
            ohlcv_df: OHLCV DataFrame
            reddit_df: Reddit posts DataFrame
            analyze_text: Whether to run FinBERT on titles

        Returns:
            DataFrame with Reddit features
        """
        if reddit_df.is_empty():
            return ohlcv_df

        # Analyze sentiment if requested
        if analyze_text and "title" in reddit_df.columns:
            if "sentiment_score" not in reddit_df.columns:
                titles = reddit_df["title"].to_list()
                sentiments = self.analyze_batch(titles)

                reddit_df = reddit_df.with_columns([
                    pl.Series([s["sentiment_score"] for s in sentiments])
                    .alias("sentiment_score"),
                ])

        # Aggregate to hourly level
        reddit_df = reddit_df.with_columns([
            pl.col("timestamp").dt.truncate("1h").alias("timestamp_hour")
        ])

        reddit_agg = reddit_df.group_by("timestamp_hour").agg([
            pl.count().alias("reddit_post_count"),
            pl.col("score").sum().alias("reddit_total_score"),
            pl.col("score").mean().alias("reddit_avg_score"),
            pl.col("num_comments").sum().alias("reddit_total_comments"),
            pl.col("upvote_ratio").mean().alias("reddit_avg_upvote_ratio"),
        ])

        # Add sentiment if available
        if "sentiment_score" in reddit_df.columns:
            sentiment_agg = reddit_df.group_by("timestamp_hour").agg([
                pl.col("sentiment_score").mean().alias("reddit_sentiment_avg"),
            ])
            reddit_agg = reddit_agg.join(sentiment_agg, on="timestamp_hour", how="left")

        # Join with OHLCV
        ohlcv_df = ohlcv_df.with_columns([
            pl.col("timestamp").dt.truncate("1h").alias("timestamp_hour")
        ])

        result = ohlcv_df.join(
            reddit_agg,
            on="timestamp_hour",
            how="left",
        ).drop("timestamp_hour")

        # Fill nulls
        for col in reddit_agg.columns:
            if col != "timestamp_hour" and col in result.columns:
                result = result.with_columns([
                    pl.col(col).fill_null(0)
                ])

        # Engagement features
        result = result.with_columns([
            (pl.col("reddit_total_score") + pl.col("reddit_total_comments") * 2)
            .alias("reddit_engagement"),
        ])

        # Momentum
        result = result.with_columns([
            pl.col("reddit_engagement").rolling_mean(60).alias("reddit_engagement_ma"),
            (pl.col("reddit_engagement") / (pl.col("reddit_engagement").rolling_mean(60) + 1))
            .alias("reddit_engagement_ratio"),
        ])

        return result

    def create_sentiment_composite(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create composite sentiment score from all available sources.

        Args:
            df: DataFrame with individual sentiment features

        Returns:
            DataFrame with composite sentiment
        """
        components = []
        weights = []

        # News sentiment (high weight)
        if "news_sentiment_avg" in df.columns:
            components.append("news_sentiment_avg")
            weights.append(0.4)

        # Fear & Greed (medium weight, inverted for contrarian)
        if "fear_greed_normalized" in df.columns:
            # Normalize to -1 to 1
            df = df.with_columns([
                (pl.col("fear_greed_normalized") * 2 - 1).alias("fg_normalized_centered")
            ])
            components.append("fg_normalized_centered")
            weights.append(0.3)

        # Reddit sentiment (lower weight)
        if "reddit_sentiment_avg" in df.columns:
            components.append("reddit_sentiment_avg")
            weights.append(0.3)

        if not components:
            return df

        # Weighted average
        weighted_sum = sum(
            pl.col(c) * w for c, w in zip(components, weights)
        )
        total_weight = sum(weights)

        df = df.with_columns([
            (weighted_sum / total_weight).alias("sentiment_composite")
        ])

        # Sentiment regime
        df = df.with_columns([
            pl.when(pl.col("sentiment_composite") > 0.3)
            .then(1)
            .when(pl.col("sentiment_composite") < -0.3)
            .then(-1)
            .otherwise(0)
            .alias("sentiment_regime")
        ])

        return df
