"""
Data ingestion module for various data sources.
"""

from src.data.ingestion.base import DataSource
from src.data.ingestion.binance_ohlcv import BinanceOHLCVCollector
from src.data.ingestion.binance_orderbook import BinanceOrderBookCollector
from src.data.ingestion.binance_futures import BinanceFuturesCollector
from src.data.ingestion.cryptopanic import CryptoPanicCollector
from src.data.ingestion.fear_greed import FearGreedCollector
from src.data.ingestion.reddit import RedditCollector

__all__ = [
    "DataSource",
    "BinanceOHLCVCollector",
    "BinanceOrderBookCollector",
    "BinanceFuturesCollector",
    "CryptoPanicCollector",
    "FearGreedCollector",
    "RedditCollector",
]
