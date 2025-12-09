#!/usr/bin/env python
"""Data collection script."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    config: str = typer.Option("config/base.yaml", help="Config file path"),
    symbols: str = typer.Option(None, help="Comma-separated symbols"),
    start_date: str = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, help="End date (YYYY-MM-DD)"),
    data_type: str = typer.Option("all", help="Data type: ohlcv, futures, sentiment, all"),
):
    """Collect market data from various sources."""
    from src.utils.config import load_config
    from src.utils.logging import setup_logging
    from src.data.storage import ParquetStore

    # Load config
    cfg = load_config(config)
    setup_logging(cfg.get("system", {}).get("log_level", "INFO"))

    # Parse parameters
    symbol_list = symbols.split(",") if symbols else cfg["data"]["symbols"]

    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start = datetime.now() - timedelta(days=365)

    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end = datetime.now()

    logger.info(f"Collecting {data_type} data for {symbol_list} from {start} to {end}")

    # Initialize storage
    store = ParquetStore(cfg["data"]["storage"]["raw_path"])

    async def collect():
        if data_type in ["ohlcv", "all"]:
            await collect_ohlcv(symbol_list, start, end, store)

        if data_type in ["futures", "all"]:
            await collect_futures(symbol_list, start, end, store)

        if data_type in ["sentiment", "all"]:
            await collect_sentiment(symbol_list, store)

    asyncio.run(collect())
    logger.info("Data collection complete")


async def collect_ohlcv(symbols, start, end, store):
    """Collect OHLCV data."""
    from src.data.ingestion import BinanceOHLCVCollector

    collector = BinanceOHLCVCollector()
    results = await collector.fetch_multiple(symbols, start, end)

    for symbol, df in results.items():
        store.save(df, "ohlcv", symbol)
        logger.info(f"Saved {len(df)} OHLCV rows for {symbol}")


async def collect_futures(symbols, start, end, store):
    """Collect futures metrics."""
    from src.data.ingestion import BinanceFuturesCollector

    collector = BinanceFuturesCollector()
    results = await collector.fetch_multiple(symbols, start, end)

    for symbol, df in results.items():
        store.save(df, "futures", symbol)
        logger.info(f"Saved {len(df)} futures rows for {symbol}")


async def collect_sentiment(symbols, store):
    """Collect sentiment data."""
    from src.data.ingestion import FearGreedCollector

    fg_collector = FearGreedCollector()
    fg_df = await fg_collector.fetch(limit=365)
    store.save(fg_df, "sentiment/fear_greed")
    logger.info(f"Saved {len(fg_df)} Fear & Greed rows")


if __name__ == "__main__":
    app()
