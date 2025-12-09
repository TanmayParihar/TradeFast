#!/usr/bin/env python
"""Data collection script."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import typer
from loguru import logger

app = typer.Typer()


@app.command("ohlcv")
def collect_ohlcv_cmd(
    config: str = typer.Option("config/base.yaml", help="Config file path"),
    symbols: str = typer.Option(None, help="Comma-separated symbols"),
    start_date: str = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, help="End date (YYYY-MM-DD)"),
):
    """Collect OHLCV price data from Binance."""
    from src.utils.config import load_config
    from src.utils.logging import setup_logging
    from src.data.storage import ParquetStore
    from src.data.ingestion import BinanceOHLCVCollector

    cfg = load_config(config)
    setup_logging(cfg.get("system", {}).get("log_level", "INFO"))

    symbol_list = symbols.split(",") if symbols else cfg["data"]["symbols"]
    start, end = _parse_dates(start_date, end_date)

    logger.info(f"Collecting OHLCV data for {symbol_list}")
    store = ParquetStore(cfg["data"]["storage"]["raw_path"])

    async def collect():
        collector = BinanceOHLCVCollector()
        results = await collector.fetch_multiple(symbol_list, start, end)
        for symbol, df in results.items():
            store.save(df, "ohlcv", symbol)
            logger.info(f"Saved {len(df)} OHLCV rows for {symbol}")

    asyncio.run(collect())
    logger.info("OHLCV collection complete")


@app.command("futures")
def collect_futures_cmd(
    config: str = typer.Option("config/base.yaml", help="Config file path"),
    symbols: str = typer.Option(None, help="Comma-separated symbols"),
    start_date: str = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, help="End date (YYYY-MM-DD)"),
):
    """Collect futures metrics (funding rate, OI, L/S ratio) from Binance."""
    from src.utils.config import load_config
    from src.utils.logging import setup_logging
    from src.data.storage import ParquetStore
    from src.data.ingestion import BinanceFuturesCollector

    cfg = load_config(config)
    setup_logging(cfg.get("system", {}).get("log_level", "INFO"))

    symbol_list = symbols.split(",") if symbols else cfg["data"]["symbols"]
    start, end = _parse_dates(start_date, end_date)

    logger.info(f"Collecting futures data for {symbol_list}")
    store = ParquetStore(cfg["data"]["storage"]["raw_path"])

    async def collect():
        collector = BinanceFuturesCollector()
        results = await collector.fetch_multiple(symbol_list, start, end)
        for symbol, df in results.items():
            store.save(df, "futures", symbol)
            logger.info(f"Saved {len(df)} futures rows for {symbol}")

    asyncio.run(collect())
    logger.info("Futures collection complete")


@app.command("sentiment")
def collect_sentiment_cmd(
    config: str = typer.Option("config/base.yaml", help="Config file path"),
    days: int = typer.Option(365, help="Number of days to fetch"),
):
    """Collect sentiment data (Fear & Greed Index)."""
    from src.utils.config import load_config
    from src.utils.logging import setup_logging
    from src.data.storage import ParquetStore
    from src.data.ingestion import FearGreedCollector

    cfg = load_config(config)
    setup_logging(cfg.get("system", {}).get("log_level", "INFO"))

    logger.info(f"Collecting Fear & Greed Index for {days} days")
    store = ParquetStore(cfg["data"]["storage"]["raw_path"])

    async def collect():
        collector = FearGreedCollector()
        fg_df = await collector.fetch(limit=days)
        store.save(fg_df, "sentiment/fear_greed")
        logger.info(f"Saved {len(fg_df)} Fear & Greed rows")

    asyncio.run(collect())
    logger.info("Sentiment collection complete")


@app.command("all")
def collect_all_cmd(
    config: str = typer.Option("config/base.yaml", help="Config file path"),
    symbols: str = typer.Option(None, help="Comma-separated symbols"),
    start_date: str = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, help="End date (YYYY-MM-DD)"),
):
    """Collect all data types (OHLCV, futures, sentiment)."""
    from src.utils.config import load_config
    from src.utils.logging import setup_logging
    from src.data.storage import ParquetStore
    from src.data.ingestion import (
        BinanceOHLCVCollector,
        BinanceFuturesCollector,
        FearGreedCollector,
    )

    cfg = load_config(config)
    setup_logging(cfg.get("system", {}).get("log_level", "INFO"))

    symbol_list = symbols.split(",") if symbols else cfg["data"]["symbols"]
    start, end = _parse_dates(start_date, end_date)

    logger.info(f"Collecting all data for {symbol_list}")
    store = ParquetStore(cfg["data"]["storage"]["raw_path"])

    async def collect():
        # OHLCV
        ohlcv_collector = BinanceOHLCVCollector()
        ohlcv_results = await ohlcv_collector.fetch_multiple(symbol_list, start, end)
        for symbol, df in ohlcv_results.items():
            store.save(df, "ohlcv", symbol)
            logger.info(f"Saved {len(df)} OHLCV rows for {symbol}")

        # Futures
        futures_collector = BinanceFuturesCollector()
        futures_results = await futures_collector.fetch_multiple(symbol_list, start, end)
        for symbol, df in futures_results.items():
            store.save(df, "futures", symbol)
            logger.info(f"Saved {len(df)} futures rows for {symbol}")

        # Sentiment
        fg_collector = FearGreedCollector()
        fg_df = await fg_collector.fetch(limit=365)
        store.save(fg_df, "sentiment/fear_greed")
        logger.info(f"Saved {len(fg_df)} Fear & Greed rows")

    asyncio.run(collect())
    logger.info("All data collection complete")


def _parse_dates(start_date: str | None, end_date: str | None) -> tuple[datetime, datetime]:
    """Parse date strings to datetime objects."""
    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start = datetime.now() - timedelta(days=365)

    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end = datetime.now()

    return start, end


if __name__ == "__main__":
    app()
