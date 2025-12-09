#!/usr/bin/env python
"""Feature engineering script."""

from pathlib import Path
import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    config: str = typer.Option("config/base.yaml", help="Config file path"),
    symbols: str = typer.Option(None, help="Comma-separated symbols"),
    output_dir: str = typer.Option("data/features", help="Output directory"),
):
    """Build features from raw data."""
    from src.utils.config import load_config
    from src.utils.logging import setup_logging
    from src.data.storage import ParquetStore
    from src.features import FeaturePipeline

    # Load config
    cfg = load_config(config)
    setup_logging(cfg.get("system", {}).get("log_level", "INFO"))

    symbol_list = symbols.split(",") if symbols else cfg["data"]["symbols"]

    # Initialize
    raw_store = ParquetStore(cfg["data"]["storage"]["raw_path"])
    feature_store = ParquetStore(output_dir)
    pipeline = FeaturePipeline(cfg.get("features", {}))

    for symbol in symbol_list:
        logger.info(f"Building features for {symbol}")

        # Load raw data
        ohlcv_df = raw_store.load("ohlcv", symbol)
        if ohlcv_df.is_empty():
            logger.warning(f"No OHLCV data for {symbol}")
            continue

        futures_df = raw_store.load("futures", symbol)
        fg_df = raw_store.load("sentiment/fear_greed")

        # Run pipeline
        features_df = pipeline.run(
            ohlcv_df=ohlcv_df,
            futures_df=futures_df if not futures_df.is_empty() else None,
            fear_greed_df=fg_df if not fg_df.is_empty() else None,
        )

        # Save features
        feature_store.save(features_df, "features", symbol)
        logger.info(f"Saved {len(features_df)} feature rows for {symbol}")

    logger.info("Feature engineering complete")


if __name__ == "__main__":
    app()
