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
    start_date: str = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, help="End date (YYYY-MM-DD)"),
):
    """Build features from raw data."""
    from src.utils.config import load_config
    from src.utils.logging import setup_logging
    from src.data.loaders import ExistingDataLoader
    from src.data.storage import ParquetStore
    from src.features import FeaturePipeline

    # Load config
    cfg = load_config(config)
    setup_logging(cfg.get("system", {}).get("log_level", "INFO"))

    # Get raw data path
    raw_path = cfg["data"]["storage"]["raw_path"]

    # Initialize data loader for existing parquet files
    try:
        loader = ExistingDataLoader(raw_path)
        available_symbols = loader.list_available_symbols()
        logger.info(f"Found {len(available_symbols)} symbols in {raw_path}: {available_symbols}")
    except FileNotFoundError:
        logger.error(f"Raw data path not found: {raw_path}")
        raise typer.Exit(1)

    # Determine which symbols to process
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    else:
        symbol_list = cfg["data"]["symbols"]

    # Filter to available symbols
    symbol_list = [s for s in symbol_list if s in available_symbols]
    if not symbol_list:
        logger.error("No matching symbols found in raw data")
        raise typer.Exit(1)

    logger.info(f"Processing symbols: {symbol_list}")

    # Initialize feature storage and pipeline
    feature_store = ParquetStore(output_dir)
    pipeline = FeaturePipeline(cfg.get("features", {}))

    for symbol in symbol_list:
        logger.info(f"Building features for {symbol}")

        # Show file info
        info = loader.get_file_info(symbol)
        if info.get("exists"):
            logger.info(f"  File: {info['file_path']}")
            logger.info(f"  Rows: {info.get('row_count', 'unknown'):,}")
            logger.info(f"  Size: {info.get('file_size_mb', 0):.2f} MB")
            if info.get("date_range"):
                logger.info(f"  Date range: {info['date_range'].get('start')} to {info['date_range'].get('end')}")

        # Load OHLCV data
        ohlcv_df = loader.load_ohlcv(symbol, start=start_date, end=end_date)
        if ohlcv_df.is_empty():
            logger.warning(f"No OHLCV data for {symbol}")
            continue

        logger.info(f"Loaded {len(ohlcv_df):,} rows for {symbol}")

        # Run feature pipeline (OHLCV only for now, other data sources can be added later)
        features_df = pipeline.run(
            ohlcv_df=ohlcv_df,
            futures_df=None,  # Add if available
            fear_greed_df=None,  # Add if available
        )

        # Save features
        feature_store.save(features_df, "features", symbol)
        logger.info(f"Saved {len(features_df):,} feature rows for {symbol}")

    logger.info("Feature engineering complete")


@app.command("info")
def show_info(
    config: str = typer.Option("config/base.yaml", help="Config file path"),
):
    """Show information about available raw data."""
    from src.utils.config import load_config
    from src.data.loaders import ExistingDataLoader

    cfg = load_config(config)
    raw_path = cfg["data"]["storage"]["raw_path"]

    try:
        loader = ExistingDataLoader(raw_path)
    except FileNotFoundError:
        logger.error(f"Raw data path not found: {raw_path}")
        raise typer.Exit(1)

    symbols = loader.list_available_symbols()
    print(f"\nRaw data directory: {raw_path}")
    print(f"Available symbols: {len(symbols)}\n")

    for symbol in symbols:
        info = loader.get_file_info(symbol)
        print(f"{symbol}:")
        print(f"  Rows: {info.get('row_count', 'unknown'):,}")
        print(f"  Size: {info.get('file_size_mb', 0):.2f} MB")
        print(f"  Columns: {info.get('columns', [])}")
        if info.get("date_range"):
            print(f"  Range: {info['date_range'].get('start')} to {info['date_range'].get('end')}")
        print()


@app.command("validate")
def validate_data(
    config: str = typer.Option("config/base.yaml", help="Config file path"),
    symbols: str = typer.Option(None, help="Comma-separated symbols"),
):
    """Validate raw data integrity."""
    from src.utils.config import load_config
    from src.data.loaders import ExistingDataLoader

    cfg = load_config(config)
    raw_path = cfg["data"]["storage"]["raw_path"]

    try:
        loader = ExistingDataLoader(raw_path)
    except FileNotFoundError:
        logger.error(f"Raw data path not found: {raw_path}")
        raise typer.Exit(1)

    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    else:
        symbol_list = loader.list_available_symbols()

    print(f"\nValidating {len(symbol_list)} symbols...\n")

    all_valid = True
    for symbol in symbol_list:
        result = loader.validate_data(symbol)
        status = "✓" if result["valid"] else "✗"
        print(f"{status} {symbol}: {result.get('row_count', 0):,} rows")
        if result.get("issues"):
            for issue in result["issues"]:
                print(f"    - {issue}")
            all_valid = False

    if all_valid:
        print("\nAll data validated successfully!")
    else:
        print("\nSome data validation issues found.")


if __name__ == "__main__":
    app()
