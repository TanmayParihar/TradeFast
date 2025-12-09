#!/usr/bin/env python
"""Live trading script."""

import asyncio
from pathlib import Path
import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    config: str = typer.Option("config/production.yaml", help="Config file path"),
    symbols: str = typer.Option(None, help="Comma-separated symbols"),
    model_dir: str = typer.Option("checkpoints", help="Model directory"),
    paper: bool = typer.Option(True, help="Paper trading mode"),
):
    """Run live trading bot."""
    from src.utils.config import load_config
    from src.utils.logging import setup_logging
    from src.execution import RiskManager

    # Setup
    cfg = load_config(config)
    setup_logging(cfg.get("system", {}).get("log_level", "INFO"))

    symbol_list = symbols.split(",") if symbols else cfg["data"]["symbols"]

    logger.info(f"Starting {'paper' if paper else 'live'} trading for {symbol_list}")

    # Initialize components
    risk_manager = RiskManager(cfg.get("strategy", {}).get("risk_management", {}))

    async def run_trading():
        while True:
            try:
                for symbol in symbol_list:
                    await process_symbol(symbol, cfg, model_dir, risk_manager, paper)

                await asyncio.sleep(60)  # 1-minute loop

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)

    asyncio.run(run_trading())


async def process_symbol(symbol, cfg, model_dir, risk_manager, paper):
    """Process a single symbol."""
    # This would be expanded to:
    # 1. Fetch latest data
    # 2. Compute features
    # 3. Generate predictions
    # 4. Check risk limits
    # 5. Execute trades
    pass


if __name__ == "__main__":
    app()
