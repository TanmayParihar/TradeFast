#!/usr/bin/env python
"""Backtesting script."""

from pathlib import Path
import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    config: str = typer.Option("config/base.yaml", help="Config file path"),
    symbols: str = typer.Option(None, help="Comma-separated symbols"),
    model_dir: str = typer.Option("checkpoints", help="Model directory"),
    output_dir: str = typer.Option("experiments", help="Output directory"),
    walk_forward: bool = typer.Option(False, help="Run walk-forward optimization"),
):
    """Run backtesting on trained models."""
    import numpy as np
    import pandas as pd
    from src.utils.config import load_config
    from src.utils.logging import setup_logging
    from src.data.storage import ParquetStore
    from src.backtest import BacktestEngine, calculate_metrics

    # Setup
    cfg = load_config(config)
    setup_logging(cfg.get("system", {}).get("log_level", "INFO"))

    symbol_list = symbols.split(",") if symbols else cfg["data"]["symbols"]
    model_path = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load features
    feature_store = ParquetStore(cfg["data"]["storage"]["features_path"])

    all_results = []

    for symbol in symbol_list:
        logger.info(f"Backtesting {symbol}")

        df = feature_store.load("features", symbol)
        if df.is_empty():
            continue

        # Load model and generate predictions
        predictions, prices = generate_predictions(symbol, df, model_path, cfg)

        if predictions is None:
            continue

        # Run backtest
        engine = BacktestEngine(cfg.get("backtest", {}))
        result = engine.run(
            prices=prices,
            signals=predictions,
            timestamps=df["timestamp"].to_numpy() if "timestamp" in df.columns else None,
        )

        logger.info(f"{symbol} Results: Sharpe={result.metrics['sharpe']:.2f}, Return={result.metrics['total_return']:.2%}")

        # Save results
        result.equity.to_csv(output_path / f"{symbol}_equity.csv")
        result.trades.to_csv(output_path / f"{symbol}_trades.csv")

        all_results.append({"symbol": symbol, **result.metrics})

    # Summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path / "backtest_summary.csv", index=False)
    logger.info(f"Backtest complete. Results saved to {output_path}")


def generate_predictions(symbol, df, model_path, cfg):
    """Generate predictions from trained model."""
    import numpy as np

    lgb_path = model_path / symbol / "lightgbm.joblib"

    if not lgb_path.exists():
        logger.warning(f"No model found for {symbol}")
        return None, None

    from src.models.architectures import LightGBMModel
    from src.models.base import ModelConfig

    config = ModelConfig(name="lightgbm", type="lightgbm", output_dim=3)
    model = LightGBMModel(config)
    model.load(lgb_path)

    # Prepare features
    feature_cols = [c for c in df.columns if c not in ["timestamp", "symbol", "open", "high", "low", "close"]]
    X = df.select(feature_cols).to_numpy()

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    # Get predictions
    proba = model.predict_proba(X)
    predictions = np.argmax(proba, axis=1)

    # Convert to signals: 0 (buy) -> 1, 1 (hold) -> 0, 2 (sell) -> -1
    signals = np.where(predictions == 0, 1, np.where(predictions == 2, -1, 0))

    prices = df["close"].to_numpy()

    return signals, prices


if __name__ == "__main__":
    app()
