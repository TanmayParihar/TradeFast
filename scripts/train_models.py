#!/usr/bin/env python
"""Model training script."""

from pathlib import Path
import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    config: str = typer.Option("config/base.yaml", help="Config file path"),
    model: str = typer.Option("all", help="Model to train: tft, mamba, lightgbm, xgboost, ensemble, all"),
    symbols: str = typer.Option(None, help="Comma-separated symbols"),
    output_dir: str = typer.Option("checkpoints", help="Output directory"),
    optimize: bool = typer.Option(False, help="Run hyperparameter optimization"),
):
    """Train trading models."""
    import numpy as np
    from src.utils.config import load_config
    from src.utils.logging import setup_logging
    from src.utils.gpu_utils import setup_gpu_optimizations
    from src.data.storage import ParquetStore
    from src.models.base import ModelConfig
    from src.models.meta_labeling import TripleBarrier
    from src.training import Trainer, PurgedKFold

    # Setup
    cfg = load_config(config)
    setup_logging(cfg.get("system", {}).get("log_level", "INFO"))
    setup_gpu_optimizations()

    symbol_list = symbols.split(",") if symbols else cfg["data"]["symbols"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load features
    feature_store = ParquetStore(cfg["data"]["storage"]["features_path"])

    for symbol in symbol_list:
        logger.info(f"Training models for {symbol}")

        df = feature_store.load("features", symbol)
        if df.is_empty():
            logger.warning(f"No features for {symbol}")
            continue

        # Generate labels
        barrier = TripleBarrier(**cfg["strategy"]["triple_barrier"])
        labels = barrier.get_class_labels(df.to_pandas())

        # Prepare data
        feature_cols = [c for c in df.columns if c not in ["timestamp", "symbol", "open", "high", "low", "close"]]
        X = df.select(feature_cols).to_numpy()
        y = labels

        # Remove NaN
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X, y = X[valid_mask], y[valid_mask]

        # Train/val split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(f"Training data: {len(X_train)}, Validation: {len(X_val)}")

        # Train models
        if model in ["lightgbm", "all"]:
            train_lightgbm(X_train, y_train, X_val, y_val, cfg, output_path / symbol)

        if model in ["mamba", "all"]:
            train_mamba(X_train, y_train, X_val, y_val, cfg, output_path / symbol)

    logger.info("Training complete")


def train_lightgbm(X_train, y_train, X_val, y_val, cfg, output_path):
    """Train LightGBM model."""
    from src.models.architectures import LightGBMModel
    from src.models.base import ModelConfig

    config = ModelConfig(
        name="lightgbm",
        type="lightgbm",
        output_dim=3,
        extra=cfg["models"]["lightgbm"],
    )

    model = LightGBMModel(config)
    model.fit(X_train, y_train, X_val, y_val)

    output_path.mkdir(parents=True, exist_ok=True)
    model.save(output_path / "lightgbm.joblib")
    logger.info("LightGBM training complete")


def train_mamba(X_train, y_train, X_val, y_val, cfg, output_path):
    """Train Mamba model."""
    import numpy as np
    from src.models.architectures import MambaModel
    from src.models.base import ModelConfig

    # Reshape for sequence model
    seq_len = cfg["models"]["mamba"].get("seq_length", 128)
    n_samples = len(X_train) - seq_len

    X_seq = np.array([X_train[i:i+seq_len] for i in range(n_samples)])
    y_seq = y_train[seq_len:]

    val_samples = len(X_val) - seq_len
    if val_samples > 0:
        X_val_seq = np.array([X_val[i:i+seq_len] for i in range(val_samples)])
        y_val_seq = y_val[seq_len:]
    else:
        X_val_seq, y_val_seq = None, None

    config = ModelConfig(
        name="mamba",
        type="mamba",
        input_dim=X_train.shape[1],
        output_dim=3,
        extra=cfg["models"]["mamba"],
    )

    model = MambaModel(config)
    model.fit(X_seq, y_seq, X_val_seq, y_val_seq)

    output_path.mkdir(parents=True, exist_ok=True)
    model.save(output_path / "mamba.pt")
    logger.info("Mamba training complete")


if __name__ == "__main__":
    app()
