"""
Training orchestrator for model training.
"""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from loguru import logger

from src.models.base import BaseModel, ModelConfig


class Trainer:
    """Training orchestrator for all model types."""

    def __init__(
        self,
        model: BaseModel,
        config: dict[str, Any] | None = None,
    ):
        self.model = model
        self.config = config or {}
        self.logger = logger.bind(module="trainer")
        self.history: dict[str, list[float]] = {}
        self.best_metric = None

    def train(
        self,
        X_train: np.ndarray | pl.DataFrame,
        y_train: np.ndarray,
        X_val: np.ndarray | pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Training history
        """
        self.logger.info(f"Starting training for {self.model.config.name}")

        # Setup MLflow tracking if configured
        if self.config.get("use_mlflow", False):
            self._setup_mlflow()

        # Train model
        self.history = self.model.fit(X_train, y_train, X_val, y_val, **kwargs)

        # Log metrics
        if self.config.get("use_mlflow", False):
            self._log_metrics()

        self.logger.info("Training complete")
        return self.history

    def train_with_cv(
        self,
        X: np.ndarray | pl.DataFrame,
        y: np.ndarray,
        cv_splitter,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """
        Train with cross-validation.

        Args:
            X: Features
            y: Labels
            cv_splitter: Cross-validation splitter

        Returns:
            Aggregated history across folds
        """
        from src.training.cross_validation import PurgedKFold

        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()

        all_histories = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X)):
            self.logger.info(f"Training fold {fold_idx + 1}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Reset model for new fold (create new instance)
            fold_history = self.model.fit(X_train, y_train, X_val, y_val, **kwargs)
            all_histories.append(fold_history)

        # Aggregate histories
        aggregated = {}
        for key in all_histories[0]:
            values = [h[key] for h in all_histories if key in h]
            if values:
                aggregated[f"{key}_mean"] = [np.mean([v[-1] for v in values])]
                aggregated[f"{key}_std"] = [np.std([v[-1] for v in values])]

        self.history = aggregated
        return self.history

    def evaluate(
        self,
        X_test: np.ndarray | pl.DataFrame,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            log_loss,
        )

        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision_macro": precision_score(y_test, predictions, average="macro"),
            "recall_macro": recall_score(y_test, predictions, average="macro"),
            "f1_macro": f1_score(y_test, predictions, average="macro"),
        }

        try:
            metrics["log_loss"] = log_loss(y_test, probabilities)
        except Exception:
            pass

        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        self.model.load(path)
        self.logger.info(f"Checkpoint loaded from {path}")

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        try:
            import mlflow

            experiment_name = self.config.get("experiment_name", "crypto_trading_bot")
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()

            # Log config
            mlflow.log_params(self.model.config.__dict__)
        except ImportError:
            self.logger.warning("MLflow not installed, skipping tracking")

    def _log_metrics(self) -> None:
        """Log metrics to MLflow."""
        try:
            import mlflow

            for key, values in self.history.items():
                if values:
                    mlflow.log_metric(key, values[-1])

            mlflow.end_run()
        except Exception as e:
            self.logger.warning(f"Failed to log to MLflow: {e}")


class EnsembleTrainer:
    """Trainer for ensemble models."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logger.bind(module="ensemble_trainer")

    def train_ensemble(
        self,
        base_models: list[BaseModel],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        cv_splitter=None,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Train ensemble base models and get OOF predictions.

        Returns:
            Tuple of (oof_predictions, training_histories)
        """
        from src.training.cross_validation import PurgedKFold

        if cv_splitter is None:
            cv_splitter = PurgedKFold(n_splits=5)

        n_samples = len(X_train)
        n_models = len(base_models)
        n_classes = 3

        oof_predictions = np.zeros((n_samples, n_models, n_classes))
        histories = []

        for model_idx, model in enumerate(base_models):
            self.logger.info(f"Training {model.config.name} ({model_idx + 1}/{n_models})")

            model_oof = np.zeros((n_samples, n_classes))

            for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train)):
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train[val_idx]
                y_fold_val = y_train[val_idx]

                model.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
                fold_pred = model.predict_proba(X_fold_val)
                model_oof[val_idx] = fold_pred

            oof_predictions[:, model_idx, :] = model_oof

            # Retrain on full data
            history = model.fit(X_train, y_train, X_val, y_val)
            histories.append(history)

        return oof_predictions, histories
