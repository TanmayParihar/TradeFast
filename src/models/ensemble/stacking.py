"""
Stacking ensemble for combining multiple models.
"""

from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from src.models.base import BaseModel, ModelConfig


class StackedEnsemble(BaseModel):
    """
    Stacking ensemble combining multiple base models.

    Uses out-of-fold predictions from base models as features
    for a meta-learner.
    """

    def __init__(
        self,
        base_models: list[BaseModel],
        meta_learner: BaseModel,
        config: ModelConfig,
    ):
        super().__init__(config)
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.oof_predictions = None

    def fit(
        self,
        X_train: np.ndarray | pl.DataFrame,
        y_train: np.ndarray,
        X_val: np.ndarray | pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        cv_splitter=None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """
        Fit the stacking ensemble.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            cv_splitter: Cross-validation splitter for OOF predictions
        """
        from src.training.cross_validation import PurgedKFold

        if isinstance(X_train, pl.DataFrame):
            X_train = X_train.to_numpy()

        n_samples = len(X_train)
        n_models = len(self.base_models)

        # Initialize OOF predictions array
        self.oof_predictions = np.zeros((n_samples, n_models * 3))  # 3 classes

        # Use provided CV or create default
        if cv_splitter is None:
            cv_splitter = PurgedKFold(n_splits=5, embargo_pct=0.02)

        # Generate OOF predictions for each base model
        for model_idx, model in enumerate(self.base_models):
            self.logger.info(f"Training base model {model_idx + 1}/{n_models}: {model.config.name}")

            oof_pred = np.zeros((n_samples, 3))

            for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train)):
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train[val_idx]
                y_fold_val = y_train[val_idx]

                # Fit base model on fold
                model.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val)

                # Get OOF predictions
                fold_pred = model.predict_proba(X_fold_val)
                oof_pred[val_idx] = fold_pred

                self.logger.debug(f"Fold {fold_idx + 1} complete")

            # Store OOF predictions
            start_col = model_idx * 3
            self.oof_predictions[:, start_col:start_col + 3] = oof_pred

        # Retrain base models on full training data
        self.logger.info("Retraining base models on full data...")
        for model in self.base_models:
            model.fit(X_train, y_train, X_val, y_val)

        # Train meta-learner on OOF predictions
        self.logger.info("Training meta-learner...")
        history = self.meta_learner.fit(
            self.oof_predictions,
            y_train,
            **kwargs,
        )

        self._is_fitted = True
        return history

    def predict(self, X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """Make predictions using the ensemble."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()

        # Get predictions from each base model
        base_predictions = []
        for model in self.base_models:
            pred = model.predict_proba(X)
            base_predictions.append(pred)

        # Concatenate as meta-features
        meta_features = np.hstack(base_predictions)

        # Meta-learner prediction
        return self.meta_learner.predict_proba(meta_features)

    def get_base_predictions(
        self, X: np.ndarray | pl.DataFrame
    ) -> dict[str, np.ndarray]:
        """Get predictions from each base model."""
        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()

        predictions = {}
        for model in self.base_models:
            predictions[model.config.name] = model.predict_proba(X)

        return predictions


class WeightedEnsemble(BaseModel):
    """
    Simple weighted average ensemble.
    """

    def __init__(
        self,
        base_models: list[BaseModel],
        weights: list[float] | None = None,
        config: ModelConfig | None = None,
    ):
        config = config or ModelConfig(name="weighted_ensemble", type="ensemble")
        super().__init__(config)

        self.base_models = base_models
        self.weights = weights or [1.0 / len(base_models)] * len(base_models)

        assert len(self.weights) == len(self.base_models)
        assert abs(sum(self.weights) - 1.0) < 1e-6

    def fit(
        self,
        X_train: np.ndarray | pl.DataFrame,
        y_train: np.ndarray,
        X_val: np.ndarray | pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """Fit all base models."""
        for model in self.base_models:
            self.logger.info(f"Training {model.config.name}")
            model.fit(X_train, y_train, X_val, y_val, **kwargs)

        self._is_fitted = True
        return {}

    def predict(self, X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """Make predictions."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """Predict weighted average probabilities."""
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        weighted_proba = None
        for model, weight in zip(self.base_models, self.weights):
            proba = model.predict_proba(X)
            if weighted_proba is None:
                weighted_proba = weight * proba
            else:
                weighted_proba += weight * proba

        return weighted_proba

    def optimize_weights(
        self,
        X_val: np.ndarray | pl.DataFrame,
        y_val: np.ndarray,
        metric: str = "accuracy",
    ) -> list[float]:
        """Optimize ensemble weights on validation set."""
        from scipy.optimize import minimize

        if isinstance(X_val, pl.DataFrame):
            X_val = X_val.to_numpy()

        # Get base predictions
        base_preds = [model.predict_proba(X_val) for model in self.base_models]
        base_preds = np.array(base_preds)  # (n_models, n_samples, n_classes)

        def objective(weights):
            weights = weights / weights.sum()  # Normalize
            weighted = np.tensordot(weights, base_preds, axes=1)
            preds = np.argmax(weighted, axis=1)

            if metric == "accuracy":
                return -np.mean(preds == y_val)
            else:
                return -np.mean(preds == y_val)

        # Optimize
        n_models = len(self.base_models)
        x0 = np.ones(n_models) / n_models
        bounds = [(0, 1) for _ in range(n_models)]
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        self.weights = list(result.x)
        self.logger.info(f"Optimized weights: {self.weights}")

        return self.weights
