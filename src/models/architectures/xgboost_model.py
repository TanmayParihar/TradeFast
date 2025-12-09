"""
XGBoost model wrapper for trading.
"""

from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from src.models.base import BaseModel, ModelConfig


class XGBoostModel(BaseModel):
    """XGBoost classifier wrapper for trading signals."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.feature_names = None

    def fit(
        self,
        X_train: np.ndarray | pl.DataFrame,
        y_train: np.ndarray,
        X_val: np.ndarray | pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """Fit XGBoost model."""
        import xgboost as xgb

        # Convert to numpy
        if isinstance(X_train, pl.DataFrame):
            self.feature_names = X_train.columns
            X_train = X_train.to_numpy()
        if isinstance(X_val, pl.DataFrame):
            X_val = X_val.to_numpy()

        # Get hyperparameters from config
        extra = self.config.extra

        params = {
            "objective": "multi:softproba",
            "num_class": self.config.output_dim,
            "eval_metric": "mlogloss",
            "booster": "gbtree",
            "n_estimators": extra.get("n_estimators", 500),
            "max_depth": extra.get("max_depth", 6),
            "learning_rate": extra.get("learning_rate", 0.05),
            "min_child_weight": extra.get("min_child_weight", 3),
            "subsample": extra.get("subsample", 0.8),
            "colsample_bytree": extra.get("colsample_bytree", 0.8),
            "colsample_bylevel": extra.get("colsample_bylevel", 1.0),
            "reg_alpha": extra.get("reg_alpha", 0.1),
            "reg_lambda": extra.get("reg_lambda", 0.1),
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": 0,
        }

        # Create model
        self.model = xgb.XGBClassifier(**params)

        # Fit with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=100,
            )
        else:
            self.model.fit(X_train, y_train)

        self._is_fitted = True

        # Return history
        history = {"train_loss": [], "val_loss": []}

        if hasattr(self.model, "evals_result_"):
            results = self.model.evals_result()
            if "validation_0" in results:
                history["val_loss"] = results["validation_0"]["mlogloss"]

        return history

    def predict(self, X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()

        return self.model.predict_proba(X)

    def get_feature_importance(
        self, importance_type: str = "gain"
    ) -> dict[str, float]:
        """Get feature importance."""
        if not self._is_fitted:
            return {}

        importance = self.model.feature_importances_

        if self.feature_names is not None:
            return dict(zip(self.feature_names, importance))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importance)}

    def save(self, path: str) -> None:
        """Save model to disk."""
        import joblib
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            "model": self.model,
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "is_fitted": self._is_fitted,
        }, path)

        self.logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        import joblib

        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data.get("feature_names")
        self._is_fitted = data.get("is_fitted", True)

        self.logger.info(f"Model loaded from {path}")


class XGBoostRegressor(BaseModel):
    """XGBoost regressor for return prediction."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.feature_names = None

    def fit(
        self,
        X_train: np.ndarray | pl.DataFrame,
        y_train: np.ndarray,
        X_val: np.ndarray | pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """Fit XGBoost regressor."""
        import xgboost as xgb

        if isinstance(X_train, pl.DataFrame):
            self.feature_names = X_train.columns
            X_train = X_train.to_numpy()
        if isinstance(X_val, pl.DataFrame):
            X_val = X_val.to_numpy()

        extra = self.config.extra

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": "gbtree",
            "n_estimators": extra.get("n_estimators", 500),
            "max_depth": extra.get("max_depth", 6),
            "learning_rate": extra.get("learning_rate", 0.05),
            "min_child_weight": extra.get("min_child_weight", 3),
            "subsample": extra.get("subsample", 0.8),
            "colsample_bytree": extra.get("colsample_bytree", 0.8),
            "reg_alpha": extra.get("reg_alpha", 0.1),
            "reg_lambda": extra.get("reg_lambda", 0.1),
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": 0,
        }

        self.model = xgb.XGBRegressor(**params)

        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=100,
            )
        else:
            self.model.fit(X_train, y_train)

        self._is_fitted = True

        return {"train_loss": [], "val_loss": []}

    def predict(self, X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """Predict returns."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """For regressors, return prediction as single column."""
        return self.predict(X).reshape(-1, 1)
