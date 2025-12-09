"""
Hyperparameter optimization using Optuna.
"""

from typing import Any, Callable

import numpy as np
from loguru import logger


class OptunaOptimizer:
    """Optuna-based hyperparameter optimization."""

    def __init__(
        self,
        model_class: type,
        param_space: dict[str, Any],
        metric: str = "accuracy",
        direction: str = "maximize",
        n_trials: int = 100,
        cv_splits: int = 5,
    ):
        """
        Initialize optimizer.

        Args:
            model_class: Model class to optimize
            param_space: Parameter search space definition
            metric: Metric to optimize
            direction: 'maximize' or 'minimize'
            n_trials: Number of optimization trials
            cv_splits: Number of CV splits
        """
        self.model_class = model_class
        self.param_space = param_space
        self.metric = metric
        self.direction = direction
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.logger = logger.bind(module="optuna_optimizer")
        self.study = None
        self.best_params = None

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Best parameters
        """
        import optuna
        from src.training.cross_validation import PurgedKFold

        def objective(trial):
            # Sample parameters
            params = self._sample_params(trial)

            # Create model config
            from src.models.base import ModelConfig
            config = ModelConfig(
                name=f"trial_{trial.number}",
                type=self.model_class.__name__,
                **params,
            )

            # Cross-validation
            cv = PurgedKFold(n_splits=self.cv_splits)
            scores = []

            for train_idx, val_idx in cv.split(X):
                X_train, X_cv_val = X[train_idx], X[val_idx]
                y_train, y_cv_val = y[train_idx], y[val_idx]

                model = self.model_class(config)
                model.fit(X_train, y_train)

                predictions = model.predict(X_cv_val)

                if self.metric == "accuracy":
                    score = (predictions == y_cv_val).mean()
                else:
                    from sklearn.metrics import f1_score
                    score = f1_score(y_cv_val, predictions, average="macro")

                scores.append(score)

            return np.mean(scores)

        # Create study
        self.study = optuna.create_study(direction=self.direction)
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        self.best_params = self.study.best_params
        self.logger.info(f"Best params: {self.best_params}")
        self.logger.info(f"Best score: {self.study.best_value}")

        return self.best_params

    def _sample_params(self, trial) -> dict[str, Any]:
        """Sample parameters based on param_space definition."""
        params = {}

        for name, spec in self.param_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(
                    name, spec["low"], spec["high"], step=spec.get("step", 1)
                )
            elif spec["type"] == "float":
                if spec.get("log", False):
                    params[name] = trial.suggest_float(
                        name, spec["low"], spec["high"], log=True
                    )
                else:
                    params[name] = trial.suggest_float(
                        name, spec["low"], spec["high"]
                    )
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])

        return params

    def get_importance(self) -> dict[str, float]:
        """Get parameter importance."""
        if self.study is None:
            return {}

        try:
            import optuna
            importance = optuna.importance.get_param_importances(self.study)
            return dict(importance)
        except Exception:
            return {}


# Example parameter spaces
LIGHTGBM_PARAM_SPACE = {
    "n_estimators": {"type": "int", "low": 100, "high": 1000},
    "num_leaves": {"type": "int", "low": 16, "high": 128},
    "max_depth": {"type": "int", "low": 4, "high": 12},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
    "reg_alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
}

MAMBA_PARAM_SPACE = {
    "d_model": {"type": "categorical", "choices": [32, 64, 128]},
    "d_state": {"type": "categorical", "choices": [32, 64, 128]},
    "n_layers": {"type": "int", "low": 2, "high": 8},
    "dropout": {"type": "float", "low": 0.0, "high": 0.5},
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
}
