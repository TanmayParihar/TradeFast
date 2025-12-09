"""
Base model class and utilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from loguru import logger


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    type: str
    input_dim: int = 0
    output_dim: int = 3  # Buy/Hold/Sell
    hidden_dim: int = 64
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10
    device: str = "cuda"
    extra: dict = field(default_factory=dict)


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logger.bind(model=config.name)
        self.model = None
        self._is_fitted = False

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray | pl.DataFrame,
        y_train: np.ndarray,
        X_val: np.ndarray | pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """
        Fit the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dictionary with training history
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions (class indices for classification)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        pass

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(self.model, nn.Module):
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config.__dict__,
                'is_fitted': self._is_fitted,
            }, path)
        else:
            import joblib
            joblib.dump({
                'model': self.model,
                'config': self.config.__dict__,
                'is_fitted': self._is_fitted,
            }, path)

        self.logger.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        if str(path).endswith('.pt') or str(path).endswith('.pth'):
            checkpoint = torch.load(path, map_location=self.config.device)
            if self.model is not None:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            self._is_fitted = checkpoint.get('is_fitted', True)
        else:
            import joblib
            data = joblib.load(path)
            self.model = data['model']
            self._is_fitted = data.get('is_fitted', True)

        self.logger.info(f"Model loaded from {path}")

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}')"


class TorchModel(BaseModel):
    """Base class for PyTorch models."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None

    def _to_tensor(self, X: np.ndarray | pl.DataFrame) -> torch.Tensor:
        """Convert input to tensor."""
        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()
        return torch.FloatTensor(X).to(self.device)

    def _setup_training(
        self,
        learning_rate: float | None = None,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Setup optimizer and scheduler."""
        lr = learning_rate or self.config.learning_rate

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.01,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )

        return optimizer, scheduler

    def _train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            X_batch, y_batch = batch
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad(set_to_none=True)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def _validate(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            X_batch, y_batch = batch
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()

            if outputs.dim() > 1:
                preds = outputs.argmax(dim=1)
            else:
                preds = (outputs > 0.5).long()

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        accuracy = correct / total if total > 0 else 0.0
        return total_loss / len(dataloader), accuracy


class EarlyStopping:
    """Early stopping handler."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        """Check if should stop and save best model."""
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self._save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def _save_checkpoint(self, model: nn.Module) -> None:
        """Save model state."""
        import copy
        self.best_model = copy.deepcopy(model.state_dict())

    def load_best(self, model: nn.Module) -> None:
        """Load best model state."""
        if self.best_model is not None:
            model.load_state_dict(self.best_model)
