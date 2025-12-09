"""
Meta-labeling model for trade quality prediction.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Any

from src.models.base import TorchModel, ModelConfig, EarlyStopping


class MetaLabelingNN(nn.Module):
    """Neural network for meta-labeling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [64, 32],
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MetaLabelingModel(TorchModel):
    """
    Meta-labeling model for predicting trade quality.

    Takes primary model predictions and additional features,
    outputs probability that the trade will be profitable.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.threshold = 0.5

    def _build_model(self, input_dim: int) -> None:
        """Build meta-labeling model."""
        extra = self.config.extra

        self.model = MetaLabelingNN(
            input_dim=input_dim,
            hidden_dims=extra.get("hidden_dims", [64, 32]),
            dropout=self.config.dropout,
        ).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """
        Fit the meta-labeling model.

        Args:
            X_train: Training features (including primary model outputs)
            y_train: Binary labels (1 = profitable trade, 0 = not)
            X_val: Validation features
            y_val: Validation labels
            sample_weight: Optional sample weights
        """
        # Build model
        self._build_model(X_train.shape[1])

        # Create datasets
        if sample_weight is not None:
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train.reshape(-1, 1)),
                torch.FloatTensor(sample_weight),
            )
        else:
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train.reshape(-1, 1)),
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val.reshape(-1, 1)),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size * 2,
                shuffle=False,
            )

        # Setup training
        optimizer, scheduler = self._setup_training()
        criterion = nn.BCELoss()

        early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            mode="min",
        )

        # Training loop
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.max_epochs):
            self.model.train()
            total_loss = 0.0

            for batch in train_loader:
                if len(batch) == 3:
                    X_batch, y_batch, weights = batch
                    weights = weights.to(self.device)
                else:
                    X_batch, y_batch = batch
                    weights = None

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(X_batch)

                if weights is not None:
                    loss = (criterion(outputs, y_batch) * weights).mean()
                else:
                    loss = criterion(outputs, y_batch)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
            history["train_loss"].append(train_loss)
            scheduler.step()

            if val_loader is not None:
                val_loss = self._validate_binary(val_loader, criterion)
                history["val_loss"].append(val_loss)

                if early_stopping(val_loss, self.model):
                    break

        if early_stopping.best_model is not None:
            early_stopping.load_best(self.model)

        self._is_fitted = True
        return history

    def _validate_binary(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        """Validate binary classification model."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                X_batch, y_batch = batch[:2]
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of profitable trade."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            proba = self.model(X_tensor)

        return proba.squeeze().cpu().numpy()

    def set_threshold(self, threshold: float) -> None:
        """Set classification threshold."""
        self.threshold = threshold

    def optimize_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = "f1",
    ) -> float:
        """Optimize classification threshold."""
        from sklearn.metrics import f1_score, precision_score, recall_score

        proba = self.predict_proba(X_val)
        best_threshold = 0.5
        best_score = 0.0

        for threshold in np.arange(0.3, 0.8, 0.05):
            preds = (proba >= threshold).astype(int)

            if metric == "f1":
                score = f1_score(y_val, preds)
            elif metric == "precision":
                score = precision_score(y_val, preds)
            elif metric == "recall":
                score = recall_score(y_val, preds)
            else:
                score = (preds == y_val).mean()

            if score > best_score:
                best_score = score
                best_threshold = threshold

        self.threshold = best_threshold
        return best_threshold


def kelly_position_size(
    pred_prob: float,
    odds: float = 1.0,
    fraction: float = 0.5,
    max_size: float = 0.1,
) -> float:
    """
    Calculate position size using Kelly Criterion.

    Args:
        pred_prob: Predicted probability of winning
        odds: Payout odds (1.0 = even odds)
        fraction: Kelly fraction (0.5 = half-Kelly)
        max_size: Maximum position size

    Returns:
        Position size as fraction of capital
    """
    if pred_prob <= 0 or pred_prob >= 1:
        return 0.0

    # Kelly formula: f* = (p * b - q) / b
    # where p = win probability, q = loss probability, b = odds
    q = 1 - pred_prob
    kelly = (pred_prob * odds - q) / odds

    if kelly <= 0:
        return 0.0

    # Apply fraction and cap
    size = kelly * fraction
    return min(size, max_size)
