"""
Meta-learner for stacking ensemble.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Any

from src.models.base import TorchModel, ModelConfig, EarlyStopping


class MetaLearnerNN(nn.Module):
    """Neural network meta-learner for stacking."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [32, 16],
        output_dim: int = 3,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MetaLearner(TorchModel):
    """
    Meta-learner wrapper for stacking ensemble.

    Takes predictions from base models as input and
    learns to combine them optimally.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None

    def _build_model(self, input_dim: int) -> None:
        """Build meta-learner model."""
        extra = self.config.extra

        self.model = MetaLearnerNN(
            input_dim=input_dim,
            hidden_dims=extra.get("hidden_dims", [32, 16]),
            output_dim=self.config.output_dim,
            dropout=self.config.dropout,
            use_batch_norm=extra.get("use_batch_norm", True),
        ).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """Fit the meta-learner."""
        # Build model based on input dimension
        input_dim = X_train.shape[1]
        self._build_model(input_dim)

        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train),
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
                torch.LongTensor(y_val),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size * 2,
                shuffle=False,
            )

        # Setup training
        optimizer, scheduler = self._setup_training()
        criterion = nn.CrossEntropyLoss()

        early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            mode="min",
        )

        # Training loop
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.max_epochs):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            scheduler.step()

            history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader, criterion)
                history["val_loss"].append(val_loss)

                if early_stopping(val_loss, self.model):
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        if early_stopping.best_model is not None:
            early_stopping.load_best(self.model)

        self._is_fitted = True
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            proba = torch.softmax(outputs, dim=1)

        return proba.cpu().numpy()

    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """Get prediction confidence (max probability)."""
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)


class ConfidenceWeightedMeta(MetaLearner):
    """
    Meta-learner that also outputs prediction confidence.

    Useful for position sizing based on model certainty.
    """

    def _build_model(self, input_dim: int) -> None:
        """Build model with confidence head."""
        extra = self.config.extra
        hidden_dims = extra.get("hidden_dims", [32, 16])

        # Shared backbone
        backbone_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            backbone_layers.append(nn.Linear(prev_dim, hidden_dim))
            backbone_layers.append(nn.BatchNorm1d(hidden_dim))
            backbone_layers.append(nn.GELU())
            backbone_layers.append(nn.Dropout(self.config.dropout))
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*backbone_layers).to(self.device)

        # Classification head
        self.class_head = nn.Linear(prev_dim, self.config.output_dim).to(self.device)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.GELU(),
            nn.Linear(prev_dim // 2, 1),
            nn.Sigmoid(),
        ).to(self.device)

        # Combine for standard interface
        self.model = nn.ModuleList([self.backbone, self.class_head, self.confidence_head])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits and confidence."""
        features = self.backbone(x)
        logits = self.class_head(features)
        confidence = self.confidence_head(features)
        return logits, confidence

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence scores.

        Returns:
            predictions: Class predictions
            probabilities: Class probabilities
            confidence: Model confidence scores
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        self.backbone.eval()
        self.class_head.eval()
        self.confidence_head.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            features = self.backbone(X_tensor)
            logits = self.class_head(features)
            confidence = self.confidence_head(features)

            proba = torch.softmax(logits, dim=1)
            preds = torch.argmax(proba, dim=1)

        return (
            preds.cpu().numpy(),
            proba.cpu().numpy(),
            confidence.squeeze().cpu().numpy(),
        )
