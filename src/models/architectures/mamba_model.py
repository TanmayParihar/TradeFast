"""
Mamba State Space Model for trading.
"""

from typing import Any
import torch.nn.functional as F
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

from src.models.base import TorchModel, ModelConfig, EarlyStopping
from src.models.encoders.price_encoder import PriceEncoder, MambaBlock

class MambaTSClassifier(nn.Module):
    """Minimal Mamba classifier for immediate testing"""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 64,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Simple architecture for testing
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Global pooling
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        x = self.dropout(x)
        
        return self.classifier(x)

class MambaTrader(nn.Module):
    """
    Mamba-based trading model.

    Uses Mamba blocks for efficient sequence modeling with
    linear complexity in sequence length.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_classes: int = 3,
        use_checkpointing: bool = True,
    ):
        super().__init__()

        # Input embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Mamba layers
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        self.use_checkpointing = use_checkpointing

        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, n_classes) - Class logits
        """
        # Embed input
        x = self.embedding(x)

        # Mamba layers
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        # Take last timestep
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)

        return self.head(x)


class MambaModel(TorchModel):
    """Wrapper for Mamba trading model."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._build_model()

    def _build_model(self) -> None:
        """Build the Mamba model."""
        extra = self.config.extra

        self.model = MambaTrader(
            input_dim=self.config.input_dim,
            d_model=extra.get("d_model", 64),
            d_state=extra.get("d_state", 64),
            d_conv=extra.get("d_conv", 4),
            expand=extra.get("expand", 2),
            n_layers=extra.get("n_layers", 4),
            dropout=self.config.dropout,
            n_classes=self.config.output_dim,
            use_checkpointing=extra.get("use_checkpointing", True),
        ).to(self.device)

        # Compile for performance
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self.logger.info("Model compiled with torch.compile")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}")

    def fit(
        self,
        X_train: np.ndarray | pl.DataFrame,
        y_train: np.ndarray,
        X_val: np.ndarray | pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """Fit the Mamba model."""
        # Convert to numpy if needed
        if isinstance(X_train, pl.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(X_val, pl.DataFrame):
            X_val = X_val.to_numpy()

        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
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
                num_workers=4,
                pin_memory=True,
            )

        # Setup training
        optimizer, scheduler = self._setup_training()
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            mode="min",
        )

        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(self.config.max_epochs):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            scheduler.step()

            history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.max_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}"
                )

                if early_stopping(val_loss, self.model):
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.max_epochs} - "
                    f"Train Loss: {train_loss:.4f}"
                )

        # Load best model
        if early_stopping.best_model is not None:
            early_stopping.load_best(self.model)

        self._is_fitted = True
        return history

    def predict(self, X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """Make predictions."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            proba = torch.softmax(outputs, dim=1)

        return proba.cpu().numpy()


class MambaRegressor(nn.Module):
    """Mamba model for return regression."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_outputs: int = 1,
    ):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, n_outputs) - Predicted returns
        """
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)

        return self.head(x)
