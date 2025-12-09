"""
Temporal Fusion Transformer model wrapper.
"""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
from loguru import logger

from src.models.base import BaseModel, ModelConfig


class TFTModel(BaseModel):
    """
    Temporal Fusion Transformer for time series forecasting.

    Wrapper around pytorch-forecasting's TFT implementation.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.dataset_params = None
        self.trainer = None

    def _prepare_dataset(
        self,
        df: pd.DataFrame,
        training: bool = True,
    ):
        """Prepare TimeSeriesDataSet for TFT."""
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer

        config = self.config.extra

        # Ensure required columns exist
        if "time_idx" not in df.columns:
            df = df.reset_index()
            df["time_idx"] = range(len(df))

        if "symbol" not in df.columns:
            df["symbol"] = "default"

        # Define time-varying features
        time_varying_known = config.get("time_varying_known", [
            "hour", "minute", "day_of_week",
            "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        ])
        time_varying_known = [c for c in time_varying_known if c in df.columns]

        time_varying_unknown = config.get("time_varying_unknown", [
            "open", "high", "low", "close", "volume", "returns",
            "rsi_14", "macd", "bb_position", "atr",
        ])
        time_varying_unknown = [c for c in time_varying_unknown if c in df.columns]

        # Target column
        target = config.get("target", "returns")
        if target not in df.columns:
            df["returns"] = df["close"].pct_change()
            target = "returns"

        encoder_length = config.get("encoder_length", 168)
        prediction_length = config.get("prediction_length", 15)

        if training:
            dataset = TimeSeriesDataSet(
                df,
                time_idx="time_idx",
                target=target,
                group_ids=["symbol"],
                min_encoder_length=encoder_length // 2,
                max_encoder_length=encoder_length,
                min_prediction_length=1,
                max_prediction_length=prediction_length,
                time_varying_known_reals=time_varying_known,
                time_varying_unknown_reals=time_varying_unknown,
                target_normalizer=GroupNormalizer(groups=["symbol"]),
                add_relative_time_idx=True,
                add_encoder_length=True,
            )
            self.dataset_params = dataset.get_parameters()
        else:
            dataset = TimeSeriesDataSet.from_parameters(
                self.dataset_params,
                df,
                predict=True,
            )

        return dataset

    def _build_model(self, dataset):
        """Build TFT model from dataset."""
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import QuantileLoss

        config = self.config.extra

        model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=config.get("learning_rate", 0.003),
            hidden_size=config.get("hidden_size", 32),
            attention_head_size=config.get("attention_head_size", 4),
            dropout=config.get("dropout", 0.2),
            hidden_continuous_size=config.get("hidden_continuous_size", 16),
            lstm_layers=config.get("lstm_layers", 2),
            output_size=config.get("output_size", 7),
            loss=QuantileLoss(quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]),
            reduce_on_plateau_patience=4,
        )

        return model

    def fit(
        self,
        X_train: np.ndarray | pl.DataFrame,
        y_train: np.ndarray,
        X_val: np.ndarray | pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """
        Fit the TFT model.

        Note: TFT uses the full DataFrame format, not separate X/y.
        Pass the full DataFrame through X_train.
        """
        import pytorch_lightning as pl_lightning
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

        # Convert to pandas if needed
        if isinstance(X_train, pl.DataFrame):
            df_train = X_train.to_pandas()
        elif isinstance(X_train, pd.DataFrame):
            df_train = X_train
        else:
            raise ValueError("TFT requires DataFrame input")

        # Prepare datasets
        train_dataset = self._prepare_dataset(df_train, training=True)

        # Validation dataset
        if X_val is not None:
            if isinstance(X_val, pl.DataFrame):
                df_val = X_val.to_pandas()
            else:
                df_val = X_val
            val_dataset = self._prepare_dataset(df_val, training=False)
        else:
            # Use last 20% of training data
            val_size = int(len(df_train) * 0.2)
            df_val = df_train.iloc[-val_size:]
            val_dataset = self._prepare_dataset(df_val, training=False)

        # Create dataloaders
        batch_size = self.config.extra.get("batch_size", 32)
        train_dataloader = train_dataset.to_dataloader(
            train=True,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
        )
        val_dataloader = val_dataset.to_dataloader(
            train=False,
            batch_size=batch_size * 2,
            num_workers=4,
            pin_memory=True,
        )

        # Build model
        self.model = self._build_model(train_dataset)

        # Callbacks
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.config.early_stopping_patience,
            mode="min",
        )
        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

        # Trainer
        self.trainer = pl_lightning.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="auto",
            devices=1,
            gradient_clip_val=1.0,
            callbacks=[early_stop, checkpoint],
            enable_progress_bar=True,
            precision="16-mixed" if torch.cuda.is_available() else 32,
        )

        # Train
        self.trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # Load best checkpoint
        if checkpoint.best_model_path:
            self.model = self.model.__class__.load_from_checkpoint(
                checkpoint.best_model_path
            )

        self._is_fitted = True

        return {
            "train_loss": [float(self.trainer.callback_metrics.get("train_loss", 0))],
            "val_loss": [float(self.trainer.callback_metrics.get("val_loss", 0))],
        }

    def predict(self, X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """Make predictions."""
        proba = self.predict_proba(X)

        # Convert quantile predictions to class predictions
        # Median prediction (index 3 for 7 quantiles)
        median_pred = proba[:, 3] if proba.ndim > 1 else proba

        # Convert to direction: positive -> buy, negative -> sell, near zero -> hold
        predictions = np.where(median_pred > 0.001, 0,  # Buy
                              np.where(median_pred < -0.001, 2, 1))  # Sell / Hold

        return predictions

    def predict_proba(self, X: np.ndarray | pl.DataFrame) -> np.ndarray:
        """Predict return quantiles."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, pl.DataFrame):
            df = X.to_pandas()
        else:
            df = X

        dataset = self._prepare_dataset(df, training=False)
        dataloader = dataset.to_dataloader(
            train=False,
            batch_size=self.config.extra.get("batch_size", 32) * 2,
            num_workers=4,
        )

        predictions = self.model.predict(dataloader)

        return predictions.numpy()

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from TFT attention."""
        if not self._is_fitted:
            return {}

        interpretation = self.model.interpret_output(
            self.trainer.predict(self.model, dataloaders=None)[0]
        )

        return {
            "encoder_importance": interpretation.get("encoder_importance", {}),
            "decoder_importance": interpretation.get("decoder_importance", {}),
        }
