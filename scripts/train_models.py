#!/usr/bin/env python3
"""
Train machine learning models for cryptocurrency trading.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import logging
import warnings
from tqdm import tqdm
import time
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Import project modules
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.gpu_utils import setup_gpu_optimizations
from src.data.storage.parquet_store import ParquetStore
from src.models.meta_labeling.triple_barrier import TripleBarrier
from src.models.architectures.lightgbm_model import LightGBMModel
from src.models.architectures.xgboost_model import XGBoostModel


def train_lightgbm(X_train, y_train, X_val, y_val, cfg, output_path, symbol):
    """Train LightGBM model with progress bar"""
    import lightgbm as lgb
    from tqdm import tqdm
    
    logger = logging.getLogger(__name__)
    logger.info(f"Training LightGBM for {symbol}")
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Get model parameters
    params = cfg["models"]["lightgbm"].copy()
    params.update({
        "objective": "multiclass",
        "num_class": len(np.unique(y_train)),
        "verbosity": -1,
        "metric": "multi_logloss"
    })
    
    # Training with progress bar
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
    ]
    
    # Custom progress callback
    class ProgressCallback:
        def __init__(self, total_iterations):
            self.pbar = tqdm(total=total_iterations, desc=f"{symbol} - LightGBM",
                           unit="iter", ncols=120)
            self.best_score = float('inf')

        def __call__(self, env):
            current_iter = env.iteration + 1  # iteration is 0-indexed
            eval_result = env.evaluation_result_list

            train_loss = None
            val_loss = None

            if eval_result:
                # LightGBM returns tuples of (dataset_name, metric_name, value, is_higher_better)
                for item in eval_result:
                    if len(item) >= 3:
                        dataset_name, metric_name, value = item[0], item[1], item[2]
                        if dataset_name == 'train':
                            train_loss = value
                        elif dataset_name == 'valid':
                            val_loss = value

            if val_loss is not None and val_loss < self.best_score:
                self.best_score = val_loss

            # Update progress bar with metrics
            self.pbar.n = current_iter
            self.pbar.set_postfix(
                train=f'{train_loss:.4f}' if train_loss else '?',
                val=f'{val_loss:.4f}' if val_loss else '?',
                best=f'{self.best_score:.4f}' if self.best_score < float('inf') else '?',
                refresh=False
            )
            self.pbar.refresh()

        def close(self):
            self.pbar.close()
    
    # Train model
    total_iterations = params.get("n_estimators", 500)
    progress_callback = ProgressCallback(total_iterations)
    
    bst = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=total_iterations,
        callbacks=callbacks + [progress_callback]
    )
    
    progress_callback.close()
    
    # Save model
    model_path = output_path / "lightgbm.joblib"
    import joblib
    joblib.dump(bst, model_path)
    logger.info(f"LightGBM model saved to {model_path}")
    
    # Print final stats
    y_pred = bst.predict(X_val)
    y_pred_class = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred_class == y_val)
    
    logger.info(f"LightGBM final validation accuracy: {accuracy:.4f}")
    
    return bst


def train_xgboost(X_train, y_train, X_val, y_val, cfg, output_path, symbol):
    """Train XGBoost model with progress bar"""
    import xgboost as xgb
    from tqdm import tqdm
    
    logger = logging.getLogger(__name__)
    logger.info(f"Training XGBoost for {symbol}")
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Get model parameters
    params = cfg["models"]["xgboost"].copy()
    params.update({
        "objective": "multi:softprob",
        "num_class": len(np.unique(y_train)),
        "eval_metric": "mlogloss",
        "verbosity": 0
    })
    
    # Training with progress bar
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    num_rounds = params.get("n_estimators", 500)
    
    # Custom callback for progress bar
    class XGBProgressCallback(xgb.callback.TrainingCallback):
        def __init__(self, symbol):
            self.symbol = symbol
            self.pbar = None
            self.best_score = float('inf')
            self.history = {'train': [], 'eval': []}

        def after_iteration(self, model, epoch, evals_log):
            if self.pbar is None:
                self.pbar = tqdm(total=num_rounds, desc=f"{self.symbol} - XGBoost",
                               unit="iter", ncols=120)

            # Get latest metrics
            train_loss = evals_log['train']['mlogloss'][-1] if 'train' in evals_log else None
            eval_loss = evals_log['eval']['mlogloss'][-1] if 'eval' in evals_log else None

            if eval_loss is not None and eval_loss < self.best_score:
                self.best_score = eval_loss

            # Update progress bar
            self.pbar.n = epoch + 1
            self.pbar.set_postfix(
                train=f'{train_loss:.4f}' if train_loss else '?',
                val=f'{eval_loss:.4f}' if eval_loss else '?',
                best=f'{self.best_score:.4f}' if self.best_score < float('inf') else '?',
                refresh=False
            )
            self.pbar.refresh()

            # Store history
            if train_loss:
                self.history['train'].append(train_loss)
            if eval_loss:
                self.history['eval'].append(eval_loss)

            return False  # Continue training

        def after_training(self, model):
            if self.pbar:
                self.pbar.close()
            return model
    
    # Train model
    progress_callback = XGBProgressCallback(symbol)

    # Use callbacks for early stopping and suppress default logging
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=watchlist,
        callbacks=[
            progress_callback,
            xgb.callback.EarlyStopping(rounds=50, save_best=True),
        ],
        verbose_eval=False  # Suppress default logging
    )
    
    # Save model
    model_path = output_path / "xgboost.joblib"
    import joblib
    joblib.dump(bst, model_path)
    logger.info(f"XGBoost model saved to {model_path}")
    
    # Print final stats
    y_pred = bst.predict(dval)
    y_pred_class = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred_class == y_val)
    
    logger.info(f"XGBoost final validation accuracy: {accuracy:.4f}")
    
    return bst


def train_mamba(X_train, y_train, X_val, y_val, cfg, output_path, symbol):
    """Train Mamba model with progress bar"""
    import torch
    import torch.nn as nn
    import numpy as np
    import logging
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
    import time
    
    logger = logging.getLogger(__name__)
    logger.info(f"Training Mamba for {symbol}")
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to float32 and handle NaN/Inf
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)

    # Replace NaN and Inf with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize features (critical for neural networks!)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Clip extreme values after standardization
    X_train = np.clip(X_train, -10, 10)
    X_val = np.clip(X_val, -10, 10)

    logger.info(f"Data stats after preprocessing - Train: mean={X_train.mean():.4f}, std={X_train.std():.4f}")
    logger.info(f"Data stats after preprocessing - Val: mean={X_val.mean():.4f}, std={X_val.std():.4f}")
    
    # Define Dataset class
    class SequenceDataset(Dataset):
        def __init__(self, X, y, seq_len):
            self.X = X
            self.y = y
            self.seq_len = seq_len
            
        def __len__(self):
            return len(self.X) - self.seq_len
        
        def __getitem__(self, idx):
            return (
                torch.FloatTensor(self.X[idx:idx+self.seq_len]),
                torch.LongTensor([self.y[idx+self.seq_len]])
            )
    
    # Get sequence length from config
    seq_len = cfg["models"]["mamba"].get("seq_length", 128)
    
    # Create datasets
    train_dataset = SequenceDataset(X_train, y_train, seq_len)
    val_dataset = SequenceDataset(X_val, y_val, seq_len)
    
    # Create data loaders
    batch_size = cfg["models"]["mamba"]["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, cfg["hardware"]["cpu"]["num_workers"]),
        pin_memory=cfg["hardware"]["cpu"]["pin_memory"]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(2, cfg["hardware"]["cpu"]["num_workers"]),
        pin_memory=cfg["hardware"]["cpu"]["pin_memory"]
    )
    
    # Import Mamba model
    try:
        from src.models.architectures.mamba_model import MambaTSClassifier
    except ImportError:
        logger.warning("MambaTSClassifier not found, using fallback implementation")
        
        # Fallback implementation
        class MambaTSClassifier(nn.Module):
            def __init__(self, input_dim, num_classes, d_model=64, d_state=64, 
                         d_conv=4, expand=2, n_layers=4, dropout=0.1):
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
            
            def forward(self, x):
                x = x.transpose(1, 2)
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x).squeeze(-1)
                x = self.dropout(x)
                return self.classifier(x)
    
    # Initialize model
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    model_config = cfg["models"]["mamba"]
    model = MambaTSClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=model_config["d_model"],
        d_state=model_config["d_state"],
        d_conv=model_config["d_conv"],
        expand=model_config["expand"],
        n_layers=model_config["n_layers"],
        dropout=model_config["dropout"]
    )
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Using device: {device}")

    # Calculate class weights for imbalanced data
    from collections import Counter
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    num_classes = len(class_counts)
    class_weights = torch.FloatTensor([
        total_samples / (num_classes * class_counts.get(i, 1))
        for i in range(num_classes)
    ]).to(device)

    logger.info(f"Class distribution: {dict(class_counts)}")
    logger.info(f"Class weights: {class_weights.tolist()}")

    # Training setup with higher initial learning rate
    initial_lr = model_config["learning_rate"] * 3  # 3x default LR for faster initial learning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=cfg["training"]["optimizer"]["weight_decay"]
    )

    # Learning rate scheduler - cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Use weighted cross entropy for class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop with progress bar
    epochs = cfg["training"]["epochs"]
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    early_stopping_patience = cfg["training"]["early_stopping_patience"]
    checkpoint_every = 5  # Save checkpoint every N epochs
    start_epoch = 0

    # Create checkpoint directory
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint to resume from
    resume_checkpoint = checkpoint_dir / "latest_checkpoint.pt"
    if resume_checkpoint.exists():
        logger.info(f"Found checkpoint, resuming training...")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

    # Training statistics
    train_history = []
    val_history = []

    # Create main progress bar for epochs - use ncols to show stats properly
    epoch_pbar = tqdm(range(start_epoch, epochs), desc=f"{symbol} - Mamba",
                     unit="epoch", ncols=140, initial=start_epoch, total=epochs)
    
    for epoch in epoch_pbar:
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Batch progress bar
        batch_pbar = tqdm(train_loader, desc=f"  Train",
                         leave=False, unit="batch", ncols=100)
        
        for batch_X, batch_y in batch_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                cfg["training"]["gradient_clip"]
            )
            
            optimizer.step()
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
            # Update batch progress
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.4f}' if train_total > 0 else '0.0000'
            })
        
        batch_pbar.close()
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"  Valid",
                           leave=False, unit="batch", ncols=100)
            
            for batch_X, batch_y in val_pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).squeeze()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                
                # Update validation progress
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{val_correct/val_total:.4f}' if val_total > 0 else '0.0000'
                })
            
            val_pbar.close()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        
        # Step learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Store history
        train_history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_acc': train_accuracy,
            'val_loss': avg_val_loss,
            'val_acc': val_accuracy,
            'lr': current_lr
        })

        # Update epoch progress bar with LR
        epoch_pbar.set_postfix({
            'loss': f'{avg_train_loss:.4f}',
            'acc': f'{train_accuracy:.2%}',
            'v_loss': f'{avg_val_loss:.4f}',
            'v_acc': f'{val_accuracy:.2%}',
            'lr': f'{current_lr:.2e}'
        })

        # Early stopping check
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_val_acc = val_accuracy
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'history': train_history,
                'config': model_config
            }, output_path / "mamba_best.pt")
            epoch_pbar.set_description(f"{symbol} - Mamba ★")
        else:
            patience_counter += 1

        # Save periodic checkpoint every N epochs
        if (epoch + 1) % checkpoint_every == 0 or epoch == start_epoch:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'history': train_history,
                'config': model_config,
                'patience_counter': patience_counter
            }, checkpoint_path)
            # Also save as latest for resume
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'history': train_history,
                'config': model_config,
                'patience_counter': patience_counter
            }, checkpoint_dir / "latest_checkpoint.pt")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            epoch_pbar.set_description(f"{symbol} - Mamba (Early Stop)")
            break
    
    epoch_pbar.close()
    
    # Save final model
    torch.save(model.state_dict(), output_path / "mamba_final.pt")

    # Save scaler for inference
    import joblib
    joblib.dump(scaler, output_path / "mamba_scaler.joblib")

    # Save training history
    history_df = pd.DataFrame(train_history)
    history_path = output_path / "mamba_history.csv"
    history_df.to_csv(history_path, index=False)

    logger.info(f"Mamba model saved to {output_path}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final validation accuracy: {val_accuracy:.4f}")

    return model


def train_tft(X_train, y_train, X_val, y_val, cfg, output_path, symbol):
    """Train Temporal Fusion Transformer with progress bar and checkpointing"""
    import torch
    import torch.nn as nn
    import numpy as np
    import logging
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    logger = logging.getLogger(__name__)
    logger.info(f"Training TFT for {symbol}")

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to float32 and handle NaN/Inf
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)

    # Replace NaN and Inf with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize features (critical for neural networks!)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Clip extreme values after standardization
    X_train = np.clip(X_train, -10, 10)
    X_val = np.clip(X_val, -10, 10)

    logger.info(f"Data stats after preprocessing - Train: mean={X_train.mean():.4f}, std={X_train.std():.4f}")
    logger.info(f"Data stats after preprocessing - Val: mean={X_val.mean():.4f}, std={X_val.std():.4f}")

    # Define Dataset class for sequences
    class SequenceDataset(Dataset):
        def __init__(self, X, y, seq_len):
            self.X = X
            self.y = y
            self.seq_len = seq_len

        def __len__(self):
            return len(self.X) - self.seq_len

        def __getitem__(self, idx):
            return (
                torch.FloatTensor(self.X[idx:idx+self.seq_len]),
                torch.LongTensor([self.y[idx+self.seq_len]])
            )

    # Get sequence length from config
    seq_len = cfg["models"]["tft"].get("encoder_length", 168)

    # Create datasets
    train_dataset = SequenceDataset(X_train, y_train, seq_len)
    val_dataset = SequenceDataset(X_val, y_val, seq_len)

    # Create data loaders
    batch_size = cfg["models"]["tft"]["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, cfg["hardware"]["cpu"]["num_workers"]),
        pin_memory=cfg["hardware"]["cpu"]["pin_memory"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(2, cfg["hardware"]["cpu"]["num_workers"]),
        pin_memory=cfg["hardware"]["cpu"]["pin_memory"]
    )

    # Simple TFT-style model (transformer-based)
    class SimpleTFT(nn.Module):
        def __init__(self, input_dim, num_classes, hidden_size=32, n_heads=4, n_layers=2, dropout=0.2):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_size)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            self.pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            # x: (batch, seq_len, features)
            x = self.input_proj(x)
            x = self.transformer(x)
            # Pool over sequence dimension
            x = x.transpose(1, 2)  # (batch, hidden, seq)
            x = self.pool(x).squeeze(-1)
            x = self.dropout(x)
            return self.classifier(x)

    # Initialize model
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model_config = cfg["models"]["tft"]
    model = SimpleTFT(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_size=model_config.get("hidden_size", 32),
        n_heads=model_config.get("attention_head_size", 4),
        n_layers=model_config.get("lstm_layers", 2),
        dropout=model_config.get("dropout", 0.2)
    )

    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Using device: {device}")

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.get("learning_rate", 0.003),
        weight_decay=cfg["training"]["optimizer"]["weight_decay"]
    )

    criterion = nn.CrossEntropyLoss()

    # Training parameters
    epochs = cfg["training"]["epochs"]
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    early_stopping_patience = cfg["training"]["early_stopping_patience"]
    checkpoint_every = 5
    start_epoch = 0

    # Create checkpoint directory
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint to resume from
    resume_checkpoint = checkpoint_dir / "tft_latest_checkpoint.pt"
    if resume_checkpoint.exists():
        logger.info(f"Found checkpoint, resuming training...")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

    # Training history
    train_history = []

    # Training loop
    epoch_pbar = tqdm(range(start_epoch, epochs), desc=f"{symbol} - TFT",
                     unit="epoch", ncols=140, initial=start_epoch, total=epochs)

    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        batch_pbar = tqdm(train_loader, desc=f"  Train", leave=False, unit="batch", ncols=100)

        for batch_X, batch_y in batch_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).squeeze()

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

            batch_pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{train_correct/train_total:.4f}')

        batch_pbar.close()
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"  Valid", leave=False, unit="batch", ncols=100)

            for batch_X, batch_y in val_pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).squeeze()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

                val_pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{val_correct/val_total:.4f}')

            val_pbar.close()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total if val_total > 0 else 0

        # Store history
        train_history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_acc': train_accuracy,
            'val_loss': avg_val_loss,
            'val_acc': val_accuracy
        })

        # Update epoch progress bar
        epoch_pbar.set_postfix(
            train_loss=f'{avg_train_loss:.4f}',
            train_acc=f'{train_accuracy:.4f}',
            val_loss=f'{avg_val_loss:.4f}',
            val_acc=f'{val_accuracy:.4f}',
            best_val=f'{best_val_loss:.4f}'
        )

        # Early stopping check
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_val_acc = val_accuracy
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'history': train_history,
                'config': model_config
            }, output_path / "tft_best.pt")
            epoch_pbar.set_description(f"{symbol} - TFT ★")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if (epoch + 1) % checkpoint_every == 0 or epoch == start_epoch:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'history': train_history,
                'config': model_config,
                'patience_counter': patience_counter
            }
            torch.save(checkpoint_data, checkpoint_dir / f"tft_checkpoint_epoch_{epoch+1}.pt")
            torch.save(checkpoint_data, checkpoint_dir / "tft_latest_checkpoint.pt")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            epoch_pbar.set_description(f"{symbol} - TFT (Early Stop)")
            break

    epoch_pbar.close()

    # Save final model
    torch.save(model.state_dict(), output_path / "tft_final.pt")

    # Save scaler for inference
    import joblib
    joblib.dump(scaler, output_path / "tft_scaler.joblib")

    # Save training history
    history_df = pd.DataFrame(train_history)
    history_df.to_csv(output_path / "tft_history.csv", index=False)

    logger.info(f"TFT model saved to {output_path}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final validation accuracy: {val_accuracy:.4f}")

    return model


def main():
    """Main training function"""
    # Helper function for DataFrame checking
    def is_dataframe_empty(df):
        """Check if a DataFrame (Pandas or Polars) is empty"""
        if df is None:
            return True
        
        # Check for Polars DataFrame
        if hasattr(df, 'is_empty'):
            return df.is_empty()
        
        # Check for Pandas DataFrame
        if hasattr(df, 'empty'):
            return df.empty
        
        # Check if it's empty by length/shape
        try:
            if hasattr(df, 'shape'):
                return df.shape[0] == 0
            elif hasattr(df, '__len__'):
                return len(df) == 0
        except:
            pass
        
        # If we can't determine, assume it's not empty
        return False
    
    parser = argparse.ArgumentParser(description="Train trading models")
    parser.add_argument("--config", type=str, default="config/base.yaml",
                       help="Path to config file")
    parser.add_argument("--model", type=str, default="all",
                       choices=["lightgbm", "xgboost", "mamba", "tft", "all"],
                       help="Model to train")
    parser.add_argument("--symbols", type=str, nargs="+",
                       help="Specific symbols to train (default: all in config)")
    parser.add_argument("--optimize", action="store_true",
                       help="Perform hyperparameter optimization")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Setup logging
    setup_logging(cfg["system"]["log_level"])
    logger = logging.getLogger(__name__)
    
    # Setup GPU optimizations
    setup_gpu_optimizations()
    
    # Get symbols
    symbol_list = args.symbols or cfg["data"]["symbols"]
    
    # Create output directory
    output_dir = cfg["training"]["checkpoint_dir"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training models for {len(symbol_list)} symbols: {symbol_list}")
    
    # Training statistics
    overall_start_time = time.time()
    model_times = {}
    
    for symbol in symbol_list:
        symbol_start_time = time.time()
        logger.info(f"--- Processing {symbol} ---")
        
        # Load data
        feature_store = ParquetStore(cfg["data"]["storage"]["features_path"])
        
        try:
            # The load() method requires category and symbol parameters
            df = feature_store.load("features", symbol)
        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            
            # Try alternative: check if the category is different
            try:
                # Try "processed" category instead of "features"
                df = feature_store.load("processed", symbol)
            except Exception as e2:
                logger.error(f"Also failed with 'processed' category: {e2}")
                
                # Try to list available categories and symbols
                try:
                    categories = feature_store.list_categories()
                    logger.info(f"Available categories: {categories}")
                    
                    for cat in categories:
                        symbols_in_cat = feature_store.list_symbols(cat)
                        logger.info(f"Symbols in {cat}: {symbols_in_cat}")
                        
                        if symbol in symbols_in_cat:
                            logger.info(f"Found {symbol} in category: {cat}")
                            df = feature_store.load(cat, symbol)
                            break
                except Exception as e3:
                    logger.error(f"Could not list categories: {e3}")
                    continue
        
         # Add this conversion code here (BEFORE the empty check)
        # Convert Polars DataFrame to Pandas if needed
        if hasattr(df, 'to_pandas'):
            # It's likely a Polars DataFrame
            logger.info(f"Converting Polars DataFrame to Pandas for {symbol}")
            df = df.to_pandas()
        
        # Now you can use .empty safely
        if df is None or df.empty:
            logger.warning(f"No data found for {symbol}, skipping...")
            continue
        
        # Apply triple barrier labeling
        barrier = TripleBarrier(
            pt_mult=cfg["strategy"]["triple_barrier"]["pt_mult"],
            sl_mult=cfg["strategy"]["triple_barrier"]["sl_mult"],
            max_holding=cfg["strategy"]["triple_barrier"]["max_holding"],
            vol_span=cfg["strategy"]["triple_barrier"]["vol_span"],
            min_vol=cfg["strategy"]["triple_barrier"]["min_vol"]
        )
        
        labels = barrier.get_class_labels(df)
        
        # Prepare features - exclude non-numeric columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'returns', 'timestamp', 'symbol', 'date', 'datetime']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        X = df[feature_cols].values.astype(np.float32)
        y = labels
        
        # Split data based on timestamp column
        val_start = pd.to_datetime(cfg["data"]["validation_period"]["start"])
        if "timestamp" in df.columns:
            train_mask = df["timestamp"] < val_start
            valid_mask = df["timestamp"] >= val_start
        else:
            # Fallback to percentage split if no timestamp
            split_idx = int(len(df) * 0.8)
            train_mask = np.arange(len(df)) < split_idx
            valid_mask = np.arange(len(df)) >= split_idx
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[valid_mask]
        y_val = y[valid_mask]
        
        logger.info(f"Training data: {len(X_train)}, Validation: {len(X_val)}")
        
        # Create symbol output directory
        symbol_output_path = output_path / symbol
        symbol_output_path.mkdir(parents=True, exist_ok=True)
        
        # Rest of your code...
        
        # Train models
        model = args.model.lower()
        
        if model in ["lightgbm", "all"]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting LightGBM training for {symbol}")
            logger.info(f"{'='*60}")
            lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, cfg, symbol_output_path, symbol)
        
        if model in ["xgboost", "all"]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting XGBoost training for {symbol}")
            logger.info(f"{'='*60}")
            xgb_model = train_xgboost(X_train, y_train, X_val, y_val, cfg, symbol_output_path, symbol)
        
        if model in ["mamba", "all"]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting Mamba training for {symbol}")
            logger.info(f"{'='*60}")
            mamba_model = train_mamba(X_train, y_train, X_val, y_val, cfg, symbol_output_path, symbol)
        
        if model in ["tft", "all"]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting TFT training for {symbol}")
            logger.info(f"{'='*60}")
            tft_model = train_tft(X_train, y_train, X_val, y_val, cfg, symbol_output_path, symbol)
        
        symbol_time = time.time() - symbol_start_time
        model_times[symbol] = symbol_time
        logger.info(f"Completed {symbol} in {symbol_time:.2f} seconds")
    
    # Print summary
    total_time = time.time() - overall_start_time
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Models trained: {args.model}")
    logger.info(f"Symbols processed: {len(symbol_list)}")
    
    for symbol, symbol_time in model_times.items():
        logger.info(f"  {symbol}: {symbol_time:.2f} seconds")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()