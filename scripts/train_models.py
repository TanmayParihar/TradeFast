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
        lgb.log_evaluation(period=100)
    ]
    
    # Custom progress callback
    class ProgressCallback:
        def __init__(self, total_iterations):
            self.pbar = tqdm(total=total_iterations, desc=f"{symbol} - LightGBM", 
                           unit="iter", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            self.best_score = float('inf')
            
        def __call__(self, env):
            current_iter = env.iteration
            eval_result = env.evaluation_result_list
            
            if eval_result:
                train_loss = None
                val_loss = None
                for metric_name, metric_value, is_higher_better in eval_result:
                    if "train" in metric_name and "multi_logloss" in metric_name:
                        train_loss = metric_value
                    elif "valid" in metric_name and "multi_logloss" in metric_name:
                        val_loss = metric_value
                
                if val_loss is not None:
                    if val_loss < self.best_score:
                        self.best_score = val_loss
                    
                    # Update progress bar
                    self.pbar.set_postfix({
                        'train_loss': f'{train_loss:.4f}' if train_loss else 'N/A',
                        'val_loss': f'{val_loss:.4f}',
                        'best_val': f'{self.best_score:.4f}'
                    })
            
            self.pbar.n = current_iter
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
        num_boost_round=total_iterations,
        callbacks=callbacks,
        verbose_eval=False
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
                               unit="iter", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            
            # Get latest metrics
            train_loss = evals_log['train']['mlogloss'][-1] if 'train' in evals_log else None
            eval_loss = evals_log['eval']['mlogloss'][-1] if 'eval' in evals_log else None
            
            if eval_loss is not None and eval_loss < self.best_score:
                self.best_score = eval_loss
            
            # Update progress bar
            self.pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}' if train_loss else 'N/A',
                'val_loss': f'{eval_loss:.4f}' if eval_loss else 'N/A',
                'best_val': f'{self.best_score:.4f}'
            })
            
            self.pbar.update(1)
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
    
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=watchlist,
        callbacks=[progress_callback],
        verbose_eval=False,
        early_stopping_rounds=50
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
    
    # Convert to float32 to save memory
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    
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
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config["learning_rate"],
        weight_decay=cfg["training"]["optimizer"]["weight_decay"]
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with progress bar
    epochs = cfg["training"]["epochs"]
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = cfg["training"]["early_stopping_patience"]
    
    # Training statistics
    train_history = []
    val_history = []
    
    # Create main progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc=f"{symbol} - Mamba", 
                     unit="epoch", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    for epoch in epoch_pbar:
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Batch progress bar
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", 
                         leave=False, unit="batch", 
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        
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
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", 
                           leave=False, unit="batch",
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            
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
        
        # Store history
        train_history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_acc': train_accuracy,
            'val_loss': avg_val_loss,
            'val_acc': val_accuracy
        })
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'train_acc': f'{train_accuracy:.4f}',
            'val_loss': f'{avg_val_loss:.4f}',
            'val_acc': f'{val_accuracy:.4f}',
            'best_val': f'{best_val_loss:.4f}'
        })
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'history': train_history,
                'config': model_config
            }, output_path / "mamba_best.pt")
            epoch_pbar.set_description(f"{symbol} - Mamba (Best: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                epoch_pbar.set_description(f"{symbol} - Mamba (Early Stop)")
                break
    
    epoch_pbar.close()
    
    # Save final model
    torch.save(model.state_dict(), output_path / "mamba_final.pt")
    
    # Save training history
    history_df = pd.DataFrame(train_history)
    history_path = output_path / "mamba_history.csv"
    history_df.to_csv(history_path, index=False)
    
    logger.info(f"Mamba model saved to {output_path}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final validation accuracy: {val_accuracy:.4f}")
    
    return model


def train_tft(X_train, y_train, X_val, y_val, cfg, output_path, symbol):
    """Train Temporal Fusion Transformer with progress bar"""
    import torch
    import torch.nn as nn
    import numpy as np
    import logging
    from tqdm import tqdm
    
    logger = logging.getLogger(__name__)
    logger.info(f"Training TFT for {symbol}")
    
    # TODO: Implement TFT training
    logger.warning("TFT training not implemented yet")
    
    return None


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
        
        labels = barrier.label(df)
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'returns']]
        X = df[feature_cols].values
        y = labels
        
        # Split data
        train_mask = df.index < cfg["data"]["validation_period"]["start"]
        valid_mask = df.index >= cfg["data"]["validation_period"]["start"]
        
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