import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import wandb
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional
import math

class ImprovedTrainer:
    """Enhanced trainer with better validation and early stopping"""
    
    def __init__(self, model, cfg, device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(cfg.class_weights, device=device) if hasattr(cfg, 'class_weights') else None
        )
        self.confidence_loss = nn.MSELoss()
        
        # Optimizer with gradient clipping
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Early stopping
        self.patience = cfg.patience
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False
        
        # Gradient clipping
        self.clip_value = cfg.grad_clip
        
        # Metrics tracking
        self.train_metrics = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []
        }
        self.val_metrics = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []
        }
        
        # Initialize wandb if enabled
        if cfg.use_wandb:
            wandb.init(project="TradeFast-Mamba", config=vars(cfg))
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        if self.cfg.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.cfg.warmup_epochs,
                T_mult=2,
                eta_min=self.cfg.min_lr
            )
        elif self.cfg.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=self.cfg.min_lr
            )
        else:
            # Linear warmup then cosine decay
            def lr_lambda(epoch):
                if epoch < self.cfg.warmup_epochs:
                    return float(epoch) / float(max(1, self.cfg.warmup_epochs))
                else:
                    progress = float(epoch - self.cfg.warmup_epochs) / float(max(1, self.cfg.total_epochs - self.cfg.warmup_epochs))
                    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        return scheduler
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, confidence = self.model(data)
            
            # Calculate losses
            cls_loss = self.classification_loss(logits, target)
            
            # Confidence loss based on accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                correct = (preds == target).float()
            
            conf_loss = self.confidence_loss(confidence.squeeze(), correct)
            
            # Combined loss
            loss = cls_loss + 0.1 * conf_loss
            
            # Backward pass with gradient clipping
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Log progress
            if batch_idx % 100 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}')
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )
        
        return avg_loss, accuracy, precision, recall, f1
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                logits, confidence = self.model(data)
                
                # Calculate loss
                loss = self.classification_loss(logits, target)
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                
                # Accumulate
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_confidences.extend(confidence.squeeze().cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # Calculate balanced accuracy
        if cm.shape[0] == 2:  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (sensitivity + specificity) / 2
        else:
            balanced_acc = accuracy
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'confidence_mean': np.mean(all_confidences),
            'confidence_std': np.std(all_confidences)
        }
    
    def early_stopping_check(self, val_loss):
        """Check for early stopping"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
            # Save best model
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'best_model_{self.cfg.symbol}.pth')
            return False  # Continue training
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                print(f'Early stopping triggered after {self.current_epoch} epochs')
                self.early_stop = True
            return self.early_stop
    
    def train(self, train_loader, val_loader, num_epochs):
        """Main training loop"""
        print(f'Starting training for {num_epochs} epochs')
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc, train_prec, train_rec, train_f1 = self.train_epoch(train_loader)
            
            # Validate
            val_results = self.validate(val_loader)
            
            # Update learning rate
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_results['loss'])
            else:
                self.scheduler.step()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f'\nEpoch {epoch + 1}/{num_epochs}:')
            print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2%}, F1: {train_f1:.4f}')
            print(f'  Val   - Loss: {val_results["loss"]:.4f}, Acc: {val_results["accuracy"]:.2%}, '
                  f'Bal Acc: {val_results["balanced_accuracy"]:.2%}, F1: {val_results["f1"]:.4f}')
            print(f'  LR: {current_lr:.2e}, Confidence: {val_results["confidence_mean"]:.3f} ± {val_results["confidence_std"]:.3f}')
            
            # Wandb logging
            if self.cfg.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'train_f1': train_f1,
                    'val_loss': val_results['loss'],
                    'val_accuracy': val_results['accuracy'],
                    'val_balanced_accuracy': val_results['balanced_accuracy'],
                    'val_f1': val_results['f1'],
                    'learning_rate': current_lr,
                    'confidence_mean': val_results['confidence_mean'],
                    'confidence_std': val_results['confidence_std']
                })
            
            # Early stopping check
            if self.early_stopping_check(val_results['loss']):
                break
        
        print(f'\nTraining completed. Best validation loss: {self.best_val_loss:.4f}')
        return self.model

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32, shuffle=True):
    """Create enhanced data loaders with proper batching"""
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def analyze_class_balance(y_train, y_val):
    """Analyze and report class balance"""
    train_counts = np.bincount(y_train)
    val_counts = np.bincount(y_val)
    
    print("\nClass Distribution Analysis:")
    print(f"Train set: {dict(enumerate(train_counts))}")
    print(f"Validation set: {dict(enumerate(val_counts))}")
    
    # Calculate class weights for imbalance
    class_weights = len(y_train) / (len(np.unique(y_train)) * train_counts)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    print(f"Class weights for loss: {class_weights}")
    
    return class_weights.tolist()

def train_mamba(X_train, y_train, X_val, y_val, cfg, output_path, symbol):
    """Enhanced training function for Mamba model"""
    
    print(f"\n{'='*60}")
    print(f"Training Mamba model for {symbol}")
    print(f"{'='*60}")
    
    # Analyze data
    cfg.class_weights = analyze_class_balance(y_train, y_val)
    
    # Check data dimensions
    print(f"\nData dimensions:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    # Check for NaN values
    if np.isnan(X_train).any() or np.isnan(X_val).any():
        print("Warning: NaN values detected in data!")
        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)
    
    # Normalize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val,
        batch_size=cfg.batch_size
    )
    
    # Initialize model
    from src.models.architectures.mamba_model import EnhancedMambaTradingModel
    model = EnhancedMambaTradingModel(cfg)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Initialize trainer
    trainer = ImprovedTrainer(model, cfg, device)
    
    # Train model
    model = trainer.train(train_loader, val_loader, cfg.num_epochs)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_state': scaler,
        'cfg': cfg,
        'val_metrics': trainer.val_metrics,
    }, output_path)
    
    print(f"\nModel saved to: {output_path}")
    
    # Final evaluation
    final_results = trainer.validate(val_loader)
    
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    print(f"Validation Loss: {final_results['loss']:.4f}")
    print(f"Validation Accuracy: {final_results['accuracy']:.2%}")
    print(f"Balanced Accuracy: {final_results['balanced_accuracy']:.2%}")
    print(f"F1 Score: {final_results['f1']:.4f}")
    print(f"Precision: {final_results['precision']:.4f}")
    print(f"Recall: {final_results['recall']:.4f}")
    print(f"Confidence: {final_results['confidence_mean']:.3f} ± {final_results['confidence_std']:.3f}")
    
    if 'confusion_matrix' in final_results:
        print(f"\nConfusion Matrix:")
        print(final_results['confusion_matrix'])
    
    # Check if model is usable for trading
    if final_results['balanced_accuracy'] > 0.52 and final_results['f1'] > 0.5:
        print("\n✅ Model shows potential for trading (balanced accuracy > 52%)")
    else:
        print("\n⚠️  Model performance is below trading thresholds")
        print("   Consider: More data, Feature engineering, Model simplification")
    
    return model
