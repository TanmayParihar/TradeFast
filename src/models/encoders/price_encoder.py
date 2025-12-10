import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PriceEncoder(nn.Module):
    """Enhanced price encoder with better initialization and regularization"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Input layer with proper initialization
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        
        # Temporal encoding
        self.time_encoding = nn.Parameter(torch.zeros(1, cfg.window_size, cfg.hidden_dim))
        
        # Multi-scale convolutional features
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, padding=1),
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=5, padding=2),
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=7, padding=3),
        ])
        
        # Attention mechanism for feature selection
        self.attention = nn.MultiheadAttention(
            cfg.hidden_dim, num_heads=4, batch_first=True, dropout=0.1
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(cfg.hidden_dim)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Output projection
        self.output_proj = nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'conv' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'attention' in name:
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # Initialize time encoding
        nn.init.normal_(self.time_encoding, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, window_size, input_dim)
        Returns:
            encoded: (batch_size, window_size, hidden_dim)
        """
        batch_size, window_size, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (B, W, H)
        
        # Add time encoding
        x = x + self.time_encoding[:, :window_size, :]
        
        # Apply layer norm
        x = self.ln1(x)
        
        # Multi-scale convolutional features
        conv_features = []
        x_conv = x.transpose(1, 2)  # (B, H, W)
        
        for conv in self.conv_layers:
            feat = conv(x_conv)
            feat = F.gelu(feat)
            feat = feat.transpose(1, 2)  # (B, W, H)
            conv_features.append(feat)
        
        # Concatenate multi-scale features
        conv_features = torch.cat(conv_features, dim=-1)  # (B, W, H*3)
        
        # Project back to hidden dim
        conv_features = self.output_proj(conv_features)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)
        
        # Residual connection
        x = x + attn_out
        x = self.ln2(x)
        
        # Combine conv and attention features
        x = torch.cat([x, conv_features], dim=-1)
        
        # Final projection
        x = self.output_proj(x)
        
        return x
