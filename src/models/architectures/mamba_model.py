import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from mamba_ssm import Mamba
from ..encoders.price_encoder import PriceEncoder

class ImprovedMambaBlock(nn.Module):
    """Enhanced Mamba block with better training stability"""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = d_model * expand
        
        # Mamba layer with proper initialization
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # Normalization layers
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
        # Gated linear unit for better feature extraction
        self.glu = nn.Linear(d_model, d_model * 2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Proper weight initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        residual = x
        
        # Apply normalization
        x = self.norm(x)
        
        # Apply Mamba
        mamba_out = self.mamba(x)
        
        # Apply GLU gating
        glu_out = self.glu(x)
        gate, value = glu_out.chunk(2, dim=-1)
        glu_out = torch.sigmoid(gate) * value
        
        # Combine Mamba and GLU outputs
        x = mamba_out + glu_out
        
        # Residual connection with dropout
        x = self.dropout(x)
        output = residual + x
        
        return output

class EnhancedMambaTradingModel(nn.Module):
    """Enhanced Mamba model for trading with better training dynamics"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Price encoder
        self.price_encoder = PriceEncoder(cfg)
        
        # Multiple Mamba blocks with skip connections
        self.mamba_blocks = nn.ModuleList([
            ImprovedMambaBlock(cfg.hidden_dim, d_state=32, d_conv=4, expand=2)
            for _ in range(cfg.num_layers)
        ])
        
        # Layer normalization between blocks
        self.inter_norms = nn.ModuleList([
            nn.LayerNorm(cfg.hidden_dim)
            for _ in range(cfg.num_layers - 1)
        ])
        
        # Feature pyramid for multi-scale features
        self.pyramid_scales = [1, 2, 4]
        self.pyramid_convs = nn.ModuleList([
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim // 4, kernel_size=3, padding=1)
            for _ in self.pyramid_scales
        ])
        
        # Temporal attention for final aggregation
        self.temporal_attention = nn.MultiheadAttention(
            cfg.hidden_dim, num_heads=4, batch_first=True, dropout=0.1
        )
        
        # Output layers
        self.output_norm = nn.LayerNorm(cfg.hidden_dim * 2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(cfg.hidden_dim // 2, cfg.num_classes)
        )
        
        # Regression head for confidence/auxiliary tasks
        self.regressor = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 1),
            nn.Sigmoid()  # Confidence score
        )
        
        # Initialize weights
        self._init_weights()
        
        # Gradient checkpointing for memory efficiency
        self.use_checkpoint = cfg.use_checkpoint
        
    def _init_weights(self):
        """Initialize all weights"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, window_size, input_dim)
        Returns:
            logits: (batch_size, num_classes)
            confidence: (batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Encode price data
        x = self.price_encoder(x)  # (B, S, H)
        
        # Process through Mamba blocks
        for i, block in enumerate(self.mamba_blocks):
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False
                )
            else:
                x = block(x)
            
            # Apply normalization between blocks
            if i < len(self.inter_norms):
                x = self.inter_norms[i](x)
        
        # Extract multi-scale pyramid features
        pyramid_features = []
        x_conv = x.transpose(1, 2)  # (B, H, S)
        
        for scale, conv in zip(self.pyramid_scales, self.pyramid_convs):
            if seq_len >= scale:
                # Pool if needed
                if scale > 1:
                    pooled = F.avg_pool1d(x_conv, kernel_size=scale)
                else:
                    pooled = x_conv
                
                # Apply convolution
                feat = conv(pooled)
                feat = F.gelu(feat)
                
                # Resize to original sequence length
                if scale > 1:
                    feat = F.interpolate(feat, size=seq_len, mode='linear', align_corners=False)
                
                pyramid_features.append(feat)
        
        # Concatenate pyramid features
        if pyramid_features:
            pyramid_features = torch.cat(pyramid_features, dim=1)  # (B, H*len(scales)/4, S)
            pyramid_features = pyramid_features.transpose(1, 2)  # (B, S, H*len(scales)/4)
            
            # Combine with original features
            x = torch.cat([x, pyramid_features], dim=-1)
        
        # Apply temporal attention
        attn_out, _ = self.temporal_attention(x, x, x)
        x = x + attn_out
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (B, H*2)
        
        # Normalize before output
        x = self.output_norm(x)
        
        # Get classification logits
        logits = self.classifier(x)
        
        # Get confidence score
        confidence = self.regressor(x)
        
        return logits, confidence

# Simple wrapper for compatibility
class MambaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = EnhancedMambaTradingModel(cfg)
    
    def forward(self, x):
        logits, confidence = self.model(x)
        return logits  # Return only logits for compatibility
