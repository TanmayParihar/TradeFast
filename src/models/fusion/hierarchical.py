"""
Hierarchical fusion for multi-scale temporal aggregation.
"""

import torch
import torch.nn as nn


class HierarchicalFusion(nn.Module):
    """
    Hierarchical temporal fusion for multi-timeframe analysis.

    Aggregates information from different temporal scales
    (e.g., 1-min, 5-min, 15-min, 1-hour) into a unified representation.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_scales: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_scales = n_scales

        # Scale-specific encoders
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(n_scales)
        ])

        # Cross-scale attention
        self.cross_scale_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )

        # Scale importance weights (learnable)
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * n_scales, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        scale_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuse features from multiple temporal scales.

        Args:
            scale_features: List of (batch, d_model) tensors for each scale

        Returns:
            fused: (batch, d_model) - Hierarchically fused representation
        """
        assert len(scale_features) == self.n_scales

        # Encode each scale
        encoded_scales = []
        for i, (encoder, features) in enumerate(zip(self.scale_encoders, scale_features)):
            encoded = encoder(features)
            encoded_scales.append(encoded)

        # Stack for attention: (batch, n_scales, d_model)
        stacked = torch.stack(encoded_scales, dim=1)

        # Cross-scale attention
        attended, _ = self.cross_scale_attn(stacked, stacked, stacked)
        attended = stacked + attended  # Residual

        # Apply learned scale weights
        weights = torch.softmax(self.scale_weights, dim=0)
        weighted = attended * weights.view(1, -1, 1)

        # Flatten and fuse
        flattened = weighted.view(weighted.size(0), -1)
        fused = self.fusion(flattened)

        return fused


class TemporalPyramid(nn.Module):
    """
    Temporal pyramid for multi-scale feature extraction.

    Creates features at different temporal resolutions using
    pooling operations.
    """

    def __init__(
        self,
        d_model: int = 64,
        pool_sizes: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.pool_sizes = pool_sizes or [1, 5, 15, 60]

        # Pooling layers
        self.pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=size, stride=size)
            for size in self.pool_sizes
        ])

        # Feature transformers for each scale
        self.transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in self.pool_sizes
        ])

        # Fusion layer
        self.fusion = HierarchicalFusion(
            d_model=d_model,
            n_scales=len(self.pool_sizes),
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract and fuse multi-scale features.

        Args:
            x: (batch, seq_len, d_model) - Input sequence

        Returns:
            (batch, d_model) - Fused multi-scale features
        """
        scale_features = []

        # Transpose for pooling: (batch, d_model, seq_len)
        x_t = x.transpose(1, 2)

        for pool, transformer in zip(self.pools, self.transformers):
            # Pool to different scales
            pooled = pool(x_t)  # (batch, d_model, seq_len/size)

            # Take last timestep
            pooled = pooled[:, :, -1]  # (batch, d_model)

            # Transform
            transformed = transformer(pooled)
            scale_features.append(transformed)

        # Fuse scales
        return self.fusion(scale_features)


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence aggregation."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional (batch, seq_len) mask

        Returns:
            (batch, d_model) - Attention-pooled representation
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)

        # Weighted sum
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)
