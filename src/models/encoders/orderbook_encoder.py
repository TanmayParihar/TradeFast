"""
Order book encoder using attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrderBookAttention(nn.Module):
    """Multi-head attention for order book levels."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, n_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # Project and reshape
        Q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)

        return output, attn_weights


class OrderBookEncoder(nn.Module):
    """
    Encoder for order book data using attention.

    Processes bid/ask levels and learns relationships between them.
    """

    def __init__(
        self,
        n_levels: int = 10,
        level_dim: int = 2,  # price, quantity per level
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.level_dim = level_dim
        self.d_model = d_model

        # Input embedding for each level
        # Input: (price, quantity) pairs for bids and asks
        input_dim = level_dim * 2  # bid + ask
        self.level_embedding = nn.Linear(input_dim, d_model)

        # Positional encoding for levels (closer to mid = more important)
        self.level_pos = nn.Parameter(torch.randn(1, n_levels, d_model))

        # Side embedding (bid vs ask)
        self.side_embedding = nn.Embedding(2, d_model)

        # Transformer encoder
        self.attention_layers = nn.ModuleList([
            OrderBookAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * n_levels, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        bid_prices: torch.Tensor,
        bid_quantities: torch.Tensor,
        ask_prices: torch.Tensor,
        ask_quantities: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bid_prices: (batch, n_levels) - Bid prices sorted descending
            bid_quantities: (batch, n_levels) - Bid quantities
            ask_prices: (batch, n_levels) - Ask prices sorted ascending
            ask_quantities: (batch, n_levels) - Ask quantities
            return_attention: Whether to return attention weights

        Returns:
            output: (batch, d_model) - Encoded order book representation
            attention: Optional attention weights if return_attention=True
        """
        batch_size = bid_prices.size(0)

        # Stack bid and ask into levels
        # Each level: (bid_price, bid_qty, ask_price, ask_qty)
        levels = torch.stack([
            bid_prices,
            bid_quantities,
            ask_prices,
            ask_quantities,
        ], dim=-1)  # (batch, n_levels, 4)

        # Normalize prices relative to mid-price
        mid_price = (bid_prices[:, 0:1] + ask_prices[:, 0:1]) / 2
        bid_prices_norm = (bid_prices - mid_price) / (mid_price + 1e-10)
        ask_prices_norm = (ask_prices - mid_price) / (mid_price + 1e-10)

        levels_norm = torch.stack([
            bid_prices_norm,
            bid_quantities / (bid_quantities.sum(dim=-1, keepdim=True) + 1e-10),
            ask_prices_norm,
            ask_quantities / (ask_quantities.sum(dim=-1, keepdim=True) + 1e-10),
        ], dim=-1)

        # Embed levels
        x = self.level_embedding(levels_norm)  # (batch, n_levels, d_model)

        # Add positional encoding
        x = x + self.level_pos

        # Transformer layers
        all_attention = []
        for attn, ff, norm1, norm2 in zip(
            self.attention_layers,
            self.ff_layers,
            self.norms1,
            self.norms2,
        ):
            # Self-attention
            attn_out, attn_weights = attn(x, x, x)
            x = norm1(x + attn_out)

            # Feed-forward
            x = norm2(x + ff(x))

            if return_attention:
                all_attention.append(attn_weights)

        # Flatten and project
        x = x.view(batch_size, -1)
        output = self.output_proj(x)
        output = self.final_norm(output)

        if return_attention:
            return output, torch.stack(all_attention, dim=1)
        return output

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with flat input.

        Args:
            x: (batch, n_levels * 4) - Flattened order book features

        Returns:
            (batch, d_model)
        """
        batch_size = x.size(0)

        # Reshape to (batch, n_levels, 4)
        x = x.view(batch_size, self.n_levels, -1)

        # Split into components
        bid_prices = x[:, :, 0]
        bid_quantities = x[:, :, 1]
        ask_prices = x[:, :, 2]
        ask_quantities = x[:, :, 3]

        return self.forward(bid_prices, bid_quantities, ask_prices, ask_quantities)

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.d_model


class SimpleOrderBookEncoder(nn.Module):
    """Simpler MLP-based order book encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) - Order book features

        Returns:
            (batch, output_dim)
        """
        return self.encoder(x)

    def get_output_dim(self) -> int:
        return self.encoder[-1].normalized_shape[0]
