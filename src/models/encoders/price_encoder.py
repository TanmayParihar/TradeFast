"""
Price/OHLCV encoder using Mamba or LSTM.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class MambaBlock(nn.Module):
    """Simplified Mamba-like block for sequence modeling."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Conv layer
        self.conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)

        # Output
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        # Input projection with gate
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Conv
        x = x.transpose(1, 2)
        x = self.conv(x)[:, :, :x.size(-1)]
        x = x.transpose(1, 2)

        # Activation
        x = x * torch.sigmoid(x)  # SiLU

        # Simplified SSM-like operation
        x_dbl = self.x_proj(x)
        dt, B = x_dbl.chunk(2, dim=-1)
        dt = torch.softplus(self.dt_proj(dt))

        # Simple recurrence approximation
        x = x * dt

        # Output gate
        x = x * torch.sigmoid(z)
        x = self.out_proj(x)
        x = self.dropout(x)

        return x + residual


class PriceEncoder(nn.Module):
    """
    Encoder for price/OHLCV time series data.

    Uses a stack of Mamba blocks or LSTM layers for sequence modeling.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_mamba: bool = True,
        use_checkpointing: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_checkpointing = use_checkpointing

        # Input embedding
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Encoder layers
        if use_mamba:
            self.layers = nn.ModuleList([
                MambaBlock(d_model, d_state, d_conv, expand, dropout)
                for _ in range(n_layers)
            ])
        else:
            # Fallback to LSTM
            self.layers = nn.ModuleList([
                nn.LSTM(
                    d_model,
                    d_model // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout if n_layers > 1 else 0,
                )
                for _ in range(n_layers)
            ])
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(n_layers)
            ])

        self.use_mamba = use_mamba
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            return_sequence: If True, return full sequence; else return last hidden

        Returns:
            If return_sequence: (batch, seq_len, d_model)
            Else: (batch, d_model)
        """
        # Input projection
        x = self.input_proj(x)

        # Encoder layers
        if self.use_mamba:
            for layer in self.layers:
                if self.use_checkpointing and self.training:
                    x = checkpoint(layer, x, use_reentrant=False)
                else:
                    x = layer(x)
        else:
            for i, layer in enumerate(self.layers):
                residual = x
                x, _ = layer(x)
                x = self.layer_norms[i](x + residual)

        x = self.final_norm(x)

        if return_sequence:
            return x
        else:
            return x[:, -1, :]  # Return last timestep

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.d_model


class LSTMPriceEncoder(nn.Module):
    """LSTM-based price encoder as alternative."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim * (2 if bidirectional else 1))

    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            (batch, hidden_dim * num_directions) or (batch, seq_len, hidden_dim * num_directions)
        """
        x = self.input_proj(x)
        output, (hidden, cell) = self.lstm(x)
        output = self.norm(output)

        if return_sequence:
            return self.dropout(output)
        else:
            # Concatenate final hidden states from both directions
            if self.bidirectional:
                final = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            else:
                final = hidden[-1]
            return self.dropout(final)

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.hidden_dim * (2 if self.bidirectional else 1)
