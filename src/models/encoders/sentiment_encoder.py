"""
Sentiment encoder using FinBERT or simple embeddings.
"""

import torch
import torch.nn as nn


class SentimentEncoder(nn.Module):
    """
    Encoder for sentiment features.

    Can use pre-computed sentiment scores or FinBERT embeddings.
    """

    def __init__(
        self,
        input_dim: int = 768,  # FinBERT embedding dimension
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.1,
        use_finbert: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_finbert = use_finbert

        if use_finbert:
            # Project FinBERT embeddings
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
        else:
            # For pre-computed sentiment scores (scalar features)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
            )

        # Optional temporal attention for sequence of sentiments
        self.temporal_attention = nn.MultiheadAttention(
            output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        temporal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) or (batch, seq_len, input_dim) if temporal
            temporal: Whether input is a sequence of sentiment features

        Returns:
            (batch, output_dim)
        """
        if temporal and x.dim() == 3:
            # Process each timestep
            batch_size, seq_len, _ = x.size()
            x = x.view(-1, x.size(-1))
            x = self.encoder(x)
            x = x.view(batch_size, seq_len, -1)

            # Temporal attention
            attn_out, _ = self.temporal_attention(x, x, x)
            x = self.temporal_norm(x + attn_out)

            # Take last timestep or mean pool
            return x[:, -1, :]
        else:
            return self.encoder(x)

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.output_dim


class FinBERTEncoder(nn.Module):
    """
    Full FinBERT encoder for raw text.

    Note: For efficiency, it's recommended to pre-compute embeddings
    and use the SentimentEncoder instead.
    """

    def __init__(
        self,
        output_dim: int = 64,
        freeze_bert: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.device = device

        # Load FinBERT
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.bert = AutoModel.from_pretrained("ProsusAI/finbert")

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(768, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        texts: list[str],
    ) -> torch.Tensor:
        """
        Args:
            texts: List of text strings

        Returns:
            (batch, output_dim) - Encoded sentiment embeddings
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert(**inputs)

        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Project
        return self.projection(cls_embedding)

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.output_dim


class SentimentAggregator(nn.Module):
    """Aggregate multiple sentiment signals."""

    def __init__(
        self,
        n_sources: int = 3,  # news, reddit, fear_greed
        source_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_sources = n_sources
        self.source_dim = source_dim

        # Learnable weights for each source
        self.source_weights = nn.Parameter(torch.ones(n_sources) / n_sources)

        # Attention-based aggregation
        self.attention = nn.MultiheadAttention(
            source_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(source_dim * n_sources, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        sources: list[torch.Tensor],
        use_attention: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            sources: List of (batch, source_dim) tensors from different sources
            use_attention: Whether to use attention-based aggregation

        Returns:
            (batch, output_dim) - Aggregated sentiment embedding
        """
        # Stack sources: (batch, n_sources, source_dim)
        x = torch.stack(sources, dim=1)

        if use_attention:
            # Self-attention across sources
            attn_out, _ = self.attention(x, x, x)
            x = x + attn_out

        # Weighted sum
        weights = torch.softmax(self.source_weights, dim=0)
        x = x * weights.view(1, -1, 1)

        # Flatten and project
        x = x.view(x.size(0), -1)
        return self.output(x)

    def get_output_dim(self) -> int:
        return self.output[-1].normalized_shape[0]
