"""
Cross-modal attention fusion for multi-modal data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing different data modalities.

    Allows each modality to attend to others, learning which
    information is relevant for trading decisions.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Cross-attention layers
        self.cross_attn_price_to_orderbook = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_price_to_sentiment = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_orderbook_to_price = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Gating mechanism
        self.gate_price_ob = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.gate_fused_sentiment = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(
        self,
        price_enc: torch.Tensor,
        orderbook_enc: torch.Tensor,
        sentiment_enc: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Fuse multi-modal encodings using cross-attention.

        Args:
            price_enc: (batch, d_model) - Price encoder output
            orderbook_enc: (batch, d_model) - Order book encoder output
            sentiment_enc: (batch, d_model) - Sentiment encoder output
            return_attention: Whether to return attention weights

        Returns:
            fused: (batch, d_model) - Fused representation
            attention_weights: Optional dict of attention weights
        """
        # Add sequence dimension for attention
        price = price_enc.unsqueeze(1)  # (batch, 1, d_model)
        orderbook = orderbook_enc.unsqueeze(1)
        sentiment = sentiment_enc.unsqueeze(1)

        attention_weights = {}

        # Price attends to orderbook
        p_to_ob, attn_p_ob = self.cross_attn_price_to_orderbook(
            query=price, key=orderbook, value=orderbook
        )
        p_to_ob = self.norm1(price + p_to_ob)

        if return_attention:
            attention_weights["price_to_orderbook"] = attn_p_ob

        # Orderbook attends to price
        ob_to_p, attn_ob_p = self.cross_attn_orderbook_to_price(
            query=orderbook, key=price, value=price
        )
        ob_to_p = self.norm2(orderbook + ob_to_p)

        if return_attention:
            attention_weights["orderbook_to_price"] = attn_ob_p

        # Gated fusion of price and orderbook
        gate_po = self.gate_price_ob(
            torch.cat([p_to_ob, ob_to_p], dim=-1)
        )
        fused_po = gate_po * p_to_ob + (1 - gate_po) * ob_to_p

        # Fused attends to sentiment
        fused_to_sent, attn_f_s = self.cross_attn_price_to_sentiment(
            query=fused_po, key=sentiment, value=sentiment
        )
        fused_to_sent = self.norm3(fused_po + fused_to_sent)

        if return_attention:
            attention_weights["fused_to_sentiment"] = attn_f_s

        # Gated fusion with sentiment
        gate_fs = self.gate_fused_sentiment(
            torch.cat([fused_po, fused_to_sent], dim=-1)
        )
        fused = gate_fs * fused_po + (1 - gate_fs) * fused_to_sent

        # Feed-forward
        fused = fused + self.ffn(fused)
        fused = self.norm_ffn(fused)

        # Remove sequence dimension
        fused = fused.squeeze(1)

        if return_attention:
            return fused, attention_weights
        return fused


class MultiModalFusionNetwork(nn.Module):
    """
    Complete multi-modal fusion network combining all modalities.
    """

    def __init__(
        self,
        price_encoder: nn.Module,
        orderbook_encoder: nn.Module,
        sentiment_encoder: nn.Module,
        d_model: int = 64,
        n_heads: int = 4,
        n_fusion_layers: int = 2,
        dropout: float = 0.1,
        n_classes: int = 3,
    ):
        super().__init__()

        self.price_encoder = price_encoder
        self.orderbook_encoder = orderbook_encoder
        self.sentiment_encoder = sentiment_encoder

        # Projection layers to common dimension
        self.price_proj = nn.Linear(price_encoder.get_output_dim(), d_model)
        self.orderbook_proj = nn.Linear(orderbook_encoder.get_output_dim(), d_model)
        self.sentiment_proj = nn.Linear(sentiment_encoder.get_output_dim(), d_model)

        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            CrossModalAttention(d_model, n_heads, dropout)
            for _ in range(n_fusion_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(
        self,
        price_data: torch.Tensor,
        orderbook_data: torch.Tensor,
        sentiment_data: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through fusion network.

        Args:
            price_data: Input for price encoder
            orderbook_data: Input for orderbook encoder
            sentiment_data: Input for sentiment encoder
            return_embeddings: Whether to return fused embeddings

        Returns:
            logits: (batch, n_classes)
            embeddings: Optional fused embeddings
        """
        # Encode each modality
        price_enc = self.price_encoder(price_data)
        orderbook_enc = self.orderbook_encoder(orderbook_data)
        sentiment_enc = self.sentiment_encoder(sentiment_data)

        # Project to common dimension
        price_enc = self.price_proj(price_enc)
        orderbook_enc = self.orderbook_proj(orderbook_enc)
        sentiment_enc = self.sentiment_proj(sentiment_enc)

        # Apply fusion layers
        fused = None
        for fusion_layer in self.fusion_layers:
            fused = fusion_layer(price_enc, orderbook_enc, sentiment_enc)
            # Update encodings for next layer
            price_enc = fused

        # Classification
        logits = self.classifier(fused)

        if return_embeddings:
            return logits, fused
        return logits
