"""
Coordinate-Aware Sparse GigaPixel Processor.
Consumes sparse, routed high-resolution patches [B, K, C, P, P] along with their
normalized spatial 2D coordinates [B, K, 2].
Enables global spatial reasoning over 4K images with tiny compute footprint.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateEmbedding(nn.Module):
    """Encodes normalized (x, y) coordinates into continuous Fourier embeddings."""
    def __init__(self, embed_dim: int = 128, num_bands: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bands = num_bands
        # Random Fourier projection frequencies
        self.freqs = nn.Parameter(torch.randn(2, embed_dim // 2) * 2.0 * math.pi, requires_grad=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [B, K, 2] in [0, 1]
        # proj: coords @ freqs -> [B, K, embed_dim // 2]
        proj = torch.matmul(coords, self.freqs)
        # Sinusoidal embedding: [sin, cos]
        fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.proj(fourier)


class GigaPixelSparseTransformer(nn.Module):
    """
    Lightweight Transformer Block operating strictly on the Top-K active geometric patches.
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embed_dim: int = 128, num_heads: int = 4, depth: int = 3, num_classes: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch linear projection: [C * P * P] -> embed_dim
        self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.coord_embed = CoordinateEmbedding(embed_dim=embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification/Detection Head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, patches: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # patches: [B, K, C, P, P]
        # coords: [B, K, 2]
        B, K, C, P, P = patches.shape
        flat_patches = patches.view(B, K, C * P * P)
        
        # 1. Embed patch appearance
        token_feats = self.patch_embed(flat_patches) # [B, K, embed_dim]

        # 2. Inject global coordinate geometry
        coord_feats = self.coord_embed(coords) # [B, K, embed_dim]
        tokens = token_feats + coord_feats

        # 3. Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1) # [B, K + 1, embed_dim]

        # 4. Sparse attention (computes attention ONLY over the active K tokens!)
        tokens = self.transformer(tokens)

        # 5. Output classification or detection prediction
        cls_out = tokens[:, 0]
        logits = self.head(cls_out)
        return logits
