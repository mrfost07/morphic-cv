"""
morphic_cv.adapters.msa
Morphological Sparse Adapter (MSA) for Parameter-Efficient Fine-Tuning.
"""

import math
import torch
import torch.nn as nn
from ..morphology.ops import SoftDilation2d, SoftErosion2d


class MorphicAdapter(nn.Module):
    """
    Morphological Sparse Adapter (MSA):
    Injects non-linear topological structural bias into latent feature grids
    with < 1% trainable parameters.
    """
    def __init__(self, in_channels: int, bottleneck_dim: int = 32, alpha: float = 0.5, tau: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_dim = bottleneck_dim
        self.alpha = alpha

        # Down-projection
        self.down_proj = nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, bias=False)
        # Structural morphological operators
        self.dilation = SoftDilation2d(bottleneck_dim, kernel_size=3, tau=tau)
        self.erosion = SoftErosion2d(bottleneck_dim, kernel_size=3, tau=tau)
        self.bn = nn.BatchNorm2d(bottleneck_dim)
        # Up-projection (zero-initialized)
        self.up_proj = nn.Conv2d(bottleneck_dim, in_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        residual = x
        h = self.down_proj(x)
        grad = self.bn(self.dilation(h) - self.erosion(h))
        adapted = self.up_proj(grad)
        return residual + self.alpha * adapted
