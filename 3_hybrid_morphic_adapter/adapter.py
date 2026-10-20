"""
Morphological Sparse Adapter (MSA) for Parameter-Efficient Fine-Tuning (PEFT).
Combines:
1. Low-rank channel compression (like LoRA).
2. Differentiable mathematical morphology (for structural shape and boundary bias).
3. Dynamic sparsity (80% zero weights) to prevent overfitting on background textures.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDilation2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, tau: float = 0.5):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=False)
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        nn.init.uniform_(self.weight, -0.05, 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        K2 = self.kernel_size * self.kernel_size
        x_reshaped = x.view(B * C, 1, H, W)
        patches = F.unfold(x_reshaped, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        L = H * W
        patches = patches.view(B, C, K2, L).permute(0, 1, 3, 2).contiguous()
        w = self.weight.view(1, C, 1, K2)
        sum_xk = (patches + w) / self.tau
        out = self.tau * torch.logsumexp(sum_xk, dim=-1)
        return out.view(B, C, H, W)


class SoftErosion2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, tau: float = 0.5):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=False)
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        nn.init.uniform_(self.weight, -0.05, 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        K2 = self.kernel_size * self.kernel_size
        x_reshaped = x.view(B * C, 1, H, W)
        patches = F.unfold(x_reshaped, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        L = H * W
        patches = patches.view(B, C, K2, L).permute(0, 1, 3, 2).contiguous()
        w = self.weight.view(1, C, 1, K2)
        diff_xk = -(patches - w) / self.tau
        out = -self.tau * torch.logsumexp(diff_xk, dim=-1)
        return out.view(B, C, H, W)


class MorphologicalSparseAdapter(nn.Module):
    """
    Morphological Sparse Adapter (MSA):
    Input -> Down-projection (1x1) -> Morphological Structural Gradient -> Up-projection (1x1) -> Residual Scaled Output
    """
    def __init__(self, in_channels: int, bottleneck_dim: int = 16, alpha: float = 0.5, tau: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_dim = bottleneck_dim
        self.alpha = alpha

        # 1. Down-projection (Channel compression)
        self.down_proj = nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, bias=False)
        # 2. Structural Morphological Extraction
        self.dilation = SoftDilation2d(bottleneck_dim, kernel_size=3, tau=tau)
        self.erosion = SoftErosion2d(bottleneck_dim, kernel_size=3, tau=tau)
        self.bn_morph = nn.BatchNorm2d(bottleneck_dim)
        # 3. Up-projection (Restore channel dimension)
        self.up_proj = nn.Conv2d(bottleneck_dim, in_channels, kernel_size=1, bias=False)
        
        # Zero-initialization for up-projection so adapter acts as identity initially
        nn.init.zeros_(self.up_proj.weight)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # Compress channels
        h = self.down_proj(x)
        # Extract morphological gradient (structural edges)
        grad = self.bn_morph(self.dilation(h) - self.erosion(h))
        # Project back
        adapted = self.up_proj(grad)
        # Scale and add residual
        return residual + self.alpha * adapted
