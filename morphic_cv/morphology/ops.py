"""
morphic_cv.morphology.ops
Depthwise-separable differentiable mathematical morphology layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDilation2d(nn.Module):
    """
    Depthwise Differentiable 2D Morphological Dilation using Log-Sum-Exp:
    Dilation(X, K) = tau * ln( sum( exp( (X + K) / tau ) ) )
    """
    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, tau: float = 0.5):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=False)
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        nn.init.uniform_(self.weight, -0.05, 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        K2 = self.kernel_size * self.kernel_size
        x_reshaped = x.reshape(B * C, 1, H, W)
        patches = F.unfold(x_reshaped, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        L = H_out * W_out

        patches = patches.view(B, C, K2, L).permute(0, 1, 3, 2).contiguous()
        w = self.weight.view(1, C, 1, K2)
        sum_xk = (patches + w) / self.tau
        out = self.tau * torch.logsumexp(sum_xk, dim=-1)
        return out.view(B, C, H_out, W_out)


class SoftErosion2d(nn.Module):
    """
    Depthwise Differentiable 2D Morphological Erosion using negative Log-Sum-Exp:
    Erosion(X, K) = -tau * ln( sum( exp( -(X - K) / tau ) ) )
    """
    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, tau: float = 0.5):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=False)
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        nn.init.uniform_(self.weight, -0.05, 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        K2 = self.kernel_size * self.kernel_size
        x_reshaped = x.reshape(B * C, 1, H, W)
        patches = F.unfold(x_reshaped, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        L = H_out * W_out

        patches = patches.view(B, C, K2, L).permute(0, 1, 3, 2).contiguous()
        w = self.weight.view(1, C, 1, K2)
        diff_xk = -(patches - w) / self.tau
        out = -self.tau * torch.logsumexp(diff_xk, dim=-1)
        return out.view(B, C, H_out, W_out)


class MorphologicalGradient2d(nn.Module):
    """
    Morphological Gradient = Dilation(X) - Erosion(X)
    Extracts geometric outlines, contours, and topological boundaries natively.
    """
    def __init__(self, in_channels: int, out_channels: int = None, kernel_size: int = 3, stride: int = 1, padding: int = 1, tau: float = 0.5):
        super().__init__()
        out_channels = out_channels or in_channels
        self.dilation = SoftDilation2d(in_channels, kernel_size, stride, padding, tau)
        self.erosion = SoftErosion2d(in_channels, kernel_size, stride, padding, tau)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grad = self.dilation(x) - self.erosion(x)
        return self.norm(self.proj(grad))
