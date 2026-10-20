"""
Lightweight Dense Prediction Decoder Head.
Consumes dequantized spatial patch tokens [B, dim, H_grid, W_grid]
and produces dense full-resolution masks [B, num_classes, H_out, W_out].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DensePatchDecoder(nn.Module):
    """
    Lightweight progressive upsampling decoder for spatial tokens.
    Uses depthwise separable convolutions to keep memory below 50MB.
    """
    def __init__(self, in_channels: int = 128, num_classes: int = 5, hidden_dim: int = 64):
        super().__init__()
        # Initial projection
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(hidden_dim)
        
        # Stage 1: 2x upsample
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.point1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # Stage 2: 2x upsample
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim // 2)

        # Final segmentation head
        self.head = nn.Conv2d(hidden_dim // 2, num_classes, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, target_size: tuple = (64, 64)) -> torch.Tensor:
        # x: [B, in_channels, H_grid, W_grid]
        x = self.act(self.bn0(self.proj(x)))

        # Upsample 1
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = self.act(self.bn1(self.point1(x)))

        # Upsample 2
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.act(self.bn2(self.conv2(x)))

        # Final resize to target image resolution
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        logits = self.head(x)
        return logits
