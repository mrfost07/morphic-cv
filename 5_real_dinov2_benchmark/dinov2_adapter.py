"""
Real Foundation Model (DINOv2) with Morphological Sparse Adapters (MSA).
Freezes 100% of Meta's DINOv2-Small (22M parameters) and adapts it
using structural morphological operators on spatial patch tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSoftDilation2d(nn.Module):
    def __init__(self, channels: int, tau: float = 0.5):
        super().__init__()
        self.channels = channels
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=False)
        self.weight = nn.Parameter(torch.empty(channels, 1, 3, 3))
        nn.init.uniform_(self.weight, -0.05, 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_reshaped = x.view(B * C, 1, H, W)
        patches = F.unfold(x_reshaped, kernel_size=3, padding=1)
        L = H * W
        patches = patches.view(B, C, 9, L).permute(0, 1, 3, 2).contiguous()
        w = self.weight.view(1, C, 1, 9)
        sum_xk = (patches + w) / self.tau
        out = self.tau * torch.logsumexp(sum_xk, dim=-1)
        return out.view(B, C, H, W)


class PatchSoftErosion2d(nn.Module):
    def __init__(self, channels: int, tau: float = 0.5):
        super().__init__()
        self.channels = channels
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=False)
        self.weight = nn.Parameter(torch.empty(channels, 1, 3, 3))
        nn.init.uniform_(self.weight, -0.05, 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_reshaped = x.view(B * C, 1, H, W)
        patches = F.unfold(x_reshaped, kernel_size=3, padding=1)
        L = H * W
        patches = patches.view(B, C, 9, L).permute(0, 1, 3, 2).contiguous()
        w = self.weight.view(1, C, 1, 9)
        diff_xk = -(patches - w) / self.tau
        out = -self.tau * torch.logsumexp(diff_xk, dim=-1)
        return out.view(B, C, H, W)


class DINOv2MorphologicalAdapter(nn.Module):
    """
    Adapts DINOv2 patch tokens:
    [B, N_patches, 384] -> Reshape to 2D Grid -> Morphological Gradient -> Pool -> Residual
    """
    def __init__(self, embed_dim: int = 384, bottleneck: int = 32, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.down = nn.Conv2d(embed_dim, bottleneck, kernel_size=1, bias=False)
        self.dilation = PatchSoftDilation2d(bottleneck)
        self.erosion = PatchSoftErosion2d(bottleneck)
        self.bn = nn.BatchNorm2d(bottleneck)
        self.up = nn.Conv2d(bottleneck, embed_dim, kernel_size=1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, patch_grid: torch.Tensor) -> torch.Tensor:
        # patch_grid: [B, 384, H_p, W_p]
        residual = patch_grid
        h = self.down(patch_grid)
        grad = self.bn(self.dilation(h) - self.erosion(h))
        adapted = self.up(grad)
        return residual + self.alpha * adapted


class DINOv2LoRAAdapter(nn.Module):
    """Standard LoRA baseline on patch token grid."""
    def __init__(self, embed_dim: int = 384, rank: int = 32, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.down = nn.Conv2d(embed_dim, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, embed_dim, kernel_size=1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, patch_grid: torch.Tensor) -> torch.Tensor:
        return patch_grid + self.alpha * self.up(self.down(patch_grid))


class DINOv2AdaptedClassifier(nn.Module):
    def __init__(self, dinov2_backbone: nn.Module, num_classes: int = 10, mode: str = 'msa'):
        super().__init__()
        self.mode = mode
        self.backbone = dinov2_backbone
        
        # Freeze 100% of DINOv2 foundation weights
        for p in self.backbone.parameters():
            p.requires_grad = False

        embed_dim = 384 # DINOv2-small patch embedding dimension

        if mode == 'msa':
            self.adapter = DINOv2MorphologicalAdapter(embed_dim=embed_dim, bottleneck=32)
        elif mode == 'lora':
            self.adapter = DINOv2LoRAAdapter(embed_dim=embed_dim, rank=32)
        else:
            self.adapter = nn.Identity()

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224] (DINOv2 uses patch size 14, so 224/14 = 16x16 grid = 256 patches)
        with torch.no_grad():
            features_dict = self.backbone.forward_features(x)
            patch_tokens = features_dict['x_norm_patchtokens'] # [B, 256, 384]
            cls_token = features_dict['x_norm_clstoken']       # [B, 384]

        B, N, D = patch_tokens.shape
        grid_size = int(math.sqrt(N))
        # Reshape to 2D grid: [B, 384, 16, 16]
        grid = patch_tokens.view(B, grid_size, grid_size, D).permute(0, 3, 1, 2).contiguous()

        # Apply Adapter
        adapted_grid = self.adapter(grid) # [B, 384, 16, 16]

        # Spatial average pool
        adapted_pooled = adapted_grid.mean(dim=[-2, -1]) # [B, 384]

        # Combine with CLS token and classify
        fused = cls_token + adapted_pooled
        logits = self.classifier(fused)
        return logits

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
