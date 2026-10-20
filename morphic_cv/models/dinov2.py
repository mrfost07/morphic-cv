"""
morphic_cv.models.dinov2
High-level wrapper for fine-tuning Meta's DINOv2 with Morphological Adapters.
"""

import math
import os
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from ..adapters.msa import MorphicAdapter


class MorphicDINOv2(nn.Module):
    """
    Meta DINOv2 Vision Transformer enhanced with Morphological Sparse Adapters.
    Freezes 100% of the backbone parameters and fine-tunes only structural adapters.
    """
    def __init__(self, backbone: nn.Module, num_classes: int = 10, bottleneck_dim: int = 32, alpha: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim

        # Freeze entire backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Inferred embedding dim (384 for vit-s, 768 for vit-b, 1024 for vit-l)
        self.embed_dim = backbone.embed_dim if hasattr(backbone, 'embed_dim') else 384
        self.adapter = MorphicAdapter(in_channels=self.embed_dim, bottleneck_dim=bottleneck_dim, alpha=alpha)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    @classmethod
    def from_pretrained(cls, model_name: str = 'dinov2_vits14', num_classes: int = 10, bottleneck_dim: int = 32):
        """
        Loads Meta's official DINOv2 backbone from torch.hub and attaches MorphicAdapter.
        Example:
            model = MorphicDINOv2.from_pretrained('dinov2_vits14', num_classes=5)
        """
        backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        return cls(backbone, num_classes=num_classes, bottleneck_dim=bottleneck_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] where H and W are multiples of 14 (e.g. 224x224)
        with torch.no_grad():
            features = self.backbone.forward_features(x)
            patch_tokens = features['x_norm_patchtokens'] # [B, N_patches, embed_dim]
            cls_token = features['x_norm_clstoken']       # [B, embed_dim]

        B, N, D = patch_tokens.shape
        grid_size = int(math.sqrt(N))
        grid = patch_tokens.view(B, grid_size, grid_size, D).permute(0, 3, 1, 2).contiguous()

        # Apply Morphological Sparse Adapter
        adapted_grid = self.adapter(grid)
        adapted_pooled = adapted_grid.mean(dim=[-2, -1])

        # Fuse CLS token + structural adapter features
        fused = cls_token + adapted_pooled
        return self.classifier(fused)

    def get_parameter_summary(self) -> Dict[str, Any]:
        total_p = sum(p.numel() for p in self.parameters())
        trainable_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_p,
            "trainable_parameters": trainable_p,
            "trainable_fraction_pct": (trainable_p / total_p) * 100.0
        }

    def save_adapter(self, save_path: str):
        """Saves only the lightweight adapter and classifier weights (< 200 KB!)."""
        state_dict = {
            "adapter": self.adapter.state_dict(),
            "classifier": self.classifier.state_dict(),
            "num_classes": self.num_classes,
            "bottleneck_dim": self.bottleneck_dim
        }
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save(state_dict, save_path)
        print(f"Adapter saved successfully to: {save_path} ({os.path.getsize(save_path)/1024:.1f} KB)")

    def load_adapter(self, load_path: str):
        """Loads saved adapter and classifier weights."""
        state_dict = torch.load(load_path, map_location='cpu')
        self.adapter.load_state_dict(state_dict["adapter"])
        self.classifier.load_state_dict(state_dict["classifier"])
        print(f"Adapter loaded successfully from: {load_path}")
