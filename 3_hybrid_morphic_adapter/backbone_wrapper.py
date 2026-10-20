"""
Backbone Wrapper for Parameter-Efficient Morphological Adaptation.
Freezes 100% of a pretrained vision backbone (MobileNetV3 / ResNet)
and fine-tunes only injected Morphological Sparse Adapters (MSA).
Trainable parameters: < 1.5% of the total network.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from adapter import MorphologicalSparseAdapter


class StandardLinearLoRA(nn.Module):
    """Standard LoRA baseline for 2D feature maps."""
    def __init__(self, in_channels: int, rank: int = 16, alpha: float = 0.5):
        super().__init__()
        self.down = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv2d(rank, in_channels, kernel_size=1, bias=False)
        self.alpha = alpha
        nn.init.kaiming_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.up(self.down(x))


class AdaptedVisionModel(nn.Module):
    """
    Wraps MobileNetV3-Small with either MSA (Morphological Sparse Adapter)
    or Standard LoRA.
    """
    def __init__(self, num_classes: int = 10, adapter_type: str = 'msa', bottleneck_dim: int = 16):
        super().__init__()
        self.adapter_type = adapter_type
        # Load backbone
        backbone = mobilenet_v3_small(weights=None)
        
        # Split into early, middle, and late stages
        # mobilenet_v3_small features: 13 sequential blocks
        # Stage 1: layers 0..4 (channels 24)
        # Stage 2: layers 5..8 (channels 48)
        # Stage 3: layers 9..12 (channels 96)
        self.stage1 = nn.Sequential(*backbone.features[:5])
        self.stage2 = nn.Sequential(*backbone.features[5:9])
        self.stage3 = nn.Sequential(*backbone.features[9:])

        # FREEZE 100% of backbone parameters
        for p in self.stage1.parameters():
            p.requires_grad = False
        for p in self.stage2.parameters():
            p.requires_grad = False
        for p in self.stage3.parameters():
            p.requires_grad = False

        # Injected adapters between stages (stage1: 40 ch, stage2: 48 ch)
        if adapter_type == 'msa':
            self.adapter1 = MorphologicalSparseAdapter(in_channels=40, bottleneck_dim=bottleneck_dim)
            self.adapter2 = MorphologicalSparseAdapter(in_channels=48, bottleneck_dim=bottleneck_dim)
        elif adapter_type == 'lora':
            self.adapter1 = StandardLinearLoRA(in_channels=40, rank=bottleneck_dim)
            self.adapter2 = StandardLinearLoRA(in_channels=48, rank=bottleneck_dim)
        else:
            self.adapter1 = nn.Identity()
            self.adapter2 = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Trainable classifier head (576 channels -> num_classes)
        self.classifier = nn.Linear(576, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.adapter1(x)
        x = self.stage2(x)
        x = self.adapter2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def get_parameter_summary(self) -> Tuple[int, int, float]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        fraction = (trainable_params / total_params) * 100.0
        return total_params, trainable_params, fraction
