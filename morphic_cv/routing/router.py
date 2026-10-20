"""
morphic_cv.routing.router
Ultra-fast Morphological Saliency Router for GigaPixel (2K/4K) images.
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class SaliencyRouter(nn.Module):
    """
    Scans raw high-resolution images in milliseconds, prunes empty/uniform background,
    and extracts only the Top-K salient geometric patches with normalized 2D coordinates.
    """
    def __init__(self, patch_size: int = 16, keep_ratio: float = 0.10):
        super().__init__()
        self.patch_size = patch_size
        self.keep_ratio = keep_ratio

    def extract_energy(self, img: torch.Tensor) -> torch.Tensor:
        # Grayscale luminance
        gray = 0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]
        dilated = F.max_pool2d(gray, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-gray, kernel_size=3, stride=1, padding=1)
        return dilated - eroded

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        B, C, H, W = img.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0, f"Image dims ({H}x{W}) must be divisible by patch size {P}"

        grid_h = H // P
        grid_w = W // P
        total_patches = grid_h * grid_w
        K = max(1, int(total_patches * self.keep_ratio))

        # 1. Topological energy map
        energy_map = self.extract_energy(img)

        # 2. Patch saliency pooling
        patch_saliency = F.avg_pool2d(energy_map, kernel_size=P, stride=P).view(B, total_patches)

        # 3. Top-K indices
        topk_scores, topk_indices = torch.topk(patch_saliency, k=K, dim=-1, largest=True)

        # 4. Normalized 2D coordinates [x, y] in [0, 1]
        row_indices = topk_indices // grid_w
        col_indices = topk_indices % grid_w
        norm_y = (row_indices.float() + 0.5) / grid_h
        norm_x = (col_indices.float() + 0.5) / grid_w
        coords = torch.stack([norm_x, norm_y], dim=-1)

        # 5. Extract selected patches
        patches_all = img.view(B, C, grid_h, P, grid_w, P).permute(0, 2, 4, 1, 3, 5).contiguous()
        patches_all = patches_all.view(B, total_patches, C, P, P)

        batch_idx = torch.arange(B, device=img.device).unsqueeze(1).expand(B, K)
        selected_patches = patches_all[batch_idx, topk_indices]

        metadata = {
            "total_patches": total_patches,
            "kept_patches": K,
            "pruned_patches": total_patches - K,
            "sparsity_ratio": (1.0 - (K / total_patches)) * 100.0,
            "attention_memory_reduction": (total_patches / K) ** 2,
            "topk_indices": topk_indices,
            "saliency_map": patch_saliency.view(B, 1, grid_h, grid_w)
        }
        return selected_patches, coords, metadata
