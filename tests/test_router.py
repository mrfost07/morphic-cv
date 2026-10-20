"""
Unit tests for morphic_cv.routing
"""

import pytest
import torch
from morphic_cv.routing.router import SaliencyRouter


def test_saliency_router_shapes_and_ratio():
    router = SaliencyRouter(patch_size=16, keep_ratio=0.10)
    # 256x256 image -> 16x16 = 256 total patches
    img = torch.randn(2, 3, 256, 256)
    selected_patches, coords, meta = router(img)

    # 10% of 256 is ~25 patches
    assert meta["total_patches"] == 256
    assert meta["kept_patches"] == 25
    assert selected_patches.shape == (2, 25, 3, 16, 16)
    assert coords.shape == (2, 25, 2)
    # Coordinates must be normalized between 0.0 and 1.0
    assert (coords >= 0.0).all() and (coords <= 1.0).all()
    assert meta["attention_memory_reduction"] > 50.0


def test_saliency_router_locks_onto_anomaly():
    """Router must prioritize high-contrast geometric anomalies over smooth gradients."""
    router = SaliencyRouter(patch_size=16, keep_ratio=0.05)
    # Smooth canvas (H=128, W=128 -> 64 patches)
    canvas = torch.zeros(1, 3, 128, 128)
    
    # Inject high contrast anomaly at patch (row 2, col 2) -> (y=32..48, x=32..48)
    canvas[:, :, 36:44, 36:44] = 10.0
    selected_patches, coords, meta = router(canvas)

    # Target patch index: row 2 * 8 cols + col 2 = index 18
    target_idx = 2 * (128 // 16) + 2
    topk = meta["topk_indices"][0].tolist()
    assert target_idx in topk, f"Target patch {target_idx} was not selected in top-k: {topk}"
