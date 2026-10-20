"""
Unit tests for morphic_cv.adapters and morphic_cv.models
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
from morphic_cv.adapters.msa import MorphicAdapter
from morphic_cv.models.dinov2 import MorphicDINOv2


def test_morphic_adapter_forward_and_residual():
    adapter = MorphicAdapter(in_channels=64, bottleneck_dim=16, alpha=0.5)
    x = torch.randn(2, 64, 8, 8)
    
    # Because up_proj is zero-initialized, initial output should equal input (identity residual)
    out = adapter(x)
    assert out.shape == (2, 64, 8, 8)
    assert torch.allclose(out, x, atol=1e-5), "Adapter must start as identity pass-through!"


def test_morphic_dinov2_save_and_load(tmp_path):
    # Create tiny mock backbone simulating DINOv2 structure for quick testing
    class MockDINOv2(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 64
            # Fixed deterministic feature buffer so repeated calls return identical tokens
            self.register_buffer("fixed_patches", torch.randn(2, 256, 64))
            self.register_buffer("fixed_cls", torch.randn(2, 64))
        def forward_features(self, x):
            return {'x_norm_patchtokens': self.fixed_patches, 'x_norm_clstoken': self.fixed_cls}

    mock_backbone = MockDINOv2()
    model = MorphicDINOv2(mock_backbone, num_classes=3, bottleneck_dim=16)
    
    summary = model.get_parameter_summary()
    assert summary["trainable_parameters"] > 0
    assert summary["total_parameters"] >= summary["trainable_parameters"]

    # Test forward pass
    dummy_img = torch.randn(2, 3, 224, 224)
    logits = model(dummy_img)
    assert logits.shape == (2, 3)

    # Test save and load adapter
    save_file = str(tmp_path / "test_adapter.pt")
    model.save_adapter(save_file)
    assert os.path.exists(save_file)

    # Load into fresh model using the same backbone features
    new_model = MorphicDINOv2(mock_backbone, num_classes=3, bottleneck_dim=16)
    new_model.load_adapter(save_file)
    new_logits = new_model(dummy_img)
    assert torch.allclose(logits, new_logits, atol=1e-4)
