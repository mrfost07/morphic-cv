"""
Unit tests for morphic_cv.morphology
"""

import pytest
import torch
import torch.nn as nn
from morphic_cv.morphology.ops import SoftDilation2d, SoftErosion2d, MorphologicalGradient2d
from morphic_cv.morphology.rigl import RigLScheduler


def test_soft_dilation_shape_and_grad():
    layer = SoftDilation2d(channels=4, kernel_size=3, padding=1, tau=0.5)
    x = torch.randn(2, 4, 16, 16, requires_grad=True)
    out = layer(x)
    assert out.shape == (2, 4, 16, 16)
    
    # Verify backward pass and non-vanishing gradients
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert (x.grad.abs() > 0).any()


def test_soft_erosion_shape_and_grad():
    layer = SoftErosion2d(channels=4, kernel_size=3, padding=1, tau=0.5)
    x = torch.randn(2, 4, 16, 16, requires_grad=True)
    out = layer(x)
    assert out.shape == (2, 4, 16, 16)
    
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_morphological_gradient_cancels_uniform_texture():
    """Over flat constant values, spatial variance of morphological gradient must be zero (no edges)."""
    layer = MorphologicalGradient2d(in_channels=1, kernel_size=3, padding=1, tau=0.5)
    # Uniform constant tensor
    x = torch.ones(1, 1, 16, 16) * 5.0
    dil = layer.dilation(x)
    ero = layer.erosion(x)
    diff = dil - ero
    # Spatial variation across interior (ignoring boundary padding) must be flat (std == 0)
    interior_std = diff[:, :, 2:-2, 2:-2].std().item()
    assert interior_std < 1e-4, f"Expected uniform texture to have 0 spatial variance, got std: {interior_std}"


def test_rigl_scheduler_maintains_target_sparsity():
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    target_sparsity = 0.80
    scheduler = RigLScheduler(model, target_sparsity=target_sparsity, update_frequency=2)
    
    initial_sparsity = scheduler.get_sparsity()
    assert abs(initial_sparsity - target_sparsity) < 0.05

    # Run simulated train step with backward
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(4):
        optimizer.zero_grad()
        out = model(torch.randn(8, 32))
        loss = out.sum()
        loss.backward()
        scheduler.step()
        optimizer.step()
        scheduler.apply_mask()

    # Sparsity must strictly hold after rewiring
    final_sparsity = scheduler.get_sparsity()
    assert abs(final_sparsity - target_sparsity) < 0.05
