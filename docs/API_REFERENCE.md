# MorphicCV API Reference

Complete documentation for all classes, methods, and modules in `morphic_cv`.

---

## Table of Contents
1. [`morphic_cv.models.dinov2`](#1-morphic_cvmodelsdinov2)
2. [`morphic_cv.routing.router`](#2-morphic_cvroutingrouter)
3. [`morphic_cv.adapters.msa`](#3-morphic_cvadaptersmsa)
4. [`morphic_cv.morphology.ops`](#4-morphic_cvmorphologyops)
5. [`morphic_cv.morphology.rigl`](#5-morphic_cvmorphologyrigl)
6. [`morphic_cv.streaming`](#6-morphic_cvstreaming)

---

## 1. `morphic_cv.models.dinov2`

### `MorphicDINOv2`
```python
class MorphicDINOv2(nn.Module)
```
Meta's DINOv2 foundation Vision Transformer enhanced with injected Morphological Sparse Adapters (MSA). Freezes 100% of the backbone weights ($0$ gradient memory) and fine-tunes only structural adapters.

#### Methods:
* `from_pretrained(model_name='dinov2_vits14', num_classes=10, bottleneck_dim=32)`
  * **Args**:
    * `model_name` (*str*): DINOv2 model variant from torch.hub (`dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`).
    * `num_classes` (*int*): Number of output classification targets.
    * `bottleneck_dim` (*int*): Intermediate channel dimension of the morphological adapter (default: 32).
  * **Returns**: Initialized `MorphicDINOv2` instance.

* `forward(x: torch.Tensor) -> torch.Tensor`
  * **Args**: `x` (*torch.Tensor*): Input image tensor $[B, 3, H, W]$ where $H, W$ are multiples of 14 (e.g., $224 \times 224$).
  * **Returns**: Logits tensor $[B, \text{num\_classes}]$.

* `save_adapter(save_path: str)`
  * Serializes only the lightweight adapter weights and classification head to disk ($\approx 106\text{ KB}$).

* `load_adapter(load_path: str)`
  * Deserializes previously saved adapter weights into the model.

* `get_parameter_summary() -> Dict[str, Any]`
  * Returns total parameters, trainable parameters, and percentage of model tuned.

---

## 2. `morphic_cv.routing.router`

### `SaliencyRouter`
```python
class SaliencyRouter(nn.Module)
```
Ultra-fast topological scanning sieve that extracts the Top-$K$ high-entropy geometric patches from raw 2K/4K images.

#### Constructor:
* `__init__(patch_size=16, keep_ratio=0.10)`
  * **Args**:
    * `patch_size` (*int*): Square patch size in pixels (default: 16).
    * `keep_ratio` (*float*): Fraction of active patches to retain (e.g., `0.05` keeps top 5%, pruning 95% of background).

#### Methods:
* `forward(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]`
  * **Args**: `img` (*torch.Tensor*): Full-resolution image $[B, C, H, W]$.
  * **Returns**:
    * `selected_patches` (*torch.Tensor*): $[B, K, C, P, P]$ containing only the top-$K$ active patches.
    * `coords` (*torch.Tensor*): $[B, K, 2]$ normalized global coordinates $(x, y) \in [0, 1]^2$.
    * `metadata` (*dict*): Contains `sparsity_ratio`, `attention_memory_reduction`, and `saliency_map`.

---

## 3. `morphic_cv.adapters.msa`

### `MorphicAdapter`
```python
class MorphicAdapter(nn.Module)
```
Parameter-Efficient Fine-Tuning (PEFT) adapter using non-linear mathematical morphology.

#### Constructor:
* `__init__(in_channels, bottleneck_dim=32, alpha=0.5, tau=0.5)`
  * **Args**:
    * `in_channels` (*int*): Input channel dimension of the feature manifold.
    * `bottleneck_dim` (*int*): Compressed intermediate channel dimension.
    * `alpha` (*float*): Residual scaling multiplier.
    * `tau` (*float*): Temperature parameter for Log-Sum-Exp smoothing.

---

## 4. `morphic_cv.morphology.ops`

### `SoftDilation2d`
```python
class SoftDilation2d(nn.Module)
```
Depthwise-separable differentiable morphological dilation using Log-Sum-Exp:
$$\text{Dilation}(X, K) = \tau \ln \left( \sum \exp((X + K) / \tau) \right)$$

### `SoftErosion2d`
```python
class SoftErosion2d(nn.Module)
```
Depthwise-separable differentiable morphological erosion:
$$\text{Erosion}(X, K) = -\tau \ln \left( \sum \exp(-(X - K) / \tau) \right)$$

### `MorphologicalGradient2d`
```python
class MorphologicalGradient2d(nn.Module)
```
Computes $\text{SoftDilation}(X) - \text{SoftErosion}(X)$ followed by a $1 \times 1$ pointwise projection and batch normalization.

---

## 5. `morphic_cv.morphology.rigl`

### `RigLScheduler`
```python
class RigLScheduler
```
Dynamic Sparse Training engine that evolves network connectivity during optimization.

#### Constructor:
* `__init__(model, target_sparsity=0.80, update_frequency=100, drop_fraction=0.3, total_steps=1000, exclude_layers=None)`
  * **Args**:
    * `target_sparsity` (*float*): Enforced fraction of zero weights (e.g. `0.80` = 80% zeros).
    * `update_frequency` (*int*): Number of optimizer steps between topology rewiring steps.
    * `drop_fraction` (*float*): Fraction of active weights pruned and regrown per update.

---

## 6. `morphic_cv.streaming`

### `ProductQuantizer`
```python
class ProductQuantizer(dim=128, num_subvectors=8, num_clusters=256)
```
Compresses float32 vector tokens into 8-bit `uint8` centroid codes ($64\times$ smaller).

### `TokenShardWriter`
```python
class TokenShardWriter(output_dir, dim=128, num_subvectors=8)
```
Serializes visual patch tokens into binary memory-mapped shards on disk.

### `StreamingTokenDataset`
```python
class StreamingTokenDataset(shard_dir)
```
PyTorch `Dataset` that reads slices directly from disk via `np.memmap`. Physical RAM is allocated strictly for the active mini-batch.
