# MorphicCV FAQ & Troubleshooting Guide

Common questions, hardware specifications, and debugging advice for engineers using MorphicCV.

---

## 1. Hardware Compatibility Matrix

| Hardware Target | Supported? | Expected Performance | Notes |
| :--- | :--- | :--- | :--- |
| **Intel / AMD Laptop CPU (8GB RAM)** | ✅ **Primary Target** | 3–5 seconds / epoch | Native target. Zero GPU required. |
| **Apple Silicon (M1/M2/M3 Mac)** | ✅ Fully Supported | 1–2 seconds / epoch | Accelerates via PyTorch MPS backend. |
| **Nvidia Consumer GPU (RTX 3060/4060)** | ✅ Fully Supported | < 1 second / epoch | Automatic CUDA acceleration. |
| **Raspberry Pi 5 (8GB)** | ✅ Supported | 10–15 seconds / epoch | Ideal for edge robotics and drone payloads. |
| **Cloud Instances (t3.medium / 4GB RAM)** | ✅ Supported | 5–8 seconds / epoch | Cuts cloud costs by eliminating GPU instances. |

---

## 2. Frequently Asked Questions

### Q: Does MorphicCV require an Nvidia GPU?
**No.** MorphicCV was explicitly engineered so that every operation—including DINOv2 foundation fine-tuning and 4K saliency routing—runs comfortably on standard x86 and ARM CPUs with under 8GB of RAM.

### Q: Can I use other backbones besides DINOv2?
**Yes.** While `MorphicDINOv2` is pre-packaged for convenience, the core `MorphicAdapter` and `SaliencyRouter` can be plugged into any PyTorch model (including MobileNet, ResNet, Swin, FastViT, or CLIP).

### Q: Why is my saved adapter checkpoint so small (~106 KB)?
Because MorphicCV freezes 100% of the foundation backbone. Only the non-linear morphological adapter and the linear classification head are saved to disk. When deploying, you load the base model from cache and apply the 106 KB adapter weights on top.

### Q: What is the optimal `keep_ratio` for SaliencyRouter?
* **For dense scenes with small defects (PCB/cracks)**: Use `keep_ratio=0.05` to `0.10` (pruning 90%–95% of background).
* **For cluttered scenes (crowded streets)**: Use `keep_ratio=0.15` to `0.20`.

---

## 3. Troubleshooting & Common Pitfalls

### Issue 1: `RuntimeError: Image dimensions must be divisible by patch size`
* **Cause**: `SaliencyRouter` partitions images into square grid blocks (default: 16 pixels).
* **Fix**: Crop or pad the image so width and height are divisible by 16:
  ```python
  w, h = img.size
  img = img.crop((0, 0, (w // 16) * 16, (h // 16) * 16))
  ```

### Issue 2: `PermissionError [WinError 32]` on Windows when deleting temporary shards
* **Cause**: Windows locks memory-mapped files (`np.memmap`) until explicitly closed.
* **Fix**: Call `dataset.close()` before deleting shard directories:
  ```python
  dataset = StreamingTokenDataset(shard_dir)
  # ... training loop ...
  dataset.close()
  del dataset
  ```

### Issue 3: `IndexError: Target X is out of bounds`
* **Cause**: The number of folders in your dataset does not match the `num_classes` parameter.
* **Fix**: Ensure `num_classes` matches the number of subfolders in your dataset directory.
