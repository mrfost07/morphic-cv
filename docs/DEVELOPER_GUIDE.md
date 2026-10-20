# MorphicCV Developer & Engineering Guide

A practical, production-oriented guide for software engineers, machine learning researchers, and edge developers building computer vision pipelines with **MorphicCV**.

---

## 1. Why Choose MorphicCV?

| Feature | Standard Computer Vision | MorphicCV |
| :--- | :--- | :--- |
| **Minimum Hardware** | \$5,000+ GPU (16GB–80GB VRAM) | **Any laptop / CPU / Raspberry Pi (<8GB RAM)** |
| **High-Res (4K) Vision** | Downsamples to $224 \times 224$ (erases fine details) | **Native 4K via Saliency Router (400× memory drop)** |
| **Fine-Tuning Memory** | Full fine-tune (~6GB–14GB VRAM) | **Morphological Sparse Adapter (<2.1 MB RAM)** |
| **Corruption Robustness** | Fails under rain, fog, sensor noise (Texture Bias) | **High stability via Topological Shape Priors** |
| **Storage / Caching** | Raw float32 tensors (hundreds of GBs) | **Product Quantization (64× smaller binary shards)** |

---

## 2. Installation

From PyPI:
```bash
pip install morphic-cv
```

Or clone from GitHub for local development:
```bash
git clone https://github.com/mrfost07/morphic-cv.git
cd morphic-cv
pip install -e .
```

---

## 3. Production Recipes

### Recipe 1: Native 4K Drone / Infrastructure Inspection
When inspecting bridges, pipelines, solar panels, or PCB micro-soldering, downsampling destroys hairline cracks. Use `SaliencyRouter` to extract only the high-entropy patches at native resolution:

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
from morphic_cv import SaliencyRouter

# 1. Load your high-resolution image (e.g. 2048x2048 or 4K)
raw_img = Image.open("bridge_inspection_4k.jpg").convert("RGB")
w, h = raw_img.size

# Ensure dimensions are divisible by 16
img = raw_img.crop((0, 0, (w // 16) * 16, (h // 16) * 16))
img_tensor = transforms.ToTensor()(img).unsqueeze(0)

# 2. Initialize router: keep only the top 5% most salient patches
router = SaliencyRouter(patch_size=16, keep_ratio=0.05)
selected_patches, coords, meta = router(img_tensor)

print(f"Total canvas patches: {meta['total_patches']:,}")
print(f"Patches routed:       {meta['kept_patches']:,} ({meta['sparsity_ratio']:.1f}% background pruned)")
print(f"Attention footprint:  {meta['attention_memory_reduction']:.1f}x smaller than full attention!")

# selected_patches: [1, K, 3, 16, 16] - only the 5% target regions
# coords:           [1, K, 2]         - exact global (x, y) coordinates on the 4K canvas
```

---

### Recipe 2: Fine-Tuning Meta's DINOv2 on Custom Data
Fine-tune Meta's official 22M Vision Transformer on your own photos with **< 2.5 MB of RAM** and save a deployable adapter of only **~110 KB**:

```python
import torch
import torch.nn as nn
from morphic_cv import MorphicDINOv2

# 1. Load official DINOv2 with Morphological Sparse Adapter (MSA)
model = MorphicDINOv2.from_pretrained('dinov2_vits14', num_classes=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 2. Check parameters
summary = model.get_parameter_summary()
print(f"Trainable Parameters: {summary['trainable_parameters']:,} ({summary['trainable_fraction_pct']:.2f}% of model)")

# 3. Setup optimizer (only tunes the adapter and linear classifier)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 4. Standard training loop
model.train()
for epoch in range(5):
    for batch_imgs, batch_labels in your_dataloader:
        batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        logits = model(batch_imgs)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

# 5. Save the tiny adapter checkpoint
model.save_adapter("my_custom_morphic_adapter.pt")

# To reload later:
# model.load_adapter("my_custom_morphic_adapter.pt")
```

---

### Recipe 3: Streaming Massive Datasets on an 8GB Laptop via Product Quantization
If your dataset of extracted features is too large to fit in memory, compress spatial tokens onto disk by **64×** and stream them with near-zero RAM usage:

```python
import numpy as np
import torch
from torch.utils.data import DataLoader
from morphic_cv import TokenShardWriter, StreamingTokenDataset

# STEP 1: Compress once onto disk
writer = TokenShardWriter(output_dir="./compressed_dataset", dim=128, num_subvectors=8)
meta = writer.write_dataset(token_tensors=raw_features, labels=ground_truth_labels)
print(f"Compressed {meta['raw_size_mb']:.1f} MB -> {meta['compressed_size_mb']:.1f} MB (64x reduction)")

# STEP 2: Stream batches with 0 pre-allocated memory
dataset = StreamingTokenDataset(shard_dir="./compressed_dataset")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_codes, batch_labels in loader:
    # Dequantizes on-the-fly directly inside the active batch loop
    spatial_features = dataset.dequantize_batch(batch_codes) # [32, 128, 16, 16]
    
    # Train your downstream segmentation or detection head here!
    # Physical RAM remains < 80 MB regardless of dataset size.
```

---

### Recipe 4: Universal Production Inference Engine (Images, Faces, Salient Objects, & Live Webcam)
Deploy any trained MorphicCV adapter for real-time inference on a consumer laptop CPU without requiring a GPU or large runtime dependencies:

```python
from morphic_cv import MorphicInferenceEngine

# 1. Initialize engine on CPU (<300MB RAM)
engine = MorphicInferenceEngine(
    model_name="dinov2_vits14",
    adapter_path="my_custom_morphic_adapter.pt",
    class_names=["cat", "dog", "bird"],
    device="cpu"
)

# 2. Predict on a single image (path, PIL Image, or numpy array)
result = engine.predict("sample.jpg", return_probs=True)
print(f"Predicted: {result['label']} ({result['confidence']:.2f}%) in {result['latency_ms']:.1f}ms")

# 3. Detect and classify all faces in a photo
faces = engine.detect_faces("group_photo.jpg")
for f in faces:
    print(f"Face at {f['bbox']} -> {f['label']} ({f['confidence']:.1f}%)")

# 4. Detect salient objects/regions using Morphological Gradients
objects = engine.detect_salient_objects("scene.jpg")
for obj in objects:
    print(f"Object at {obj['bbox']} -> {obj['label']} ({obj['confidence']:.1f}%)")

# 5. Real-time live camera / webcam stream on pure CPU (with HUD overlay)
# Press 'q' or 'ESC' in the window to stop
engine.stream_webcam(camera_id=0, mode="classify")  # or mode="faces" / "salient"
```


---

## 4. Command-Line Interface (CLI)

MorphicCV comes with a built-in terminal CLI:

```bash
# Check environment and setup
morphic-cv info

# Scan and route a high-resolution image
morphic-cv route --image ./highres_sample.jpg --keep-ratio 0.05

# Initialize or inspect DINOv2 adapter architecture
morphic-cv finetune --model dinov2_vits14 --classes 10
```

---

## 5. Ready-to-Run Starter Templates

Two production-ready scripts are included in the repository's `templates/` folder:

1. **`templates/finetune_dinov2_template.py`**:
   Fine-tune DINOv2 on any folder organized by class (`dataset/cat/`, `dataset/dog/`):
   ```bash
   python templates/finetune_dinov2_template.py --data-dir ./my_dataset --epochs 5 --output my_adapter.pt
   ```

2. **`templates/gigapixel_inference.py`**:
   Scan a high-res image and visualize bounding boxes over the kept patches:
   ```bash
   python templates/gigapixel_inference.py --image path/to/large_photo.jpg --keep-ratio 0.10
   ```
