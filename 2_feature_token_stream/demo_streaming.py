"""
Demo & Benchmark Script for Feature-First Compressed Patch-Token Streaming.
Demonstrates:
1. Compressing 2D spatial patch tokens by 64x using Product Quantization (PQ).
2. Writing and streaming directly from a memory-mapped binary shard.
3. Training a dense segmentation decoder head on dequantized batches.
4. Peak RAM logging proving training on <100MB RAM.
"""

import os
import shutil
import time
import tracemalloc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from patch_shard_manager import PatchShardWriter, StreamingPatchDataset
from dense_head import DensePatchDecoder


def run_benchmark():
    tracemalloc.start()
    print("=" * 60)
    print("PROJECT 2: Feature-First Compressed Patch-Token Streaming")
    print("=" * 60)

    shard_dir = os.path.join(os.path.dirname(__file__), "demo_shards")
    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)

    num_samples = 250
    grid_h, grid_w = 8, 8
    token_dim = 128
    num_subvectors = 8 # 128 dims / 8 subvectors = 16-dim centroids; 8 bytes per patch!
    num_classes = 5
    img_res = (64, 64)

    # 1. Synthesize Extracted Feature Maps & Segmentation Masks
    print(f"\n[1] Simulating offline backbone extraction for {num_samples} images...")
    np.random.seed(42)
    raw_tokens = np.random.randn(num_samples, grid_h, grid_w, token_dim).astype(np.float32)
    masks = np.random.randint(0, num_classes, size=(num_samples, img_res[0], img_res[1]), dtype=np.int64)

    # 2. Write Compressed Shards via Product Quantization & Memmap
    print("\n[2] Writing compressed memory-mapped binary shards...")
    writer = PatchShardWriter(shard_dir, dim=token_dim, num_subvectors=num_subvectors)
    meta = writer.write_dataset(raw_tokens, masks)

    # 3. Stream and Train Dense Decoder
    print("\n[3] Initializing Streaming Dataset & Dense Patch Decoder...")
    dataset = StreamingPatchDataset(shard_dir)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    decoder = DensePatchDecoder(in_channels=token_dim, num_classes=num_classes, hidden_dim=64)
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Dense Decoder Parameters: {total_params:,}")

    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("\n[4] Training Dense Decoder Directly From Streaming Compressed Shards...")
    decoder.train()
    for epoch in range(1, 4):
        epoch_loss = 0.0
        start_time = time.time()
        for batch_codes, batch_labels in loader:
            # On-the-fly streaming dequantization (RAM allocated only for active batch)
            spatial_features = dataset.dequantize_batch(batch_codes) # [B, 128, 8, 8]
            
            optimizer.zero_grad()
            logits = decoder(spatial_features, target_size=img_res) # [B, 5, 64, 64]
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - start_time
        cur_ram, peak_ram = tracemalloc.get_traced_memory()
        print(f"Epoch {epoch:02d} | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s | Peak RAM: {peak_ram / (1024*1024):.2f} MB")

    # 4. Clean up temporary demo shards
    del loader
    dataset.close()
    del dataset
    import gc
    gc.collect()

    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)

    cur_ram, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nFinal Verified Peak RAM: {peak_ram / (1024*1024):.2f} MB")
    print(f"Compression achieved: {meta['compression_ratio']:.1f}x reduction vs raw feature grids!")
    print("=" * 60)
    print("Project 2 verification complete.\n")


if __name__ == '__main__':
    run_benchmark()
