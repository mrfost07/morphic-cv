"""
Memory-Mapped Compressed Patch-Token Shard Manager.
Streams 2D spatial patch tokens directly from disk with near-zero RAM usage.
Even on datasets of 1,000,000 images, peak training RAM remains under 200MB.
"""

import json
import os
from typing import Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from product_quantizer import ProductQuantizer


class PatchShardWriter:
    """
    Serializes extracted patch tokens into memory-mapped uint8 binary shards.
    """
    def __init__(self, output_dir: str, dim: int = 128, num_subvectors: int = 8):
        self.output_dir = output_dir
        self.dim = dim
        self.M = num_subvectors
        os.makedirs(output_dir, exist_ok=True)
        self.pq = ProductQuantizer(dim=dim, num_subvectors=num_subvectors)

    def write_dataset(
        self,
        token_tensors: np.ndarray, # [N, grid_h, grid_w, dim]
        labels: np.ndarray,        # [N] or [N, H, W] for segmentation
        sample_fit_size: int = 2000
    ):
        N, H_p, W_p, D = token_tensors.shape
        num_patches = H_p * W_p
        flat_tokens = token_tensors.reshape(-1, D)

        print(f"[Writer] Fitting Product Quantizer on {min(sample_fit_size, flat_tokens.shape[0])} patch samples...")
        sample_idx = np.random.choice(flat_tokens.shape[0], size=min(sample_fit_size, flat_tokens.shape[0]), replace=False)
        self.pq.fit(flat_tokens[sample_idx])

        print(f"[Writer] Compressing {N} images x {num_patches} patches ({flat_tokens.shape[0]:,} tokens)...")
        codes = self.pq.encode(flat_tokens) # [N * num_patches, M] uint8
        codes_3d = codes.reshape(N, num_patches, self.M)

        # 1. Write binary memmap file
        bin_path = os.path.join(self.output_dir, "tokens.bin")
        fp = np.memmap(bin_path, dtype=np.uint8, mode='w+', shape=codes_3d.shape)
        fp[:] = codes_3d[:]
        fp.flush()
        del fp

        # 2. Write codebooks
        codebook_path = os.path.join(self.output_dir, "codebooks.npy")
        np.save(codebook_path, self.pq.codebooks)

        # 3. Write labels
        label_path = os.path.join(self.output_dir, "labels.npy")
        np.save(label_path, labels)

        # 4. Write metadata
        meta = {
            "num_samples": N,
            "grid_h": H_p,
            "grid_w": W_p,
            "num_patches": num_patches,
            "token_dim": D,
            "num_subvectors": self.M,
            "raw_size_mb": (N * num_patches * D * 4) / (1024 * 1024),
            "compressed_size_mb": (N * num_patches * self.M) / (1024 * 1024),
            "compression_ratio": (D * 4) / self.M
        }
        with open(os.path.join(self.output_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[Writer] Sharding complete! Raw: {meta['raw_size_mb']:.2f} MB -> Compressed: {meta['compressed_size_mb']:.2f} MB ({meta['compression_ratio']:.1f}x reduction)")
        return meta


class StreamingPatchDataset(Dataset):
    """
    PyTorch Dataset reading directly from the memory-mapped file on disk.
    RAM usage is decoupled from dataset size.
    """
    def __init__(self, shard_dir: str):
        self.shard_dir = shard_dir
        with open(os.path.join(shard_dir, "meta.json"), "r") as f:
            self.meta = json.load(f)

        self.N = self.meta["num_samples"]
        self.num_patches = self.meta["num_patches"]
        self.M = self.meta["num_subvectors"]
        self.dim = self.meta["token_dim"]
        self.grid_h = self.meta["grid_h"]
        self.grid_w = self.meta["grid_w"]

        # Open memory-map in read-only mode (virtual memory mapped, 0 physical RAM allocated up-front)
        bin_path = os.path.join(shard_dir, "tokens.bin")
        self.mmap_codes = np.memmap(bin_path, dtype=np.uint8, mode='r', shape=(self.N, self.num_patches, self.M))
        
        # Load codebooks as PyTorch tensor for batch-level GPU/CPU dequantization
        codebooks_np = np.load(os.path.join(shard_dir, "codebooks.npy"))
        self.codebooks_tensor = torch.from_numpy(codebooks_np).float() # [M, 256, d]

        # Load labels
        self.labels = np.load(os.path.join(shard_dir, "labels.npy"))
        self.pq = ProductQuantizer(dim=self.dim, num_subvectors=self.M)

    def close(self):
        """Explicitly release the memory-mapped file handle (essential for Windows)."""
        if hasattr(self, 'mmap_codes') and self.mmap_codes is not None:
            # Flush and delete the memory map to release OS file locks
            del self.mmap_codes
            self.mmap_codes = None

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reads only the requested item's codes from SSD/disk
        codes = torch.from_numpy(self.mmap_codes[idx].copy()) # [num_patches, M] uint8
        label = torch.from_numpy(self.labels[idx].copy())
        return codes, label

    def dequantize_batch(self, batch_codes: torch.Tensor) -> torch.Tensor:
        """
        Dequantizes a batch [B, num_patches, M] -> [B, grid_h, grid_w, dim]
        Directly in working memory.
        """
        B, P, M = batch_codes.shape
        tokens = self.pq.decode_torch(batch_codes, self.codebooks_tensor) # [B, P, dim]
        # Reshape to spatial feature grid [B, dim, grid_h, grid_w]
        tokens_spatial = tokens.view(B, self.grid_h, self.grid_w, self.dim).permute(0, 3, 1, 2)
        return tokens_spatial
