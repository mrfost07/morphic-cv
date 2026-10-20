"""
morphic_cv.streaming.shard
Memory-mapped shard reader and writer.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .quantizer import ProductQuantizer


class TokenShardWriter:
    def __init__(self, output_dir: str, dim: int = 128, num_subvectors: int = 8):
        self.output_dir = output_dir
        self.dim = dim
        self.M = num_subvectors
        os.makedirs(output_dir, exist_ok=True)
        self.pq = ProductQuantizer(dim=dim, num_subvectors=num_subvectors)

    def write_dataset(self, token_tensors: np.ndarray, labels: np.ndarray, sample_fit_size: int = 2000):
        N, H_p, W_p, D = token_tensors.shape
        num_patches = H_p * W_p
        flat_tokens = token_tensors.reshape(-1, D)

        sample_idx = np.random.choice(flat_tokens.shape[0], size=min(sample_fit_size, flat_tokens.shape[0]), replace=False)
        self.pq.fit(flat_tokens[sample_idx])

        codes = self.pq.encode(flat_tokens).reshape(N, num_patches, self.M)

        bin_path = os.path.join(self.output_dir, "tokens.bin")
        fp = np.memmap(bin_path, dtype=np.uint8, mode='w+', shape=codes.shape)
        fp[:] = codes[:]
        fp.flush()
        del fp

        np.save(os.path.join(self.output_dir, "codebooks.npy"), self.pq.codebooks)
        np.save(os.path.join(self.output_dir, "labels.npy"), labels)

        meta = {
            "num_samples": N,
            "grid_h": H_p,
            "grid_w": W_p,
            "num_patches": num_patches,
            "token_dim": D,
            "num_subvectors": self.M,
            "compression_ratio": (D * 4) / self.M
        }
        with open(os.path.join(self.output_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return meta


class StreamingTokenDataset(Dataset):
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

        bin_path = os.path.join(shard_dir, "tokens.bin")
        self.mmap_codes = np.memmap(bin_path, dtype=np.uint8, mode='r', shape=(self.N, self.num_patches, self.M))
        codebooks_np = np.load(os.path.join(shard_dir, "codebooks.npy"))
        self.codebooks_tensor = torch.from_numpy(codebooks_np).float()
        self.labels = np.load(os.path.join(shard_dir, "labels.npy"))
        self.pq = ProductQuantizer(dim=self.dim, num_subvectors=self.M)

    def close(self):
        if hasattr(self, 'mmap_codes') and self.mmap_codes is not None:
            del self.mmap_codes
            self.mmap_codes = None

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        codes = torch.from_numpy(self.mmap_codes[idx].copy())
        lbl = self.labels[idx]
        label = torch.tensor(lbl) if np.isscalar(lbl) else torch.from_numpy(lbl.copy())
        return codes, label

    def dequantize_batch(self, batch_codes: torch.Tensor) -> torch.Tensor:
        B, P, M = batch_codes.shape
        tokens = self.pq.decode_torch(batch_codes, self.codebooks_tensor)
        return tokens.view(B, self.grid_h, self.grid_w, self.dim).permute(0, 3, 1, 2)
