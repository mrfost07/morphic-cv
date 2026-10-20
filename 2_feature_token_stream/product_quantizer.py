"""
Pure PyTorch/NumPy Product Quantizer (PQ).
Compresses high-dimensional visual patch tokens into tiny byte codes.

Mathematical formulation:
Given a token vector x in R^D:
1. Split x into M sub-vectors: [u_1, u_2, ..., u_M], each in R^(D/M).
2. For each sub-space m in {1..M}, find the closest centroid in a learned codebook C_m of size K=256.
3. Represent x as an M-tuple of 8-bit integers (uint8): [c_1, c_2, ..., c_M].

Compression ratio:
A D=384 float32 vector (1,536 bytes) compressed with M=16 bytes yields 96x compression!
"""

import numpy as np
import torch
import torch.nn as nn


class ProductQuantizer:
    def __init__(self, dim: int = 128, num_subvectors: int = 8, num_clusters: int = 256):
        assert dim % num_subvectors == 0, f"dim {dim} must be divisible by num_subvectors {num_subvectors}"
        self.dim = dim
        self.M = num_subvectors
        self.d = dim // num_subvectors
        self.K = num_clusters
        # Codebooks: [M, K, d]
        self.codebooks = None

    def fit(self, vectors: np.ndarray, n_iter: int = 15):
        """
        Fits K-means clustering per sub-space on a sample of token vectors.
        vectors shape: [N, dim]
        """
        N, D = vectors.shape
        assert D == self.dim
        # Reshape to [N, M, d] -> transpose to [M, N, d]
        sub_vecs = vectors.reshape(N, self.M, self.d).transpose(1, 0, 2)
        
        codebooks = np.zeros((self.M, self.K, self.d), dtype=np.float32)

        for m in range(self.M):
            X_m = sub_vecs[m] # [N, d]
            # Simple, fast K-means
            indices = np.random.choice(N, size=self.K, replace=(N < self.K))
            centroids = X_m[indices].copy()
            
            for _ in range(n_iter):
                # Compute pairwise squared Euclidean distances: [N, K]
                # (x - c)^2 = x^2 + c^2 - 2 x c^T
                dists = (
                    np.sum(X_m ** 2, axis=1, keepdims=True) +
                    np.sum(centroids ** 2, axis=1, keepdims=True).T -
                    2.0 * np.dot(X_m, centroids.T)
                )
                labels = np.argmin(dists, axis=1) # [N]
                
                # Update centroids
                for k in range(self.K):
                    mask = (labels == k)
                    if np.any(mask):
                        centroids[k] = X_m[mask].mean(axis=0)
            codebooks[m] = centroids
            
        self.codebooks = codebooks

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Quantizes vectors [N, dim] into uint8 codes [N, M].
        """
        assert self.codebooks is not None, "Quantizer must be fitted first!"
        N, D = vectors.shape
        sub_vecs = vectors.reshape(N, self.M, self.d).transpose(1, 0, 2)
        codes = np.zeros((N, self.M), dtype=np.uint8)

        for m in range(self.M):
            X_m = sub_vecs[m] # [N, d]
            centroids = self.codebooks[m] # [K, d]
            dists = (
                np.sum(X_m ** 2, axis=1, keepdims=True) +
                np.sum(centroids ** 2, axis=1, keepdims=True).T -
                2.0 * np.dot(X_m, centroids.T)
            )
            codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)
            
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Reconstructs approximate vectors [N, dim] from uint8 codes [N, M].
        """
        assert self.codebooks is not None, "Quantizer must be fitted first!"
        N, M = codes.shape
        reconstructed = np.zeros((N, M, self.d), dtype=np.float32)

        for m in range(self.M):
            idx = codes[:, m] # [N]
            reconstructed[:, m, :] = self.codebooks[m][idx]

        return reconstructed.reshape(N, self.dim)

    def decode_torch(self, codes_tensor: torch.Tensor, codebooks_tensor: torch.Tensor) -> torch.Tensor:
        """
        Fast on-the-fly PyTorch dequantization directly inside DataLoader or GPU/CPU batch.
        codes_tensor: [B, num_patches, M] (torch.uint8 or torch.long)
        codebooks_tensor: [M, K, d] (torch.float32)
        Returns: [B, num_patches, dim]
        """
        B, P, M = codes_tensor.shape
        codes_long = codes_tensor.long()
        d = codebooks_tensor.shape[-1]
        
        # Gather centroids
        decompressed_parts = []
        for m in range(M):
            # codebooks_tensor[m]: [K, d]
            # codes_long[:, :, m]: [B, P]
            part = codebooks_tensor[m][codes_long[:, :, m]] # [B, P, d]
            decompressed_parts.append(part)
            
        return torch.cat(decompressed_parts, dim=-1) # [B, P, M*d = dim]
