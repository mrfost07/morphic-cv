"""
morphic_cv.streaming.quantizer
Dependency-free Product Quantizer for visual patch token compression.
"""

import numpy as np
import torch


class ProductQuantizer:
    def __init__(self, dim: int = 128, num_subvectors: int = 8, num_clusters: int = 256):
        assert dim % num_subvectors == 0
        self.dim = dim
        self.M = num_subvectors
        self.d = dim // num_subvectors
        self.K = num_clusters
        self.codebooks = None

    def fit(self, vectors: np.ndarray, n_iter: int = 15):
        N, D = vectors.shape
        sub_vecs = vectors.reshape(N, self.M, self.d).transpose(1, 0, 2)
        codebooks = np.zeros((self.M, self.K, self.d), dtype=np.float32)

        for m in range(self.M):
            X_m = sub_vecs[m]
            indices = np.random.choice(N, size=self.K, replace=(N < self.K))
            centroids = X_m[indices].copy()
            for _ in range(n_iter):
                dists = (
                    np.sum(X_m ** 2, axis=1, keepdims=True) +
                    np.sum(centroids ** 2, axis=1, keepdims=True).T -
                    2.0 * np.dot(X_m, centroids.T)
                )
                labels = np.argmin(dists, axis=1)
                for k in range(self.K):
                    mask = (labels == k)
                    if np.any(mask):
                        centroids[k] = X_m[mask].mean(axis=0)
            codebooks[m] = centroids
        self.codebooks = codebooks

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        N, D = vectors.shape
        sub_vecs = vectors.reshape(N, self.M, self.d).transpose(1, 0, 2)
        codes = np.zeros((N, self.M), dtype=np.uint8)
        for m in range(self.M):
            X_m = sub_vecs[m]
            centroids = self.codebooks[m]
            dists = (
                np.sum(X_m ** 2, axis=1, keepdims=True) +
                np.sum(centroids ** 2, axis=1, keepdims=True).T -
                2.0 * np.dot(X_m, centroids.T)
            )
            codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)
        return codes

    def decode_torch(self, codes_tensor: torch.Tensor, codebooks_tensor: torch.Tensor) -> torch.Tensor:
        B, P, M = codes_tensor.shape
        codes_long = codes_tensor.long()
        parts = [codebooks_tensor[m][codes_long[:, :, m]] for m in range(M)]
        return torch.cat(parts, dim=-1)
