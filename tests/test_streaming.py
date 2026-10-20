"""
Unit tests for morphic_cv.streaming
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
import torch
from morphic_cv.streaming.quantizer import ProductQuantizer
from morphic_cv.streaming.shard import TokenShardWriter, StreamingTokenDataset


def test_product_quantizer_encode_decode():
    dim = 64
    M = 4 # 4 subvectors of 16 dims
    pq = ProductQuantizer(dim=dim, num_subvectors=M, num_clusters=32)
    
    # Random training vectors
    X = np.random.randn(200, dim).astype(np.float32)
    pq.fit(X, n_iter=5)

    codes = pq.encode(X[:10])
    assert codes.shape == (10, M)
    assert codes.dtype == np.uint8

    # Test PyTorch batch dequantization
    codes_tensor = torch.from_numpy(codes).unsqueeze(1) # [10, 1, M]
    codebooks_tensor = torch.from_numpy(pq.codebooks).float() # [M, 32, 16]
    decoded = pq.decode_torch(codes_tensor, codebooks_tensor) # [10, 1, 64]
    assert decoded.shape == (10, 1, dim)


def test_streaming_token_dataset(tmp_path):
    shard_dir = str(tmp_path / "test_shards")
    writer = TokenShardWriter(shard_dir, dim=32, num_subvectors=4)
    
    # 20 samples, 4x4 spatial patches, 32 dims
    tokens = np.random.randn(20, 4, 4, 32).astype(np.float32)
    labels = np.random.randint(0, 5, size=(20,))
    meta = writer.write_dataset(tokens, labels, sample_fit_size=50)

    assert meta["num_samples"] == 20
    assert meta["compression_ratio"] > 1.0

    dataset = StreamingTokenDataset(shard_dir)
    assert len(dataset) == 20
    codes, label = dataset[0]
    assert codes.shape == (16, 4)

    # Test batch dequantization
    batch_codes = torch.stack([dataset[i][0] for i in range(4)]) # [4, 16, 4]
    spatial_grid = dataset.dequantize_batch(batch_codes)
    assert spatial_grid.shape == (4, 32, 4, 4)

    dataset.close()
