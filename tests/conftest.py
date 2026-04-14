"""Shared fixtures for TurboAgent test suite."""

import pytest
import torch

from turboagent.quant.turboquant import TurboQuantKVCache


@pytest.fixture
def kv_cache_turbo3():
    """A small turbo3 cache for unit tests (1 layer, short context)."""
    return TurboQuantKVCache(
        bit_mode="turbo3",
        device="cpu",
        head_dim=128,
        num_layers=1,
        max_context=4096,
    )


@pytest.fixture
def kv_cache_turbo4():
    """A small turbo4 cache for unit tests (1 layer, short context)."""
    return TurboQuantKVCache(
        bit_mode="turbo4",
        device="cpu",
        head_dim=128,
        num_layers=1,
        max_context=4096,
    )


@pytest.fixture
def sample_kv_pair():
    """Generate a (seq_len=1024, head_dim=128) FP16 key/value pair."""
    k = torch.randn(1024, 128, dtype=torch.float16)
    v = torch.randn(1024, 128, dtype=torch.float16)
    return k, v


@pytest.fixture
def long_kv_pair():
    """Generate a (seq_len=4096, head_dim=128) FP16 key/value pair."""
    k = torch.randn(4096, 128, dtype=torch.float16)
    v = torch.randn(4096, 128, dtype=torch.float16)
    return k, v
