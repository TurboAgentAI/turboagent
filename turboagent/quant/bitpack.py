"""
Bit-packing layer for TurboQuant compressed KV cache.

Packs the output of turboquant-kv (int64 indices + float32 signs) into
a compact bit-packed format for memory-efficient storage between turns.

Storage format per KV pair (K or V):
  - idx: packed into ceil(mse_bits/8) bytes per element
  - sign: 1 bit per element, packed 8-per-byte
  - norm: float16 per vector (1 value per seq position per head)
  - gamma: float16 per vector

For turbo3 (bits=3): MSE uses 2 bits, QJL uses 1 bit → 3 bits/element
For turbo4 (bits=4): MSE uses 3 bits, QJL uses 1 bit → 4 bits/element

Achieves ~3-4x compression over FP16 baseline (vs 0.2x with unpacked format).
"""

import math
from typing import Dict, Tuple

import torch
import numpy as np


def pack_kv_dict(kv_dict: Dict[str, torch.Tensor], mse_bits: int) -> Dict[str, bytes | torch.Tensor]:
    """
    Pack a turboquant-kv output dict into compact bit-packed storage.

    Args:
        kv_dict: Output from TurboQuantProd.quantize_kv() with keys
                 k_idx, k_norm, k_sign, k_gamma, v_idx, v_norm, v_sign, v_gamma.
        mse_bits: Number of MSE bits (bits - 1 from TurboQuantProd).

    Returns:
        Dict with packed byte tensors and metadata for unpacking.
    """
    packed = {"mse_bits": mse_bits}

    for prefix in ("k", "v"):
        idx = kv_dict[f"{prefix}_idx"]
        sign = kv_dict[f"{prefix}_sign"]
        norm = kv_dict[f"{prefix}_norm"]
        gamma = kv_dict[f"{prefix}_gamma"]

        shape = idx.shape  # (batch, heads, seq, dim)

        # Pack indices (mse_bits per element)
        packed[f"{prefix}_idx_packed"] = _pack_indices(idx.cpu(), mse_bits)
        # Pack signs (1 bit per element)
        packed[f"{prefix}_sign_packed"] = _pack_signs(sign.cpu())
        # Norms and gammas: just downcast to float16
        packed[f"{prefix}_norm"] = norm.cpu().half()
        packed[f"{prefix}_gamma"] = gamma.cpu().half()
        packed[f"{prefix}_shape"] = shape

    return packed


def unpack_kv_dict(
    packed: Dict, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> Dict[str, torch.Tensor]:
    """
    Unpack bit-packed storage back to turboquant-kv format.

    Returns dict compatible with TurboQuantProd.decompress() /
    quantized_attention_fused_auto().
    """
    mse_bits = packed["mse_bits"]
    result = {}

    for prefix in ("k", "v"):
        shape = packed[f"{prefix}_shape"]

        # Unpack indices
        result[f"{prefix}_idx"] = _unpack_indices(
            packed[f"{prefix}_idx_packed"], shape, mse_bits
        ).to(device=device, dtype=torch.int64)

        # Unpack signs
        result[f"{prefix}_sign"] = _unpack_signs(
            packed[f"{prefix}_sign_packed"], shape
        ).to(device=device, dtype=dtype)

        # Norms and gammas: upcast back to target dtype
        result[f"{prefix}_norm"] = packed[f"{prefix}_norm"].to(device=device, dtype=dtype)
        result[f"{prefix}_gamma"] = packed[f"{prefix}_gamma"].to(device=device, dtype=dtype)

    return result


def packed_size_bytes(packed: Dict) -> int:
    """Total storage size of the packed representation in bytes."""
    total = 0
    for key, val in packed.items():
        if isinstance(val, torch.Tensor):
            total += val.element_size() * val.numel()
        elif isinstance(val, (bytes, bytearray, np.ndarray)):
            total += len(val)
    return total


# ---------------------------------------------------------------------------
# Index packing: N-bit integers into byte arrays
# ---------------------------------------------------------------------------

def _pack_indices(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Pack integer indices (values 0..2^bits-1) into a compact byte tensor.

    For bits <= 4, we pack floor(8/bits) values per byte.
    For bits=2: 4 values per byte
    For bits=3: 2 values per byte (wastes 2 bits, but fast)
    For bits=4: 2 values per byte (exact)

    Using a simple scheme: values_per_byte = 8 // bits
    """
    flat = idx.reshape(-1).to(torch.uint8)
    n = flat.numel()

    if bits <= 4:
        vals_per_byte = 8 // bits
        # Pad to multiple of vals_per_byte
        pad_n = (vals_per_byte - (n % vals_per_byte)) % vals_per_byte
        if pad_n > 0:
            flat = torch.cat([flat, torch.zeros(pad_n, dtype=torch.uint8)])

        flat = flat.reshape(-1, vals_per_byte)
        packed = torch.zeros(flat.shape[0], dtype=torch.uint8)
        for i in range(vals_per_byte):
            packed |= (flat[:, i] << (i * bits))

        return packed
    else:
        # For bits > 4, just store as uint8 (values fit in one byte for bits <= 8)
        return flat


def _unpack_indices(packed: torch.Tensor, shape: tuple, bits: int) -> torch.Tensor:
    """Unpack byte tensor back to integer indices."""
    n_elements = 1
    for s in shape:
        n_elements *= s
    mask = (1 << bits) - 1

    if bits <= 4:
        vals_per_byte = 8 // bits
        flat = torch.zeros(packed.numel() * vals_per_byte, dtype=torch.int64)
        for i in range(vals_per_byte):
            flat[i::vals_per_byte] = ((packed >> (i * bits)) & mask).to(torch.int64)
        return flat[:n_elements].reshape(shape)
    else:
        return packed[:n_elements].to(torch.int64).reshape(shape)


# ---------------------------------------------------------------------------
# Sign packing: {-1, +1} → 1 bit per element
# ---------------------------------------------------------------------------

def _pack_signs(sign: torch.Tensor) -> torch.Tensor:
    """Pack {-1.0, +1.0} tensor into 1-bit-per-element byte tensor."""
    flat = (sign.reshape(-1) > 0).to(torch.uint8)
    n = flat.numel()

    # Pad to multiple of 8
    pad_n = (8 - (n % 8)) % 8
    if pad_n > 0:
        flat = torch.cat([flat, torch.zeros(pad_n, dtype=torch.uint8)])

    flat = flat.reshape(-1, 8)
    packed = torch.zeros(flat.shape[0], dtype=torch.uint8)
    for i in range(8):
        packed |= (flat[:, i] << i)

    return packed


def _unpack_signs(packed: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Unpack 1-bit signs back to {-1.0, +1.0} tensor."""
    n_elements = 1
    for s in shape:
        n_elements *= s

    flat = torch.zeros(packed.numel() * 8, dtype=torch.float32)
    for i in range(8):
        bits = ((packed >> i) & 1).to(torch.float32)
        flat[i::8] = bits * 2.0 - 1.0  # 0 → -1.0, 1 → +1.0

    return flat[:n_elements].reshape(shape)
