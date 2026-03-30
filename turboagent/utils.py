"""
Shared utility functions for TurboAgent.

Helpers for model resolution, token estimation, memory profiling,
and common operations used across backends and agents.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("turboagent.utils")


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def resolve_model_path(model_id: str, cache_dir: Optional[str] = None) -> str:
    """
    Resolve a model identifier to a local file path.

    Handles three cases:
      1. Local path (already exists) — returned as-is.
      2. HuggingFace hub ID — downloads via huggingface_hub if available.
      3. Short alias (e.g., "llama-70b") — looked up in a known alias table.

    Args:
        model_id: HuggingFace ID, local path, or short alias.
        cache_dir: Override default cache directory.

    Returns:
        Absolute path to the model file / directory.
    """
    # Case 1: already a local path
    if os.path.exists(model_id):
        return os.path.abspath(model_id)

    # Case 2: try huggingface_hub download
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            model_id,
            cache_dir=cache_dir,
            local_files_only=False,
        )
        logger.info(f"Downloaded model to: {path}")
        return path
    except ImportError:
        logger.warning("huggingface_hub not installed; cannot download from Hub.")
    except Exception as e:
        logger.warning(f"Failed to download '{model_id}' from Hub: {e}")

    raise FileNotFoundError(
        f"Could not resolve model '{model_id}'. Provide a local path or "
        f"install huggingface_hub: pip install huggingface_hub"
    )


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str, method: str = "heuristic") -> int:
    """
    Fast token count estimation.

    Args:
        text: Input string.
        method: "heuristic" (~4 chars/token) or "whitespace" (word count * 1.3).

    Returns:
        Estimated token count.
    """
    if method == "whitespace":
        return int(len(text.split()) * 1.3)
    # Default heuristic: ~4 characters per token (works well for English + code)
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Memory profiling
# ---------------------------------------------------------------------------

def get_gpu_memory_info() -> dict:
    """
    Return current GPU memory usage.

    Returns:
        Dict with total_gb, used_gb, free_gb (zeros if no GPU).
    """
    if not torch.cuda.is_available():
        return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0}

    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    free = total - reserved

    return {
        "total_gb": total / 1e9,
        "used_gb": allocated / 1e9,
        "free_gb": free / 1e9,
    }


def estimate_kv_memory_gb(
    num_layers: int,
    head_dim: int,
    seq_len: int,
    bit_mode: str = "turbo3",
) -> dict:
    """
    Estimate KV cache memory for a given configuration.

    Returns:
        Dict with fp16_gb (uncompressed baseline) and turbo_gb (compressed).
    """
    # FP16: 2 bytes * head_dim * seq_len * num_layers * 2 (K + V)
    fp16_bytes = 2 * head_dim * seq_len * num_layers * 2
    fp16_gb = fp16_bytes / 1e9

    compression = {"turbo3": 4.9, "turbo4": 3.8}.get(bit_mode, 4.9)
    turbo_gb = fp16_gb / compression

    return {
        "fp16_gb": fp16_gb,
        "turbo_gb": turbo_gb,
        "compression_ratio": compression,
        "savings_gb": fp16_gb - turbo_gb,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_chat_ml(messages: list) -> str:
    """Format a message list as ChatML (common fallback template)."""
    formatted = ""
    for m in messages:
        formatted += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted
