"""
Option C — Streaming KV Decompression (zero intra-turn compression).

Design philosophy
-----------------
TurboQuant compression is *inter-turn* only.  Within a single turn:

  • Prefill K/V is accumulated in raw CPU buffers (model's native dtype) — no compression.
  • Decode K/V is appended to those same buffers — no compression.
  • At the END of the turn, ``finalize_decode()`` compresses everything into
    turbo ONCE, for multi-turn persistence.

Why not compress during the turn?
  Compressing the prefill before decode (as the first implementation tried)
  introduces ~8% noise (0.92 cosine-sim) into EVERY attention query during
  decode.  For unique short strings like "ALPHA-7734", that level of noise
  is enough to cause the model to miss the needle.  Raw FP16 buffers during
  the current turn eliminate this problem entirely while adding only negligible
  system-RAM cost (≤ 1 GB for 16k-token contexts on a typical 7B model).

Previous-turn KV *is* compressed (stored in turbo from their ``finalize_decode``).
The decompression noise for prior turns is unavoidable, but it affects turns
far in the past and is tolerable for conversational use.

GPU peak: ≈ one layer's full K + V tensors (ephemeral, during attention only).
CPU RAM:  raw buffers for the current turn (model dtype, 2 B/elem for bfloat16/float16)
          + turbo-compressed KV for all prior turns (≈ 5.8× smaller than FP16).

HuggingFace contract
--------------------
StreamingDynamicCache satisfies the full ``DynamicCache`` interface:
  ``update(k, v, layer_idx)``  — build & return full GPU KV; buffer new tokens
  ``get_seq_length()``         — past tokens before the current forward pass
  ``get_usable_length()``      — same
  ``get_mask_sizes()``         — (kv_length, 0) for causal mask sizing
  ``seen_tokens`` property     — same as get_seq_length()
  ``__len__``                  — number of layers
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

from turboagent.quant.turboquant import TurboQuantKVCache

logger = logging.getLogger("turboagent.quant.streaming")


class StreamingDynamicCache:
    """
    DynamicCache-compatible KV cache that keeps the CURRENT TURN in raw FP16
    CPU buffers and compresses to TurboQuantKVCache only at the end of the turn.

    Pass directly as ``past_key_values`` to any HuggingFace model that calls
    ``cache.update(k, v, layer_idx)`` internally.

    Lifecycle::

        cache = StreamingDynamicCache(...)
        # --- Prefill (chunked or single) ---
        for chunk in prompt_chunks:
            out = model(..., past_key_values=cache)
            cache = out.past_key_values   # same object returned by model
        cache.start_decode()              # flip to decode mode (no-op compression)
        # --- Decode loop ---
        for step in range(max_new_tokens):
            out = model(..., past_key_values=cache)
        cache.finalize_decode()           # compress current turn → turbo ONCE

    Args:
        turbo_cache:        Pre-existing TurboQuantKVCache (previous turns' KV).
        layer_shapes:       Per-layer (n_kv_heads, head_dim) from TorchEngine.
        layer_devices:      Per-layer device (retained for API compatibility).
        dtype:              Model dtype (bfloat16 / float16).
        default_n_kv_heads: Fallback n_kv_heads for layers not in layer_shapes.
        default_head_dim:   Fallback head_dim for layers not in layer_shapes.
    """

    def __init__(
        self,
        turbo_cache: TurboQuantKVCache,
        layer_shapes: Dict[int, Tuple[int, int]],
        layer_devices: Dict[int, torch.device],
        dtype: torch.dtype,
        default_n_kv_heads: int,
        default_head_dim: int,
    ) -> None:
        self._turbo = turbo_cache
        self._layer_shapes = layer_shapes
        self._layer_devices = layer_devices       # kept for API compat
        self._dtype = dtype
        self._default_n_kv_heads = default_n_kv_heads
        self._default_head_dim = default_head_dim
        self._n_layers = turbo_cache.num_layers

        # ── Current-turn raw FP16 buffers (prefill, CPU) ──────────────────────
        # Shape per layer: (1, n_kv_heads, seq, head_dim), dtype=float16, CPU.
        # Grows by chunk_size per prefill forward pass (at layer 0).
        self._prefill_k: List[Optional[torch.Tensor]] = [None] * self._n_layers
        self._prefill_v: List[Optional[torch.Tensor]] = [None] * self._n_layers
        self._prefill_seq_len: int = 0  # tokens accumulated in prefill buffer

        # ── Current-turn raw FP16 buffers (decode, CPU) ───────────────────────
        # Grows by 1 per decode step (at layer 0).
        self._is_decoding: bool = False
        self._decode_k: Optional[List[Optional[torch.Tensor]]] = None
        self._decode_v: Optional[List[Optional[torch.Tensor]]] = None
        self._decode_seq_len: int = 0

    # ------------------------------------------------------------------
    # DynamicCache interface required by HuggingFace Transformers
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build and return full K/V for this layer's attention.

        Prefill mode: [turbo_past] + [prefill_so_far] + [new_chunk]
        Decode mode:  [turbo_past] + [prefill_all]  + [decode_so_far] + [new_token]

        Turbo is NEVER updated here.  Raw buffers grow by the new slice each call.
        The returned tensors are ephemeral; Python GC frees them after attention.

        Returns
        -------
        (full_key, full_value) on GPU, shape (1, n_kv_heads, total_seq, head_dim).
        """
        nkh, hd = self._layer_shapes.get(
            layer_idx, (self._default_n_kv_heads, self._default_head_dim)
        )
        dev = key_states.device
        turbo_seq = self._turbo._seq_len  # previous turns; never changes during a turn

        parts_k: list = []
        parts_v: list = []

        # ── (1) Previous turns from turbo ────────────────────────────────────
        if turbo_seq > 0 and self._turbo._cache_k[layer_idx] is not None:
            k_flat, v_flat = self._turbo.get(layer_idx, turbo_seq)
            k_t = (
                k_flat.to(device=dev, dtype=self._dtype)
                .view(turbo_seq, nkh, hd)
                .permute(1, 0, 2)
                .unsqueeze(0)
            )
            v_t = (
                v_flat.to(device=dev, dtype=self._dtype)
                .view(turbo_seq, nkh, hd)
                .permute(1, 0, 2)
                .unsqueeze(0)
            )
            parts_k.append(k_t)
            parts_v.append(v_t)

        # ── (2) This turn's prefill buffer (raw FP16, CPU) ───────────────────
        if self._prefill_k[layer_idx] is not None:
            parts_k.append(self._prefill_k[layer_idx].to(device=dev, dtype=self._dtype))
            parts_v.append(self._prefill_v[layer_idx].to(device=dev, dtype=self._dtype))

        # ── (3) This turn's decode buffer (raw FP16, CPU) — decode mode only ──
        if self._is_decoding and self._decode_k[layer_idx] is not None:
            parts_k.append(self._decode_k[layer_idx].to(device=dev, dtype=self._dtype))
            parts_v.append(self._decode_v[layer_idx].to(device=dev, dtype=self._dtype))

        # ── (4) Current new tokens ────────────────────────────────────────────
        parts_k.append(key_states)
        parts_v.append(value_states)

        full_k = torch.cat(parts_k, dim=2) if len(parts_k) > 1 else parts_k[0]
        full_v = torch.cat(parts_v, dim=2) if len(parts_v) > 1 else parts_v[0]

        # ── Buffer the new slice on CPU in the model's native dtype ──────────
        # Storing in self._dtype (bfloat16 for GPU models) avoids the
        # bfloat16→float16→bfloat16 round-trip that the previous .half() call
        # caused.  Both bfloat16 and float16 are 2 bytes, so memory usage is
        # unchanged; precision loss is eliminated.
        new_k_cpu = key_states.detach().cpu().to(dtype=self._dtype)
        new_v_cpu = value_states.detach().cpu().to(dtype=self._dtype)

        if not self._is_decoding:
            # Prefill mode: grow the prefill buffer
            if self._prefill_k[layer_idx] is None:
                self._prefill_k[layer_idx] = new_k_cpu
                self._prefill_v[layer_idx] = new_v_cpu
            else:
                self._prefill_k[layer_idx] = torch.cat(
                    [self._prefill_k[layer_idx], new_k_cpu], dim=2
                )
                self._prefill_v[layer_idx] = torch.cat(
                    [self._prefill_v[layer_idx], new_v_cpu], dim=2
                )
            if layer_idx == 0:
                self._prefill_seq_len += key_states.shape[2]
        else:
            # Decode mode: grow the decode buffer
            if self._decode_k[layer_idx] is None:
                self._decode_k[layer_idx] = new_k_cpu
                self._decode_v[layer_idx] = new_v_cpu
            else:
                self._decode_k[layer_idx] = torch.cat(
                    [self._decode_k[layer_idx], new_k_cpu], dim=2
                )
                self._decode_v[layer_idx] = torch.cat(
                    [self._decode_v[layer_idx], new_v_cpu], dim=2
                )
            if layer_idx == 0:
                self._decode_seq_len += 1

        return full_k, full_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """
        Tokens committed before the current forward pass.

        = turbo (previous turns) + prefill buffer + decode buffer.
        Always reflects the correct positional offset for the next forward pass.
        """
        return self._turbo._seq_len + self._prefill_seq_len + self._decode_seq_len

    def get_usable_length(self, new_seq_len: int, layer_idx: int = 0) -> int:
        """Usable past length (full context — no sliding window)."""
        return self._turbo._seq_len + self._prefill_seq_len + self._decode_seq_len

    def get_mask_sizes(
        self,
        cache_position: "torch.Tensor",
        layer_idx: Optional[int] = None,
    ) -> tuple:
        """
        (kv_length, kv_offset) for causal mask construction.

        kv_length = all past tokens + the current step's tokens.
        """
        new_seq_len = cache_position.shape[-1]
        total_past = (
            self._turbo._seq_len + self._prefill_seq_len + self._decode_seq_len
        )
        return total_past + new_seq_len, 0

    def __len__(self) -> int:
        return self._n_layers

    @property
    def seen_tokens(self) -> int:
        return self._turbo._seq_len + self._prefill_seq_len + self._decode_seq_len

    @seen_tokens.setter
    def seen_tokens(self, value: int) -> None:
        # Some transformers versions write this; ignore it
        pass

    # ------------------------------------------------------------------
    # Phase-transition methods (called by TorchEngine)
    # ------------------------------------------------------------------

    def start_decode(self) -> None:
        """
        Transition from prefill to decode mode.

        No compression happens here.  The prefill buffer stays in raw FP16
        on CPU and will be used as-is during decode attention.  This avoids
        the quantization noise that would result from compressing before decode.

        ``finalize_decode()`` handles compression at the end of the turn.
        """
        self._is_decoding = True
        self._decode_k = [None] * self._n_layers
        self._decode_v = [None] * self._n_layers
        self._decode_seq_len = 0
        logger.debug(
            f"start_decode(): prefill={self._prefill_seq_len} tokens buffered, "
            f"switching to decode mode (no compression until finalize_decode)"
        )

    def finalize_decode(self) -> None:
        """
        Compress the current turn's raw K/V into turbo ONCE, for persistence.

        Combines: previous-turn turbo  +  this-turn prefill buffer
                                       +  this-turn decode buffer
        → single turbo.update() call per layer.

        After this call ``turbo._seq_len`` covers all tokens from all turns.
        The raw buffers are cleared.  Call at the very end of Phase 4 (after
        the decode loop), before reading ``kv_cache._seq_len`` as the token count.
        """
        if not self._is_decoding and self._prefill_seq_len == 0:
            return  # nothing to persist

        old_turbo_seq = self._turbo._seq_len  # previous turns only

        for layer_idx in range(self._n_layers):
            nkh, hd = self._layer_shapes.get(
                layer_idx, (self._default_n_kv_heads, self._default_head_dim)
            )
            parts: list = []

            # Previous turns (already in turbo)
            if old_turbo_seq > 0 and self._turbo._cache_k[layer_idx] is not None:
                k_old, v_old = self._turbo.get(layer_idx, old_turbo_seq)
                parts.append((k_old.cpu(), v_old.cpu()))

            # This turn's prefill (model native dtype on CPU)
            if self._prefill_k[layer_idx] is not None:
                pk = self._prefill_k[layer_idx]  # (1, nkh, prefill_seq, hd) CPU
                pv = self._prefill_v[layer_idx]
                # .half() here: turbo compression expects FP16 input
                pk_flat = pk.squeeze(0).permute(1, 0, 2).reshape(-1, nkh * hd).half()
                pv_flat = pv.squeeze(0).permute(1, 0, 2).reshape(-1, nkh * hd).half()
                parts.append((pk_flat, pv_flat))

            # This turn's decode tokens (model native dtype on CPU)
            if self._decode_k is not None and self._decode_k[layer_idx] is not None:
                dk = self._decode_k[layer_idx]  # (1, nkh, decode_seq, hd) CPU
                dv = self._decode_v[layer_idx]
                # .half() here: turbo compression expects FP16 input
                dk_flat = dk.squeeze(0).permute(1, 0, 2).reshape(-1, nkh * hd).half()
                dv_flat = dv.squeeze(0).permute(1, 0, 2).reshape(-1, nkh * hd).half()
                parts.append((dk_flat, dv_flat))

            if not parts:
                continue

            if len(parts) == 1:
                full_k, full_v = parts[0]
            else:
                full_k = torch.cat([p[0] for p in parts], dim=0)
                full_v = torch.cat([p[1] for p in parts], dim=0)

            self._turbo.update(layer_idx, full_k, full_v)

        # Clear raw buffers
        self._prefill_k = [None] * self._n_layers
        self._prefill_v = [None] * self._n_layers
        self._prefill_seq_len = 0
        if self._decode_k is not None:
            self._decode_k = [None] * self._n_layers
            self._decode_v = [None] * self._n_layers
        self._decode_seq_len = 0

        logger.debug(
            f"finalize_decode(): {self._turbo._seq_len} total tokens → turbo "
            f"({self._turbo.memory_usage_gb()*1000:.1f} MB compressed)"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_kv(tensor: torch.Tensor) -> torch.Tensor:
        """
        ``(1, n_heads, seq, head_dim)`` → ``(seq, n_heads * head_dim)`` in FP16.

        Kept for test compatibility.
        """
        t = tensor.squeeze(0)        # (n_heads, seq, head_dim)
        t = t.permute(1, 0, 2)       # (seq, n_heads, head_dim)
        return t.reshape(t.shape[0], -1).contiguous().half()

    def __repr__(self) -> str:
        phase = "decode" if self._is_decoding else "prefill"
        total = self._prefill_seq_len + self._decode_seq_len
        return (
            f"StreamingDynamicCache("
            f"phase={phase}, "
            f"turbo_seq={self._turbo._seq_len}, "
            f"buffered={total}, "
            f"layers={self._n_layers}, "
            f"compressed={self._turbo.memory_usage_gb()*1000:.1f} MB)"
        )
