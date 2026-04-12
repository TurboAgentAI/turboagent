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

Stage 3 — Pinned memory + async prefetch
-----------------------------------------
CPU buffers (_prefill_k/v, _decode_k/v) are stored in pinned (page-locked)
memory.  Pinning lets the DMA engine read directly from host pages without a
shadow copy, reducing H2D bandwidth to the PCIe wire limit.

Additionally, at the end of ``update(layer_N)``, Stage 3 kicks off an async
transfer of layer N+1's data on a dedicated CUDA copy stream.  This transfer
overlaps with the GPU's attention computation for layer N:

  CPU: [dequant(N+1)]→[pin]→[kick H2D on copy_stream]→[return from update(N)]
  GPU: [layer_N attention ≈ 5-10ms]
  DMA: (copy_stream)→[H2D copy of N+1 ≈ 2-4ms]

When ``update(layer_N+1)`` is called, ``current_stream.wait_stream(copy_stream)``
ensures the transfer is complete before the tensor is consumed.  The wait is
typically a no-op (transfer finished during layer N attention).

Both turbo-past KV and the stable prefill buffer are prefetched this way.
The decode buffer (1 token/step) is small and transferred inline.

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
    DynamicCache-compatible KV cache that keeps the CURRENT TURN in raw CPU
    buffers (model's native dtype) and compresses to TurboQuantKVCache only
    at the end of the turn.

    Pass directly as ``past_key_values`` to any HuggingFace model that calls
    ``cache.update(k, v, layer_idx)`` internally.

    Stage 3 features:
      - CPU buffers stored in pinned memory (faster H2D transfers via direct DMA).
      - Async prefetch of layer N+1 on a dedicated CUDA copy stream, overlapping
        with GPU attention computation on layer N.

    Lifecycle::

        cache = StreamingDynamicCache(...)
        # --- Prefill (chunked or single) ---
        for chunk in prompt_chunks:
            out = model(..., past_key_values=cache)
            cache = out.past_key_values   # same object returned by model
        cache.start_decode()              # pin prefill buffers; flip to decode mode
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

        # ── Current-turn raw CPU buffers (prefill) ────────────────────────────
        # Shape per layer: (1, n_kv_heads, seq, head_dim), model dtype, CPU.
        # Grows by chunk_size per prefill forward pass (at layer 0).
        # Pinned after start_decode() for fast decode-time H2D transfers.
        self._prefill_k: List[Optional[torch.Tensor]] = [None] * self._n_layers
        self._prefill_v: List[Optional[torch.Tensor]] = [None] * self._n_layers
        self._prefill_seq_len: int = 0

        # ── Current-turn raw CPU buffers (decode) ─────────────────────────────
        # Grows by 1 per decode step (at layer 0). Stored pinned for fast H2D.
        self._is_decoding: bool = False
        self._decode_k: Optional[List[Optional[torch.Tensor]]] = None
        self._decode_v: Optional[List[Optional[torch.Tensor]]] = None
        self._decode_seq_len: int = 0

        # ── Stage 3: pinned memory + async prefetch ───────────────────────────
        # Pinned memory requires CUDA (page-locking is a CUDA host concept).
        self._use_pinned: bool = torch.cuda.is_available()
        # Dedicated CUDA stream for H2D KV transfers (async, separate from compute).
        self._copy_stream: Optional["torch.cuda.Stream"] = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        # Prefetched GPU tensors for turbo past: {layer_idx: (k_gpu, v_gpu)}.
        # Populated by update(N) for layer N+1; consumed by update(N+1).
        self._pf_turbo: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Prefetched GPU tensors for the stable prefill buffer (decode only).
        self._pf_prefill: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

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

        Stage 3 path (CUDA only):
          • Turbo past and prefill buffer are consumed from async-prefetched GPU
            tensors (transferred on copy_stream while previous layer's attention ran).
          • On cache miss (layer 0 or first step), falls back to synchronous transfer.
          • At the end, kicks off async prefetch for layer_idx + 1.

        Returns
        -------
        (full_key, full_value) on GPU, shape (1, n_kv_heads, total_seq, head_dim).
        """
        nkh, hd = self._layer_shapes.get(
            layer_idx, (self._default_n_kv_heads, self._default_head_dim)
        )
        dev = key_states.device
        turbo_seq = self._turbo._seq_len  # previous turns; never changes during a turn

        # Safety: clear stale prefetch at the start of each new forward pass.
        # In normal operation the dicts should already be empty (all entries
        # consumed within the same forward pass that created them), but this
        # guards against edge cases like early-exit or exception recovery.
        if layer_idx == 0:
            self._pf_turbo.clear()
            self._pf_prefill.clear()

        parts_k: list = []
        parts_v: list = []

        # ── (1) Previous turns from turbo ────────────────────────────────────
        if turbo_seq > 0 and self._turbo._cache_k[layer_idx] is not None:
            if layer_idx in self._pf_turbo:
                # Async transfer was kicked off at the end of update(layer_idx-1).
                # Sync the copy stream into the compute stream before use.
                torch.cuda.current_stream().wait_stream(self._copy_stream)
                k_t, v_t = self._pf_turbo.pop(layer_idx)
            else:
                # Synchronous fallback: layer 0, or first decode step after resume.
                k_flat, v_flat = self._turbo.get(layer_idx, turbo_seq)
                if self._use_pinned:
                    if not k_flat.is_pinned():
                        k_flat = k_flat.pin_memory()
                    if not v_flat.is_pinned():
                        v_flat = v_flat.pin_memory()
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

        # ── (2) This turn's prefill buffer (raw CPU) ─────────────────────────
        if self._prefill_k[layer_idx] is not None:
            if self._is_decoding and layer_idx in self._pf_prefill:
                # Async transfer was kicked off at end of update(layer_idx-1).
                torch.cuda.current_stream().wait_stream(self._copy_stream)
                pk_gpu, pv_gpu = self._pf_prefill.pop(layer_idx)
            else:
                # Direct transfer — non_blocking=True is safe on the compute
                # stream because subsequent GPU ops in this stream serialize after it.
                pk_gpu = self._prefill_k[layer_idx].to(
                    device=dev, dtype=self._dtype, non_blocking=self._use_pinned
                )
                pv_gpu = self._prefill_v[layer_idx].to(
                    device=dev, dtype=self._dtype, non_blocking=self._use_pinned
                )
            parts_k.append(pk_gpu)
            parts_v.append(pv_gpu)

        # ── (3) This turn's decode buffer (raw CPU) — decode mode only ────────
        if self._is_decoding and self._decode_k[layer_idx] is not None:
            parts_k.append(
                self._decode_k[layer_idx].to(
                    device=dev, dtype=self._dtype, non_blocking=self._use_pinned
                )
            )
            parts_v.append(
                self._decode_v[layer_idx].to(
                    device=dev, dtype=self._dtype, non_blocking=self._use_pinned
                )
            )

        # ── (4) Current new tokens ────────────────────────────────────────────
        parts_k.append(key_states)
        parts_v.append(value_states)

        full_k = torch.cat(parts_k, dim=2) if len(parts_k) > 1 else parts_k[0]
        full_v = torch.cat(parts_v, dim=2) if len(parts_v) > 1 else parts_v[0]

        # ── Buffer the new slice on CPU (pinned for fast H2D on next step) ────
        # Storing in self._dtype (bfloat16 for GPU models) avoids the
        # bfloat16→float16→bfloat16 round-trip that caused decode divergence.
        new_k_cpu = key_states.detach().cpu().to(dtype=self._dtype)
        new_v_cpu = value_states.detach().cpu().to(dtype=self._dtype)
        if self._use_pinned:
            new_k_cpu = new_k_cpu.pin_memory()
            new_v_cpu = new_v_cpu.pin_memory()

        if not self._is_decoding:
            # Prefill mode: grow the prefill buffer
            if self._prefill_k[layer_idx] is None:
                self._prefill_k[layer_idx] = new_k_cpu
                self._prefill_v[layer_idx] = new_v_cpu
            else:
                merged_k = torch.cat([self._prefill_k[layer_idx], new_k_cpu], dim=2)
                merged_v = torch.cat([self._prefill_v[layer_idx], new_v_cpu], dim=2)
                # torch.cat may produce non-pinned output; re-pin if needed.
                if self._use_pinned and not merged_k.is_pinned():
                    merged_k = merged_k.pin_memory()
                    merged_v = merged_v.pin_memory()
                self._prefill_k[layer_idx] = merged_k
                self._prefill_v[layer_idx] = merged_v
            if layer_idx == 0:
                self._prefill_seq_len += key_states.shape[2]
        else:
            # Decode mode: grow the decode buffer
            if self._decode_k[layer_idx] is None:
                self._decode_k[layer_idx] = new_k_cpu
                self._decode_v[layer_idx] = new_v_cpu
            else:
                merged_k = torch.cat([self._decode_k[layer_idx], new_k_cpu], dim=2)
                merged_v = torch.cat([self._decode_v[layer_idx], new_v_cpu], dim=2)
                if self._use_pinned and not merged_k.is_pinned():
                    merged_k = merged_k.pin_memory()
                    merged_v = merged_v.pin_memory()
                self._decode_k[layer_idx] = merged_k
                self._decode_v[layer_idx] = merged_v
            if layer_idx == 0:
                self._decode_seq_len += 1

        # ── Stage 3: kick off async prefetch for layer_idx + 1 ───────────────
        # This runs while the GPU computes layer_idx's attention, overlapping
        # CPU dequant + H2D DMA with GPU compute.
        next_layer = layer_idx + 1
        if next_layer < self._n_layers and self._copy_stream is not None:
            self._kick_prefetch(next_layer, dev, turbo_seq)

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

        No turbo compression happens here.  The prefill buffer stays in raw CPU
        memory and will be used as-is during decode attention.  This avoids the
        quantization noise that would result from compressing before decode.

        Stage 3: pins all prefill buffers now (once) so decode-loop H2D
        transfers go at full PCIe speed via direct DMA reads.

        ``finalize_decode()`` handles turbo compression at the end of the turn.
        """
        self._is_decoding = True
        self._decode_k = [None] * self._n_layers
        self._decode_v = [None] * self._n_layers
        self._decode_seq_len = 0

        # Pin prefill buffers once for fast decode-time H2D transfers.
        # Pinning is O(buffer_size) — done once rather than every decode step.
        if self._use_pinned:
            n_pinned = 0
            for i in range(self._n_layers):
                if self._prefill_k[i] is not None and not self._prefill_k[i].is_pinned():
                    self._prefill_k[i] = self._prefill_k[i].pin_memory()
                    self._prefill_v[i] = self._prefill_v[i].pin_memory()
                    n_pinned += 1
            if n_pinned:
                logger.debug(f"start_decode(): pinned {n_pinned} prefill buffers for fast H2D")

        # Clear any residual prefetch state
        self._pf_turbo.clear()
        self._pf_prefill.clear()

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

        # Clear raw buffers and prefetch state
        self._prefill_k = [None] * self._n_layers
        self._prefill_v = [None] * self._n_layers
        self._prefill_seq_len = 0
        if self._decode_k is not None:
            self._decode_k = [None] * self._n_layers
            self._decode_v = [None] * self._n_layers
        self._decode_seq_len = 0
        self._pf_turbo.clear()
        self._pf_prefill.clear()

        logger.debug(
            f"finalize_decode(): {self._turbo._seq_len} total tokens → turbo "
            f"({self._turbo.memory_usage_gb()*1000:.1f} MB compressed)"
        )

    # ------------------------------------------------------------------
    # Stage 3: async prefetch helpers
    # ------------------------------------------------------------------

    def _kick_prefetch(
        self,
        next_layer: int,
        dev: torch.device,
        turbo_seq: int,
    ) -> None:
        """
        Start an async H2D transfer of next_layer's turbo past and (during
        decode) prefill buffer on the copy stream.

        Called at the end of ``update(next_layer - 1)`` so the DMA transfer
        overlaps with GPU attention compute on layer next_layer - 1.

        Only active during decode (prefill buffer is still growing during
        prefill, making it unsafe to prefetch ahead).
        """
        if not self._is_decoding:
            return  # prefill buffer is still growing — don't race on it

        nkh, hd = self._layer_shapes.get(
            next_layer, (self._default_n_kv_heads, self._default_head_dim)
        )

        # ── Async prefetch turbo past for next_layer ──────────────────────────
        if (
            turbo_seq > 0
            and next_layer not in self._pf_turbo
            and self._turbo._cache_k[next_layer] is not None
        ):
            k_flat, v_flat = self._turbo.get(next_layer, turbo_seq)  # CPU dequant
            if not k_flat.is_pinned():
                k_flat = k_flat.pin_memory()
            if not v_flat.is_pinned():
                v_flat = v_flat.pin_memory()
            with torch.cuda.stream(self._copy_stream):
                k_gpu = k_flat.to(device=dev, dtype=self._dtype, non_blocking=True)
                v_gpu = v_flat.to(device=dev, dtype=self._dtype, non_blocking=True)
            # .view / .permute / .unsqueeze are metadata ops (no GPU kernel).
            k_t = k_gpu.view(turbo_seq, nkh, hd).permute(1, 0, 2).unsqueeze(0)
            v_t = v_gpu.view(turbo_seq, nkh, hd).permute(1, 0, 2).unsqueeze(0)
            self._pf_turbo[next_layer] = (k_t, v_t)

        # ── Async prefetch prefill buffer for next_layer ──────────────────────
        # The prefill buffer is stable during decode, so it's safe to prefetch.
        if (
            next_layer not in self._pf_prefill
            and self._prefill_k[next_layer] is not None
        ):
            pk = self._prefill_k[next_layer]
            pv = self._prefill_v[next_layer]
            with torch.cuda.stream(self._copy_stream):
                pk_gpu = pk.to(device=dev, dtype=self._dtype, non_blocking=True)
                pv_gpu = pv.to(device=dev, dtype=self._dtype, non_blocking=True)
            self._pf_prefill[next_layer] = (pk_gpu, pv_gpu)

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
            f"compressed={self._turbo.memory_usage_gb()*1000:.1f} MB, "
            f"pinned={self._use_pinned})"
        )
