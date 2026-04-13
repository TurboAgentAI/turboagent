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
CPU buffers are stored in pinned (page-locked) memory.  Pinning lets the DMA
engine read directly from host pages without a shadow copy, reducing H2D
bandwidth to the PCIe wire limit.

At the end of ``update(layer_N)``, Stage 3 kicks off an async transfer of
layer N+1's data on a dedicated CUDA copy stream.  This transfer overlaps with
the GPU's attention computation for layer N:

  CPU: [dequant(N+1)]→[kick H2D on copy_stream]→[return from update(N)]
  GPU: [layer_N attention ≈ 5-10ms]
  DMA: (copy_stream)→[H2D copy of N+1 ≈ 2-4ms]

When ``update(layer_N+1)`` is called, ``current_stream.wait_stream(copy_stream)``
ensures the transfer is complete before the tensor is consumed.  The wait is
typically a no-op (transfer finished during layer N attention).

Stage 4 — Block-allocated pinned buffers
-----------------------------------------
Instead of growing CPU buffers via ``torch.cat()`` on every update, Stage 4
pre-allocates in fixed-size *blocks* (default: 16 tokens) and fills in-place.

  Stage 3 (per-token decode): 64 steps × 28 layers × 2 (K+V) =
      3,584 torch.cat() + pin_memory() calls, each O(step) copy.

  Stage 4 (block-allocated):
      grow occurs every ``block_size`` decode steps:
        4 grows × 28 layers = 112 alloc+copy events (16× less frequent).
      All other steps: a single in-place slice assignment (no allocation, no copy).

Grow strategy: capacity rounded up to the next block boundary.  On grow,
all layers are extended together so subsequent layers in the same forward
pass skip the grow check.  Both prefill and decode buffers use this strategy.

A ``_prefill_write_ptr`` / ``_decode_write_ptr`` is set at layer 0 and reused
by all subsequent layers in the same forward pass, avoiding a double-advance
of the sequence-length counter.

GPU peak: ≈ one layer's full K + V tensors (ephemeral, during attention only).
CPU RAM:  block-allocated pinned buffers for the current turn
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
import math
from typing import Dict, List, Optional, Tuple

import torch

from turboagent.quant.turboquant import TurboQuantKVCache

try:
    from transformers import DynamicCache as _DynamicCacheBase
except ImportError:
    # Non-HuggingFace environment — define a no-op base so the class still works.
    class _DynamicCacheBase:  # type: ignore[no-redef]
        def __init__(self):
            pass

logger = logging.getLogger("turboagent.quant.streaming")

# Default block size (tokens) for buffer growth.
# 16 tokens = one typical PagedAttention page.
DEFAULT_BLOCK_SIZE: int = 16


class StreamingDynamicCache(_DynamicCacheBase):
    """
    DynamicCache-compatible KV cache that keeps the CURRENT TURN in
    block-allocated pinned CPU buffers and compresses to TurboQuantKVCache
    only at the end of the turn.

    Pass directly as ``past_key_values`` to any HuggingFace model that calls
    ``cache.update(k, v, layer_idx)`` internally.

    Stage 3+4 features:
      - Block-allocated pinned buffers: grow in ``block_size``-token increments,
        filled in-place — no torch.cat() per step.
      - Async prefetch of layer N+1 on a CUDA copy stream, overlapping
        H2D transfer with GPU attention on layer N.

    Lifecycle::

        cache = StreamingDynamicCache(...)
        # --- Prefill (chunked or single) ---
        for chunk in prompt_chunks:
            out = model(..., past_key_values=cache)
        cache.start_decode()              # flip to decode mode (no compression)
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
        block_size:         Token granularity for buffer growth (default 16).
    """

    def __init__(
        self,
        turbo_cache: TurboQuantKVCache,
        layer_shapes: Dict[int, Tuple[int, int]],
        layer_devices: Dict[int, torch.device],
        dtype: torch.dtype,
        default_n_kv_heads: int,
        default_head_dim: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        # Initialise the DynamicCache base so isinstance(self, DynamicCache) is
        # True.  This is required in transformers ≥ 4.46 where Gemma 4 (and
        # other models) check the cache type to decide whether to use SDPA or
        # fall back to eager attention, and whether to create a HybridCache
        # pre-allocation.  Without this, the model ignores our passed cache and
        # allocates its own — which OOMs for long contexts (O(seq²) eager attn).
        super().__init__()
        # DynamicCache.__init__ sets key_cache/value_cache/seen_tokens; our
        # update() override keeps those lists empty and handles all KV routing.
        self._turbo = turbo_cache
        self._layer_shapes = layer_shapes
        self._layer_devices = layer_devices
        self._dtype = dtype
        self._default_n_kv_heads = default_n_kv_heads
        self._default_head_dim = default_head_dim
        self._n_layers = turbo_cache.num_layers
        self._block_size = block_size

        # ── Stage 4: block-allocated pinned prefill buffers ───────────────────
        # Shape when allocated: (1, n_kv_heads, capacity, head_dim), pinned CPU.
        # Capacity grows in _block_size-token increments as chunks arrive.
        # Filled in-place via slice assignment — no torch.cat() per chunk.
        self._prefill_buf_k: List[Optional[torch.Tensor]] = [None] * self._n_layers
        self._prefill_buf_v: List[Optional[torch.Tensor]] = [None] * self._n_layers
        self._prefill_cap: int = 0        # allocated capacity (tokens)
        self._prefill_seq_len: int = 0    # tokens written so far
        # Write pointer for the current forward pass (set at layer 0,
        # reused by layers 1-N to avoid a double-advance of _prefill_seq_len).
        self._prefill_write_ptr: int = 0

        # ── Stage 4: block-allocated pinned decode buffers ────────────────────
        # Same structure; grows every block_size decode steps instead of every step.
        self._is_decoding: bool = False
        self._decode_buf_k: Optional[List[Optional[torch.Tensor]]] = None
        self._decode_buf_v: Optional[List[Optional[torch.Tensor]]] = None
        self._decode_cap: int = 0
        self._decode_seq_len: int = 0
        self._decode_write_ptr: int = 0

        # ── Stage 3: pinned memory + async prefetch ───────────────────────────
        self._use_pinned: bool = torch.cuda.is_available()
        self._copy_stream: Optional["torch.cuda.Stream"] = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        # Prefetched GPU tensors for layer N+1; populated at end of update(N).
        self._pf_turbo: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
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
        Decode mode:  [turbo_past] + [prefill_all]   + [decode_so_far] + [new_token]

        Stage 4 path (both CUDA and CPU):
          • New tokens are written directly into pre-allocated block buffers
            via in-place slice assignment — no torch.cat(), no pin_memory() call.
          • Buffer grows (all layers at once) only when capacity is exhausted,
            which happens at most once every ``block_size`` decode steps.

        Stage 3 path (CUDA only):
          • Turbo past and prefill buffer consumed from async-prefetched GPU
            tensors (transferred on copy_stream while previous layer's attention ran).
          • At the end, kicks off async prefetch for layer_idx + 1.

        Returns
        -------
        (full_key, full_value) on GPU, shape (1, n_kv_heads, total_seq, head_dim).
        """
        nkh, hd = self._layer_shapes.get(
            layer_idx, (self._default_n_kv_heads, self._default_head_dim)
        )
        dev = key_states.device
        turbo_seq = self._turbo._seq_len
        new_n = key_states.shape[2]

        # ── Layer-0 housekeeping for this forward pass ────────────────────────
        if layer_idx == 0:
            # Clear stale prefetch entries from any interrupted prior pass.
            self._pf_turbo.clear()
            self._pf_prefill.clear()

            if not self._is_decoding:
                # Record write position before advancing _prefill_seq_len,
                # so all layers in this pass write at the same offset.
                self._prefill_write_ptr = self._prefill_seq_len
                needed = self._prefill_write_ptr + new_n
                if needed > self._prefill_cap:
                    self._grow_prefill_bufs(needed)
            else:
                self._decode_write_ptr = self._decode_seq_len
                needed = self._decode_write_ptr + new_n
                if needed > self._decode_cap:
                    self._grow_decode_bufs(needed)

        parts_k: list = []
        parts_v: list = []

        # ── (1) Previous turns from turbo ────────────────────────────────────
        if turbo_seq > 0 and self._turbo._cache_k[layer_idx] is not None:
            if layer_idx in self._pf_turbo:
                # Async transfer completed on copy_stream — sync before use.
                torch.cuda.current_stream().wait_stream(self._copy_stream)
                k_t, v_t = self._pf_turbo.pop(layer_idx)
            else:
                # Synchronous fallback: layer 0 or non-CUDA machine.
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

        # ── (2) This turn's prefill buffer ────────────────────────────────────
        # Use _prefill_write_ptr (set at layer 0 BEFORE advancing _prefill_seq_len)
        # so all layers in the same forward pass see the same "past" token count.
        # During decode, _prefill_write_ptr == _prefill_seq_len (set by start_decode).
        if self._prefill_write_ptr > 0 and self._prefill_buf_k[layer_idx] is not None:
            if self._is_decoding and layer_idx in self._pf_prefill:
                torch.cuda.current_stream().wait_stream(self._copy_stream)
                pk_gpu, pv_gpu = self._pf_prefill.pop(layer_idx)
            else:
                pk = self._prefill_buf_k[layer_idx][:, :, :self._prefill_write_ptr, :]
                pv = self._prefill_buf_v[layer_idx][:, :, :self._prefill_write_ptr, :]
                pk_gpu = pk.to(device=dev, dtype=self._dtype, non_blocking=self._use_pinned)
                pv_gpu = pv.to(device=dev, dtype=self._dtype, non_blocking=self._use_pinned)
            parts_k.append(pk_gpu)
            parts_v.append(pv_gpu)

        # ── (3) This turn's decode buffer — decode mode only ──────────────────
        # Use _decode_write_ptr for the same reason: set at layer 0 before advancing.
        if self._is_decoding and self._decode_write_ptr > 0 and self._decode_buf_k[layer_idx] is not None:
            dk = self._decode_buf_k[layer_idx][:, :, :self._decode_write_ptr, :]
            dv = self._decode_buf_v[layer_idx][:, :, :self._decode_write_ptr, :]
            parts_k.append(dk.to(device=dev, dtype=self._dtype, non_blocking=self._use_pinned))
            parts_v.append(dv.to(device=dev, dtype=self._dtype, non_blocking=self._use_pinned))

        # ── (4) Current new tokens ────────────────────────────────────────────
        parts_k.append(key_states)
        parts_v.append(value_states)

        full_k = torch.cat(parts_k, dim=2) if len(parts_k) > 1 else parts_k[0]
        full_v = torch.cat(parts_v, dim=2) if len(parts_v) > 1 else parts_v[0]

        # ── Stage 4: in-place write into pre-allocated block buffer ──────────
        # The buffer is already sized (grown at layer 0 if needed).
        # key_states is on GPU; detach + cpu + dtype-cast, then slice-assign.
        new_k_cpu = key_states.detach().cpu().to(dtype=self._dtype)
        new_v_cpu = value_states.detach().cpu().to(dtype=self._dtype)

        if not self._is_decoding:
            pos = self._prefill_write_ptr
            self._prefill_buf_k[layer_idx][:, :, pos:pos + new_n, :] = new_k_cpu
            self._prefill_buf_v[layer_idx][:, :, pos:pos + new_n, :] = new_v_cpu
            if layer_idx == 0:
                self._prefill_seq_len += new_n
        else:
            pos = self._decode_write_ptr
            self._decode_buf_k[layer_idx][:, :, pos:pos + new_n, :] = new_k_cpu
            self._decode_buf_v[layer_idx][:, :, pos:pos + new_n, :] = new_v_cpu
            if layer_idx == 0:
                self._decode_seq_len += new_n

        # ── Stage 3: async prefetch for layer_idx + 1 ────────────────────────
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
        """(kv_length, kv_offset) for causal mask construction."""
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
        pass  # some transformers versions write this; ignore it

    # ------------------------------------------------------------------
    # Phase-transition methods (called by TorchEngine)
    # ------------------------------------------------------------------

    def start_decode(self) -> None:
        """
        Transition from prefill to decode mode.

        No turbo compression.  Prefill buffers stay in raw CPU memory (already
        pinned from Stage 4 block allocation) and are used as-is during decode
        attention.  ``finalize_decode()`` handles turbo compression at turn end.

        Stage 3+4: prefill buffers are already pinned (allocated that way).
        No re-pinning needed here — just init decode state and clear prefetch.
        """
        self._is_decoding = True
        self._decode_buf_k = [None] * self._n_layers
        self._decode_buf_v = [None] * self._n_layers
        self._decode_cap = 0
        self._decode_seq_len = 0
        self._decode_write_ptr = 0

        # Lock the prefill write pointer to the total prefill length.
        # During decode, _prefill_write_ptr acts as a stable "how many past prefill
        # tokens to include" sentinel — it no longer changes across forward passes.
        self._prefill_write_ptr = self._prefill_seq_len

        # Clear any residual prefetch state
        self._pf_turbo.clear()
        self._pf_prefill.clear()

        logger.debug(
            f"start_decode(): prefill={self._prefill_seq_len} tokens buffered "
            f"({self._prefill_cap} cap, {self._prefill_cap // self._block_size} blocks), "
            f"switching to decode mode"
        )

    def finalize_decode(self) -> None:
        """
        Compress the current turn's raw K/V into turbo ONCE, for persistence.

        Combines: previous-turn turbo  +  this-turn prefill buffer
                                       +  this-turn decode buffer
        → single turbo.update() call per layer.

        After this call ``turbo._seq_len`` covers all tokens from all turns.
        Raw buffers and block allocations are released.
        """
        if not self._is_decoding and self._prefill_seq_len == 0:
            return  # nothing to persist

        old_turbo_seq = self._turbo._seq_len

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
            if self._prefill_buf_k[layer_idx] is not None and self._prefill_seq_len > 0:
                pk = self._prefill_buf_k[layer_idx][:, :, :self._prefill_seq_len, :]
                pv = self._prefill_buf_v[layer_idx][:, :, :self._prefill_seq_len, :]
                # .half(): turbo compression expects FP16 input
                pk_flat = pk.squeeze(0).permute(1, 0, 2).reshape(-1, nkh * hd).half()
                pv_flat = pv.squeeze(0).permute(1, 0, 2).reshape(-1, nkh * hd).half()
                parts.append((pk_flat, pv_flat))

            # This turn's decode tokens (model native dtype on CPU)
            if (self._decode_buf_k is not None
                    and self._decode_buf_k[layer_idx] is not None
                    and self._decode_seq_len > 0):
                dk = self._decode_buf_k[layer_idx][:, :, :self._decode_seq_len, :]
                dv = self._decode_buf_v[layer_idx][:, :, :self._decode_seq_len, :]
                dk_flat = dk.squeeze(0).permute(1, 0, 2).reshape(-1, nkh * hd).half()
                dv_flat = dv.squeeze(0).permute(1, 0, 2).reshape(-1, nkh * hd).half()
                parts.append((dk_flat, dv_flat))

            if not parts:
                continue

            full_k = torch.cat([p[0] for p in parts], dim=0) if len(parts) > 1 else parts[0][0]
            full_v = torch.cat([p[1] for p in parts], dim=0) if len(parts) > 1 else parts[0][1]
            self._turbo.update(layer_idx, full_k, full_v)

        # Release block buffers and reset all state
        self._prefill_buf_k = [None] * self._n_layers
        self._prefill_buf_v = [None] * self._n_layers
        self._prefill_cap = 0
        self._prefill_seq_len = 0
        self._prefill_write_ptr = 0

        self._decode_buf_k = [None] * self._n_layers
        self._decode_buf_v = [None] * self._n_layers
        self._decode_cap = 0
        self._decode_seq_len = 0
        self._decode_write_ptr = 0

        self._pf_turbo.clear()
        self._pf_prefill.clear()

        logger.debug(
            f"finalize_decode(): {self._turbo._seq_len} total tokens → turbo "
            f"({self._turbo.memory_usage_gb()*1000:.1f} MB compressed)"
        )

    # ------------------------------------------------------------------
    # Stage 4: block-buffer growth
    # ------------------------------------------------------------------

    def _grow_prefill_bufs(self, needed: int) -> None:
        """
        Grow all layers' prefill block buffers to accommodate ``needed`` tokens.

        Called only at layer_idx == 0, so all subsequent layers in the same
        forward pass skip the grow check (capacity is already sufficient).
        Allocates with pinned memory when CUDA is available.
        """
        n_blocks = math.ceil(needed / self._block_size)
        new_cap = n_blocks * self._block_size
        used = self._prefill_seq_len  # tokens already written; copy them over

        for layer_idx in range(self._n_layers):
            nkh, hd = self._layer_shapes.get(
                layer_idx, (self._default_n_kv_heads, self._default_head_dim)
            )
            new_k = torch.empty(1, nkh, new_cap, hd, dtype=self._dtype)
            new_v = torch.empty(1, nkh, new_cap, hd, dtype=self._dtype)
            if self._use_pinned:
                new_k = new_k.pin_memory()
                new_v = new_v.pin_memory()
            # Copy existing data (none on first grow)
            if used > 0 and self._prefill_buf_k[layer_idx] is not None:
                new_k[:, :, :used, :] = self._prefill_buf_k[layer_idx][:, :, :used, :]
                new_v[:, :, :used, :] = self._prefill_buf_v[layer_idx][:, :, :used, :]
            self._prefill_buf_k[layer_idx] = new_k
            self._prefill_buf_v[layer_idx] = new_v

        self._prefill_cap = new_cap
        logger.debug(
            f"Prefill buffer grown: {used} → {new_cap} tokens "
            f"({n_blocks} blocks × {self._block_size})"
        )

    def _grow_decode_bufs(self, needed: int) -> None:
        """
        Grow all layers' decode block buffers to accommodate ``needed`` tokens.

        During decode (1 token/step), this triggers every ``block_size`` steps —
        16× less frequently than the previous per-step torch.cat() approach.
        """
        n_blocks = math.ceil(needed / self._block_size)
        new_cap = n_blocks * self._block_size
        used = self._decode_seq_len

        for layer_idx in range(self._n_layers):
            nkh, hd = self._layer_shapes.get(
                layer_idx, (self._default_n_kv_heads, self._default_head_dim)
            )
            new_k = torch.empty(1, nkh, new_cap, hd, dtype=self._dtype)
            new_v = torch.empty(1, nkh, new_cap, hd, dtype=self._dtype)
            if self._use_pinned:
                new_k = new_k.pin_memory()
                new_v = new_v.pin_memory()
            if used > 0 and self._decode_buf_k[layer_idx] is not None:
                new_k[:, :, :used, :] = self._decode_buf_k[layer_idx][:, :, :used, :]
                new_v[:, :, :used, :] = self._decode_buf_v[layer_idx][:, :, :used, :]
            self._decode_buf_k[layer_idx] = new_k
            self._decode_buf_v[layer_idx] = new_v

        self._decode_cap = new_cap
        logger.debug(
            f"Decode buffer grown: {used} → {new_cap} tokens "
            f"({n_blocks} blocks × {self._block_size})"
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
        Start an async H2D transfer of next_layer's turbo past and prefill buffer
        on the copy stream, overlapping with GPU attention on the current layer.

        Only active during decode (prefill buffer is stable; during prefill it
        is still growing and unsafe to prefetch ahead).
        """
        if not self._is_decoding:
            return

        nkh, hd = self._layer_shapes.get(
            next_layer, (self._default_n_kv_heads, self._default_head_dim)
        )

        # Async prefetch turbo past for next_layer
        if (
            turbo_seq > 0
            and next_layer not in self._pf_turbo
            and self._turbo._cache_k[next_layer] is not None
        ):
            k_flat, v_flat = self._turbo.get(next_layer, turbo_seq)
            if not k_flat.is_pinned():
                k_flat = k_flat.pin_memory()
            if not v_flat.is_pinned():
                v_flat = v_flat.pin_memory()
            with torch.cuda.stream(self._copy_stream):
                k_gpu = k_flat.to(device=dev, dtype=self._dtype, non_blocking=True)
                v_gpu = v_flat.to(device=dev, dtype=self._dtype, non_blocking=True)
            k_t = k_gpu.view(turbo_seq, nkh, hd).permute(1, 0, 2).unsqueeze(0)
            v_t = v_gpu.view(turbo_seq, nkh, hd).permute(1, 0, 2).unsqueeze(0)
            self._pf_turbo[next_layer] = (k_t, v_t)

        # Async prefetch stable prefill buffer for next_layer.
        # Use _prefill_write_ptr (= _prefill_seq_len after start_decode) so the
        # prefetch slice matches what update() will consume during decode.
        if (
            next_layer not in self._pf_prefill
            and self._prefill_write_ptr > 0
            and self._prefill_buf_k[next_layer] is not None
        ):
            pk = self._prefill_buf_k[next_layer][:, :, :self._prefill_write_ptr, :]
            pv = self._prefill_buf_v[next_layer][:, :, :self._prefill_write_ptr, :]
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
        t = tensor.squeeze(0)
        t = t.permute(1, 0, 2)
        return t.reshape(t.shape[0], -1).contiguous().half()

    def __repr__(self) -> str:
        phase = "decode" if self._is_decoding else "prefill"
        total = self._prefill_seq_len + self._decode_seq_len
        prefill_blocks = self._prefill_cap // self._block_size
        decode_blocks = self._decode_cap // self._block_size
        return (
            f"StreamingDynamicCache("
            f"phase={phase}, "
            f"turbo_seq={self._turbo._seq_len}, "
            f"buffered={total}, "
            f"prefill_blocks={prefill_blocks}, "
            f"decode_blocks={decode_blocks}, "
            f"block_size={self._block_size}, "
            f"compressed={self._turbo.memory_usage_gb()*1000:.1f} MB, "
            f"pinned={self._use_pinned})"
        )
