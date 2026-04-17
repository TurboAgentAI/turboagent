# TurboAgent v1.1.0 — Streaming KV

Release date: 2026-04-15

## Highlights

**Streaming KV (`cpu_streaming` mode) is fully validated.** Run large models at long contexts on consumer and multi-GPU hardware by keeping KV cache on CPU (pinned memory) and streaming layers to GPU on demand. Only one layer's KV lives on GPU at a time during attention.

- **50/50 NIAH (Needle-in-a-Haystack) depths PASS** across three model families and four context lengths
- **5.22x–5.80x KV compression** via TurboQuant, applied lossless per-turn
- **Contexts up to 96k tokens validated** on multi-GPU; up to 16k on single 24 GB GPU
- **Zero intra-turn quantization loss** — current turn kept as raw BF16; prior turns turbo-compressed

## Validation matrix

| Model | Hardware | Context lengths | Result | Compression |
|-------|----------|-----------------|--------|-------------|
| Qwen2.5-7B-Instruct | RTX 4090 (24 GB) | 16k | 5/5 PASS | 5.22x |
| Qwen2.5-32B-Instruct (BF16) | 2× RTX PRO 6000 Blackwell (192 GiB) | 4k / 16k / 32k / 65k / 96k | 25/25 PASS | 5.28x |
| Gemma 4 31B-it | 2× RTX PRO 6000 Blackwell (192 GiB) | 16k / 32k / 65k / 96k | 20/20 PASS | 5.80x |
| **Total** | | | **50/50 PASS** | |

Representative peaks (Qwen2.5-32B BF16 @ 96k): 24,709 MB FP16 KV → 4,681 MB compressed; peak VRAM 34.85 GiB.

## What's new

### `cpu_streaming` KV cache mode

New storage mode for `TorchEngine`: set `kv_storage="cpu_streaming"` to keep KV on pinned CPU memory and stream it to GPU layer-by-layer. Use `kv_storage="auto"` to switch to streaming automatically when estimated FP16 KV would exceed 60% of free VRAM.

### Block-allocated pinned buffers (Stage 4)

- `_prefill_buf_k/v` and `_decode_buf_k/v`: fixed-size block-allocated pinned buffers (default block_size=16)
- Decode grows every 16 steps instead of every step — 16× reduction in allocation frequency
- In-place writes: no `torch.cat`, no `pin_memory()` calls during steady-state decode
- Write-pointer pattern locked at layer-0 per forward pass to prevent double-counting across layers

### Async layer-N+1 prefetch (Stage 3)

- Dedicated `_copy_stream` for H2D transfers overlaps with GPU attention compute
- End of `update(layer_N)` kicks dequant + async copy for layer N+1
- Prefetches both turbo-past and stable prefill buffer
- Single-GPU expected decode speedup: ~252 ms → ~144 ms/step at 12k context

### Multi-GPU support

- Per-device pinned-to-GPU transfers honor each layer's actual `device_map` placement
- Disables `_copy_stream` when model spans multiple GPUs (avoids cross-device stream issues)
- `empty_cache()` moved to post-prefill to avoid racing with unsequenced H2D DMA
- `non_blocking=False` for prefill/decode H2D when multi-GPU (without copy_stream, async gives no benefit but introduces races)

### Phase contract (zero compression mid-turn)

| Phase | Behavior | Turbo writes |
|-------|----------|--------------|
| Prefill (any # chunks) | buffer raw K/V on CPU in model dtype | 0 |
| `start_decode()` | flips `_is_decoding=True`, inits decode buffers | 0 |
| Decode (any # steps) | buffer new tokens on CPU | 0 |
| `finalize_decode()` | compress `turbo_past + prefill + decode` once per layer | 1 |

Current turn achieves `cos_sim = 1.0` vs. the GPU `DynamicCache` path at every decode step.

## Bug fixes

- **BF16 round-trip loss** — buffers now stored as `self._dtype` (BF16 for GPU models) instead of `.half()`; eliminates rounding divergence at close-margin greedy decisions
- **Empty cache guard** — `kv_cache._seq_len==0` is now always treated as a fresh turn, regardless of `_prev_input_ids`
- **transformers 5.x compat** — pre-patch `attn_implementation` via `AutoConfig` so it propagates to `text_config`; handle `get_mask_sizes` receiving either `int` or `Tensor`
- **Multi-GPU prefetch target** — `_kick_prefetch` now looks up next layer's device from `layer_devices` and skips at device boundaries
- **SDPA path** for Gemma 4 31B multi-GPU cpu_streaming

## Breaking changes

None. `kv_storage` defaults to `"gpu"` (classic v1.0 behavior).

## Tests

**60/60 streaming-KV unit tests passing** (35 base + 11 Stage 3 async/pinned + 14 Stage 4 block-buffer). Total suite: 148 passed, 8 integration tests deselected (require live model download).

## Acknowledgments

Multi-GPU validation performed on Vast.ai (2× RTX PRO 6000 Blackwell, 192 GiB).

## What's next (v1.2)

- Merge with Compton/BitTorch (4-bit weight quantization + desktop GUI)
- FlexTensor integration (automatic weight GPU↔CPU offloading)
- Target: 70B on RTX 4090 with 200k+ context
