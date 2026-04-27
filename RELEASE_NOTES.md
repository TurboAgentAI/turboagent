# TurboAgent v1.1.0 — NF4 + Streaming KV

Release date: 2026-04-24 (PyPI publish)

## Headline

**Qwen2.5-32B-Instruct runs on a single 24 GB GPU at 65k context, 100% retrieval accuracy at every needle depth tested.**

NF4 4-bit weight quantization (via bitsandbytes) cuts the 32B model's weights from ~64 GB BF16 to ~18 GB. Streaming KV keeps the compressed KV cache on CPU pinned memory and streams only the slice needed for each attention computation. The combination fits a 32B model on a single RTX 4090 — the first time TurboAgent has done this.

## v1.1.0 validation matrix

| Model | Hardware | VRAM | Lengths | Result | Compression |
|---|---|---|---|---|---|
| **Qwen2.5-32B (NF4)** | Single RTX 4090 | 24 GB | 16k / 32k / 65k | **15/15 PASS** | 5.28x |
| Qwen2.5-32B (BF16) | 2× RTX PRO 6000 | 192 GB | 4k / 16k / 32k / 65k / 96k | 25/25 PASS | 5.28x |
| Gemma 4 31B (BF16) | 2× RTX PRO 6000 | 192 GB | 16k / 32k / 65k / 96k | 20/20 PASS | 5.80x |
| Qwen2.5-7B (BF16) | Single RTX 4090 | 24 GB | 16k | 5/5 PASS | 5.22x |
| **Total** | | | | **65/65 PASS** | |

NIAH (RULER-style, 5 needle depths × N context lengths) tests retrieval fidelity. Downstream task validation (LongBench, RULER full suite, agent-loop coherence) is planned for v1.2.

### Single-RTX-4090 NF4 details (the headline result)

| Context | Result | Decode | Compressed KV | Peak VRAM | Headroom |
|---|---|---|---|---|---|
| 16,000 tokens | 5/5 PASS | 111 tok/s | 782 MB | 18.80 GiB | 4.7 GiB |
| 32,000 tokens | 5/5 PASS | 91 tok/s | 1,562 MB | 19.67 GiB | 3.8 GiB |
| 65,000 tokens | 5/5 PASS | 59 tok/s | 3,170 MB | 21.47 GiB | 2.0 GiB |
| 96,000 tokens | OOM | — | — | — | 827 MB free |

96k OOMs at the hardware ceiling (1.44 GiB allocation needed, 827 MB free). 65k is the reliable maximum for 32B NF4 on a single 24 GB GPU. Decode throughput drops from 111 → 59 tok/s between 16k and 65k as per-decode-step PCIe traffic grows linearly with sequence length.

## What's new

### NF4 weight quantization (`quantize_weights="nf4"`)

`TorchEngine` accepts `quantize_weights="nf4"` (or `"int4"`) and constructs a `BitsAndBytesConfig` with double-quant + BF16 compute dtype, injected via `quantization_config` into `from_pretrained()`. The hardware detector recommends NF4 automatically for 30B+ models on 24 GB / 16 GB / 8–12 GB GPU tiers. Install via the new `bnb` extra: `pip install "turboagent-ai[torch,bnb,native]==1.1.0"`.

### `cpu_streaming` KV cache mode

New storage mode for `TorchEngine`: set `kv_storage="cpu_streaming"` to keep KV on pinned CPU memory and stream it to GPU layer-by-layer. Use `kv_storage="auto"` to switch to streaming automatically when estimated FP16 KV would exceed 60% of free VRAM.

### Block-allocated pinned buffers (Stage 4)

- `_prefill_buf_k/v` and `_decode_buf_k/v`: fixed-size block-allocated pinned buffers (default `block_size=16`)
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
|---|---|---|
| Prefill (any # chunks) | buffer raw K/V on CPU in model dtype | 0 |
| `start_decode()` | flips `_is_decoding=True`, inits decode buffers | 0 |
| Decode (any # steps) | buffer new tokens on CPU | 0 |
| `finalize_decode()` | compress `turbo_past + prefill + decode` once per layer | 1 |

Current turn achieves `cos_sim = 1.0` vs. the GPU `DynamicCache` path at every decode step.

## Reproducibility

The exact NIAH harness is included in this release at [`examples/niah/`](examples/niah/). Two canonical recipes:

```bash
# NF4 + Streaming KV — single 24 GB GPU, headline result
python examples/niah/run_niah.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --quantize-weights nf4 \
    --kv-storage cpu_streaming \
    --lengths 16000,32000,65000

# BF16 + Streaming KV — multi-GPU workstation
python examples/niah/run_niah.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --kv-storage cpu_streaming \
    --lengths 16000,32000,65000,96000
```

See [`examples/niah/README.md`](examples/niah/README.md) for the full protocol, hardware requirements, and a "what this benchmark does and doesn't validate" section.

## Bug fixes

- **BF16 round-trip loss** — buffers now stored as `self._dtype` (BF16 for GPU models) instead of `.half()`; eliminates rounding divergence at close-margin greedy decisions
- **Empty cache guard** — `kv_cache._seq_len==0` is now always treated as a fresh turn, regardless of `_prev_input_ids`
- **transformers 5.x compat** — pre-patch `attn_implementation` via `AutoConfig` so it propagates to `text_config`; handle `get_mask_sizes` receiving either `int` or `Tensor`
- **Multi-GPU prefetch target** — `_kick_prefetch` now looks up next layer's device from `layer_devices` and skips at device boundaries
- **SDPA path** for Gemma 4 31B multi-GPU cpu_streaming
- **NF4 + single-GPU max_memory** — when `quantize_weights` is set, the single-GPU `max_memory` budget no longer includes a CPU spill entry, preventing accelerate from dispatching modules to CPU/disk where bitsandbytes refuses to load them

## Breaking changes

None. `kv_storage` defaults to `"gpu"` (classic v1.0 behavior). `quantize_weights` defaults to `None` (BF16).

## Tests

**60/60 streaming-KV unit tests passing** (35 base + 11 Stage 3 async/pinned + 14 Stage 4 block-buffer). NF4 weight-quant test suite: 9/9 passing. Total suite: 157 passed, 8 integration tests deselected (require live model download).

## Acknowledgments

Multi-GPU validation performed on Vast.ai (2× RTX PRO 6000 Blackwell, 192 GiB). NF4 single-4090 validation performed on Vast.ai (RTX 4090 24 GB).

## What's next (v1.2)

- BitTorch fine-tuning integration (QLoRA / LoRA / DPO on the same NF4-quantized base)
- FlexTensor automatic GPU↔CPU weight offloading
- Llama-3.1-70B target: 200k+ context on a single 4090, 3–8 tok/s decode
- Tauri/React desktop GUI
- Expanded evaluation: LongBench, RULER full suite, agent-loop coherence, head-to-head throughput vs. vLLM / llama.cpp / ExLlama2
