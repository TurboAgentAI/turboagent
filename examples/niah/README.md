# NIAH benchmark — reproducing the v1.1.0 retrieval results

This directory contains the needle-in-a-haystack (NIAH) harness used to validate
TurboAgent v1.1.0. It produces the per-(context-length, depth) numbers reported
on [turboagent.to](https://turboagent.to) and in `RELEASE_NOTES.md`.

## Protocol

RULER-style needle-in-a-haystack:

- **Context lengths:** configurable; the v1.1.0 release covers 16k, 32k, 65k, 96k tokens
- **Needle depths:** 0%, 25%, 50%, 75%, 100% of the haystack
- **Needle:** randomly generated alphanumeric authorization code (e.g. `XK-9234`),
  embedded in a `[CRITICAL INFORMATION] ... [END CRITICAL INFORMATION]` bracketed phrase
- **Haystack:** a short natural-language paragraph repeated to the target length
- **Decode:** `temperature=0.0` (greedy), `max_new_tokens=64`
- **Grading:** exact-substring match of the needle code in the model's response

The script reports per-depth pass/fail, peak VRAM, KV cache compression ratio,
and decode throughput in tok/s.

## Reproducing the v1.1.0 headline result

**Qwen2.5-32B at 65k context on a single 24 GB GPU** (NF4 weights + Streaming KV).
This is the headline v1.1.0 number on the website.

```bash
pip install "turboagent-ai[torch,bnb,native]==1.1.0"

python examples/niah/run_niah.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --quantize-weights nf4 \
    --kv-storage cpu_streaming \
    --lengths 16000,32000,65000 \
    --output niah_nf4_qwen32b.json
```

Expected output on a single RTX 4090 (24 GB):

| Context | Result | Decode | Peak VRAM | Compression |
|---|---|---|---|---|
| 16k | 5/5 PASS | ~111 tok/s | 18.80 GiB | 5.28x |
| 32k | 5/5 PASS | ~91 tok/s  | 19.67 GiB | 5.28x |
| 65k | 5/5 PASS | ~59 tok/s  | 21.47 GiB | 5.28x |

96k OOMs at the hardware ceiling on 24 GB (1.44 GiB allocation needed, 827 MB free).

## Reproducing the BF16 multi-GPU sweep

**Qwen2.5-32B / Gemma 4 31B at 96k context** on a workstation with 2× RTX PRO 6000
Blackwell (192 GiB total).

```bash
# Qwen2.5-32B BF16, full sweep
python examples/niah/run_niah.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --kv-storage cpu_streaming \
    --lengths 4000,16000,32000,65000,96000 \
    --output niah_bf16_qwen32b.json

# Gemma 4 31B BF16, full sweep
python examples/niah/run_niah.py \
    --model google/gemma-4-31B-it \
    --kv-storage cpu_streaming \
    --lengths 16000,32000,65000,96000 \
    --output niah_bf16_gemma4.json
```

Expected: 25/25 PASS for Qwen2.5-32B, 20/20 PASS for Gemma 4 31B.

## CLI reference

```
--model               HuggingFace model ID (default: Qwen/Qwen2.5-32B-Instruct)
--quantize-weights    nf4 | int4 | none  (default: none = BF16)
--kv-storage          gpu | cpu_streaming  (default: cpu_streaming)
--lengths             Comma-separated context lengths (default: 16000,32000,65000)
--depths              Comma-separated needle depths (default: 0.0,0.25,0.5,0.75,1.0)
--max-new-tokens      Decode budget (default: 64)
--prefill-chunk-size  Prefill chunk size in tokens (default: 512)
--seed                RNG seed for needle generation (default: 0)
--output              JSON output path (default: niah_results.json)
```

## What this benchmark does and doesn't validate

NIAH measures **retrieval fidelity** — can the model find an exact-string needle
buried in a long context? It does **not** measure:

- Multi-hop reasoning across long context
- Chain-of-thought coherence
- JSON-schema adherence under compression
- Multi-turn drift over many compress/decompress cycles
- Throughput head-to-head with vLLM, llama.cpp, or other backends

LongBench / RULER / agentic downstream evaluations are scheduled for v1.2. See
the limitations section on [turboagent.to](https://turboagent.to) for the full list.

## Hardware requirements

| Configuration | Min VRAM | Tested on |
|---|---|---|
| Qwen2.5-7B BF16, 16k | 24 GB | Single RTX 4090 |
| **Qwen2.5-32B NF4, 16k–65k** | **24 GB** | **Single RTX 4090** |
| Qwen2.5-32B BF16, 4k–96k | ~80 GB | 2× RTX PRO 6000 Blackwell (192 GiB) |
| Gemma 4 31B BF16, 16k–96k | ~80 GB | 2× RTX PRO 6000 Blackwell (192 GiB) |

The NF4 path requires `bitsandbytes` (install via the `bnb` extra). Multi-GPU
runs use `device_map="auto"` with TurboAgent's built-in `max_memory` budget.
