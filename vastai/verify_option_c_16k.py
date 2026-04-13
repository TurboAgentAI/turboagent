"""
Option C — 16k NIAH validation on Vast.ai (24 GB+ GPU).

Validates StreamingDynamicCache (cpu_streaming mode) at 16k tokens on a GPU
that is large enough to run the model but would be overwhelmed at much larger
contexts — the exact scenario Option C is designed for.

What this tests:
  1. cpu_streaming path: 16k tokens × 5 needle depths — must find needle at all depths
  2. GPU DynamicCache path (comparison, optional via NIAH_COMPARE_GPU=1)
  3. Reports VRAM usage, compression ratio, decode timing for both paths

Model: Qwen/Qwen2.5-7B-Instruct (default) — ~14 GB BF16, fast, fits on 24 GB.
       Override: NIAH_MODEL=Qwen/Qwen2.5-14B-Instruct python vastai/verify_option_c_16k.py

Hardware: 24 GB+ GPU (RTX 4090, A10G, L4, RTX 6000 Ada, etc.)
          RTX 3060 12 GB cannot run 16k context (model fills 11.9 GiB).

Usage:
    python vastai/verify_option_c_16k.py
    NIAH_COMPARE_GPU=1 python vastai/verify_option_c_16k.py  # also test GPU path
    NIAH_MODEL=Qwen/Qwen2.5-14B-Instruct python vastai/verify_option_c_16k.py
"""

import gc
import json
import os
import random
import string
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

import torch

MODEL_ID = os.environ.get("NIAH_MODEL", "Qwen/Qwen2.5-7B-Instruct")
TARGET_TOKENS = int(os.environ.get("NIAH_TARGET", "16000"))
DEPTHS = [0.0, 0.25, 0.50, 0.75, 1.0]
MAX_TOKENS_OUT = 64
COMPARE_GPU = os.environ.get("NIAH_COMPARE_GPU", "0") == "1"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_HAYSTACK = (
    "The researchers at the institute continued their careful study of complex "
    "systems and adaptive behaviors, documenting each observation in meticulous "
    "detail across thousands of pages of notes and experimental records. "
    "Every new finding challenged previous assumptions about the nature of "
    "these dynamic processes and their long-term evolution over time. "
)


def generate_needle():
    """Return (needle_text, code_to_retrieve)."""
    code = (
        "".join(random.choices(string.ascii_uppercase, k=2))
        + "-"
        + "".join(random.choices(string.digits, k=4))
    )
    phrase = (
        f"\n\n[CRITICAL INFORMATION] The secret authorization code hidden in "
        f"this document is {code}. Remember it for the verification check at "
        f"the end. [END CRITICAL INFORMATION]\n\n"
    )
    return phrase, code


def make_prompt(target_tokens, tokenizer, depth):
    """Build a prompt of ~target_tokens with needle placed at fractional depth."""
    suffix = (
        "\n\n---\n\nVERIFICATION CHECK: What is the secret authorization code "
        "mentioned in this document? Respond with ONLY the code, nothing else."
    )
    needle_phrase, code = generate_needle()

    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle_phrase, add_special_tokens=False)
    target_body = target_tokens - len(suffix_ids) - len(needle_ids)

    filler_ids = tokenizer.encode(_HAYSTACK, add_special_tokens=False)
    reps = target_body // len(filler_ids) + 2
    body_ids = (filler_ids * reps)[:target_body]
    body_text = tokenizer.decode(body_ids)

    split = int(len(body_text) * depth)
    prompt = body_text[:split] + needle_phrase + body_text[split:] + suffix

    # Trim if > 2% over budget (one correction pass)
    actual = len(tokenizer.encode(prompt))
    if actual > int(target_tokens * 1.02):
        excess = actual - target_tokens
        tail_ids = tokenizer.encode(body_text[split:], add_special_tokens=False)
        tail = tokenizer.decode(tail_ids[:-excess])
        prompt = body_text[:split] + needle_phrase + tail + suffix

    return prompt, code


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------

def vram_summary():
    if not torch.cuda.is_available():
        return "  (no CUDA)"
    lines = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        used = total - free
        lines.append(
            f"  GPU {i}: {props.name} — {used/1024**3:.2f}/{total/1024**3:.1f} GiB used"
        )
    return "\n".join(lines)


def peak_vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    return max(
        torch.cuda.max_memory_allocated(i) / 1024**3
        for i in range(torch.cuda.device_count())
    )


def reset_peak():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


# ---------------------------------------------------------------------------
# Single-depth NIAH run
# ---------------------------------------------------------------------------

def run_one(engine, kv_cache, target_tokens, depth, label):
    """Run one NIAH at a single depth. Returns result dict."""
    kv_cache.clear()
    engine._prev_input_ids = None
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

    prompt, code = make_prompt(target_tokens, engine.tokenizer, depth)
    actual_toks = len(engine.tokenizer.encode(prompt))

    reset_peak()
    t0 = time.time()
    response, metrics = engine.generate_chat(
        [{"role": "user", "content": prompt}], kv_cache
    )
    elapsed = time.time() - t0

    found = code in response
    compressed_mb = kv_cache.memory_usage_gb() * 1000
    fp16_mb = kv_cache.fp16_baseline_gb() * 1000
    ratio = fp16_mb / compressed_mb if compressed_mb > 0 else 0.0

    return {
        "label": label,
        "depth": depth,
        "target_tokens": target_tokens,
        "actual_tokens": actual_toks,
        "code": code,
        "found": found,
        "response_preview": response[:120],
        "elapsed_s": round(elapsed, 2),
        "peak_vram_gb": round(peak_vram_gb(), 3),
        "tokens_cached": metrics.get("total_tokens_cached", 0),
        "compressed_mb": round(compressed_mb, 1),
        "fp16_mb": round(fp16_mb, 1),
        "compression_ratio": round(ratio, 2),
    }


# ---------------------------------------------------------------------------
# Run a full sweep (all depths) for one engine + cache
# ---------------------------------------------------------------------------

def run_sweep(engine, kv_cache, target_tokens, label):
    results = []
    print(f"\n{'='*70}")
    print(f"PATH: {label} — {target_tokens // 1000}k tokens × {len(DEPTHS)} depths")
    print(f"{'='*70}")

    for depth in DEPTHS:
        try:
            r = run_one(engine, kv_cache, target_tokens, depth, label)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  depth={depth:.2f}: OOM — {str(e)[:100]}")
            results.append({"depth": depth, "found": False, "oom": True, "label": label})
            continue
        except Exception as e:
            print(f"  depth={depth:.2f}: ERROR — {type(e).__name__}: {str(e)[:100]}")
            results.append({"depth": depth, "found": False, "error": str(e), "label": label})
            continue

        status = "PASS" if r["found"] else "FAIL"
        print(
            f"  depth={depth:.2f}: {status} | "
            f"tokens={r['actual_tokens']:,} | "
            f"time={r['elapsed_s']:.1f}s | "
            f"{r['compression_ratio']:.2f}x | "
            f"KV={r['compressed_mb']:.0f} MB | "
            f"peak={r['peak_vram_gb']:.2f} GiB"
        )
        if not r["found"]:
            print(f"    Expected: {r['code']} | Got: {r['response_preview']!r}")
        results.append(r)

    n_pass = sum(1 for r in results if r.get("found"))
    n_total = len(results)
    print(f"\n  {label}: {n_pass}/{n_total} depths PASS")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print(f"Option C — {TARGET_TOKENS // 1000}k NIAH Validation")
    print(f"  Model:  {MODEL_ID}")
    print(f"  Target: {TARGET_TOKENS:,} tokens | Depths: {DEPTHS}")
    print(f"  Compare GPU path: {'yes' if COMPARE_GPU else 'no'}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name} ({p.total_memory / 1024**3:.1f} GiB)")
    print("=" * 70)

    from turboagent.backends.torch import TorchEngine
    from turboagent.quant.turboquant import TurboQuantKVCache

    # ── Load model (cpu_streaming) ──────────────────────────────────────────
    print(f"\nLoading model with kv_storage=cpu_streaming ...")
    t0 = time.time()
    engine = TorchEngine(
        model_id=MODEL_ID,
        kv_storage="cpu_streaming",
        max_tokens=MAX_TOKENS_OUT,
        temperature=0.0,
        prefill_chunk_size=512,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Layers: {engine._n_layers} | KV heads: {engine._n_kv_heads} | Head dim: {engine._head_dim}")
    print(vram_summary())

    flat_dim = engine._n_kv_heads * engine._head_dim
    n_layers = engine._n_layers
    max_ctx = TARGET_TOKENS + 2000
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Streaming sweep ────────────────────────────────────────────────────
    cache_stream = TurboQuantKVCache(
        bit_mode="turbo3", device=dev,
        head_dim=flat_dim, num_layers=n_layers, max_context=max_ctx,
    )
    stream_results = run_sweep(engine, cache_stream, TARGET_TOKENS, "cpu_streaming")

    # ── Optional GPU comparison ────────────────────────────────────────────
    gpu_results = []
    if COMPARE_GPU:
        print(f"\nReloading model with kv_storage=gpu ...")
        # Free the streaming engine first to reclaim VRAM
        del engine
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        t0 = time.time()
        engine_gpu = TorchEngine(
            model_id=MODEL_ID,
            kv_storage="gpu",
            max_tokens=MAX_TOKENS_OUT,
            temperature=0.0,
            prefill_chunk_size=512,
        )
        print(f"  Loaded in {time.time() - t0:.1f}s")
        print(vram_summary())

        cache_gpu = TurboQuantKVCache(
            bit_mode="turbo3", device=dev,
            head_dim=flat_dim, num_layers=n_layers, max_context=max_ctx,
        )
        gpu_results = run_sweep(engine_gpu, cache_gpu, TARGET_TOKENS, "gpu")

    # ── Summary ─────────────────────────────────────────────────────────────
    stream_pass = sum(1 for r in stream_results if r.get("found"))
    stream_total = len(stream_results)

    good = [r for r in stream_results if r.get("compression_ratio", 0) > 0]
    avg_ratio = sum(r["compression_ratio"] for r in good) / len(good) if good else 0.0
    avg_fp16 = sum(r["fp16_mb"] for r in good) / len(good) if good else 0.0
    avg_comp = sum(r["compressed_mb"] for r in good) / len(good) if good else 0.0
    avg_vram = sum(r.get("peak_vram_gb", 0) for r in good) / len(good) if good else 0.0
    avg_time = sum(r.get("elapsed_s", 0) for r in stream_results) / stream_total if stream_total else 0.0

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  cpu_streaming:    {stream_pass}/{stream_total} depths PASS")
    if COMPARE_GPU:
        gpu_pass = sum(1 for r in gpu_results if r.get("found"))
        print(f"  gpu DynamicCache: {gpu_pass}/{len(gpu_results)} depths PASS")
    print(f"  Avg compression:  {avg_ratio:.2f}x  ({avg_fp16:.0f} MB FP16 → {avg_comp:.0f} MB compressed)")
    print(f"  Avg peak VRAM:    {avg_vram:.2f} GiB  (streaming path)")
    print(f"  Avg time/depth:   {avg_time:.1f}s")

    passed = stream_pass == stream_total
    verdict = "OPTION C 16k PASS" if passed else "OPTION C 16k FAIL"
    print(f"\n{verdict}")

    # ── Save results ─────────────────────────────────────────────────────────
    out = {
        "model": MODEL_ID,
        "target_tokens": TARGET_TOKENS,
        "verdict": verdict,
        "streaming": {
            "pass": stream_pass,
            "total": stream_total,
            "avg_compression_ratio": avg_ratio,
            "avg_fp16_mb": avg_fp16,
            "avg_compressed_mb": avg_comp,
            "avg_peak_vram_gb": avg_vram,
            "results": stream_results,
        },
    }
    if COMPARE_GPU:
        out["gpu"] = {
            "pass": sum(1 for r in gpu_results if r.get("found")),
            "total": len(gpu_results),
            "results": gpu_results,
        }

    results_file = f"/workspace/option_c_{TARGET_TOKENS // 1000}k_results.json"
    try:
        with open(results_file, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Results saved: {results_file}")
    except Exception as e:
        # Not on /workspace (e.g. local run) — save next to script
        alt = os.path.join(os.path.dirname(__file__), f"option_c_{TARGET_TOKENS // 1000}k_results.json")
        try:
            with open(alt, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  Results saved: {alt}")
        except Exception:
            print(f"  (Could not save results: {e})")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
