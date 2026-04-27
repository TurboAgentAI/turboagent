"""
TurboAgent Needle-in-a-Haystack (NIAH) benchmark.

Reproduces the v1.1.0 retrieval-fidelity numbers reported on turboagent.to and
in RELEASE_NOTES.md. RULER-style protocol: N context lengths × 5 needle depths,
random alphanumeric needle, repeated natural-language haystack, exact-string
match grading.

Two canonical configurations:

    # NF4 + Streaming KV — single 24 GB GPU, headline v1.1.0 result
    python examples/niah/run_niah.py \\
        --model Qwen/Qwen2.5-32B-Instruct \\
        --quantize-weights nf4 \\
        --kv-storage cpu_streaming \\
        --lengths 16000,32000,65000

    # BF16 + Streaming KV — multi-GPU workstation
    python examples/niah/run_niah.py \\
        --model Qwen/Qwen2.5-32B-Instruct \\
        --kv-storage cpu_streaming \\
        --lengths 16000,32000,65000,96000

Output: JSON with per-(length, depth) result rows including peak VRAM,
compression ratio, and decode timing. Stdout shows a pass/fail line per depth.
"""

import argparse
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


_HAYSTACK = (
    "The most important thing for a startup founder to understand is that the "
    "default outcome is failure. Most startups fail. The ones that succeed do "
    "so because they built something people want and got it in front of those "
    "people faster than the alternatives. Everything else is secondary. "
    "The hardest part of building a startup is not the technology; it is "
    "figuring out what to build. Most founders spend too much time on the "
    "wrong problems and optimize prematurely. Funding is overrated. The amount "
    "of money you raise has very little correlation with how successful your "
    "company will become. What matters is the cost structure you build and the "
    "speed at which you can iterate. "
)


def parse_args():
    p = argparse.ArgumentParser(
        description="TurboAgent NIAH benchmark — reproduces v1.1.0 retrieval results.",
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct",
                   help="HuggingFace model ID")
    p.add_argument("--quantize-weights", choices=["nf4", "int4", "none"], default="none",
                   help="4-bit weight quantization via bitsandbytes (default: none = BF16)")
    p.add_argument("--kv-storage", choices=["gpu", "cpu_streaming"], default="cpu_streaming",
                   help="KV cache placement (default: cpu_streaming)")
    p.add_argument("--lengths", default="16000,32000,65000",
                   help="Comma-separated context lengths in tokens")
    p.add_argument("--depths", default="0.0,0.25,0.5,0.75,1.0",
                   help="Comma-separated needle depths (fractions of context)")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--prefill-chunk-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="niah_results.json",
                   help="Path to write JSON results")
    return p.parse_args()


def generate_needle(rng):
    code = (
        "".join(rng.choices(string.ascii_uppercase, k=2))
        + "-"
        + "".join(rng.choices(string.digits, k=4))
    )
    phrase = (
        f"\n\n[CRITICAL INFORMATION] The secret authorization code hidden in "
        f"this document is {code}. Remember it for the verification check at "
        f"the end. [END CRITICAL INFORMATION]\n\n"
    )
    return phrase, code


def make_prompt(target_tokens, tokenizer, depth, rng):
    suffix = (
        "\n\n---\n\nVERIFICATION CHECK: What is the secret authorization code "
        "mentioned in this document? Respond with ONLY the code, nothing else."
    )
    needle_phrase, code = generate_needle(rng)

    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle_phrase, add_special_tokens=False)
    target_body = target_tokens - len(suffix_ids) - len(needle_ids)

    filler_ids = tokenizer.encode(_HAYSTACK, add_special_tokens=False)
    reps = target_body // len(filler_ids) + 2
    body_ids = (filler_ids * reps)[:target_body]
    body_text = tokenizer.decode(body_ids)

    split = int(len(body_text) * depth)
    prompt = body_text[:split] + needle_phrase + body_text[split:] + suffix

    actual = len(tokenizer.encode(prompt))
    if actual > int(target_tokens * 1.02):
        excess = actual - target_tokens
        tail_ids = tokenizer.encode(body_text[split:], add_special_tokens=False)
        tail = tokenizer.decode(tail_ids[:-excess])
        prompt = body_text[:split] + needle_phrase + tail + suffix

    return prompt, code


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


def run_one(engine, kv_cache, target_tokens, depth, rng):
    kv_cache.clear()
    engine._prev_input_ids = None
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

    prompt, code = make_prompt(target_tokens, engine.tokenizer, depth, rng)
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
    n_new = metrics.get("tokens_generated", 0) or 0
    decode_tps = (n_new / elapsed) if elapsed > 0 and n_new > 0 else None

    return {
        "depth": depth,
        "target_tokens": target_tokens,
        "actual_tokens": actual_toks,
        "code": code,
        "found": found,
        "response_preview": response[:120],
        "elapsed_s": round(elapsed, 2),
        "decode_tok_per_s": round(decode_tps, 1) if decode_tps else None,
        "peak_vram_gb": round(peak_vram_gb(), 3),
        "tokens_cached": metrics.get("total_tokens_cached", 0),
        "compressed_mb": round(compressed_mb, 1),
        "fp16_mb": round(fp16_mb, 1),
        "compression_ratio": round(ratio, 2),
    }


def run_sweep(engine, kv_cache, target_tokens, depths, rng):
    print(f"\n{'='*70}")
    print(f"NIAH @ {target_tokens // 1000}k tokens × {len(depths)} depths")
    print(f"{'='*70}")

    results = []
    for depth in depths:
        try:
            r = run_one(engine, kv_cache, target_tokens, depth, rng)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  depth={depth:.2f}: OOM — {str(e)[:80]}")
            results.append({"depth": depth, "found": False, "oom": True})
            continue
        except Exception as e:
            print(f"  depth={depth:.2f}: ERROR — {type(e).__name__}: {str(e)[:80]}")
            results.append({"depth": depth, "found": False, "error": str(e)})
            continue

        status = "PASS" if r["found"] else "FAIL"
        decode_str = f"{r['decode_tok_per_s']} tok/s" if r["decode_tok_per_s"] else "—"
        print(
            f"  depth={depth:.2f}: {status} | "
            f"tokens={r['actual_tokens']:,} | "
            f"time={r['elapsed_s']:.1f}s | "
            f"decode={decode_str} | "
            f"{r['compression_ratio']:.2f}x | "
            f"KV={r['compressed_mb']:.0f} MB | "
            f"peak={r['peak_vram_gb']:.2f} GiB"
        )
        if not r["found"]:
            print(f"    Expected: {r['code']} | Got: {r['response_preview']!r}")
        results.append(r)

    n_pass = sum(1 for r in results if r.get("found"))
    print(f"\n  Aggregate: {n_pass}/{len(results)} depths PASS")
    return results


def main():
    args = parse_args()

    quantize_weights = None if args.quantize_weights == "none" else args.quantize_weights
    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [float(x) for x in args.depths.split(",")]

    print("=" * 70)
    print("TurboAgent NIAH benchmark")
    print(f"  Model:    {args.model}")
    print(f"  Weights:  {quantize_weights or 'BF16 (no weight quantization)'}")
    print(f"  KV:       {args.kv_storage} + TurboQuant turbo3")
    print(f"  Lengths:  {lengths}")
    print(f"  Depths:   {depths}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name} ({p.total_memory / 1024**3:.1f} GiB)")
    print("=" * 70)

    from turboagent.backends.torch import TorchEngine
    from turboagent.quant.turboquant import TurboQuantKVCache

    print(f"\nLoading {args.model}...")
    t0 = time.time()
    engine = TorchEngine(
        model_id=args.model,
        kv_storage=args.kv_storage,
        quantize_weights=quantize_weights,
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s ({load_time/60:.1f} min)")
    print(f"  Layers: {engine._n_layers} | KV heads: {engine._n_kv_heads} | Head dim: {engine._head_dim}")
    print(vram_summary())

    flat_dim = engine._n_kv_heads * engine._head_dim
    n_layers = engine._n_layers
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)

    all_results = {}
    total_pass = 0
    total_tests = 0

    for length in lengths:
        max_ctx = length + 2000
        cache = TurboQuantKVCache(
            bit_mode="turbo3", device=dev,
            head_dim=flat_dim, num_layers=n_layers, max_context=max_ctx,
        )
        results = run_sweep(engine, cache, length, depths, rng)
        n_pass = sum(1 for r in results if r.get("found"))
        total_pass += n_pass
        total_tests += len(results)
        all_results[f"{length // 1000}k"] = {
            "pass": n_pass, "total": len(results), "results": results,
        }
        del cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if all(r.get("oom") for r in results):
            print(f"\n  All depths OOM at {length}; skipping remaining lengths.")
            break

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Total: {total_pass}/{total_tests} depths PASS")
    for length_key, data in all_results.items():
        good = [r for r in data["results"] if r.get("compression_ratio", 0) > 0]
        avg_ratio = sum(r["compression_ratio"] for r in good) / len(good) if good else 0
        avg_vram = sum(r["peak_vram_gb"] for r in good) / len(good) if good else 0
        avg_time = sum(r["elapsed_s"] for r in data["results"]) / len(data["results"]) if data["results"] else 0
        decode_vals = [r.get("decode_tok_per_s") for r in good if r.get("decode_tok_per_s")]
        avg_decode = sum(decode_vals) / len(decode_vals) if decode_vals else None
        decode_str = f"{avg_decode:.1f} tok/s" if avg_decode else "—"
        print(
            f"  {length_key}: {data['pass']}/{data['total']} PASS | "
            f"{avg_ratio:.2f}x compression | "
            f"peak {avg_vram:.2f} GiB | "
            f"decode {decode_str} | "
            f"avg {avg_time:.1f}s/depth"
        )

    out = {
        "model": args.model,
        "quantize_weights": quantize_weights,
        "kv_storage": args.kv_storage,
        "lengths": lengths,
        "depths": depths,
        "seed": args.seed,
        "load_time_s": round(load_time, 1),
        "total_pass": total_pass,
        "total_tests": total_tests,
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results: {args.output}")

    if total_pass != total_tests:
        sys.exit(1)


if __name__ == "__main__":
    main()
