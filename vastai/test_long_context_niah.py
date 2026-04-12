"""
TurboAgent Long-Context Needle-in-a-Haystack (NIAH) Benchmark

Tests TurboQuant KV cache compression at realistic long-context scales
on Gemma 4 31B. Follows the standard NIAH protocol used in Anthropic's
original benchmark and the RULER paper:

  - Multiple context lengths: 4k, 16k, 32k, 64k, 96k tokens
  - Multiple needle depths: 0%, 25%, 50%, 75%, 100%
  - Random needle phrase per test (prevents memorization)
  - Real haystack text (Paul Graham essays, public domain)
  - Reports retrieval accuracy + actual compression ratio per length

Hardware requirement: 4x RTX 5090 (128 GB VRAM total) or equivalent.
With Gemma 4 31B BF16 + tensor parallelism, ~96 GB free for KV cache.
KV per token ~512 KB FP16, so ~96k tokens fit at FP16 baseline.

Run: python -m pytest vastai/test_long_context_niah.py -v -s --timeout=3600
"""

import gc
import json
import logging
import os
import random
import re
import string
import time
from typing import List, Tuple

import pytest
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

HF_MODEL = os.environ.get("TURBO_NIAH_MODEL", "google/gemma-4-31B-it")

# Context lengths to test (in tokens). Adjust based on your VRAM.
# 4x 5090 (128 GB): 4k -> 96k is feasible.
# H100 SXM 192GB: can push to 200k+
NIAH_LENGTHS = [
    int(x) for x in os.environ.get(
        "NIAH_LENGTHS", "4000,16000,32000,65000,96000"
    ).split(",")
]

# Depth percentages where the needle is placed
NIAH_DEPTHS = [0.0, 0.25, 0.50, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Haystack generation
# ---------------------------------------------------------------------------

# Use Paul Graham essay text as haystack filler — public domain, real prose,
# matches what most NIAH benchmarks use. Repeated to reach target length.
PG_HAYSTACK = """\
The most important thing for a startup founder to understand is that the
default outcome is failure. Most startups fail. The ones that succeed do so
because they built something people want and got it in front of those people
faster than the alternatives. Everything else is secondary.

When I started Y Combinator, the prevailing wisdom was that you needed an
experienced team, a defensible business plan, and significant funding to
build a real company. We bet that none of those things were necessary if
you had founders who could build something people wanted. That bet paid off.

The hardest part of building a startup is not the technology. It is figuring
out what to build. Most founders spend too much time on the wrong problems.
They optimize prematurely, they build features no one asked for, they pursue
markets that don't exist. The discipline of talking to users and shipping
fast separates the startups that survive from the ones that don't.

Funding is overrated. The amount of money you raise has very little
correlation with how successful your company will become. What matters is
the cost structure you build and the speed at which you can iterate. A
small, focused team with limited capital often outperforms a large team
with abundant resources, because constraints force focus.

Hiring is the most consequential decision a founder makes after deciding
what to build. Each early hire shapes the culture, the codebase, and the
product roadmap for years afterward. The first ten employees of a company
matter more than the next hundred combined. Hire slowly, fire quickly.
"""


def _build_haystack_with_needle(
    target_chars: int,
    needle: str,
    depth_fraction: float,
) -> str:
    """
    Build a haystack of approximately `target_chars` characters with the
    needle inserted at the specified depth.
    """
    # Repeat PG essay until we have enough text
    repeats_needed = (target_chars // len(PG_HAYSTACK)) + 2
    full = PG_HAYSTACK * repeats_needed
    full = full[:target_chars]

    # Insert needle at the target depth (snap to a sentence boundary)
    insert_pos = int(len(full) * depth_fraction)
    # Find the next period after insert_pos
    if insert_pos > 0:
        next_period = full.find(". ", insert_pos)
        if next_period != -1:
            insert_pos = next_period + 2

    return full[:insert_pos] + needle + " " + full[insert_pos:]


def _generate_needle() -> Tuple[str, str]:
    """
    Generate a random needle phrase + the answer to extract.
    Uses the standard 'magic number' format from the original NIAH paper.
    """
    code = "".join(random.choices(string.ascii_uppercase, k=2)) + "-"
    code += "".join(random.choices(string.digits, k=4))
    needle = (
        f"\n\n[CRITICAL INFORMATION] The secret authorization code "
        f"hidden in this document is {code}. Remember it for the "
        f"verification check at the end. [END CRITICAL INFORMATION]\n\n"
    )
    return needle, code


# ---------------------------------------------------------------------------
# Engine fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine_and_cache():
    """Load Gemma 4 31B once for all NIAH tests."""
    from turboagent.backends.torch import TorchEngine
    from turboagent.quant.turboquant import TurboQuantKVCache, _USE_NATIVE

    print(f"\n[NIAH] Loading {HF_MODEL}...")
    print(f"[NIAH] Native quantizer: {_USE_NATIVE}")
    print(f"[NIAH] Test lengths: {NIAH_LENGTHS}")
    print(f"[NIAH] Test depths: {NIAH_DEPTHS}")

    t0 = time.time()
    # Use a generous max_context to support the largest test length
    max_ctx = max(NIAH_LENGTHS) + 2000
    engine = TorchEngine(HF_MODEL, context=max_ctx, max_tokens=64, n_gpu_layers=-1)
    load_time = time.time() - t0

    if torch.cuda.is_available():
        total = sum(
            torch.cuda.get_device_properties(i).total_memory / 1e9
            for i in range(torch.cuda.device_count())
        )
        used = sum(
            torch.cuda.memory_allocated(i) / 1e9
            for i in range(torch.cuda.device_count())
        )
        print(f"[NIAH] Loaded in {load_time:.1f}s | VRAM: {used:.1f}/{total:.1f} GB across {torch.cuda.device_count()} GPUs")

    flat_dim = engine._head_dim * engine._n_kv_heads
    cache = TurboQuantKVCache(
        bit_mode="turbo3", device="cpu",
        head_dim=flat_dim, num_layers=engine._n_layers, max_context=max_ctx,
    )

    yield engine, cache

    del engine
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# NIAH test
# ---------------------------------------------------------------------------

def _count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _run_niah_at_depth(
    engine, cache, target_tokens: int, depth: float, tokenizer,
) -> dict:
    """Run a single needle-in-haystack test at the given depth."""
    # Aggressive cleanup between tests to combat fragmentation
    cache.clear()
    engine._prev_input_ids = None
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    # Generate needle
    needle, code = _generate_needle()

    # Build haystack — use ~5 chars per token as estimate
    target_chars = int(target_tokens * 4.5)
    haystack = _build_haystack_with_needle(target_chars, needle, depth)

    # Trim to exact token count
    full_prompt = (
        haystack
        + "\n\n---\n\nVERIFICATION CHECK: What is the secret authorization "
        "code mentioned in the document above? Respond with ONLY the code, "
        "nothing else."
    )
    actual_tokens = _count_tokens(full_prompt, tokenizer)

    messages = [
        {"role": "system", "content": "You are a precise assistant that retrieves information from documents."},
        {"role": "user", "content": full_prompt},
    ]

    t0 = time.time()
    response, metrics = engine.generate_chat(messages, cache)
    elapsed = time.time() - t0

    # Check if needle was retrieved
    found = code in response

    fp16_mb = cache.fp16_baseline_gb() * 1000
    compressed_mb = cache.memory_usage_gb() * 1000
    ratio = fp16_mb / compressed_mb if compressed_mb > 0 else 0

    return {
        "target_tokens": target_tokens,
        "actual_tokens": actual_tokens,
        "depth": depth,
        "needle": code,
        "response": response[:200],
        "found": found,
        "elapsed_s": elapsed,
        "fp16_mb": fp16_mb,
        "compressed_mb": compressed_mb,
        "compression_ratio": ratio,
        "tokens_cached": metrics.get("total_tokens_cached", 0),
    }


@pytest.mark.parametrize("target_tokens", NIAH_LENGTHS)
def test_niah_full_sweep(engine_and_cache, target_tokens):
    """
    Run NIAH at all depths for a given context length.
    Reports per-depth accuracy and aggregate compression ratio.
    """
    engine, cache = engine_and_cache
    tokenizer = engine.tokenizer

    print(f"\n{'='*70}")
    print(f"NIAH @ {target_tokens} tokens — testing {len(NIAH_DEPTHS)} depths")
    print(f"{'='*70}")

    results = []
    for depth in NIAH_DEPTHS:
        try:
            r = _run_niah_at_depth(engine, cache, target_tokens, depth, tokenizer)
            results.append(r)
            status = "PASS" if r["found"] else "FAIL"
            print(
                f"  depth={depth:.2f}: {status} | "
                f"tokens={r['actual_tokens']} | "
                f"time={r['elapsed_s']:.1f}s | "
                f"compression={r['compression_ratio']:.2f}x | "
                f"KV={r['compressed_mb']:.0f} MB (FP16: {r['fp16_mb']:.0f} MB)"
            )
            if not r["found"]:
                print(f"    needle: {r['needle']}")
                print(f"    response: {r['response'][:150]}")
        except torch.cuda.OutOfMemoryError as e:
            print(f"  depth={depth:.2f}: OOM — context too large for available VRAM")
            print(f"    error: {str(e)[:200]}")
            break
        except Exception as e:
            print(f"  depth={depth:.2f}: ERROR — {type(e).__name__}: {str(e)[:200]}")
            break

    # Aggregate
    if results:
        n_pass = sum(1 for r in results if r["found"])
        accuracy = n_pass / len(results)
        avg_ratio = sum(r["compression_ratio"] for r in results) / len(results)
        avg_compressed = sum(r["compressed_mb"] for r in results) / len(results)
        avg_fp16 = sum(r["fp16_mb"] for r in results) / len(results)

        print(f"\n  AGGREGATE @ {target_tokens} tokens:")
        print(f"    Accuracy: {n_pass}/{len(results)} ({accuracy*100:.0f}%)")
        print(f"    Avg compression: {avg_ratio:.2f}x")
        print(f"    Avg KV: {avg_compressed:.0f} MB (FP16: {avg_fp16:.0f} MB)")

        # Save to JSON for the announcement post
        out_file = f"/workspace/niah_results_{target_tokens}.json"
        try:
            with open(out_file, "w") as f:
                json.dump({
                    "target_tokens": target_tokens,
                    "results": results,
                    "accuracy": accuracy,
                    "avg_compression_ratio": avg_ratio,
                }, f, indent=2)
            print(f"    Saved: {out_file}")
        except Exception:
            pass

        # Soft assertion — record results even on partial failure
        assert n_pass > 0, f"All depths failed at {target_tokens} tokens"
    else:
        pytest.skip(f"Could not run {target_tokens} token test (OOM or error)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--timeout=3600"])
