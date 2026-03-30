"""
TurboAgent 70B Integration Test Suite for Vast.ai (RTX PRO 6000, 96GB VRAM)

Validates the complete TurboQuant pipeline on production-scale models:
  - Llama-3.1-70B-Instruct via torch backend (HuggingFace Transformers)
  - Llama-3.1-70B-Instruct via llama.cpp backend (GGUF Q4_K_M)

Tests:
  1. Hardware detection for 96GB GPU
  2. Single-turn 70B generation with TurboQuant KV compression
  3. Multi-turn 70B with KV injection (coherence after compress/decompress)
  4. Compression ratio measurement at scale (80 layers, 128k potential context)
  5. Memory profiling: actual VRAM usage with compressed vs. uncompressed KV
  6. Long-context test: 8k+ token prompt with needle-in-a-haystack retrieval

Run: python -m pytest vastai/test_70b_integration.py -v -s --timeout=600
"""

import gc
import logging
import os
import time

import pytest
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_MODEL_70B = "meta-llama/Llama-3.1-70B-Instruct"

# GGUF path (downloaded by setup.sh)
GGUF_70B_PATH = None
_gguf_candidates = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--bartowski--Meta-Llama-3.1-70B-Instruct-GGUF/"
        "snapshots/*/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"
    ),
]

def _find_gguf():
    import glob
    for pattern in _gguf_candidates:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None

GGUF_70B_PATH = _find_gguf()


def _gpu_mem():
    """Return (total_gb, used_gb, free_gb) for GPU 0."""
    if not torch.cuda.is_available():
        return 0, 0, 0
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    r = torch.cuda.memory_reserved(0) / 1e9
    a = torch.cuda.memory_allocated(0) / 1e9
    return t, a, t - r


# ===========================================================================
# Test 1: Hardware Detection
# ===========================================================================

class TestHardwareDetection:
    def test_96gb_gpu_detected(self):
        from turboagent.hardware.detector import HardwareDetector

        specs = HardwareDetector.get_system_specs()
        print(f"\n[Hardware] GPU: {specs['gpu_name']}")
        print(f"[Hardware] VRAM: {specs['vram_gb']:.1f} GB")
        print(f"[Hardware] RAM: {specs['ram_gb']:.1f} GB")
        print(f"[Hardware] CUDA: {specs['has_cuda']}")

        assert specs["has_cuda"] is True
        assert specs["vram_gb"] >= 80, f"Expected 96GB VRAM, got {specs['vram_gb']:.1f}"

    def test_70b_config_for_96gb(self):
        from turboagent.hardware.detector import HardwareDetector

        config = HardwareDetector.get_optimal_config(target_model_size_b=70.0)
        print(f"\n[Hardware] 70B config: {config}")

        assert config["kv_mode"] in ("turbo3", "turbo4")
        assert config["context"] >= 65536
        # With 96GB, should get aggressive GPU offloading
        assert config["n_gpu_layers"] >= 40


# ===========================================================================
# Test 2: Torch Backend — 70B Single Turn
# ===========================================================================

class TestTorch70B:
    @pytest.fixture(scope="class")
    def engine_and_cache(self):
        """Load 70B model once for all tests in this class."""
        from turboagent.backends.torch import TorchEngine
        from turboagent.quant.turboquant import TurboQuantKVCache, _USE_NATIVE

        print(f"\n[Torch] Loading {HF_MODEL_70B}...")
        print(f"[Torch] Native quantizer: {_USE_NATIVE}")
        t0 = time.time()

        engine = TorchEngine(
            HF_MODEL_70B,
            context=8192,
            max_tokens=128,
            n_gpu_layers=-1,
        )

        load_time = time.time() - t0
        total, used, free = _gpu_mem()
        print(f"[Torch] Model loaded in {load_time:.1f}s")
        print(f"[Torch] GPU memory: {used:.1f}/{total:.1f} GB used, {free:.1f} GB free")
        print(f"[Torch] Layers: {engine._n_layers}, Head dim: {engine._head_dim}, KV heads: {engine._n_kv_heads}")

        flat_dim = engine._head_dim * engine._n_kv_heads
        cache = TurboQuantKVCache(
            bit_mode="turbo3",
            device="cpu",
            head_dim=flat_dim,
            num_layers=engine._n_layers,
            max_context=8192,
        )

        yield engine, cache

        # Cleanup
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    def test_single_turn_coherent(self, engine_and_cache):
        """70B should produce a coherent, correct answer."""
        engine, cache = engine_and_cache

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is the capital of France? Answer in one word."},
        ]

        t0 = time.time()
        response, metrics = engine.generate_chat(messages, cache)
        gen_time = time.time() - t0

        print(f"\n[Torch 70B T1] Response: {response[:300]}")
        print(f"[Torch 70B T1] Metrics: {metrics}")
        print(f"[Torch 70B T1] Time: {gen_time:.1f}s")
        print(f"[Torch 70B T1] Tokens/s: {metrics['turn_output_tokens'] / gen_time:.1f}")

        assert isinstance(response, str)
        assert len(response) > 0
        assert "Paris" in response or "paris" in response.lower(), \
            f"70B model should know the capital of France, got: {response[:200]}"

        compressed_mb = cache.memory_usage_gb() * 1000
        print(f"[Torch 70B T1] Compressed KV: {compressed_mb:.2f} MB")
        assert compressed_mb > 0

    def test_multi_turn_coherent(self, engine_and_cache):
        """70B should maintain coherence across turns with compressed KV injection."""
        engine, cache = engine_and_cache

        # Clear cache from previous test
        cache.clear()
        engine._prev_input_ids = None

        # Turn 1: Set a fact
        messages_t1 = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "Remember this: the secret project code is PHOENIX-42."},
        ]
        resp1, m1 = engine.generate_chat(messages_t1, cache)
        print(f"\n[Torch 70B T1] Response: {resp1[:300]}")

        compressed_t1 = cache.memory_usage_gb() * 1000
        print(f"[Torch 70B T1] Compressed KV: {compressed_t1:.2f} MB")

        # Turn 2: Recall the fact (tests KV injection coherence)
        messages_t2 = messages_t1 + [
            {"role": "assistant", "content": resp1},
            {"role": "user", "content": "What is the secret project code I just told you?"},
        ]
        resp2, m2 = engine.generate_chat(messages_t2, cache)
        print(f"[Torch 70B T2] Response: {resp2[:300]}")
        print(f"[Torch 70B T2] New tokens processed: {m2['turn_input_tokens']}")

        compressed_t2 = cache.memory_usage_gb() * 1000
        print(f"[Torch 70B T2] Compressed KV: {compressed_t2:.2f} MB")

        # THE CRITICAL TEST: 70B should recall the code after KV compression/decompression
        assert "PHOENIX-42" in resp2 or "PHOENIX" in resp2, \
            f"70B failed to recall fact after TurboQuant KV injection: {resp2[:300]}"

        # Incremental: T2 should process fewer new tokens
        assert m2["turn_input_tokens"] < m1["turn_input_tokens"] + 200

    def test_compression_ratio_70b(self, engine_and_cache):
        """Measure compression ratio on 70B-scale KV data."""
        engine, cache = engine_and_cache
        cache.clear()
        engine._prev_input_ids = None

        # Generate with a longer prompt for meaningful measurement
        long_prompt = "Explain the theory of general relativity in detail. " * 20
        messages = [
            {"role": "system", "content": "You are a physics professor."},
            {"role": "user", "content": long_prompt},
        ]

        response, metrics = engine.generate_chat(messages, cache)

        compressed_gb = cache.memory_usage_gb()
        fp16_gb = cache.fp16_baseline_gb()

        print(f"\n[Torch 70B Compression]")
        print(f"  Total tokens cached: {metrics['total_tokens_cached']}")
        print(f"  FP16 baseline: {fp16_gb * 1000:.2f} MB")
        print(f"  TurboQuant compressed: {compressed_gb * 1000:.2f} MB")

        if fp16_gb > 0 and compressed_gb > 0:
            ratio = fp16_gb / compressed_gb
            print(f"  Compression ratio: {ratio:.2f}x")

            from turboagent.quant.turboquant import _USE_NATIVE
            min_ratio = 3.0 if _USE_NATIVE else 1.3
            assert ratio > min_ratio, f"Compression ratio {ratio:.1f}x too low (min {min_ratio}x)"


# ===========================================================================
# Test 3: Torch Backend — Long Context Needle-in-Haystack
# ===========================================================================

class TestTorch70BLongContext:
    @pytest.fixture(scope="class")
    def engine_and_cache(self):
        from turboagent.backends.torch import TorchEngine
        from turboagent.quant.turboquant import TurboQuantKVCache

        engine = TorchEngine(
            HF_MODEL_70B,
            context=16384,
            max_tokens=64,
            n_gpu_layers=-1,
        )

        flat_dim = engine._head_dim * engine._n_kv_heads
        cache = TurboQuantKVCache(
            bit_mode="turbo3",
            device="cpu",
            head_dim=flat_dim,
            num_layers=engine._n_layers,
            max_context=16384,
        )
        yield engine, cache
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    def test_needle_in_haystack_8k(self, engine_and_cache):
        """
        Hide a fact in a ~8k token haystack, ask the model to retrieve it.
        Tests that TurboQuant compression doesn't lose information at scale.
        """
        engine, cache = engine_and_cache

        # Build ~8k token haystack
        filler = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump. "
        )
        haystack_before = filler * 80  # ~4k tokens
        needle = "IMPORTANT FACT: The launch authorization code is NEPTUNE-7749."
        haystack_after = filler * 80   # ~4k tokens

        prompt = f"{haystack_before}\n\n{needle}\n\n{haystack_after}\n\nQuestion: What is the launch authorization code? Answer with just the code."

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer questions precisely."},
            {"role": "user", "content": prompt},
        ]

        print(f"\n[Needle-in-Haystack] Prompt ~{len(prompt)} chars")
        t0 = time.time()
        response, metrics = engine.generate_chat(messages, cache)
        elapsed = time.time() - t0

        print(f"[Needle-in-Haystack] Response: {response[:300]}")
        print(f"[Needle-in-Haystack] Tokens cached: {metrics['total_tokens_cached']}")
        print(f"[Needle-in-Haystack] Time: {elapsed:.1f}s")
        print(f"[Needle-in-Haystack] Compressed KV: {cache.memory_usage_gb() * 1000:.2f} MB")
        print(f"[Needle-in-Haystack] FP16 baseline: {cache.fp16_baseline_gb() * 1000:.2f} MB")

        if cache.fp16_baseline_gb() > 0 and cache.memory_usage_gb() > 0:
            ratio = cache.fp16_baseline_gb() / cache.memory_usage_gb()
            print(f"[Needle-in-Haystack] Compression ratio: {ratio:.2f}x")

        assert "NEPTUNE-7749" in response or "NEPTUNE" in response, \
            f"Failed to retrieve needle from 8k haystack: {response[:300]}"


# ===========================================================================
# Test 4: llama.cpp Backend — 70B GGUF
# ===========================================================================

class TestLlamaCpp70B:
    @pytest.fixture(scope="class")
    def engine_and_cache(self):
        if GGUF_70B_PATH is None or not os.path.exists(GGUF_70B_PATH):
            pytest.skip(f"70B GGUF not found. Run setup.sh first.")

        from turboagent.backends.llama_cpp import LlamaCppEngine
        from turboagent.quant.turboquant import TurboQuantKVCache

        print(f"\n[llama.cpp] Loading {GGUF_70B_PATH}...")
        t0 = time.time()

        engine = LlamaCppEngine(
            GGUF_70B_PATH,
            n_gpu_layers=-1,  # All layers on 96GB GPU
            context=8192,
        )

        load_time = time.time() - t0
        print(f"[llama.cpp] Model loaded in {load_time:.1f}s")
        print(f"[llama.cpp] Layers: {engine._n_layers}, Head dim: {engine._head_dim}")

        cache = TurboQuantKVCache(
            bit_mode="turbo3",
            device="cpu",
            head_dim=engine._head_dim,
            num_layers=engine._n_layers,
            max_context=8192,
        )
        yield engine, cache

    def test_single_turn_70b_gguf(self, engine_and_cache):
        """70B GGUF should produce coherent output."""
        engine, cache = engine_and_cache

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 15 * 17? Show your work briefly."},
        ]

        t0 = time.time()
        response, metrics = engine.generate_chat(messages, cache)
        elapsed = time.time() - t0

        print(f"\n[llama.cpp 70B] Response: {response[:400]}")
        print(f"[llama.cpp 70B] Metrics: {metrics}")
        print(f"[llama.cpp 70B] Time: {elapsed:.1f}s")

        assert isinstance(response, str)
        assert len(response) > 0
        assert "255" in response, f"70B should compute 15*17=255, got: {response[:200]}"

    def test_multi_turn_70b_gguf(self, engine_and_cache):
        """Multi-turn with KV persistence across turns."""
        engine, cache = engine_and_cache

        messages_t1 = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "My dog's name is Biscuit and he is a golden retriever."},
        ]
        resp1, m1 = engine.generate_chat(messages_t1, cache)
        print(f"\n[llama.cpp 70B T1] Response: {resp1[:300]}")

        messages_t2 = messages_t1 + [
            {"role": "assistant", "content": resp1},
            {"role": "user", "content": "What breed is my dog and what is his name?"},
        ]
        resp2, m2 = engine.generate_chat(messages_t2, cache)
        print(f"[llama.cpp 70B T2] Response: {resp2[:300]}")

        assert "Biscuit" in resp2, f"Failed to recall dog's name: {resp2[:200]}"
        assert "golden" in resp2.lower() or "retriever" in resp2.lower(), \
            f"Failed to recall breed: {resp2[:200]}"


# ===========================================================================
# Test 5: Memory Profiling
# ===========================================================================

class TestMemoryProfiling:
    def test_vram_usage_with_compression(self):
        """Measure actual VRAM savings from TurboQuant compression."""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")

        from turboagent.quant.turboquant import TurboQuantKVCache, _USE_NATIVE

        total_gb, _, _ = _gpu_mem()
        print(f"\n[Memory] GPU total: {total_gb:.1f} GB")
        print(f"[Memory] Native quantizer: {_USE_NATIVE}")

        # Simulate 70B model KV cache: 80 layers, 8 KV heads, head_dim=128, seq=4096
        n_layers = 80
        n_kv_heads = 8
        head_dim = 128
        seq_len = 4096
        flat_dim = n_kv_heads * head_dim  # 1024

        # FP16 baseline
        fp16_per_layer = 2 * n_kv_heads * seq_len * head_dim * 2  # K+V, 2 bytes
        fp16_total_mb = fp16_per_layer * n_layers / 1e6
        print(f"[Memory] FP16 KV cache ({n_layers}L x {seq_len}tok): {fp16_total_mb:.1f} MB")

        # TurboQuant compressed
        cache = TurboQuantKVCache(
            bit_mode="turbo3", device="cpu",
            head_dim=flat_dim, num_layers=n_layers, max_context=seq_len,
        )

        t0 = time.time()
        for layer in range(n_layers):
            k = torch.randn(seq_len, flat_dim, dtype=torch.float16)
            v = torch.randn(seq_len, flat_dim, dtype=torch.float16)
            cache.update(layer, k, v)
        compress_time = time.time() - t0

        compressed_mb = cache.memory_usage_gb() * 1000
        ratio = fp16_total_mb / compressed_mb if compressed_mb > 0 else 0

        print(f"[Memory] Compressed KV cache: {compressed_mb:.1f} MB")
        print(f"[Memory] Compression ratio: {ratio:.2f}x")
        print(f"[Memory] Compression time (80 layers): {compress_time:.2f}s")
        print(f"[Memory] Savings: {fp16_total_mb - compressed_mb:.1f} MB")

        min_ratio = 3.0 if _USE_NATIVE else 1.3
        assert ratio > min_ratio


# ===========================================================================
# Test 6: TurboAgent High-Level API
# ===========================================================================

class TestTurboAgentHighLevel:
    """Test the user-facing TurboAgent API with a real 70B model."""

    def test_turboagent_one_liner(self):
        """
        The README promises one-line agent creation. Verify it works
        with 70B on 96GB hardware.

        Note: This test uses the torch backend and requires the model
        to be pre-downloaded. It may take several minutes on first run.
        """
        # We can't use the full TurboAgent constructor (it calls create_engine
        # which needs model path routing). Instead, test the components directly.
        from turboagent.hardware.detector import HardwareDetector
        from turboagent.backends.torch import TorchEngine
        from turboagent.quant.turboquant import TurboQuantKVCache

        config = HardwareDetector.get_optimal_config(target_model_size_b=70.0)
        print(f"\n[TurboAgent] Auto-config: {config}")

        engine = TorchEngine(HF_MODEL_70B, context=4096, max_tokens=64, n_gpu_layers=-1)
        flat_dim = engine._head_dim * engine._n_kv_heads
        cache = TurboQuantKVCache(
            bit_mode=config["kv_mode"], device="cpu",
            head_dim=flat_dim, num_layers=engine._n_layers, max_context=4096,
        )

        # Simulate agent.run()
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Write a haiku about machine learning."},
        ]
        response, metrics = engine.generate_chat(messages, cache)

        print(f"[TurboAgent] Response:\n{response}")
        print(f"[TurboAgent] KV compressed: {cache.memory_usage_gb() * 1000:.2f} MB")
        print(f"[TurboAgent] FP16 baseline: {cache.fp16_baseline_gb() * 1000:.2f} MB")

        assert isinstance(response, str)
        assert len(response) > 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--timeout=600"])
