"""
TurboAgent Large Model Integration Test Suite for Vast.ai (RTX PRO 6000, 96GB VRAM)

Validates the complete TurboQuant pipeline on production-scale models.

Default model: Qwen2.5-32B-Instruct (~64GB in BF16, fits in 96GB VRAM)
  - 64 layers, GQA attention, head_dim=128
  - Large enough for meaningful TurboQuant validation
  - Well above the <7B quality degradation threshold

Set TURBO_TEST_MODEL env var to override:
  - TURBO_TEST_MODEL=qwen32  → Qwen/Qwen2.5-32B-Instruct (default, ungated)
  - TURBO_TEST_MODEL=qwen72  → Qwen/Qwen2.5-72B-Instruct (needs 4-bit or larger GPU)
  - TURBO_TEST_MODEL=llama   → meta-llama/Llama-3.1-70B-Instruct (gated)

Run: python -m pytest vastai/test_70b_integration.py -v -s --timeout=600
"""

import gc
import glob
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
# Model configuration
# ---------------------------------------------------------------------------

_MODEL_CHOICE = os.environ.get("TURBO_TEST_MODEL", "qwen32").lower()

_MODEL_CONFIGS = {
    "qwen32": {
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "gguf_repo": "bartowski/Qwen2.5-32B-Instruct-GGUF",
        "gguf_file": "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        "label": "Qwen2.5-32B-Instruct",
    },
    "qwen72": {
        "hf_id": "Qwen/Qwen2.5-72B-Instruct",
        "gguf_repo": "bartowski/Qwen2.5-72B-Instruct-GGUF",
        "gguf_file": "Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        "label": "Qwen2.5-72B-Instruct",
    },
    "llama": {
        "hf_id": "meta-llama/Llama-3.1-70B-Instruct",
        "gguf_repo": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "gguf_file": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
        "label": "Llama-3.1-70B-Instruct",
    },
}

_CFG = _MODEL_CONFIGS[_MODEL_CHOICE]
HF_MODEL = _CFG["hf_id"]
_LABEL = _CFG["label"]


def _find_gguf():
    for base in [
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/workspace/.hf_home/hub",
        os.environ.get("HF_HOME", ""),
    ]:
        if not base:
            continue
        pattern = os.path.join(
            base,
            f"models--{_CFG['gguf_repo'].replace('/', '--')}",
            "snapshots", "*", _CFG["gguf_file"],
        )
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def _gpu_mem():
    if not torch.cuda.is_available():
        return 0, 0, 0
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    r = torch.cuda.memory_reserved(0) / 1e9
    a = torch.cuda.memory_allocated(0) / 1e9
    return t, a, t - r


print(f"[Config] Model: {_LABEL} (TURBO_TEST_MODEL={_MODEL_CHOICE})")
print(f"[Config] HF: {HF_MODEL}")
print(f"[Config] GGUF: {_find_gguf() or 'not found yet'}")


# ===========================================================================
# Test 1: Hardware Detection
# ===========================================================================

class TestHardwareDetection:
    def test_gpu_detected(self):
        from turboagent.hardware.detector import HardwareDetector
        specs = HardwareDetector.get_system_specs()
        print(f"\n[HW] GPU: {specs['gpu_name']} | VRAM: {specs['vram_gb']:.1f} GB | RAM: {specs['ram_gb']:.1f} GB")
        assert specs["has_cuda"] is True
        assert specs["vram_gb"] >= 80

    def test_config_for_large_gpu(self):
        from turboagent.hardware.detector import HardwareDetector
        config = HardwareDetector.get_optimal_config(target_model_size_b=70.0)
        print(f"\n[HW] Config: {config}")
        assert config["kv_mode"] in ("turbo3", "turbo4")
        assert config["context"] >= 65536
        assert config["n_gpu_layers"] == -1 or config["n_gpu_layers"] >= 40


# ===========================================================================
# Test 2: Torch Backend — Full BF16 model
# ===========================================================================

class TestTorchLargeModel:
    @pytest.fixture(scope="class")
    def engine_and_cache(self):
        from turboagent.backends.torch import TorchEngine
        from turboagent.quant.turboquant import TurboQuantKVCache, _USE_NATIVE

        print(f"\n[Torch] Loading {HF_MODEL} (BF16)...")
        print(f"[Torch] Native quantizer: {_USE_NATIVE}")
        t0 = time.time()

        engine = TorchEngine(HF_MODEL, context=4096, max_tokens=128, n_gpu_layers=-1)

        load_time = time.time() - t0
        total, used, free = _gpu_mem()
        print(f"[Torch] Loaded in {load_time:.1f}s | VRAM: {used:.1f}/{total:.1f} GB | Free: {free:.1f} GB")
        print(f"[Torch] Layers: {engine._n_layers} | Head dim: {engine._head_dim} | KV heads: {engine._n_kv_heads}")

        flat_dim = engine._head_dim * engine._n_kv_heads
        cache = TurboQuantKVCache(
            bit_mode="turbo3", device="cpu",
            head_dim=flat_dim, num_layers=engine._n_layers, max_context=4096,
        )
        yield engine, cache
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    def test_single_turn_coherent(self, engine_and_cache):
        engine, cache = engine_and_cache
        cache.clear()
        engine._prev_input_ids = None

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is the capital of France? Answer in one word."},
        ]

        t0 = time.time()
        response, metrics = engine.generate_chat(messages, cache)
        elapsed = time.time() - t0

        print(f"\n[Torch T1] Response: {response[:300]}")
        print(f"[Torch T1] Time: {elapsed:.1f}s | Tokens/s: {metrics['turn_output_tokens']/elapsed:.1f}")
        print(f"[Torch T1] KV compressed: {cache.memory_usage_gb()*1000:.2f} MB")

        assert "Paris" in response or "paris" in response.lower()
        assert cache.memory_usage_gb() > 0

    def test_multi_turn_recall(self, engine_and_cache):
        """The critical test: recall a fact after TurboQuant compress/decompress."""
        engine, cache = engine_and_cache
        cache.clear()
        engine._prev_input_ids = None

        # Turn 1
        msg1 = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "Remember this: the secret project code is PHOENIX-42."},
        ]
        resp1, m1 = engine.generate_chat(msg1, cache)
        print(f"\n[Torch T1] Response: {resp1[:300]}")
        print(f"[Torch T1] KV: {cache.memory_usage_gb()*1000:.2f} MB")

        # Turn 2 — recall after KV injection
        msg2 = msg1 + [
            {"role": "assistant", "content": resp1},
            {"role": "user", "content": "What is the secret project code I just told you?"},
        ]
        resp2, m2 = engine.generate_chat(msg2, cache)
        print(f"[Torch T2] Response: {resp2[:300]}")
        print(f"[Torch T2] New tokens: {m2['turn_input_tokens']} | KV: {cache.memory_usage_gb()*1000:.2f} MB")

        assert "PHOENIX-42" in resp2 or "PHOENIX" in resp2, \
            f"Failed to recall after TurboQuant KV injection: {resp2[:300]}"

    def test_compression_ratio(self, engine_and_cache):
        engine, cache = engine_and_cache
        cache.clear()
        engine._prev_input_ids = None

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain the theory of general relativity. " * 10},
        ]
        response, metrics = engine.generate_chat(messages, cache)

        compressed_gb = cache.memory_usage_gb()
        fp16_gb = cache.fp16_baseline_gb()

        print(f"\n[Compression] Tokens: {metrics['total_tokens_cached']}")
        print(f"[Compression] FP16: {fp16_gb*1000:.2f} MB | Turbo3: {compressed_gb*1000:.2f} MB")

        if fp16_gb > 0 and compressed_gb > 0:
            ratio = fp16_gb / compressed_gb
            print(f"[Compression] Ratio: {ratio:.2f}x")
            from turboagent.quant.turboquant import _USE_NATIVE
            min_ratio = 3.0 if _USE_NATIVE else 1.3
            assert ratio > min_ratio


# ===========================================================================
# Test 3: llama.cpp Backend — GGUF
# ===========================================================================

class TestLlamaCppLargeModel:
    @pytest.fixture(scope="class")
    def engine_and_cache(self):
        gguf_path = _find_gguf()
        if not gguf_path or not os.path.exists(gguf_path):
            pytest.skip("GGUF not found. Run setup.sh first.")

        from turboagent.backends.llama_cpp import LlamaCppEngine
        from turboagent.quant.turboquant import TurboQuantKVCache

        print(f"\n[llama.cpp] Loading {gguf_path}...")
        t0 = time.time()
        engine = LlamaCppEngine(gguf_path, n_gpu_layers=-1, context=4096)
        print(f"[llama.cpp] Loaded in {time.time()-t0:.1f}s | Layers: {engine._n_layers} | Head dim: {engine._head_dim}")

        cache = TurboQuantKVCache(
            bit_mode="turbo3", device="cpu",
            head_dim=engine._head_dim, num_layers=engine._n_layers, max_context=4096,
        )
        yield engine, cache

    def test_single_turn(self, engine_and_cache):
        engine, cache = engine_and_cache
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 15 * 17? Show your work briefly."},
        ]
        t0 = time.time()
        response, metrics = engine.generate_chat(messages, cache)
        print(f"\n[llama.cpp] Response: {response[:400]}")
        print(f"[llama.cpp] Time: {time.time()-t0:.1f}s | Metrics: {metrics}")
        assert len(response) > 0
        assert "255" in response

    def test_multi_turn_recall(self, engine_and_cache):
        engine, cache = engine_and_cache
        msg1 = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "My dog's name is Biscuit and he is a golden retriever."},
        ]
        resp1, _ = engine.generate_chat(msg1, cache)
        print(f"\n[llama.cpp T1] {resp1[:300]}")

        msg2 = msg1 + [
            {"role": "assistant", "content": resp1},
            {"role": "user", "content": "What breed is my dog and what is his name?"},
        ]
        resp2, _ = engine.generate_chat(msg2, cache)
        print(f"[llama.cpp T2] {resp2[:300]}")
        assert "Biscuit" in resp2


# ===========================================================================
# Test 4: Memory Profiling (synthetic, no model load needed)
# ===========================================================================

class TestMemoryProfiling:
    def test_kv_compression_at_scale(self):
        from turboagent.quant.turboquant import TurboQuantKVCache, _USE_NATIVE

        # Simulate 64-layer model (Qwen2.5-32B) with 4096 tokens
        n_layers, seq_len, flat_dim = 64, 4096, 1024
        fp16_mb = n_layers * 2 * seq_len * flat_dim * 2 / 1e6

        cache = TurboQuantKVCache(
            bit_mode="turbo3", device="cpu",
            head_dim=flat_dim, num_layers=n_layers, max_context=seq_len,
        )

        t0 = time.time()
        for layer in range(n_layers):
            k = torch.randn(seq_len, flat_dim, dtype=torch.float16)
            v = torch.randn(seq_len, flat_dim, dtype=torch.float16)
            cache.update(layer, k, v)
        elapsed = time.time() - t0

        compressed_mb = cache.memory_usage_gb() * 1000
        ratio = fp16_mb / compressed_mb if compressed_mb > 0 else 0

        print(f"\n[Memory] {n_layers}L x {seq_len}tok x {flat_dim}dim")
        print(f"[Memory] FP16: {fp16_mb:.1f} MB | Compressed: {compressed_mb:.1f} MB | Ratio: {ratio:.2f}x")
        print(f"[Memory] Time: {elapsed:.1f}s | Native: {_USE_NATIVE}")

        min_ratio = 3.0 if _USE_NATIVE else 1.3
        assert ratio > min_ratio


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--timeout=600"])
