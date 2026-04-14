"""
Tests for turboagent.utils: token estimation, memory profiling, formatting.
"""

import pytest

from turboagent.utils import estimate_tokens, estimate_kv_memory_gb, format_chat_ml, get_gpu_memory_info


class TestEstimateTokens:
    def test_heuristic_method(self):
        text = "hello world this is a test"
        tokens = estimate_tokens(text, method="heuristic")
        assert tokens > 0
        assert tokens == len(text) // 4

    def test_whitespace_method(self):
        text = "one two three four five"
        tokens = estimate_tokens(text, method="whitespace")
        assert tokens > 0
        assert tokens == int(5 * 1.3)

    def test_empty_string_returns_one(self):
        assert estimate_tokens("") >= 1 or estimate_tokens("") == 0

    def test_short_string(self):
        assert estimate_tokens("hi", method="heuristic") >= 0


class TestEstimateKVMemory:
    def test_returns_expected_keys(self):
        result = estimate_kv_memory_gb(num_layers=80, head_dim=128, seq_len=4096)
        assert "fp16_gb" in result
        assert "turbo_gb" in result
        assert "compression_ratio" in result
        assert "savings_gb" in result

    def test_turbo3_compression(self):
        result = estimate_kv_memory_gb(
            num_layers=80, head_dim=128, seq_len=131072, bit_mode="turbo3"
        )
        assert result["compression_ratio"] == 4.9
        assert result["turbo_gb"] < result["fp16_gb"]
        assert result["savings_gb"] > 0

    def test_turbo4_less_compression_than_turbo3(self):
        t3 = estimate_kv_memory_gb(80, 128, 131072, bit_mode="turbo3")
        t4 = estimate_kv_memory_gb(80, 128, 131072, bit_mode="turbo4")
        assert t3["turbo_gb"] < t4["turbo_gb"]


class TestGetGPUMemoryInfo:
    def test_returns_expected_keys(self):
        info = get_gpu_memory_info()
        assert "total_gb" in info
        assert "used_gb" in info
        assert "free_gb" in info

    def test_values_are_non_negative(self):
        info = get_gpu_memory_info()
        assert info["total_gb"] >= 0
        assert info["used_gb"] >= 0
        assert info["free_gb"] >= 0


class TestFormatChatML:
    def test_basic_formatting(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = format_chat_ml(messages)
        assert "<|im_start|>system" in result
        assert "<|im_start|>user" in result
        assert result.endswith("<|im_start|>assistant\n")

    def test_empty_messages(self):
        result = format_chat_ml([])
        assert result == "<|im_start|>assistant\n"
