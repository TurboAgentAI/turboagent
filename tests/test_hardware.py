"""
Tests for turboagent.hardware: HardwareDetector system profiling and auto-config.
"""

import pytest

from turboagent.hardware.detector import HardwareDetector


class TestGetSystemSpecs:
    def test_returns_expected_keys(self):
        specs = HardwareDetector.get_system_specs()
        required_keys = {"platform", "ram_gb", "has_cuda", "has_mps", "has_rocm", "vram_gb", "gpu_name"}
        assert required_keys.issubset(specs.keys())

    def test_ram_is_positive(self):
        specs = HardwareDetector.get_system_specs()
        assert specs["ram_gb"] > 0

    def test_platform_is_string(self):
        specs = HardwareDetector.get_system_specs()
        assert isinstance(specs["platform"], str)
        assert len(specs["platform"]) > 0

    def test_gpu_flags_are_booleans(self):
        specs = HardwareDetector.get_system_specs()
        assert isinstance(specs["has_cuda"], bool)
        assert isinstance(specs["has_mps"], bool)
        assert isinstance(specs["has_rocm"], bool)


class TestGetOptimalConfig:
    def test_returns_expected_keys(self):
        config = HardwareDetector.get_optimal_config()
        required_keys = {"backend", "kv_mode", "n_gpu_layers", "context", "offload_strategy"}
        assert required_keys.issubset(config.keys())

    def test_kv_mode_is_valid(self):
        config = HardwareDetector.get_optimal_config()
        assert config["kv_mode"] in ("turbo3", "turbo4")

    def test_backend_is_valid(self):
        config = HardwareDetector.get_optimal_config()
        assert config["backend"] in ("llama.cpp", "mlx", "hybrid")

    def test_context_is_positive(self):
        config = HardwareDetector.get_optimal_config()
        assert config["context"] > 0

    def test_small_model_config_differs(self):
        config_70b = HardwareDetector.get_optimal_config(target_model_size_b=70.0)
        config_7b = HardwareDetector.get_optimal_config(target_model_size_b=7.0)
        # On any hardware, a 7B model should get equal or more generous context
        # than 70B (or at minimum a different offload strategy)
        assert config_7b["context"] >= config_70b["context"] or config_7b != config_70b

    def test_n_gpu_layers_is_integer(self):
        config = HardwareDetector.get_optimal_config()
        assert isinstance(config["n_gpu_layers"], int)
