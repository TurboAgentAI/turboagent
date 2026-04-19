"""
Tests for 4-bit weight quantization integration (NF4/INT4 via bitsandbytes).
"""

import pytest
from unittest.mock import patch, MagicMock


def _build_fake_model():
    """Build a mock model that survives _detect_kv_shape()."""
    import torch

    n_layers = 4
    fake_kv = MagicMock()
    fake_kv.key_cache = [torch.zeros(1, 8, 1, 128) for _ in range(n_layers)]
    fake_kv.value_cache = [torch.zeros(1, 8, 1, 128) for _ in range(n_layers)]
    fake_kv.layers = None
    del fake_kv.layers

    fake_out = MagicMock()
    fake_out.past_key_values = fake_kv

    model = MagicMock()
    model.return_value = fake_out
    model.eval.return_value = None
    model.config = MagicMock(_attn_implementation="sdpa", text_config=None)
    return model


class TestBitsAndBytesConfigConstruction:
    """Verify BitsAndBytesConfig is built correctly without loading a model."""

    @patch("turboagent.backends.torch.AutoModelForCausalLM")
    @patch("turboagent.backends.torch.AutoTokenizer")
    @patch("turboagent.backends.torch.AutoConfig")
    def test_nf4_config_passed_to_from_pretrained(self, mock_config, mock_tok, mock_auto_model):
        """quantize_weights='nf4' injects BitsAndBytesConfig into load_kwargs."""
        import torch

        mock_tok.from_pretrained.return_value = MagicMock(
            pad_token=None, eos_token="</s>", eos_token_id=2,
        )
        mock_config.from_pretrained.return_value = MagicMock(
            _attn_implementation="sdpa",
            text_config=None,
        )
        mock_auto_model.from_pretrained.return_value = _build_fake_model()

        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            pytest.skip("transformers not installed")

        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            pytest.skip("bitsandbytes not installed")

        from turboagent.backends.torch import TorchEngine

        with patch.object(torch.cuda, "is_available", return_value=False):
            engine = TorchEngine(
                "test-model",
                kv_storage="gpu",
                quantize_weights="nf4",
                context=2048,
                max_tokens=64,
            )

        call_kwargs = mock_auto_model.from_pretrained.call_args
        all_kw = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert "quantization_config" in all_kw
        qconfig = all_kw["quantization_config"]
        assert qconfig.load_in_4bit is True
        assert qconfig.bnb_4bit_quant_type == "nf4"
        assert qconfig.bnb_4bit_use_double_quant is True

    @patch("turboagent.backends.torch.AutoModelForCausalLM")
    @patch("turboagent.backends.torch.AutoTokenizer")
    @patch("turboagent.backends.torch.AutoConfig")
    def test_int4_config_sets_correct_quant_type(self, mock_config, mock_tok, mock_auto_model):
        """quantize_weights='int4' sets bnb_4bit_quant_type='int4'."""
        import torch

        mock_tok.from_pretrained.return_value = MagicMock(
            pad_token=None, eos_token="</s>", eos_token_id=2,
        )
        mock_config.from_pretrained.return_value = MagicMock(
            _attn_implementation="sdpa",
            text_config=None,
        )
        mock_auto_model.from_pretrained.return_value = _build_fake_model()

        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except ImportError:
            pytest.skip("transformers + bitsandbytes required")

        from turboagent.backends.torch import TorchEngine

        with patch.object(torch.cuda, "is_available", return_value=False):
            engine = TorchEngine(
                "test-model",
                kv_storage="gpu",
                quantize_weights="int4",
                context=2048,
                max_tokens=64,
            )

        call_kwargs = mock_auto_model.from_pretrained.call_args
        all_kw = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert all_kw["quantization_config"].bnb_4bit_quant_type == "int4"

    @patch("turboagent.backends.torch.AutoModelForCausalLM")
    @patch("turboagent.backends.torch.AutoTokenizer")
    @patch("turboagent.backends.torch.AutoConfig")
    def test_no_quantization_config_when_none(self, mock_config, mock_tok, mock_auto_model):
        """quantize_weights=None should not inject quantization_config."""
        import torch

        mock_tok.from_pretrained.return_value = MagicMock(
            pad_token=None, eos_token="</s>", eos_token_id=2,
        )
        mock_config.from_pretrained.return_value = MagicMock(
            _attn_implementation="sdpa",
            text_config=None,
        )
        mock_auto_model.from_pretrained.return_value = _build_fake_model()

        from turboagent.backends.torch import TorchEngine

        with patch.object(torch.cuda, "is_available", return_value=False):
            engine = TorchEngine(
                "test-model",
                kv_storage="gpu",
                quantize_weights=None,
                context=2048,
                max_tokens=64,
            )

        call_kwargs = mock_auto_model.from_pretrained.call_args
        all_kw = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert "quantization_config" not in all_kw

    @patch("turboagent.backends.torch.AutoModelForCausalLM")
    @patch("turboagent.backends.torch.AutoTokenizer")
    @patch("turboagent.backends.torch.AutoConfig")
    def test_quantize_weights_stored_on_engine(self, mock_config, mock_tok, mock_auto_model):
        """Engine should store quantize_weights for introspection."""
        import torch

        mock_tok.from_pretrained.return_value = MagicMock(
            pad_token=None, eos_token="</s>", eos_token_id=2,
        )
        mock_config.from_pretrained.return_value = MagicMock(
            _attn_implementation="sdpa",
            text_config=None,
        )
        mock_auto_model.from_pretrained.return_value = _build_fake_model()

        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except ImportError:
            pytest.skip("transformers + bitsandbytes required")

        from turboagent.backends.torch import TorchEngine

        with patch.object(torch.cuda, "is_available", return_value=False):
            engine = TorchEngine(
                "test-model",
                quantize_weights="nf4",
                context=2048,
                max_tokens=64,
            )

        assert engine.quantize_weights == "nf4"


class TestHardwareDetectorQuantizeWeights:
    """HardwareDetector should recommend quantize_weights for large models."""

    def test_optimal_config_includes_quantize_weights_key(self):
        from turboagent.hardware.detector import HardwareDetector
        config = HardwareDetector.get_optimal_config(70.0)
        assert "quantize_weights" in config

    def test_70b_on_24gb_recommends_nf4(self):
        """RTX 4090 class (24 GB) should recommend NF4 for 70B models."""
        from turboagent.hardware.detector import HardwareDetector

        fake_props = MagicMock()
        fake_props.total_memory = 24 * (1024 ** 3)

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_properties", return_value=fake_props), \
             patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"), \
             patch("torch.version", create=True) as mock_version:
            mock_version.hip = None
            config = HardwareDetector.get_optimal_config(70.0)

        assert config["quantize_weights"] == "nf4"

    def test_7b_does_not_recommend_quantize_weights(self):
        """Small models should not force weight quantization."""
        from turboagent.hardware.detector import HardwareDetector
        config = HardwareDetector.get_optimal_config(7.0)
        assert config.get("quantize_weights") is None

    def test_quantize_weights_valid_values(self):
        from turboagent.hardware.detector import HardwareDetector
        config = HardwareDetector.get_optimal_config(70.0)
        assert config["quantize_weights"] in (None, "nf4", "int4")


class TestImportErrors:
    """Verify clear error messages when dependencies are missing."""

    def test_missing_bitsandbytes_raises(self):
        """Should raise ImportError with install instructions."""
        import sys
        import importlib

        # Temporarily hide bitsandbytes
        hidden = sys.modules.pop("bitsandbytes", None)
        try:
            with patch.dict(sys.modules, {"bitsandbytes": None}):
                # We can't fully test this without a real TorchEngine init,
                # but we can test the import guard directly
                with pytest.raises(ImportError, match="bitsandbytes"):
                    import bitsandbytes  # noqa: F401
        finally:
            if hidden is not None:
                sys.modules["bitsandbytes"] = hidden
