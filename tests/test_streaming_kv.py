"""
Tests for StreamingDynamicCache (Streaming KV — zero intra-turn compression).

Design under test
-----------------
  • Prefill: update() accumulates raw FP16 per layer on CPU.  Turbo unchanged.
  • start_decode(): flips to decode mode.  No compression.  Prefill buffer kept.
  • Decode: update() uses [turbo_past | prefill_buffer | decode_buffer | new_token].
            New tokens appended to decode buffer.  Turbo unchanged.
  • finalize_decode(): compresses [turbo_past + prefill + decode] → turbo ONCE.

Turbo is NEVER updated during a forward pass — only at finalize_decode().
This guarantees zero quantization noise during the current turn's inference.
"""

import pytest
import torch

from turboagent.quant.turboquant import TurboQuantKVCache
from turboagent.quant.streaming_kv import StreamingDynamicCache


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

N_LAYERS = 4
N_KV_HEADS = 4
HEAD_DIM = 64
SEQ_LEN = 128
DTYPE = torch.float16
DEVICE = "cpu"


def _make_cache(seq_len: int = SEQ_LEN) -> TurboQuantKVCache:
    """Build and populate a TurboQuantKVCache on CPU."""
    cache = TurboQuantKVCache(
        bit_mode="turbo3",
        device=DEVICE,
        head_dim=N_KV_HEADS * HEAD_DIM,
        num_layers=N_LAYERS,
        max_context=16384,
    )
    for layer_idx in range(N_LAYERS):
        k = torch.randn(seq_len, N_KV_HEADS * HEAD_DIM, dtype=DTYPE)
        v = torch.randn(seq_len, N_KV_HEADS * HEAD_DIM, dtype=DTYPE)
        cache.update(layer_idx, k, v)
    return cache


def _make_layer_shapes():
    return {i: (N_KV_HEADS, HEAD_DIM) for i in range(N_LAYERS)}


def _make_layer_devices():
    return {i: torch.device(DEVICE) for i in range(N_LAYERS)}


def _make_sdc(seq_len: int = SEQ_LEN) -> StreamingDynamicCache:
    return StreamingDynamicCache(
        turbo_cache=_make_cache(seq_len),
        layer_shapes=_make_layer_shapes(),
        layer_devices=_make_layer_devices(),
        dtype=DTYPE,
        default_n_kv_heads=N_KV_HEADS,
        default_head_dim=HEAD_DIM,
    )


def _full_prefill(sdc: StreamingDynamicCache, n_new: int) -> None:
    """Simulate a single-chunk prefill forward pass of n_new tokens."""
    new_k = torch.randn(1, N_KV_HEADS, n_new, HEAD_DIM, dtype=DTYPE)
    new_v = torch.randn(1, N_KV_HEADS, n_new, HEAD_DIM, dtype=DTYPE)
    for layer_idx in range(N_LAYERS):
        sdc.update(new_k, new_v, layer_idx)


def _decode_step(sdc: StreamingDynamicCache) -> None:
    """Simulate one autoregressive decode step."""
    new_k = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
    new_v = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
    for layer_idx in range(N_LAYERS):
        sdc.update(new_k, new_v, layer_idx)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestStreamingDynamicCacheInit:
    def test_basic_init(self):
        sdc = _make_sdc()
        assert len(sdc) == N_LAYERS
        assert sdc.get_seq_length() == SEQ_LEN

    def test_empty_cache_init(self):
        empty_cache = TurboQuantKVCache(
            bit_mode="turbo3", device=DEVICE,
            head_dim=N_KV_HEADS * HEAD_DIM, num_layers=N_LAYERS
        )
        sdc = StreamingDynamicCache(
            turbo_cache=empty_cache,
            layer_shapes=_make_layer_shapes(),
            layer_devices=_make_layer_devices(),
            dtype=DTYPE,
            default_n_kv_heads=N_KV_HEADS,
            default_head_dim=HEAD_DIM,
        )
        assert sdc.get_seq_length() == 0

    def test_repr(self):
        sdc = _make_sdc()
        r = repr(sdc)
        assert "StreamingDynamicCache" in r
        assert "turbo_seq=" in r


# ---------------------------------------------------------------------------
# update() — prefill mode
# ---------------------------------------------------------------------------


class TestUpdatePrefillMode:
    def test_returns_correct_shapes_single_new_token(self):
        sdc = _make_sdc()
        new_k = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        new_v = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        for layer_idx in range(N_LAYERS):
            full_k, full_v = sdc.update(new_k, new_v, layer_idx)
            assert full_k.shape == (1, N_KV_HEADS, SEQ_LEN + 1, HEAD_DIM)
            assert full_v.shape == full_k.shape

    def test_returns_correct_shapes_multi_new_tokens(self):
        sdc = _make_sdc()
        new_tokens = 32
        new_k = torch.randn(1, N_KV_HEADS, new_tokens, HEAD_DIM, dtype=DTYPE)
        new_v = torch.randn(1, N_KV_HEADS, new_tokens, HEAD_DIM, dtype=DTYPE)
        for layer_idx in range(N_LAYERS):
            full_k, full_v = sdc.update(new_k, new_v, layer_idx)
            assert full_k.shape == (1, N_KV_HEADS, SEQ_LEN + new_tokens, HEAD_DIM)

    def test_from_empty_cache(self):
        empty_cache = TurboQuantKVCache(
            bit_mode="turbo3", device=DEVICE,
            head_dim=N_KV_HEADS * HEAD_DIM, num_layers=N_LAYERS
        )
        sdc = StreamingDynamicCache(
            turbo_cache=empty_cache,
            layer_shapes=_make_layer_shapes(),
            layer_devices=_make_layer_devices(),
            dtype=DTYPE,
            default_n_kv_heads=N_KV_HEADS,
            default_head_dim=HEAD_DIM,
        )
        n = 64
        new_k = torch.randn(1, N_KV_HEADS, n, HEAD_DIM, dtype=DTYPE)
        new_v = torch.randn(1, N_KV_HEADS, n, HEAD_DIM, dtype=DTYPE)
        for layer_idx in range(N_LAYERS):
            full_k, full_v = sdc.update(new_k, new_v, layer_idx)
            assert full_k.shape == (1, N_KV_HEADS, n, HEAD_DIM)

    def test_chunked_prefill_accumulates(self):
        sdc = _make_sdc(seq_len=16)
        chunk_size = 8
        for chunk in range(3):
            new_k = torch.randn(1, N_KV_HEADS, chunk_size, HEAD_DIM, dtype=DTYPE)
            new_v = torch.randn(1, N_KV_HEADS, chunk_size, HEAD_DIM, dtype=DTYPE)
            expected_len = 16 + (chunk + 1) * chunk_size
            for layer_idx in range(N_LAYERS):
                full_k, full_v = sdc.update(new_k, new_v, layer_idx)
                assert full_k.shape[2] == expected_len

    def test_turbo_never_updated_during_prefill(self):
        """Turbo must remain frozen during ALL prefill chunks."""
        sdc = _make_sdc(seq_len=16)
        original = sdc._turbo._seq_len
        _full_prefill(sdc, n_new=8)
        assert sdc._turbo._seq_len == original
        _full_prefill(sdc, n_new=8)
        assert sdc._turbo._seq_len == original

    def test_prefill_seq_len_tracks_correctly(self):
        sdc = _make_sdc(seq_len=16)
        assert sdc._prefill_seq_len == 0
        _full_prefill(sdc, n_new=8)
        assert sdc._prefill_seq_len == 8
        _full_prefill(sdc, n_new=8)
        assert sdc._prefill_seq_len == 16


# ---------------------------------------------------------------------------
# start_decode() — prefill → decode transition
# ---------------------------------------------------------------------------


class TestStartDecode:
    def test_sets_is_decoding(self):
        sdc = _make_sdc()
        assert not sdc._is_decoding
        sdc.start_decode()
        assert sdc._is_decoding

    def test_initialises_decode_buffers(self):
        sdc = _make_sdc()
        sdc.start_decode()
        assert sdc._decode_buf_k is not None
        assert len(sdc._decode_buf_k) == N_LAYERS
        assert sdc._decode_seq_len == 0

    def test_does_NOT_compress_to_turbo(self):
        """
        start_decode() must NOT compress the prefill buffer.
        Compression only happens in finalize_decode().
        """
        sdc = _make_sdc(seq_len=16)
        _full_prefill(sdc, n_new=8)
        original_turbo_seq = sdc._turbo._seq_len  # 16

        sdc.start_decode()
        assert sdc._turbo._seq_len == original_turbo_seq, (
            "start_decode() must not write to turbo — compression is deferred "
            "to finalize_decode() to prevent intra-turn quantization noise."
        )

    def test_keeps_prefill_buffer_intact(self):
        """Prefill buffer must survive start_decode() for use during decode."""
        sdc = _make_sdc(seq_len=16)
        _full_prefill(sdc, n_new=8)
        assert sdc._prefill_seq_len == 8
        assert sdc._prefill_buf_k[0] is not None

        sdc.start_decode()
        assert sdc._prefill_seq_len == 8, "prefill buffer must not be cleared"
        assert sdc._prefill_buf_k[0] is not None

    def test_get_seq_length_unchanged(self):
        sdc = _make_sdc(seq_len=16)
        _full_prefill(sdc, n_new=8)
        before = sdc.get_seq_length()  # 16 + 8 = 24
        sdc.start_decode()
        assert sdc.get_seq_length() == before  # still 24


# ---------------------------------------------------------------------------
# update() — decode mode
# ---------------------------------------------------------------------------


class TestUpdateDecodeMode:
    def _make_decoding_sdc(self, turbo_seq=16, n_prefill=8):
        """Return an SDC already in decode mode with a populated prefill buffer."""
        sdc = _make_sdc(seq_len=turbo_seq)
        _full_prefill(sdc, n_new=n_prefill)
        sdc.start_decode()
        return sdc

    def test_decode_shapes_grow_by_one(self):
        sdc = self._make_decoding_sdc(turbo_seq=16, n_prefill=8)
        total_before = 24  # 16 turbo + 8 prefill

        for step in range(5):
            new_k = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
            new_v = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
            for layer_idx in range(N_LAYERS):
                full_k, full_v = sdc.update(new_k, new_v, layer_idx)
                assert full_k.shape[2] == total_before + step + 1

    def test_turbo_NEVER_updated_during_decode(self):
        """
        The core correctness guarantee: no quantization noise during decode.
        Turbo must never change during decode — not after 50 steps.
        """
        sdc = self._make_decoding_sdc(turbo_seq=16, n_prefill=8)
        fixed_turbo_seq = sdc._turbo._seq_len  # 16

        for _ in range(50):
            _decode_step(sdc)
            assert sdc._turbo._seq_len == fixed_turbo_seq, (
                "turbo._seq_len changed during decode — quantization noise bug!"
            )

    def test_decode_seq_len_tracks(self):
        sdc = self._make_decoding_sdc()
        assert sdc._decode_seq_len == 0
        for i in range(5):
            _decode_step(sdc)
            assert sdc._decode_seq_len == i + 1

    def test_get_seq_length_during_decode(self):
        sdc = self._make_decoding_sdc(turbo_seq=16, n_prefill=8)
        assert sdc.get_seq_length() == 24

        for i in range(5):
            assert sdc.get_seq_length() == 24 + i
            _decode_step(sdc)
        assert sdc.get_seq_length() == 29

    def test_prefill_buffer_included_in_decode_attention(self):
        """
        The decode path must include the prefill buffer in every attention
        context — NOT just turbo.  This is the key fix for the needle
        retrieval failure caused by compressing before decode.
        """
        sdc = StreamingDynamicCache(
            turbo_cache=TurboQuantKVCache(
                bit_mode="turbo3", device=DEVICE,
                head_dim=N_KV_HEADS * HEAD_DIM, num_layers=N_LAYERS,
                max_context=8192,
            ),
            layer_shapes=_make_layer_shapes(),
            layer_devices=_make_layer_devices(),
            dtype=DTYPE,
            default_n_kv_heads=N_KV_HEADS,
            default_head_dim=HEAD_DIM,
        )
        # Prefill 16 tokens
        _full_prefill(sdc, n_new=16)
        sdc.start_decode()

        # First decode step — full_k must contain all 16 prefill tokens + 1 new
        new_k = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        new_v = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        full_k, _ = sdc.update(new_k, new_v, layer_idx=0)
        assert full_k.shape[2] == 17, (
            f"Expected 17 (16 prefill + 1 decode), got {full_k.shape[2]}. "
            "Prefill buffer must be included in decode attention context."
        )

    def test_zero_quantization_noise_for_current_turn(self):
        """
        The first token in the prefill buffer must be retrievable with
        cosine similarity > 0.999 even after 50 decode steps.

        Old design (compress in start_decode): cosine sim dropped to ~NaN.
        New design (no intra-turn compression): cosine sim is essentially 1.0.
        """
        cache = TurboQuantKVCache(
            bit_mode="turbo3", device=DEVICE,
            head_dim=N_KV_HEADS * HEAD_DIM, num_layers=N_LAYERS, max_context=8192
        )
        sdc = StreamingDynamicCache(
            turbo_cache=cache,
            layer_shapes=_make_layer_shapes(),
            layer_devices=_make_layer_devices(),
            dtype=DTYPE,
            default_n_kv_heads=N_KV_HEADS,
            default_head_dim=HEAD_DIM,
        )

        # Prefill a known key
        k_ref = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        v_ref = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        for layer_idx in range(N_LAYERS):
            sdc.update(k_ref, v_ref, layer_idx)
        sdc.start_decode()

        k_ref_flat = k_ref.squeeze(0).permute(1, 0, 2).reshape(1, -1).float()

        for step in range(50):
            new_k = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
            new_v = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
            full_k, _ = sdc.update(new_k, new_v, layer_idx=0)

            # Extract first token (from prefill buffer) — should be lossless
            first_k = full_k[:, :, 0, :]  # (1, nkh, hd)
            first_k_flat = first_k.squeeze(0).reshape(1, -1).float()

            cos = torch.nn.functional.cosine_similarity(first_k_flat, k_ref_flat).item()
            assert cos > 0.999, (
                f"step {step}: cosine similarity {cos:.6f} < 0.999. "
                f"Current-turn tokens should be lossless (raw FP16 buffer)."
            )
            for layer_idx in range(1, N_LAYERS):
                sdc.update(new_k, new_v, layer_idx)


# ---------------------------------------------------------------------------
# finalize_decode()
# ---------------------------------------------------------------------------


class TestFinalizeDecode:
    def _run_full_turn(self, turbo_seq=16, n_prefill=8, n_decode=5):
        sdc = _make_sdc(seq_len=turbo_seq)
        _full_prefill(sdc, n_new=n_prefill)
        sdc.start_decode()
        for _ in range(n_decode):
            _decode_step(sdc)
        sdc.finalize_decode()
        return sdc

    def test_turbo_includes_all_tokens(self):
        """After finalize_decode(), turbo._seq_len = turbo_past + prefill + decode."""
        sdc = self._run_full_turn(turbo_seq=16, n_prefill=8, n_decode=5)
        assert sdc._turbo._seq_len == 29  # 16 + 8 + 5

    def test_raw_buffers_cleared(self):
        sdc = self._run_full_turn(turbo_seq=16, n_prefill=8, n_decode=5)
        assert sdc._prefill_seq_len == 0
        assert sdc._decode_seq_len == 0
        assert all(t is None for t in sdc._prefill_buf_k)
        assert all(t is None for t in sdc._decode_buf_k)

    def test_no_decode_case(self):
        """finalize_decode() with 0 decode steps still compresses prefill."""
        sdc = _make_sdc(seq_len=0)
        _full_prefill(sdc, n_new=8)
        sdc.start_decode()
        assert sdc._turbo._seq_len == 0  # not yet compressed
        sdc.finalize_decode()
        assert sdc._turbo._seq_len == 8  # prefill compressed

    def test_no_prefill_no_decode_is_noop(self):
        sdc = _make_sdc(seq_len=16)
        original = sdc._turbo._seq_len
        sdc.finalize_decode()
        assert sdc._turbo._seq_len == original

    def test_compresses_only_once_per_turn(self):
        """Turbo must be written exactly once: in finalize_decode, not earlier."""
        sdc = _make_sdc(seq_len=0)
        _full_prefill(sdc, n_new=8)
        _full_prefill(sdc, n_new=8)
        _full_prefill(sdc, n_new=8)
        sdc.start_decode()
        assert sdc._turbo._seq_len == 0  # still frozen
        for _ in range(5):
            _decode_step(sdc)
        assert sdc._turbo._seq_len == 0  # still frozen
        sdc.finalize_decode()
        assert sdc._turbo._seq_len == 29  # 24 prefill + 5 decode, compressed once


# ---------------------------------------------------------------------------
# get_seq_length, get_mask_sizes, seen_tokens
# ---------------------------------------------------------------------------


class TestSequenceLengthMethods:
    def test_get_seq_length_from_turbo(self):
        sdc = StreamingDynamicCache(
            turbo_cache=_make_cache(64),
            layer_shapes=_make_layer_shapes(),
            layer_devices=_make_layer_devices(),
            dtype=DTYPE,
            default_n_kv_heads=N_KV_HEADS,
            default_head_dim=HEAD_DIM,
        )
        assert sdc.get_seq_length() == 64

    def test_seen_tokens_equals_get_seq_length(self):
        sdc = _make_sdc(seq_len=32)
        assert sdc.seen_tokens == sdc.get_seq_length() == 32

    def test_get_usable_length(self):
        sdc = StreamingDynamicCache(
            turbo_cache=_make_cache(50),
            layer_shapes=_make_layer_shapes(),
            layer_devices=_make_layer_devices(),
            dtype=DTYPE,
            default_n_kv_heads=N_KV_HEADS,
            default_head_dim=HEAD_DIM,
        )
        assert sdc.get_usable_length(10) == 50

    def test_get_mask_sizes_prefill(self):
        sdc = _make_sdc(seq_len=16)
        _full_prefill(sdc, n_new=8)
        cache_pos = torch.arange(4)
        kv_len, offset = sdc.get_mask_sizes(cache_pos)
        assert kv_len == 28  # past(16+8) + new(4)
        assert offset == 0

    def test_get_mask_sizes_decode(self):
        sdc = _make_sdc(seq_len=16)
        _full_prefill(sdc, n_new=8)
        sdc.start_decode()
        _decode_step(sdc)  # _decode_seq_len = 1
        cache_pos = torch.arange(1)
        kv_len, offset = sdc.get_mask_sizes(cache_pos)
        assert kv_len == 26  # past(16+8+1) + new(1)
        assert offset == 0

    def test_seen_tokens_setter_is_ignored(self):
        sdc = _make_sdc(seq_len=32)
        sdc.seen_tokens = 9999
        assert sdc.seen_tokens == 32


# ---------------------------------------------------------------------------
# Correctness: streaming output vs. reference
# ---------------------------------------------------------------------------


class TestStreamingCorrectness:
    @pytest.fixture
    def populated_cache(self):
        return _make_cache(SEQ_LEN)

    def test_new_tokens_passed_through_exact(self, populated_cache):
        """New K/V tokens must be identical in streaming output (no quantization)."""
        sdc = StreamingDynamicCache(
            turbo_cache=populated_cache,
            layer_shapes=_make_layer_shapes(),
            layer_devices=_make_layer_devices(),
            dtype=DTYPE,
            default_n_kv_heads=N_KV_HEADS,
            default_head_dim=HEAD_DIM,
        )
        new_k = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        new_v = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        full_k, full_v = sdc.update(new_k, new_v, layer_idx=0)
        assert torch.allclose(full_k[:, :, -1:, :], new_k)
        assert torch.allclose(full_v[:, :, -1:, :], new_v)

    def test_past_tokens_from_turbo_cosine_similarity(self, populated_cache):
        """
        Past tokens decompressed from turbo (previous turns) have cosine sim > 0.95.
        This is the expected quality for one turbo compression cycle.
        """
        k_ref, v_ref = populated_cache.get(layer_idx=0, seq_len=SEQ_LEN)
        sdc = StreamingDynamicCache(
            turbo_cache=populated_cache,
            layer_shapes=_make_layer_shapes(),
            layer_devices=_make_layer_devices(),
            dtype=DTYPE,
            default_n_kv_heads=N_KV_HEADS,
            default_head_dim=HEAD_DIM,
        )
        new_k = torch.zeros(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        new_v = torch.zeros(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        full_k, _ = sdc.update(new_k, new_v, layer_idx=0)

        past_k = full_k[:, :, :-1, :].squeeze(0)
        past_k = past_k.permute(1, 0, 2).reshape(SEQ_LEN, -1).float()
        k_ref_f = k_ref.float()
        cos_sim = torch.nn.functional.cosine_similarity(past_k, k_ref_f, dim=-1)
        avg_cos = cos_sim.mean().item()
        assert avg_cos > 0.95

    def test_current_turn_tokens_lossless(self, populated_cache):
        """
        Current-turn tokens (prefill buffer) must be lossless.
        Cosine similarity > 0.999 even after start_decode().
        """
        sdc = StreamingDynamicCache(
            turbo_cache=populated_cache,
            layer_shapes=_make_layer_shapes(),
            layer_devices=_make_layer_devices(),
            dtype=DTYPE,
            default_n_kv_heads=N_KV_HEADS,
            default_head_dim=HEAD_DIM,
        )
        # Prefill with known tokens
        known_k = torch.randn(1, N_KV_HEADS, 16, HEAD_DIM, dtype=DTYPE)
        known_v = torch.randn(1, N_KV_HEADS, 16, HEAD_DIM, dtype=DTYPE)
        for layer_idx in range(N_LAYERS):
            sdc.update(known_k, known_v, layer_idx)

        sdc.start_decode()

        # Retrieve via a decode step
        new_k = torch.zeros(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        new_v = torch.zeros(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        full_k, _ = sdc.update(new_k, new_v, layer_idx=0)

        # The "prefill" slot is after the turbo_past (SEQ_LEN tokens)
        prefill_start = SEQ_LEN
        retrieved = full_k[:, :, prefill_start:prefill_start+16, :]  # (1,nkh,16,hd)
        cos = torch.nn.functional.cosine_similarity(
            retrieved.reshape(-1).float(),
            known_k.reshape(-1).float(),
            dim=0,
        ).item()
        assert cos > 0.999, (
            f"Current-turn prefill tokens should be lossless. cos={cos:.6f}"
        )

    def test_full_turn_end_to_end(self):
        """
        Complete turn: empty turbo → prefill 32 tokens (2 chunks) →
        start_decode → 5 decode steps → finalize_decode.
        Final turbo._seq_len must be 37.
        """
        sdc = StreamingDynamicCache(
            turbo_cache=TurboQuantKVCache(
                bit_mode="turbo3", device=DEVICE,
                head_dim=N_KV_HEADS * HEAD_DIM, num_layers=N_LAYERS,
                max_context=8192,
            ),
            layer_shapes=_make_layer_shapes(),
            layer_devices=_make_layer_devices(),
            dtype=DTYPE,
            default_n_kv_heads=N_KV_HEADS,
            default_head_dim=HEAD_DIM,
        )

        _full_prefill(sdc, n_new=16)
        _full_prefill(sdc, n_new=16)
        assert sdc._prefill_seq_len == 32
        assert sdc._turbo._seq_len == 0  # frozen during prefill

        sdc.start_decode()
        assert sdc._turbo._seq_len == 0  # still frozen — compression deferred
        assert sdc._prefill_seq_len == 32  # prefill buffer intact

        for i in range(5):
            assert sdc.get_seq_length() == 32 + i
            _decode_step(sdc)
        assert sdc._turbo._seq_len == 0  # still frozen during decode

        sdc.finalize_decode()
        assert sdc._turbo._seq_len == 37  # 32 prefill + 5 decode, compressed once
        assert sdc._prefill_seq_len == 0   # buffers cleared
        assert sdc._decode_seq_len == 0


# ---------------------------------------------------------------------------
# Stage 3: pinned memory + async prefetch infrastructure
# ---------------------------------------------------------------------------


class TestStage3Infrastructure:
    """
    Tests for Stage 3 additions: pinned-memory buffers and async prefetch.

    Most of these run on CPU (DEVICE = "cpu") where pinned memory and CUDA
    streams are unavailable.  The tests verify:
      • _use_pinned and _copy_stream are correctly initialised for the platform.
      • Prefetch dicts (_pf_turbo, _pf_prefill) exist and behave correctly on CPU.
      • _kick_prefetch is a no-op during prefill and on CPU.
      • Dicts are cleared by finalize_decode() and at the start of each layer-0 call.
      • Correctness is unchanged (existing 35 tests still hold with Stage 3 code).
    """

    def test_use_pinned_false_on_cpu(self):
        """On a CPU-only machine, pinned memory must not be requested."""
        sdc = _make_sdc()
        # CUDA is not available in the test environment (DEVICE="cpu")
        assert sdc._use_pinned == torch.cuda.is_available()

    def test_copy_stream_none_on_cpu(self):
        """copy_stream must be None when CUDA is unavailable."""
        sdc = _make_sdc()
        if not torch.cuda.is_available():
            assert sdc._copy_stream is None

    def test_prefetch_dicts_exist(self):
        """_pf_turbo and _pf_prefill dicts must be created on init."""
        sdc = _make_sdc()
        assert hasattr(sdc, "_pf_turbo")
        assert hasattr(sdc, "_pf_prefill")
        assert isinstance(sdc._pf_turbo, dict)
        assert isinstance(sdc._pf_prefill, dict)

    def test_prefetch_dicts_empty_on_init(self):
        sdc = _make_sdc()
        assert len(sdc._pf_turbo) == 0
        assert len(sdc._pf_prefill) == 0

    def test_kick_prefetch_noop_during_prefill(self):
        """
        _kick_prefetch must be a no-op when not in decode mode.
        Prefetch dicts must stay empty throughout prefill.
        """
        sdc = _make_sdc(seq_len=16)
        _full_prefill(sdc, n_new=8)
        _full_prefill(sdc, n_new=8)
        assert len(sdc._pf_turbo) == 0, "_pf_turbo should be empty during prefill"
        assert len(sdc._pf_prefill) == 0, "_pf_prefill should be empty during prefill"

    def test_prefetch_dicts_empty_after_full_decode_step_on_cpu(self):
        """
        On CPU (no copy stream), no entries should accumulate in prefetch dicts.
        """
        sdc = _make_sdc(seq_len=16)
        _full_prefill(sdc, n_new=8)
        sdc.start_decode()

        for _ in range(3):
            _decode_step(sdc)
            assert len(sdc._pf_turbo) == 0
            assert len(sdc._pf_prefill) == 0

    def test_start_decode_clears_prefetch_dicts(self):
        """start_decode() must clear any residual prefetch state."""
        sdc = _make_sdc(seq_len=16)
        # Manually inject stale entries to verify they're cleared
        sdc._pf_turbo[3] = (None, None)
        sdc._pf_prefill[2] = (None, None)

        _full_prefill(sdc, n_new=8)
        sdc.start_decode()

        assert len(sdc._pf_turbo) == 0
        assert len(sdc._pf_prefill) == 0

    def test_finalize_decode_clears_prefetch_dicts(self):
        """finalize_decode() must clear prefetch dicts."""
        sdc = _make_sdc(seq_len=16)
        _full_prefill(sdc, n_new=8)
        sdc.start_decode()
        for _ in range(3):
            _decode_step(sdc)

        # Manually inject stale entries
        sdc._pf_turbo[0] = (None, None)
        sdc._pf_prefill[1] = (None, None)

        sdc.finalize_decode()
        assert len(sdc._pf_turbo) == 0
        assert len(sdc._pf_prefill) == 0

    def test_layer0_update_clears_stale_prefetch(self):
        """
        update(layer_idx=0) must clear stale prefetch dicts from the previous
        forward pass.  This is a defensive guard for exception recovery.

        After update(0), the dict may contain a fresh entry for layer 1
        (kicked off by _kick_prefetch on CUDA machines), but the stale
        entries injected for arbitrary layers (5, 3) must be gone.
        """
        sdc = _make_sdc(seq_len=16)
        sdc.start_decode()

        # Inject stale entries (simulating a prior forward pass that was interrupted)
        sdc._pf_turbo[5] = (None, None)
        sdc._pf_prefill[3] = (None, None)

        # A layer-0 update must evict the stale entries
        new_k = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        new_v = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        sdc.update(new_k, new_v, layer_idx=0)

        # Stale keys (5, 3) must be gone
        assert 5 not in sdc._pf_turbo, "stale pf_turbo[5] not cleared by update(0)"
        assert 3 not in sdc._pf_prefill, "stale pf_prefill[3] not cleared by update(0)"
        # All remaining keys must be valid layer indices (i.e., layer 1 on CUDA)
        for k in sdc._pf_turbo:
            assert 0 < k < N_LAYERS, f"unexpected key {k} in _pf_turbo"
        for k in sdc._pf_prefill:
            assert 0 < k < N_LAYERS, f"unexpected key {k} in _pf_prefill"

    def test_repr_includes_pinned(self):
        sdc = _make_sdc()
        r = repr(sdc)
        assert "pinned=" in r

    def test_correctness_unchanged_with_stage3_code(self):
        """
        Stage 3 must not change the output of update().
        Run a full turn and verify turbo._seq_len matches expected.
        """
        sdc = StreamingDynamicCache(
            turbo_cache=TurboQuantKVCache(
                bit_mode="turbo3", device=DEVICE,
                head_dim=N_KV_HEADS * HEAD_DIM, num_layers=N_LAYERS,
                max_context=8192,
            ),
            layer_shapes=_make_layer_shapes(),
            layer_devices=_make_layer_devices(),
            dtype=DTYPE,
            default_n_kv_heads=N_KV_HEADS,
            default_head_dim=HEAD_DIM,
        )
        _full_prefill(sdc, n_new=32)
        sdc.start_decode()
        for _ in range(8):
            _decode_step(sdc)
        sdc.finalize_decode()
        assert sdc._turbo._seq_len == 40  # 32 prefill + 8 decode


# ---------------------------------------------------------------------------
# Stage 4 — block-allocated pinned buffers
# ---------------------------------------------------------------------------

from turboagent.quant.streaming_kv import DEFAULT_BLOCK_SIZE


def _make_fresh_sdc(block_size: int = DEFAULT_BLOCK_SIZE) -> "StreamingDynamicCache":
    """SDC with an empty turbo cache (no prior KV) so seq_len starts at 0."""
    empty_cache = TurboQuantKVCache(
        bit_mode="turbo3", device=DEVICE,
        head_dim=N_KV_HEADS * HEAD_DIM, num_layers=N_LAYERS,
        max_context=16384,
    )
    return StreamingDynamicCache(
        turbo_cache=empty_cache,
        layer_shapes=_make_layer_shapes(),
        layer_devices=_make_layer_devices(),
        dtype=DTYPE,
        default_n_kv_heads=N_KV_HEADS,
        default_head_dim=HEAD_DIM,
        block_size=block_size,
    )


class TestStage4BlockBuffers:
    # ------------------------------------------------------------------
    # Capacity is always a multiple of block_size
    # ------------------------------------------------------------------

    def test_prefill_cap_is_block_multiple(self):
        """After any prefill, _prefill_cap must be a multiple of block_size."""
        bs = 8
        sdc = _make_fresh_sdc(block_size=bs)
        for n_new in [3, 5, 10, 7]:
            _full_prefill(sdc, n_new=n_new)
        assert sdc._prefill_cap % bs == 0

    def test_decode_cap_is_block_multiple(self):
        """After any decode steps, _decode_cap must be a multiple of block_size."""
        bs = 8
        sdc = _make_fresh_sdc(block_size=bs)
        _full_prefill(sdc, n_new=4)
        sdc.start_decode()
        for _ in range(5):
            _decode_step(sdc)
        assert sdc._decode_cap % bs == 0

    def test_prefill_cap_minimum_covers_written_tokens(self):
        """_prefill_cap >= _prefill_seq_len at all times."""
        sdc = _make_fresh_sdc(block_size=16)
        for n_new in [1, 7, 16, 20, 3]:
            _full_prefill(sdc, n_new=n_new)
            assert sdc._prefill_cap >= sdc._prefill_seq_len

    # ------------------------------------------------------------------
    # Grow frequency — decode should only grow every block_size steps
    # ------------------------------------------------------------------

    def test_decode_grows_only_at_block_boundaries(self):
        """
        _decode_cap must increase exactly once every block_size decode steps
        (first grow at step 1, then at step block_size+1, etc.).
        """
        bs = 4
        sdc = _make_fresh_sdc(block_size=bs)
        _full_prefill(sdc, n_new=8)
        sdc.start_decode()

        prev_cap = 0
        grow_steps = []
        for step in range(1, bs * 3 + 1):
            _decode_step(sdc)
            if sdc._decode_cap != prev_cap:
                grow_steps.append(step)
                prev_cap = sdc._decode_cap

        # Should grow at steps 1, bs+1, 2*bs+1 (first token triggers alloc,
        # then each subsequent block boundary)
        assert len(grow_steps) == 3, f"expected 3 grows, got {grow_steps}"
        for i in range(1, len(grow_steps)):
            gap = grow_steps[i] - grow_steps[i - 1]
            assert gap == bs, f"grow gap {gap} != block_size {bs} at grow {i}"

    def test_decode_no_grow_within_block(self):
        """Within a single block, capacity must not change."""
        bs = 8
        sdc = _make_fresh_sdc(block_size=bs)
        _full_prefill(sdc, n_new=4)
        sdc.start_decode()

        _decode_step(sdc)  # first step triggers alloc → cap = bs
        cap_after_first = sdc._decode_cap
        assert cap_after_first == bs

        for _ in range(bs - 1):  # steps 2 .. bs: still within block
            _decode_step(sdc)
            assert sdc._decode_cap == cap_after_first, "capacity changed within block"

    # ------------------------------------------------------------------
    # Data integrity — contents preserved across grows
    # ------------------------------------------------------------------

    def test_prefill_data_preserved_across_grow(self):
        """
        Data written in chunk 1 must still be readable after a grow triggered
        by chunk 2 (if it overflows the current capacity).
        """
        bs = 4
        sdc = _make_fresh_sdc(block_size=bs)

        # Write exactly bs tokens → cap = bs, no overflow
        k1 = torch.randn(1, N_KV_HEADS, bs, HEAD_DIM, dtype=DTYPE)
        v1 = torch.randn(1, N_KV_HEADS, bs, HEAD_DIM, dtype=DTYPE)
        for layer_idx in range(N_LAYERS):
            sdc.update(k1, v1, layer_idx)

        saved_k1 = sdc._prefill_buf_k[0][:, :, :bs, :].clone()

        # Write one more token → triggers grow (bs+1 > bs)
        k2 = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        v2 = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        for layer_idx in range(N_LAYERS):
            sdc.update(k2, v2, layer_idx)

        # Original data for layer 0 must still match
        actual_k1 = sdc._prefill_buf_k[0][:, :, :bs, :]
        assert torch.allclose(actual_k1, saved_k1), "prefill data corrupted after grow"

    def test_decode_data_preserved_across_grow(self):
        """Data written in decode step 1 must survive a grow at step block_size+1."""
        bs = 4
        sdc = _make_fresh_sdc(block_size=bs)
        _full_prefill(sdc, n_new=4)
        sdc.start_decode()

        # Capture the k written at decode step 1
        k_step1 = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        v_step1 = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        for layer_idx in range(N_LAYERS):
            sdc.update(k_step1, v_step1, layer_idx)

        saved = sdc._decode_buf_k[0][:, :, 0:1, :].clone()

        # Fill the rest of block 1 and trigger block 2 grow
        for _ in range(bs):  # steps 2 .. bs+1 (step bs+1 triggers grow)
            _decode_step(sdc)

        actual = sdc._decode_buf_k[0][:, :, 0:1, :]
        assert torch.allclose(actual, saved), "decode data corrupted after grow"

    # ------------------------------------------------------------------
    # write_ptr locking
    # ------------------------------------------------------------------

    def test_prefill_write_ptr_set_at_layer0(self):
        """_prefill_write_ptr must equal old _prefill_seq_len at layer 0."""
        sdc = _make_fresh_sdc(block_size=16)
        # After first prefill chunk of n tokens, write_ptr should be 0 (was seq_len before advance)
        n = 8
        k = torch.randn(1, N_KV_HEADS, n, HEAD_DIM, dtype=DTYPE)
        v = torch.randn(1, N_KV_HEADS, n, HEAD_DIM, dtype=DTYPE)

        # Capture write_ptr right after layer 0 (before any layer-1 call)
        sdc.update(k, v, layer_idx=0)
        write_ptr_after_layer0 = sdc._prefill_write_ptr
        seq_len_after_layer0 = sdc._prefill_seq_len

        # write_ptr must be the OLD seq_len (0), seq_len must now be n
        assert write_ptr_after_layer0 == 0
        assert seq_len_after_layer0 == n

    def test_start_decode_locks_prefill_write_ptr(self):
        """start_decode() must set _prefill_write_ptr = _prefill_seq_len."""
        sdc = _make_fresh_sdc(block_size=16)
        n = 24
        _full_prefill(sdc, n_new=n)
        assert sdc._prefill_seq_len == n
        sdc.start_decode()
        assert sdc._prefill_write_ptr == n, (
            f"write_ptr {sdc._prefill_write_ptr} != prefill_seq_len {n} after start_decode()"
        )

    def test_prefill_write_ptr_stable_across_all_layers(self):
        """
        All layers of the same forward pass must see the same _prefill_write_ptr.
        After the pass, seq_len advances but write_ptr stays at old value until
        the *next* layer-0 call.
        """
        sdc = _make_fresh_sdc(block_size=16)
        n1 = 10
        _full_prefill(sdc, n_new=n1)
        # write_ptr was set at layer 0 and should still equal 0 after full pass
        assert sdc._prefill_write_ptr == 0
        assert sdc._prefill_seq_len == n1

        # Second prefill chunk: write_ptr becomes n1 at layer-0, seq_len becomes n1+n2
        n2 = 6
        k = torch.randn(1, N_KV_HEADS, n2, HEAD_DIM, dtype=DTYPE)
        v = torch.randn(1, N_KV_HEADS, n2, HEAD_DIM, dtype=DTYPE)
        sdc.update(k, v, layer_idx=0)
        assert sdc._prefill_write_ptr == n1  # locked to old seq_len
        assert sdc._prefill_seq_len == n1 + n2

    # ------------------------------------------------------------------
    # repr includes Stage 4 fields
    # ------------------------------------------------------------------

    def test_repr_includes_block_fields(self):
        sdc = _make_fresh_sdc(block_size=16)
        _full_prefill(sdc, n_new=20)
        sdc.start_decode()
        for _ in range(3):
            _decode_step(sdc)
        r = repr(sdc)
        assert "prefill_blocks=" in r, "repr missing prefill_blocks"
        assert "decode_blocks=" in r, "repr missing decode_blocks"
        assert "block_size=" in r, "repr missing block_size"

    def test_repr_block_counts_are_accurate(self):
        """prefill_blocks and decode_blocks in repr must match cap // block_size."""
        bs = 8
        sdc = _make_fresh_sdc(block_size=bs)
        _full_prefill(sdc, n_new=bs + 1)  # forces 2 blocks
        sdc.start_decode()
        for _ in range(bs + 1):  # forces 2 decode blocks
            _decode_step(sdc)

        expected_pblocks = sdc._prefill_cap // bs
        expected_dblocks = sdc._decode_cap // bs
        r = repr(sdc)
        assert f"prefill_blocks={expected_pblocks}" in r
        assert f"decode_blocks={expected_dblocks}" in r

    # ------------------------------------------------------------------
    # Correctness: Stage 4 must not change turbo output
    # ------------------------------------------------------------------

    def test_turbo_seq_len_after_full_turn(self):
        """turbo._seq_len must equal prefill+decode tokens after finalize_decode()."""
        sdc = _make_fresh_sdc(block_size=16)
        n_prefill, n_decode = 32, 12
        _full_prefill(sdc, n_new=n_prefill)
        sdc.start_decode()
        for _ in range(n_decode):
            _decode_step(sdc)
        sdc.finalize_decode()
        assert sdc._turbo._seq_len == n_prefill + n_decode

    def test_block_buffers_cleared_after_finalize(self):
        """finalize_decode() must clear decode buffers (cap and seq_len reset to 0)."""
        sdc = _make_fresh_sdc(block_size=8)
        _full_prefill(sdc, n_new=16)
        sdc.start_decode()
        for _ in range(10):
            _decode_step(sdc)
        sdc.finalize_decode()
        assert sdc._decode_seq_len == 0
        assert sdc._decode_cap == 0
        assert sdc._prefill_seq_len == 0
        assert sdc._prefill_cap == 0
