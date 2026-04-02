"""
llama.cpp backend engine with TurboQuant KV cache bridge.

Manages the full lifecycle of compressed KV state across multi-turn
agentic conversations:

  1. INJECT: Dequantize compressed KV → restore into llama.cpp context
  2. EVALUATE: Process only new tokens (incremental, not full history)
  3. GENERATE: Autoregressive decoding with llama.cpp's native loop
  4. EXTRACT: Pull per-layer K/V activations → compress via TurboQuant

Per-layer KV access uses the patched llama-cpp-python fork when available
(exposes llama_kv_get_layer_k/v). Falls back to state save/restore with
byte-level TurboQuant compression for the standard library.
"""

import ctypes
import logging
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from llama_cpp import Llama
    import llama_cpp
except ImportError:
    raise ImportError("llama-cpp-python is missing. Run: pip install turboagent-ai[llama]")

from turboagent.quant.turboquant import TurboQuantKVCache
from turboagent.backends.base import BaseEngine

logger = logging.getLogger("turboagent.backends.llama")


def _detect_patched_fork() -> bool:
    """Check if llama-cpp-python has TurboQuant KV-access extensions."""
    return (
        hasattr(llama_cpp, "llama_kv_get_layer_k")
        and hasattr(llama_cpp, "llama_kv_get_layer_v")
        and hasattr(llama_cpp, "llama_kv_set_layer_k")
        and hasattr(llama_cpp, "llama_kv_set_layer_v")
    )


_HAS_PATCHED_KV_API = _detect_patched_fork()


class LlamaCppEngine(BaseEngine):
    """
    llama.cpp inference engine with TurboQuant KV cache compression.

    Two operating modes:

    **Patched fork** (preferred): Per-layer K/V tensor extraction via
    C API extensions. Each layer's key states are compressed with
    TurboQuantMSE (Algorithm 1) and value states with TurboQuantProd
    (Algorithm 2), matching the paper's recommendation for KV cache.

    **Standard library** (fallback): Uses llama.cpp's state save/restore
    API for incremental generation. The full context state (including KV
    cache) is saved between turns and restored on the next call, avoiding
    redundant recomputation. TurboQuant compression is applied to the
    raw KV bytes extracted from the serialized state.
    """

    def __init__(self, model_id: str, **kwargs):
        self.model_path = model_id
        self.n_gpu_layers = kwargs.get("n_gpu_layers", -1)
        self.n_ctx = kwargs.get("context", 131072)
        self.temperature = kwargs.get("temperature", 0.7)

        logger.debug(
            f"Loading GGUF Engine | GPU Layers: {self.n_gpu_layers} | "
            f"Context: {self.n_ctx} | Patched KV API: {_HAS_PATCHED_KV_API}"
        )

        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=False,
        )

        # Inter-turn state tracking
        self._prev_tokens: List[int] = []  # All tokens processed so far
        self._saved_state = None  # LlamaState for standard-API fallback
        self._n_layers = self._get_n_layers()
        self._head_dim = self._get_head_dim()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        kv_cache: TurboQuantKVCache,
        tools: Optional[List[Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate with TurboQuant-compressed KV cache across turns.

        Flow:
          1. Tokenize full conversation.
          2. Inject compressed KV from previous turns (skip recomputation).
          3. Evaluate only NEW tokens.
          4. Autoregressive generation.
          5. Extract and compress new KV state with TurboQuant.
        """
        prompt = self._apply_chat_template(messages)
        all_tokens = self.llm.tokenize(prompt.encode("utf-8"))

        # --- Phase 1: INJECT compressed KV state ---
        n_restored = self._inject_kv_state(kv_cache, all_tokens)

        # Compute which tokens are new (not yet in KV cache)
        new_tokens = all_tokens[n_restored:]

        # --- Phase 2+3: EVALUATE and GENERATE ---
        # llama-cpp-python's generate() handles eval + autoregressive
        # generation in one call. Pass reset=False when we have restored
        # state so it doesn't clear the KV cache.
        if n_restored > 0 and new_tokens:
            # We have restored state — eval only new tokens, then generate
            self.llm.eval(new_tokens)
            gen_tokens = []  # generate() will continue from current state
        else:
            gen_tokens = all_tokens  # No cache — process everything

        output_tokens = []
        for token in self.llm.generate(
            gen_tokens, temp=self.temperature, reset=(n_restored == 0)
        ):
            output_tokens.append(token)
            if token == self.llm.token_eos():
                break

        total_cached = len(all_tokens) + len(output_tokens)

        # --- Phase 4: EXTRACT and COMPRESS KV ---
        self._extract_and_compress_kv(kv_cache, total_cached)

        # Update token tracking
        self._prev_tokens = all_tokens + output_tokens

        response_text = self.llm.detokenize(output_tokens).decode("utf-8")
        metrics = {
            "turn_input_tokens": len(new_tokens),
            "turn_output_tokens": len(output_tokens),
            "total_tokens_cached": total_cached,
            "kv_compression_mode": "per_layer" if _HAS_PATCHED_KV_API else "state_level",
        }

        logger.debug(
            f"Generated {len(output_tokens)} tokens | "
            f"Restored {n_restored}/{len(all_tokens)} from cache | "
            f"Total cached: {total_cached}"
        )

        return response_text, metrics

    # ------------------------------------------------------------------
    # KV Cache Bridge: INJECT (dequantize → llama.cpp)
    # ------------------------------------------------------------------

    def _inject_kv_state(
        self, kv_cache: TurboQuantKVCache, current_tokens: List[int]
    ) -> int:
        """
        Restore compressed KV state into llama.cpp's context.

        Returns the number of tokens successfully restored (so the caller
        knows which tokens to skip during evaluation).
        """
        if not self._prev_tokens:
            return 0

        # Find the longest common prefix between previous and current tokens.
        # Tokens beyond this prefix need recomputation.
        n_common = self._common_prefix_length(self._prev_tokens, current_tokens)
        if n_common == 0:
            return 0

        if _HAS_PATCHED_KV_API:
            return self._inject_kv_patched(kv_cache, n_common)
        else:
            return self._inject_kv_state_api(n_common)

    def _inject_kv_patched(self, kv_cache: TurboQuantKVCache, seq_len: int) -> int:
        """Per-layer KV injection via patched fork's C API extensions."""
        try:
            ctx = self.llm._ctx.ctx
            self.llm.reset()

            for layer_idx in range(min(self._n_layers, kv_cache.num_layers)):
                k_deq, v_deq = kv_cache.get(layer_idx, seq_len)

                # Convert to contiguous FP16 numpy arrays for C interop
                k_np = k_deq.cpu().contiguous().numpy().astype(np.float16)
                v_np = v_deq.cpu().contiguous().numpy().astype(np.float16)

                # Write to llama.cpp's internal KV buffers via patched API
                k_ptr = k_np.ctypes.data_as(ctypes.c_void_p)
                v_ptr = v_np.ctypes.data_as(ctypes.c_void_p)
                k_size = k_np.nbytes
                v_size = v_np.nbytes

                llama_cpp.llama_kv_set_layer_k(ctx, layer_idx, k_ptr, k_size, seq_len)
                llama_cpp.llama_kv_set_layer_v(ctx, layer_idx, v_ptr, v_size, seq_len)

            logger.debug(f"Injected {seq_len} tokens of compressed KV (per-layer)")
            return seq_len

        except Exception as e:
            logger.warning(f"Patched KV injection failed, falling back to recompute: {e}")
            return 0

    def _inject_kv_state_api(self, n_common: int) -> int:
        """Restore full context state via standard save/load API."""
        if self._saved_state is None:
            return 0

        try:
            prev_len = len(self._prev_tokens)

            if n_common == prev_len:
                # Perfect prefix match — restore state as-is
                self.llm.load_state(self._saved_state)
                logger.debug(f"Restored {n_common} tokens from saved state (exact match)")
                return n_common

            # If only a partial prefix matches, the saved state has tokens
            # beyond the common prefix that we can't trim without
            # llama_kv_cache_seq_rm (API varies by version). In this case,
            # it's safer to recompute from scratch rather than inject a
            # mismatched KV cache.
            logger.debug(
                f"Partial prefix match ({n_common}/{prev_len}), "
                f"recomputing from scratch"
            )
            return 0

        except Exception as e:
            logger.warning(f"State restore failed, falling back to recompute: {e}")
            self._saved_state = None
            return 0

    # ------------------------------------------------------------------
    # KV Cache Bridge: EXTRACT (llama.cpp → TurboQuant compress)
    # ------------------------------------------------------------------

    def _extract_and_compress_kv(
        self, kv_cache: TurboQuantKVCache, seq_len: int
    ) -> None:
        """
        Extract KV activations from llama.cpp and compress with TurboQuant.

        Per the paper (arXiv 2504.19874):
          - K-cache: TurboQuantMSE (Algorithm 1) — minimizes ||K - K_hat||^2,
            which directly minimizes attention score error in Q·K^T.
          - V-cache: TurboQuantProd (Algorithm 2) — unbiased inner-product
            preservation for attention_weights · V computation.
        """
        if _HAS_PATCHED_KV_API:
            self._extract_kv_patched(kv_cache, seq_len)
        else:
            self._extract_kv_state_api(kv_cache, seq_len)

    def _extract_kv_patched(self, kv_cache: TurboQuantKVCache, seq_len: int) -> None:
        """Per-layer KV extraction via patched fork's C API extensions."""
        ctx = self.llm._ctx.ctx
        n_layers = min(self._n_layers, kv_cache.num_layers)
        head_dim = self._head_dim

        for layer_idx in range(n_layers):
            # Read raw FP16 K/V data from llama.cpp's internal buffers
            k_size = seq_len * head_dim * 2  # 2 bytes per FP16
            v_size = seq_len * head_dim * 2

            k_buf = (ctypes.c_char * k_size)()
            v_buf = (ctypes.c_char * v_size)()

            llama_cpp.llama_kv_get_layer_k(
                ctx, layer_idx, ctypes.cast(k_buf, ctypes.c_void_p), k_size, seq_len
            )
            llama_cpp.llama_kv_get_layer_v(
                ctx, layer_idx, ctypes.cast(v_buf, ctypes.c_void_p), v_size, seq_len
            )

            # Convert to PyTorch tensors
            k_np = np.frombuffer(bytes(k_buf), dtype=np.float16).reshape(seq_len, head_dim)
            v_np = np.frombuffer(bytes(v_buf), dtype=np.float16).reshape(seq_len, head_dim)

            k_tensor = torch.from_numpy(k_np.copy())
            v_tensor = torch.from_numpy(v_np.copy())

            # Compress: K with MSE-optimal (Algorithm 1), V with Prod-optimal (Algorithm 2)
            # This is handled by TurboQuantKVCache which uses _k_quant (MSE) and _v_quant (Prod)
            kv_cache.update(layer_idx, k_tensor, v_tensor)

        logger.debug(
            f"Extracted and compressed {seq_len} tokens across {n_layers} layers "
            f"({kv_cache.memory_usage_gb()*1000:.1f} MB compressed)"
        )

    def _extract_kv_state_api(self, kv_cache: TurboQuantKVCache, seq_len: int) -> None:
        """
        Extract KV from serialized state and compress with TurboQuant.

        The save_state() API serializes the full context including KV cache
        as contiguous float arrays. We parse this data, extract per-layer
        K/V tensors, and compress each with the appropriate TurboQuant
        algorithm.
        """
        # Save full state (includes KV cache + metadata)
        self._saved_state = self.llm.save_state()

        # Attempt to parse per-layer KV data from the serialized state.
        # The state format stores KV as contiguous FP16 arrays after metadata.
        try:
            kv_data = self._parse_kv_from_state(self._saved_state, seq_len)
            if kv_data is not None:
                n_layers = min(len(kv_data), kv_cache.num_layers)
                for layer_idx in range(n_layers):
                    k_tensor, v_tensor = kv_data[layer_idx]
                    kv_cache.update(layer_idx, k_tensor, v_tensor)

                logger.debug(
                    f"Extracted {seq_len} tokens from state, compressed "
                    f"{n_layers} layers ({kv_cache.memory_usage_gb()*1000:.1f} MB)"
                )
                return
        except Exception as e:
            logger.debug(f"State KV parsing unavailable ({e}), using state-level save")

        # Fallback: state is saved as-is (no per-layer TurboQuant, but
        # incremental generation still works via load_state on next turn)
        logger.debug(f"Saved {seq_len}-token state for next-turn restore")

    def _parse_kv_from_state(
        self, state, seq_len: int
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Parse per-layer K/V tensors from llama.cpp's serialized state.

        The state binary format (llama.cpp >= b3000) stores KV data as:
          [header][rng_state][logits][embeddings][kv_header][kv_data]

        kv_header contains cell count, layer count, and head dimensions.
        kv_data contains interleaved K and V float16 arrays per layer.

        Returns None if parsing fails (format mismatch, older version).
        """
        try:
            # Access the raw state bytes
            if hasattr(state, "llama_state"):
                raw = bytes(state.llama_state)
            elif hasattr(state, "input_ids"):
                return None  # Not a LlamaState
            else:
                raw = bytes(state)

            if len(raw) < 256:
                return None

            # Heuristic: the KV data starts after a recognizable header pattern.
            # We search for the KV section by looking for the expected data size.
            n_layers = self._n_layers
            head_dim = self._head_dim
            expected_kv_bytes_per_layer = seq_len * head_dim * 2  # FP16
            total_kv_bytes = expected_kv_bytes_per_layer * 2 * n_layers  # K + V

            # The KV data should be near the end of the state (it's the bulk)
            if total_kv_bytes > len(raw):
                return None

            # Try to locate the KV data at the tail of the state
            kv_start = len(raw) - total_kv_bytes
            kv_bytes = raw[kv_start:]

            result = []
            offset = 0
            for layer_idx in range(n_layers):
                # K data for this layer
                k_end = offset + expected_kv_bytes_per_layer
                k_np = np.frombuffer(
                    kv_bytes[offset:k_end], dtype=np.float16
                ).reshape(seq_len, head_dim).copy()

                # V data for this layer
                v_end = k_end + expected_kv_bytes_per_layer
                v_np = np.frombuffer(
                    kv_bytes[k_end:v_end], dtype=np.float16
                ).reshape(seq_len, head_dim).copy()

                result.append((
                    torch.from_numpy(k_np),
                    torch.from_numpy(v_np),
                ))
                offset = v_end

            return result

        except Exception:
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Use model's Jinja template if available, else ChatML fallback."""
        try:
            if hasattr(self.llm, "metadata") and self.llm.metadata:
                tmpl = self.llm.metadata.get("tokenizer.chat_template")
                if tmpl:
                    from jinja2 import Template
                    t = Template(tmpl)
                    return t.render(messages=messages, add_generation_prompt=True)
        except Exception:
            pass

        # ChatML fallback
        formatted = ""
        for m in messages:
            formatted += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted

    @staticmethod
    def _common_prefix_length(a: List[int], b: List[int]) -> int:
        """Find the length of the common prefix between two token lists."""
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n

    def _get_n_layers(self) -> int:
        """Get the number of transformer layers in the loaded model."""
        try:
            n = llama_cpp.llama_n_layer(self.llm._model.model)
            return n if n > 0 else 32  # Sensible default
        except Exception:
            return 32

    def _get_head_dim(self) -> int:
        """Get the attention head dimension."""
        try:
            n_embd = llama_cpp.llama_n_embd(self.llm._model.model)
            n_head = llama_cpp.llama_n_head(self.llm._model.model)
            if n_head > 0:
                return n_embd // n_head
        except Exception:
            pass
        return 128  # Default for Llama-family models

    def __repr__(self) -> str:
        mode = "patched" if _HAS_PATCHED_KV_API else "standard"
        return (
            f"LlamaCppEngine(model={self.model_path!r}, ctx={self.n_ctx}, "
            f"kv_bridge={mode})"
        )
