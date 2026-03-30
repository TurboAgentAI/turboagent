"""
Pure-PyTorch backend engine with TurboQuant KV cache bridge.

Uses HuggingFace Transformers with direct interception of the
past_key_values (DynamicCache) returned by the model's forward pass.

KV cache lifecycle:
  1. INJECT: Dequantize compressed KV → build DynamicCache → pass as
     past_key_values to model.forward()
  2. FORWARD: Model processes only new tokens (incremental prefill)
  3. GENERATE: Custom autoregressive loop with per-step KV extraction
  4. EXTRACT: Pull per-layer K/V from model output → compress via
     TurboQuant (K: Algorithm 1 MSE-optimal, V: Algorithm 2 Prod-optimal)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from turboagent.quant.turboquant import TurboQuantKVCache
from turboagent.backends.base import BaseEngine

logger = logging.getLogger("turboagent.backends.torch")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError(
        "HuggingFace Transformers is not installed. "
        "Install it via: pip install turboagent[torch]"
    )

# DynamicCache is the standard KV cache class in transformers >= 4.38
try:
    from transformers import DynamicCache
except ImportError:
    DynamicCache = None


class TorchEngine(BaseEngine):
    """
    Pure-PyTorch research backend with TurboQuant KV cache compression.

    Intercepts the model's past_key_values after each forward pass to
    compress per-layer K/V tensors with TurboQuant. On subsequent calls,
    the compressed cache is dequantized and injected as past_key_values,
    so the model only processes new tokens.

    Per the paper (arXiv 2504.19874):
      - K-cache: TurboQuantMSE (Algorithm 1) for MSE-optimal compression
      - V-cache: TurboQuantProd (Algorithm 2) for unbiased inner-product

    Args:
        model_id: HuggingFace model ID or local path.
        **kwargs: Hardware config from HardwareDetector.
    """

    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        self.n_ctx = kwargs.get("context", 131072)
        self.max_new_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.9)
        n_gpu_layers = kwargs.get("n_gpu_layers", -1)

        # Determine dtype and device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        logger.debug(
            f"Loading Torch Engine | Model: {model_id} | "
            f"Device: {self.device} | Dtype: {self.dtype}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = "auto" if n_gpu_layers == -1 else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=self.dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        if device_map is None:
            self.model.to(self.device)
        self.model.eval()

        # Inter-turn state
        self._prev_input_ids: Optional[torch.Tensor] = None
        self._n_layers = self.model.config.num_hidden_layers
        self._head_dim = getattr(
            self.model.config, "head_dim",
            self.model.config.hidden_size // self.model.config.num_attention_heads,
        )
        self._n_kv_heads = getattr(
            self.model.config, "num_key_value_heads",
            self.model.config.num_attention_heads,
        )

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
          1. Tokenize and find new tokens (diff from previous turn).
          2. Inject compressed KV as past_key_values.
          3. Prefill new tokens via model.forward().
          4. Autoregressive generation loop.
          5. Extract and compress final KV state.
        """
        prompt = self._apply_chat_template(messages)
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.n_ctx,
        )["input_ids"].to(self.device)

        # --- Phase 1: Determine new tokens ---
        n_prev, new_input_ids = self._compute_token_diff(input_ids)

        # Cap n_prev to what the TurboQuant cache actually holds
        # (the cache may have fewer tokens than the token-diff suggests,
        # because the model's cache length can differ from our token count)
        if n_prev > 0 and kv_cache._seq_len > 0:
            n_prev = min(n_prev, kv_cache._seq_len)
            new_input_ids = input_ids[:, n_prev:]

        # --- Phase 2: INJECT compressed KV as past_key_values ---
        past_key_values = None
        if n_prev > 0:
            past_key_values = self._inject_kv_from_cache(kv_cache, n_prev)

        # --- Phase 3: PREFILL new tokens ---
        with torch.inference_mode():
            prefill_out = self.model(
                input_ids=new_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

        # The model returns updated past_key_values containing ALL tokens
        # (injected + newly prefilled)
        current_cache = prefill_out.past_key_values
        next_token_logits = prefill_out.logits[:, -1, :]

        # --- Phase 4: GENERATE response tokens ---
        output_ids = []
        for _ in range(self.max_new_tokens):
            next_token = self._sample_token(next_token_logits)
            output_ids.append(next_token.item())

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Forward pass for next token (single token, using cache)
            # Ensure input_ids is exactly (batch=1, seq=1)
            step_input = next_token.view(1, 1)
            with torch.inference_mode():
                step_out = self.model(
                    input_ids=step_input,
                    past_key_values=current_cache,
                    use_cache=True,
                )
            current_cache = step_out.past_key_values
            next_token_logits = step_out.logits[:, -1, :]

        # --- Phase 5: EXTRACT and COMPRESS KV ---
        # The actual cache length is determined by what the model stored,
        # not our token count (last generated token may not be in cache yet).
        actual_cache_seq_len = current_cache[0][0].shape[-2]  # (batch, heads, seq, dim)
        self._extract_and_compress_kv(current_cache, kv_cache, actual_cache_seq_len)
        total_seq_len = actual_cache_seq_len

        # Update state tracking
        all_ids = torch.cat([
            input_ids,
            torch.tensor([output_ids], device=self.device),
        ], dim=1)
        self._prev_input_ids = all_ids

        response_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        metrics = {
            "turn_input_tokens": new_input_ids.shape[1],
            "turn_output_tokens": len(output_ids),
            "total_tokens_cached": total_seq_len,
            "kv_compressed_mb": kv_cache.memory_usage_gb() * 1000,
        }

        logger.debug(
            f"Generated {len(output_ids)} tokens | "
            f"Restored {n_prev} from TurboQuant cache | "
            f"Total: {total_seq_len} | "
            f"Compressed: {metrics['kv_compressed_mb']:.1f} MB"
        )

        return response_text, metrics

    # ------------------------------------------------------------------
    # KV Cache Bridge: INJECT (TurboQuant → past_key_values)
    # ------------------------------------------------------------------

    def _inject_kv_from_cache(self, kv_cache: TurboQuantKVCache, seq_len: int):
        """
        Dequantize compressed KV state and build past_key_values.

        Returns a DynamicCache (or list of tuples) suitable for passing
        to model.forward(past_key_values=...).
        """
        n_layers = min(self._n_layers, kv_cache.num_layers)

        if DynamicCache is not None:
            cache = DynamicCache()
            for layer_idx in range(n_layers):
                k_deq, v_deq = kv_cache.get(layer_idx, seq_len)

                # Reshape from (seq_len, head_dim) to (batch=1, n_kv_heads, seq_len, head_dim)
                k = self._reshape_for_model(k_deq, seq_len)
                v = self._reshape_for_model(v_deq, seq_len)

                cache.update(k, v, layer_idx)

            logger.debug(f"Injected {seq_len} tokens from TurboQuant cache (DynamicCache)")
            return cache
        else:
            # Fallback: list of (K, V) tuples
            past = []
            for layer_idx in range(n_layers):
                k_deq, v_deq = kv_cache.get(layer_idx, seq_len)
                k = self._reshape_for_model(k_deq, seq_len)
                v = self._reshape_for_model(v_deq, seq_len)
                past.append((k, v))

            logger.debug(f"Injected {seq_len} tokens from TurboQuant cache (tuple list)")
            return tuple(past)

    # ------------------------------------------------------------------
    # KV Cache Bridge: EXTRACT (past_key_values → TurboQuant)
    # ------------------------------------------------------------------

    def _extract_and_compress_kv(
        self, past_key_values, kv_cache: TurboQuantKVCache, seq_len: int
    ) -> None:
        """
        Extract per-layer K/V tensors from model output and compress.

        Per the paper (arXiv 2504.19874):
          - K tensors: compressed with TurboQuantMSE (Algorithm 1)
            Minimizes ||K - K_hat||^2 → optimal for Q·K^T attention scores
          - V tensors: compressed with TurboQuantProd (Algorithm 2)
            Unbiased inner-product preservation → optimal for attn_weights · V

        This separation is critical: using MSE-optimal for V would introduce
        bias in the attention output (proven in Section 3.2 of the paper).
        """
        n_layers = min(self._n_layers, kv_cache.num_layers)

        # Both DynamicCache and legacy tuple format support indexing:
        #   cache[layer_idx] → (key_tensor, value_tensor)
        # Each tensor has shape (batch, n_kv_heads, seq_len, head_dim)
        for layer_idx in range(n_layers):
            k, v = past_key_values[layer_idx]

            # Flatten to (seq_len, n_kv_heads * head_dim) for TurboQuant
            k_flat = self._flatten_from_model(k)
            v_flat = self._flatten_from_model(v)

            kv_cache.update(layer_idx, k_flat, v_flat)

        logger.debug(
            f"Compressed {seq_len} tokens across {n_layers} layers → "
            f"{kv_cache.memory_usage_gb()*1000:.1f} MB "
            f"(FP16 baseline: {kv_cache.fp16_baseline_gb()*1000:.1f} MB)"
        )

    # ------------------------------------------------------------------
    # Tensor reshaping: (seq, dim) ↔ (batch, heads, seq, head_dim)
    # ------------------------------------------------------------------

    def _reshape_for_model(self, flat: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Reshape (seq_len, n_kv_heads * head_dim) → (1, n_kv_heads, seq_len, head_dim)
        for injection into HuggingFace attention layers.
        """
        flat = flat.to(device=self.device, dtype=self.dtype)
        return flat.view(seq_len, self._n_kv_heads, self._head_dim).permute(1, 0, 2).unsqueeze(0)

    def _flatten_from_model(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reshape (1, n_kv_heads, seq_len, head_dim) → (seq_len, n_kv_heads * head_dim)
        for TurboQuant compression.
        """
        # (1, n_heads, seq, head_dim) → (seq, n_heads, head_dim) → (seq, n_heads * head_dim)
        t = tensor.squeeze(0)  # (n_heads, seq, head_dim)
        t = t.permute(1, 0, 2)  # (seq, n_heads, head_dim)
        return t.reshape(t.shape[0], -1).contiguous().half()

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _compute_token_diff(
        self, input_ids: torch.Tensor
    ) -> Tuple[int, torch.Tensor]:
        """
        Find the common prefix with previous turn's tokens.

        Returns (n_prev, new_tokens) where n_prev is how many tokens
        are already in the compressed cache, and new_tokens are the
        tokens that need fresh computation.
        """
        if self._prev_input_ids is None:
            return 0, input_ids

        prev = self._prev_input_ids.squeeze(0)
        curr = input_ids.squeeze(0)
        n = min(len(prev), len(curr))

        n_common = 0
        for i in range(n):
            if prev[i] != curr[i]:
                break
            n_common += 1

        new_tokens = input_ids[:, n_common:]

        if new_tokens.shape[1] == 0:
            # Identical prompt — return at least the last token to avoid empty forward
            new_tokens = input_ids[:, -1:]
            n_common = max(0, n_common - 1)

        return n_common, new_tokens

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next token with temperature and top-p."""
        if self.temperature <= 0:
            return logits.argmax(dim=-1)

        logits = logits / self.temperature

        # Top-p (nucleus) sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Use tokenizer's built-in template if available, else ChatML fallback."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        formatted = ""
        for m in messages:
            formatted += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted

    def __repr__(self) -> str:
        return (
            f"TorchEngine(model={self.model_id!r}, device={self.device}, "
            f"ctx={self.n_ctx}, layers={self._n_layers})"
        )
