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
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from turboagent.quant.turboquant import TurboQuantKVCache
from turboagent.quant.streaming_kv import StreamingDynamicCache
from turboagent.backends.base import BaseEngine

logger = logging.getLogger("turboagent.backends.torch")

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError(
        "HuggingFace Transformers is not installed. "
        "Install it via: pip install turboagent-ai[torch]"
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
        kv_storage: KV storage strategy.
            ``"gpu"``            — classic path: decompress all layers to GPU
                                   upfront, inject as DynamicCache (v1.0).
            ``"cpu_streaming"``  — keep compressed KV on CPU,
                                   stream one layer at a time to GPU during the
                                   forward pass (StreamingDynamicCache).
            ``"auto"``           — choose ``"cpu_streaming"`` when estimated KV
                                   size exceeds available VRAM, else ``"gpu"``.
        **kwargs: Hardware config from HardwareDetector.
    """

    def __init__(
        self,
        model_id: str,
        kv_storage: Literal["auto", "gpu", "cpu_streaming"] = "auto",
        **kwargs,
    ):
        self.model_id = model_id
        self.kv_storage = kv_storage
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

        # Reduce CUDA memory fragmentation for long-context workloads.
        # max_split_size_mb prevents the allocator from creating large
        # unfreeable splits that lead to fragmentation OOMs.
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True,max_split_size_mb:256",
        )

        device_map = "auto" if n_gpu_layers == -1 else None

        # Build max_memory dict to control GPU/CPU weight sharding.
        #
        # Single GPU:
        #   device_map="auto" without a cap greedily fills GPU 0, leaving no
        #   room for KV cache or activations.  We always cap single-GPU usage
        #   to leave a headroom buffer.
        #   - cpu_streaming: KV lives on CPU (~few MB on GPU), reserve 0.5 GiB
        #   - gpu / auto:    KV lives on GPU, reserve 2 GiB for KV + activations
        #
        # Multi-GPU:
        #   Balance sharding across all GPUs at 55% each, as before.
        max_memory = None
        if device_map == "auto" and torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if n_gpus == 1:
                total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if kv_storage == "cpu_streaming":
                    # Streaming: KV never accumulates on GPU; only tiny compute
                    # buffers needed. Reserve 0.5 GiB for activations/fragmentation.
                    reserve_gib = 0.5
                else:
                    # GPU path: need headroom for KV cache + activations.
                    reserve_gib = 2.0
                cap_gib = max(int(total_gib - reserve_gib), int(total_gib * 0.75))
                max_memory = {0: f"{cap_gib}GiB", "cpu": "80GiB"}
                logger.info(
                    f"Single-GPU max_memory: {cap_gib} GiB "
                    f"(kv_storage={kv_storage!r}, reserve={reserve_gib} GiB)"
                )
            elif n_gpus > 1:
                max_memory = {}
                for i in range(n_gpus):
                    total_gib = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    cap_gib = int(total_gib * 0.55)  # leave 45% for activations
                    max_memory[i] = f"{cap_gib}GiB"
                max_memory["cpu"] = "200GiB"
                logger.info(f"Multi-GPU balanced sharding via max_memory: {max_memory}")

        # Use SDPA (PyTorch's Flash Attention 2) by default — critical for
        # long-context inference. The default "eager" attention materializes
        # the full N^2 attention matrix which OOMs beyond ~8k tokens.
        attn_impl = kwargs.get("attn_implementation", "sdpa")

        # In transformers 5.x, attn_implementation="sdpa" passed to from_pretrained()
        # does not propagate to text_config._attn_implementation for composite models
        # like Gemma 4. The attention class is selected at __init__ time by reading
        # text_config._attn_implementation, so we must patch it BEFORE model construction.
        pre_config = None
        try:
            pre_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            pre_config._attn_implementation = attn_impl
            text_cfg_pre = getattr(pre_config, "text_config", None)
            if text_cfg_pre is not None:
                text_cfg_pre._attn_implementation = attn_impl
                logger.info(
                    f"Pre-patched text_config._attn_implementation={attn_impl!r} "
                    f"(transformers 5.x composite model fix)"
                )
            else:
                logger.info(f"Pre-patched config._attn_implementation={attn_impl!r}")
        except Exception as e:
            logger.warning(
                f"Could not pre-load config for attn_implementation patching "
                f"(will rely on kwarg only): {e}"
            )

        load_kwargs = dict(
            dtype=self.dtype,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
        if pre_config is not None:
            load_kwargs["config"] = pre_config
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        except (ValueError, ImportError) as e:
            logger.warning(
                f"Failed to load with {attn_impl} attention ({e}). "
                f"Falling back to eager (will OOM at long contexts)."
            )
            load_kwargs.pop("attn_implementation", None)
            load_kwargs.pop("config", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        # Verify what attention implementation was actually applied.
        # Check text_config first (used by composite models like Gemma 4);
        # fall back to top-level config.
        actual_attn = "unknown"
        try:
            text_cfg = getattr(self.model.config, "text_config", None)
            if text_cfg is not None:
                actual_attn = getattr(text_cfg, "_attn_implementation", "unknown")
                top_attn = getattr(self.model.config, "_attn_implementation", None)
                actual_attn = f"top:{top_attn} / text:{actual_attn}"
            else:
                actual_attn = getattr(self.model.config, "_attn_implementation", "unknown")
        except Exception:
            pass

        print(f"[TorchEngine] ATTENTION IMPL IN USE: {actual_attn}")
        logger.info(f"Model loaded. Attention implementation: {actual_attn}")

        if device_map is None:
            self.model.to(self.device)
        self.model.eval()

        # Chunked prefill — feed long prompts in slices to avoid OOM.
        # Smaller chunks = lower peak activation memory but more overhead.
        # 1024 is conservative for tight VRAM situations (Gemma 4 on 32GB cards).
        self.prefill_chunk_size = int(kwargs.get("prefill_chunk_size", 1024))

        # Inter-turn state
        self._prev_input_ids: Optional[torch.Tensor] = None

        # Detect actual KV dimensions by running a 1-token forward pass.
        # Inspects all layers to handle heterogeneous shapes (Gemma 3/4
        # interleaved sliding-window attention has varying KV dims per layer).
        (
            self._n_layers,
            self._n_kv_heads,
            self._head_dim,
            self._layer_shapes,
            self._layer_devices,
        ) = self._detect_kv_shape()
        logger.info(
            f"Detected: {self._n_layers} layers, most common shape: "
            f"{self._n_kv_heads} kv_heads x {self._head_dim} head_dim"
        )
        unique_devices = set(str(d) for d in self._layer_devices.values())
        if len(unique_devices) > 1:
            logger.info(
                f"Tensor parallelism detected: layers spread across "
                f"{len(unique_devices)} devices: {sorted(unique_devices)}"
            )

    def _detect_kv_shape(self) -> Tuple[int, int, int, Dict[int, Tuple[int, int]], Dict[int, torch.device]]:
        """
        Run a 1-token forward pass and inspect past_key_values for ALL layers
        to determine the (n_layers, n_kv_heads, head_dim_per_head).

        Inspects every layer because some models (Gemma 3/4 with interleaved
        sliding-window attention) have heterogeneous KV shapes across layers.
        Returns the maximum dimensions found and logs all unique shapes.
        """
        dummy = torch.tensor([[self.tokenizer.eos_token_id or 0]], device=self.device)
        with torch.inference_mode():
            out = self.model(input_ids=dummy, use_cache=True)
        pkv = out.past_key_values

        # Helper to extract (K, V) from any past_key_values format
        def _get_layer_kv(idx):
            if hasattr(pkv, "layers"):
                layer = pkv.layers[idx]
                return layer.keys, layer.values
            if hasattr(pkv, "key_cache"):
                return pkv.key_cache[idx], pkv.value_cache[idx]
            return pkv[idx][0], pkv[idx][1]

        # Count layers
        if hasattr(pkv, "layers"):
            n_layers = len(pkv.layers)
        elif hasattr(pkv, "key_cache"):
            n_layers = len(pkv.key_cache)
        else:
            n_layers = len(pkv)

        # Inspect every layer and record per-layer shape AND device
        layer_shapes: Dict[int, Tuple[int, int]] = {}
        layer_devices: Dict[int, torch.device] = {}
        unique_shapes: Dict[Tuple[int, int], int] = {}
        for i in range(n_layers):
            k, _ = _get_layer_kv(i)
            if k.dim() != 4:
                raise RuntimeError(
                    f"Unexpected K tensor shape at layer {i}: {k.shape}. "
                    f"Expected 4D (batch, heads, seq, dim)."
                )
            shape_key = (k.shape[1], k.shape[3])
            layer_shapes[i] = shape_key
            layer_devices[i] = k.device  # capture the actual device for this layer
            unique_shapes[shape_key] = unique_shapes.get(shape_key, 0) + 1

        # Log what we found
        logger.info(f"Inspected {n_layers} layers. Unique KV shapes:")
        for (kvh, hd), count in sorted(unique_shapes.items(), key=lambda x: -x[1]):
            logger.info(f"  ({kvh} kv_heads, {hd} head_dim) -> {count} layers")

        if len(unique_shapes) > 1:
            logger.warning(
                f"Heterogeneous KV shapes detected ({len(unique_shapes)} variants). "
                f"Per-layer quantizers will be created lazily."
            )

        # Return the most common shape (used as default sizing hint)
        most_common = max(unique_shapes.items(), key=lambda x: x[1])
        n_kv_heads, head_dim = most_common[0]

        return n_layers, n_kv_heads, head_dim, layer_shapes, layer_devices

    @staticmethod
    def _resolve_text_config(config):
        """
        Find the transformer config object.

        Modern multimodal models (Gemma 3n, Gemma 4, Llama 3.2 Vision) wrap
        the text-decoder config inside .text_config. Older models expose the
        params directly on the top-level config. Try the nested location
        first, then fall back to the top-level config.
        """
        for attr in ("text_config", "decoder_config", "language_config"):
            sub = getattr(config, attr, None)
            if sub is not None and hasattr(sub, "num_hidden_layers"):
                return sub
        if hasattr(config, "num_hidden_layers"):
            return config
        raise AttributeError(
            f"Could not find num_hidden_layers in {type(config).__name__}. "
            f"Tried: text_config, decoder_config, language_config, top-level."
        )

    # ------------------------------------------------------------------
    # Streaming KV injection
    # ------------------------------------------------------------------

    def _resolve_kv_storage(self, kv_cache: TurboQuantKVCache) -> str:
        """
        Resolve "auto" → "gpu" or "cpu_streaming" based on available VRAM.

        Estimates the decompressed KV size for this cache state.  If it
        exceeds 60% of free VRAM we switch to streaming automatically.
        """
        if self.kv_storage != "auto":
            return self.kv_storage

        if not torch.cuda.is_available():
            return "gpu"  # CPU-only: no streaming benefit

        # Estimate FP16 KV size for ALL layers at current seq_len
        seq_len = max(kv_cache._seq_len, 1)
        fp16_bytes = 0
        for layer_idx in range(self._n_layers):
            nkh, hd = self._layer_shapes.get(
                layer_idx, (self._n_kv_heads, self._head_dim)
            )
            # K + V, batch=1, dtype=2 bytes
            fp16_bytes += 2 * 1 * nkh * seq_len * hd * 2

        free_bytes = torch.cuda.mem_get_info()[0]  # (free, total)
        threshold = 0.60 * free_bytes

        if fp16_bytes > threshold:
            logger.info(
                f"KV size {fp16_bytes/1e9:.2f} GB > {threshold/1e9:.2f} GB "
                f"threshold ({free_bytes/1e9:.2f} GB free). "
                f"Switching to cpu_streaming."
            )
            return "cpu_streaming"

        return "gpu"

    def _build_streaming_cache(self, kv_cache: TurboQuantKVCache) -> StreamingDynamicCache:
        """
        Wrap an existing TurboQuantKVCache in a StreamingDynamicCache.

        The StreamingDynamicCache satisfies the full DynamicCache contract
        (``update()``, ``get_seq_length()``, ``__len__()``) and can be passed
        directly as ``past_key_values`` to any HuggingFace model.
        """
        return StreamingDynamicCache(
            turbo_cache=kv_cache,
            layer_shapes=self._layer_shapes,
            layer_devices=self._layer_devices,
            dtype=self.dtype,
            default_n_kv_heads=self._n_kv_heads,
            default_head_dim=self._head_dim,
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

        # Cap n_prev to what the TurboQuant cache actually holds.
        # If the cache is empty (e.g., a fresh cache passed after a previous
        # call used a different cache object), treat this as a fresh turn.
        if kv_cache._seq_len == 0:
            n_prev = 0
            new_input_ids = input_ids
        elif n_prev > 0:
            n_prev = min(n_prev, kv_cache._seq_len)
            new_input_ids = input_ids[:, n_prev:]

        # --- Phase 2: INJECT compressed KV as past_key_values ---
        storage_mode = self._resolve_kv_storage(kv_cache)
        past_key_values = None
        if storage_mode == "cpu_streaming":
            # Use StreamingDynamicCache even on the first
            # turn (n_prev == 0).  This prevents GPU KV accumulation during
            # long prefills: each layer's KV is compressed to CPU immediately
            # rather than building up 28 × seq × kv_heads × head_dim FP16 on GPU.
            #
            # With n_prev == 0 the cache starts empty; update() just stores new
            # tokens per-layer without decompressing any past.
            past_key_values = self._build_streaming_cache(kv_cache)
            if n_prev > 0:
                logger.info(
                    f"Streaming KV active: streaming {n_prev} past tokens from CPU "
                    f"({kv_cache.memory_usage_gb()*1000:.1f} MB compressed)"
                )
            else:
                logger.info("Streaming KV active: layer-wise prefill (first turn, no past)")
        elif n_prev > 0:
            past_key_values = self._inject_kv_from_cache(kv_cache, n_prev)

        # --- Phase 3: PREFILL new tokens ---
        # Chunked prefill: process the prompt in slices to bound peak memory.
        # This is mandatory for long contexts; without it, transformers
        # materializes (batch, seq, vocab) logits in FP32 which OOMs.
        n_new = new_input_ids.shape[1]
        chunk_size = self.prefill_chunk_size if self.prefill_chunk_size > 0 else n_new
        n_chunks = (n_new + chunk_size - 1) // chunk_size

        logger.info(
            f"Prefill: {n_new} new tokens in {n_chunks} chunks of {chunk_size} "
            f"(prefill_chunk_size={self.prefill_chunk_size})"
        )
        if n_chunks == 1 and n_new > 4096:
            logger.warning(
                f"WARNING: prefill is one big chunk of {n_new} tokens. "
                f"chunked prefill is not active. self.prefill_chunk_size={self.prefill_chunk_size}"
            )

        current_cache = past_key_values
        next_token_logits = None

        for chunk_idx, chunk_start in enumerate(range(0, n_new, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, n_new)
            chunk = new_input_ids[:, chunk_start:chunk_end]
            is_last_chunk = chunk_end == n_new

            logger.debug(
                f"  prefill chunk {chunk_idx + 1}/{n_chunks}: "
                f"tokens [{chunk_start}:{chunk_end}] (size {chunk.shape[1]})"
            )

            forward_kwargs = dict(
                input_ids=chunk,
                past_key_values=current_cache,
                use_cache=True,
                return_dict=True,
            )

            # Only the last chunk needs logits. Try to skip computing them
            # entirely for non-final chunks via num_logits_to_keep.
            if not is_last_chunk:
                # Try modern arg, then legacy arg, fall back to no flag
                for arg_name in ("num_logits_to_keep", "logits_to_keep"):
                    try:
                        with torch.inference_mode():
                            chunk_out = self.model(
                                **forward_kwargs, **{arg_name: 1}
                            )
                        break
                    except TypeError:
                        continue
                else:
                    with torch.inference_mode():
                        chunk_out = self.model(**forward_kwargs)
            else:
                # Final chunk — try num_logits_to_keep=1 to limit logits memory
                for arg_name in ("num_logits_to_keep", "logits_to_keep"):
                    try:
                        with torch.inference_mode():
                            chunk_out = self.model(
                                **forward_kwargs, **{arg_name: 1}
                            )
                        break
                    except TypeError:
                        continue
                else:
                    with torch.inference_mode():
                        chunk_out = self.model(**forward_kwargs)

            current_cache = chunk_out.past_key_values
            if is_last_chunk:
                next_token_logits = chunk_out.logits[:, -1, :].clone()

            del chunk_out
            # Do NOT call torch.cuda.empty_cache() between chunks.
            # On multi-GPU with async H2D transfers (non_blocking=True in
            # StreamingDynamicCache.update), calling empty_cache() while the
            # CUDA allocator is still managing in-flight DMA buffers causes
            # intermittent cudaErrorLaunchFailure at long contexts (32k+).
            # PyTorch's allocator recycles memory automatically; a single
            # empty_cache() after the full prefill loop is sufficient.

        # Single cache flush after entire prefill — safe here since all chunk
        # tensors are freed and no async transfers are in flight.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if next_token_logits is None:
            raise RuntimeError("Prefill produced no logits — empty input?")

        # Switch streaming cache to decode mode: compress prefill buffer → turbo
        # once, then freeze turbo for the duration of the decode loop.
        # This must happen BEFORE the first decode step so that get_seq_length()
        # returns the correct offset (turbo_seq = all prefill tokens).
        if storage_mode == "cpu_streaming" and isinstance(current_cache, StreamingDynamicCache):
            current_cache.start_decode()
            logger.info(
                f"StreamingDynamicCache: prefill → turbo "
                f"({kv_cache._seq_len} tokens, {kv_cache.memory_usage_gb()*1000:.1f} MB), "
                f"switching to decode mode"
            )

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
        # Streaming mode: finalize_decode() compresses the raw decode-token
        # buffer into turbo ONCE (avoids re-compression noise), then
        # kv_cache._seq_len is the authoritative token count.
        # Classic mode: extract from DynamicCache GPU tensors and compress now.
        if isinstance(current_cache, StreamingDynamicCache):
            current_cache.finalize_decode()
            actual_cache_seq_len = kv_cache._seq_len
            logger.debug(
                f"Streaming: decode tokens compressed into turbo "
                f"({actual_cache_seq_len} tokens, "
                f"{kv_cache.memory_usage_gb()*1000:.1f} MB)"
            )
        else:
            actual_cache_seq_len = self._get_cache_seq_len(current_cache)
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

        Uses per-layer shapes (`self._layer_shapes`) to handle models with
        heterogeneous KV dimensions across layers.
        """
        n_layers = min(self._n_layers, kv_cache.num_layers)

        def _build_layer(layer_idx: int):
            k_deq, v_deq = kv_cache.get(layer_idx, seq_len)
            nkh, hd = self._layer_shapes.get(
                layer_idx, (self._n_kv_heads, self._head_dim)
            )
            # Place KV on the same device as this layer's weights
            # (critical for tensor-parallel models with device_map="auto")
            dev = self._layer_devices.get(layer_idx, self.device)
            k = self._reshape_for_model(k_deq, seq_len, nkh, hd, device=dev)
            v = self._reshape_for_model(v_deq, seq_len, nkh, hd, device=dev)
            return k, v

        if DynamicCache is not None:
            cache = DynamicCache()
            for layer_idx in range(n_layers):
                k, v = _build_layer(layer_idx)
                cache.update(k, v, layer_idx)
            logger.debug(f"Injected {seq_len} tokens (DynamicCache)")
            return cache
        else:
            past = tuple(_build_layer(i) for i in range(n_layers))
            logger.debug(f"Injected {seq_len} tokens (tuple list)")
            return past

    # ------------------------------------------------------------------
    # KV Cache Bridge: EXTRACT (past_key_values → TurboQuant)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_kv_layer(past_key_values, layer_idx) -> tuple:
        """
        Extract (key, value) tensors for a layer from past_key_values.

        Handles multiple transformers versions:
          - Legacy: past_key_values[i] → (K, V)
          - DynamicCache (old): cache[i] → (K, V)
          - DynamicCache (new): cache.layers[i].keys / .values
        """
        # Try subscript first (works on legacy tuples and older DynamicCache)
        try:
            item = past_key_values[layer_idx]
            if isinstance(item, tuple) and len(item) == 2:
                return item
        except (TypeError, IndexError):
            pass

        # Try .layers attribute (newer DynamicCache with DynamicLayer)
        if hasattr(past_key_values, "layers"):
            layer = past_key_values.layers[layer_idx]
            return layer.keys, layer.values

        # Try to_legacy_cache conversion
        if hasattr(past_key_values, "to_legacy_cache"):
            legacy = past_key_values.to_legacy_cache()
            return legacy[layer_idx]

        raise TypeError(
            f"Cannot extract KV from {type(past_key_values).__name__}. "
            f"Unsupported past_key_values format."
        )

    @staticmethod
    def _get_cache_seq_len(past_key_values) -> int:
        """Get the sequence length stored in the cache."""
        try:
            k, _ = TorchEngine._get_kv_layer(past_key_values, 0)
            return k.shape[-2]
        except Exception:
            return 0

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

        for layer_idx in range(n_layers):
            k, v = self._get_kv_layer(past_key_values, layer_idx)

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

    def _reshape_for_model(
        self, flat: torch.Tensor, seq_len: int,
        n_kv_heads: Optional[int] = None, head_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Reshape (seq_len, n_kv_heads * head_dim) → (1, n_kv_heads, seq_len, head_dim)
        for injection into HuggingFace attention layers.

        If n_kv_heads / head_dim are not provided, falls back to detected values.
        Per-layer overrides handle:
          - heterogeneous KV shapes (Gemma 3/4 interleaved attention)
          - tensor parallelism (each layer's KV must live on its layer's device)
        """
        target_device = device if device is not None else self.device
        flat = flat.to(device=target_device, dtype=self.dtype)
        nkh = n_kv_heads if n_kv_heads is not None else self._n_kv_heads
        hd = head_dim if head_dim is not None else self._head_dim
        return flat.view(seq_len, nkh, hd).permute(1, 0, 2).unsqueeze(0)

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
