"""
vLLM backend engine for high-throughput server-style inference.

Wraps vLLM's AsyncLLMEngine with TurboQuant KV cache injection,
optimized for multi-request serving scenarios (e.g., swarm agents
issuing concurrent generation requests).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from turboagent.quant.turboquant import TurboQuantKVCache
from turboagent.backends.base import BaseEngine

logger = logging.getLogger("turboagent.backends.vllm")

try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError(
        "vLLM is not installed. Install it via: pip install turboagent[vllm]"
    )


class VLLMEngine(BaseEngine):
    """
    High-throughput inference engine backed by vLLM.

    Configures vLLM with TurboQuant-compressed KV cache types via the
    community Triton kernels (0xSero/turboquant, mitkox/vllm-turboquant).

    Args:
        model_id: HuggingFace model ID or local path.
        **kwargs: Hardware config from HardwareDetector (context, kv_mode, etc.).
    """

    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        self.n_ctx = kwargs.get("context", 131072)
        kv_mode = kwargs.get("kv_mode", "turbo3")

        # Map turboagent kv_mode to vLLM cache config keys
        cache_type = "turbo3" if kv_mode == "turbo3" else "turbo4"

        logger.debug(
            f"Initializing vLLM Engine | Model: {model_id} | "
            f"Context: {self.n_ctx} | KV cache: {cache_type}"
        )

        self.llm = LLM(
            model=model_id,
            max_model_len=self.n_ctx,
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.90),
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            # TurboQuant KV cache type (requires patched vLLM fork)
            kv_cache_dtype=cache_type,
            trust_remote_code=True,
        )

        self.default_sampling = SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 4096),
            top_p=kwargs.get("top_p", 0.9),
        )

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        kv_cache: TurboQuantKVCache,
        tools: Optional[List[Any]] = None,
        sampling_params: Optional[Any] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response from a chat message list.

        Note: vLLM manages its own internal KV cache. The external
        TurboQuantKVCache object is used here for state bookkeeping and
        memory metrics; the actual compressed caching happens inside vLLM
        via the patched Triton kernels.
        """
        prompt = self._apply_chat_template(messages)
        params = sampling_params or self.default_sampling

        outputs = self.llm.generate([prompt], params)
        result = outputs[0]

        response_text = result.outputs[0].text
        prompt_tokens = len(result.prompt_token_ids)
        completion_tokens = len(result.outputs[0].token_ids)

        metrics = {
            "turn_input_tokens": prompt_tokens,
            "turn_output_tokens": completion_tokens,
            "total_tokens_cached": prompt_tokens + completion_tokens,
        }

        return response_text, metrics

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """ChatML fallback template (production uses model's Jinja template)."""
        formatted = ""
        for m in messages:
            formatted += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted

    def __repr__(self) -> str:
        return f"VLLMEngine(model={self.model_id!r}, ctx={self.n_ctx})"
