"""
Abstract base class for all TurboAgent inference backends.

Every backend (llama.cpp, vLLM, PyTorch) must implement this interface
so the factory in create_engine() and the agent classes can treat them
interchangeably.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from turboagent.quant.turboquant import TurboQuantKVCache


class BaseEngine(ABC):
    """
    Unified inference engine interface.

    All backends must implement `generate_chat()` with this exact signature
    so that TurboAgent, TurboSwarm, and the create_engine() factory can
    work with any backend transparently.
    """

    @abstractmethod
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        kv_cache: TurboQuantKVCache,
        tools: Optional[List[Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response from a chat message list.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            kv_cache: Shared TurboQuantKVCache for compressed KV state.
            tools: Optional list of tool callables for agentic workflows.

        Returns:
            Tuple of (response_text, metrics_dict).
            metrics_dict must include at minimum:
                - "turn_input_tokens": int
                - "turn_output_tokens": int
                - "total_tokens_cached": int
        """
        ...
