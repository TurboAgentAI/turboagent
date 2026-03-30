import logging
from typing import Any, Dict, List, Optional, Callable

from turboagent.quant.turboquant import TurboQuantKVCache
from turboagent.hardware.detector import HardwareDetector
# Assuming a unified factory exists in backends/
from turboagent.backends import create_engine 

logger = logging.getLogger("turboagent.agents")

class TurboAgent:
    """
    Persistent, TurboQuant-backed agent with long-context memory.

    This class orchestrates multi-turn conversations, tool usage, and RAG
    by seamlessly managing a compressed KV cache across the agent's lifespan.

    This is part of the open-source TurboAgent core (MIT license).
    Commercial extensions (SSO, audit logging, advanced swarm governance) are
    available via TurboAgent Enterprise: https://turboagent.dev/enterprise
    """

    def __init__(
        self,
        model_id: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        # 1. Hardware-Aware Auto-Tuning
        self.config = HardwareDetector.get_optimal_config()
        self.config.update(kwargs)  # Allow explicit developer overrides

        logger.info(f"Initializing TurboAgent with model: {model_id}")
        logger.debug(f"Active Hardware/Backend Config: {self.config}")

        # 2. Unified Backend Engine
        # Auto-selects llama.cpp, vLLM, or pure Torch based on detector
        self.engine = create_engine(model_id, **self.config)

        # 3. Initialize the Compressed KV Cache
        self.kv_cache = TurboQuantKVCache(
            bit_mode=self.config.get("kv_mode", "turbo3"),
            max_context=self.config.get("context", 131072)
        )

        # 4. Multi-Turn Memory Structures
        self.system_prompt = system_prompt or "You are a helpful, analytical AI assistant."
        self.history: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        self.episodic_memory: Dict[str, Any] = {}  # For persistent RAG/snapshot states
        self.current_kv_tokens = 0

        # 5. Enterprise Extensions (feature-flagged)
        self._enterprise_audit = None
        try:
            from turboagent.enterprise.license import is_enterprise_license_valid, has_feature
            if is_enterprise_license_valid() and has_feature("audit_logging"):
                from turboagent.enterprise.audit import AuditLogger
                self._enterprise_audit = AuditLogger()
                logger.info("Enterprise audit logging enabled.")
        except ImportError:
            pass

    def run(self, prompt: str, tools: Optional[List[Callable]] = None) -> str:
        """
        Executes a multi-turn reasoning step. 
        
        Injects context, manages tool calls, and updates the shared, 
        zero-eviction compressed KV cache seamlessly. [cite: 333, 334]
        """
        self.history.append({"role": "user", "content": prompt})

        # Pre-flight Reality Check: Prevent hard OOM crashes
        # Even with 6x compression, hardware limits exist.
        estimated_new_tokens = self._estimate_tokens(prompt)
        if self.current_kv_tokens + estimated_new_tokens > self.kv_cache.max_context:
            logger.warning("Approaching maximum hardware context limit. Triggering episodic compression...")
            self._compress_history_to_episodic()

        # Engine Generation Phase
        # The engine handles the tokenization and interacts with the stateful self.kv_cache
        # to avoid re-computing past states.
        response, turn_metrics = self.engine.generate_chat(
            messages=self.history,
            kv_cache=self.kv_cache,
            tools=tools
        )

        # Update State
        self.history.append({"role": "assistant", "content": response})
        self.current_kv_tokens = turn_metrics.get("total_tokens_cached", self.current_kv_tokens + estimated_new_tokens)

        # Enterprise audit trail
        if self._enterprise_audit:
            self._enterprise_audit.log(
                actor="TurboAgent",
                action="run",
                detail=f"prompt_len={len(prompt)}, response_len={len(response)}",
                metadata=turn_metrics,
            )

        return response

    def _estimate_tokens(self, text: str) -> int:
        """A fast heuristic calculation to pre-flight memory safety (~4 chars per token)."""
        return len(text) // 4

    def _compress_history_to_episodic(self) -> None:
        """
        Safety fallback: when approaching the hardware context limit, preserve
        the system prompt and recent turns while compressing older messages
        into an episodic summary that gets prepended to future prompts.

        Strategy (sliding window with summary):
          1. Keep the system prompt (index 0).
          2. Summarize the oldest N messages into a compact text block.
          3. Store the summary in episodic_memory.
          4. Remove the old messages from history.
          5. Inject a synthetic "context_summary" message at position 1.
          6. Clear the KV cache so it rebuilds from the trimmed history.
        """
        if len(self.history) <= 3:
            # Too few messages to compress — nothing to prune
            return

        # Keep system prompt + last 4 messages (2 user-assistant pairs)
        n_keep_tail = 4
        system_msg = self.history[0]
        tail = self.history[-n_keep_tail:]
        middle = self.history[1:-n_keep_tail]

        if not middle:
            return

        # Build a compact summary of the pruned messages
        summary_parts = []
        for msg in middle:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate long messages to keep the summary tight
            if len(content) > 200:
                content = content[:200] + "..."
            summary_parts.append(f"[{role}]: {content}")

        summary_text = "Previous conversation summary:\n" + "\n".join(summary_parts)

        # Store in episodic memory (accumulates across compressions)
        compression_id = f"compression_{len(self.episodic_memory)}"
        self.episodic_memory[compression_id] = {
            "n_messages_compressed": len(middle),
            "summary": summary_text,
        }

        # Rebuild history: system + summary + recent tail
        summary_msg = {"role": "system", "content": summary_text}
        self.history = [system_msg, summary_msg] + tail

        # Reset KV token count — the engine will reprocess the shorter history
        self.current_kv_tokens = self._estimate_tokens(
            " ".join(m["content"] for m in self.history)
        )
        self.kv_cache.clear()

        logger.info(
            f"Compressed {len(middle)} messages into episodic memory. "
            f"History reduced to {len(self.history)} messages, "
            f"~{self.current_kv_tokens} tokens."
        )
        
    def reset_memory(self) -> None:
        """Clears the KV cache and conversation history."""
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.episodic_memory.clear()
        self.current_kv_tokens = 0
        self.kv_cache.clear()