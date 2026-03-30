"""
Multi-agent swarm orchestration with shared TurboQuant KV pool.

Enables collaborative reasoning across specialist agents (e.g., researcher +
critic + executor) that share a single compressed KV cache, dramatically
reducing memory vs. independent agent contexts.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from turboagent.quant.turboquant import TurboQuantKVCache
from turboagent.hardware.detector import HardwareDetector
from turboagent.backends import create_engine

logger = logging.getLogger("turboagent.agents.swarm")


class SwarmAgent:
    """
    A lightweight agent participant within a TurboSwarm.

    Each SwarmAgent has its own role/system prompt and tool set but shares the
    swarm's KV cache and engine, so collaborative context is preserved across
    agent boundaries without duplication.
    """

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt or f"You are {name}, a specialist in {role}."
        self.tools = tools or []
        self.turn_count = 0

    def __repr__(self) -> str:
        return f"SwarmAgent(name={self.name!r}, role={self.role!r}, turns={self.turn_count})"


class TurboSwarm:
    """
    Multi-agent swarm with a shared, compressed KV pool.

    All agents in the swarm share one TurboQuantKVCache and one inference
    engine. Turns are orchestrated round-robin or via a custom routing
    function. The shared cache means Agent B can attend to context generated
    by Agent A without re-encoding — TurboQuant's inner-product fidelity
    ensures attention accuracy across agent boundaries.

    This is part of the open-source TurboAgent core (MIT license).
    For advanced governance, RBAC, audit logging, and compliance exports,
    see SecureMultiAgentSwarm in TurboAgent Enterprise:
    https://turboagent.dev/enterprise

    Args:
        model_id: HuggingFace hub ID or local path.
        agents: List of SwarmAgent participants.
        router: Optional callable(history, agents) -> SwarmAgent that picks
                the next agent to act. Defaults to round-robin.
        max_rounds: Safety cap on orchestration rounds.
        **kwargs: Forwarded to HardwareDetector / create_engine.
    """

    def __init__(
        self,
        model_id: str,
        agents: List[SwarmAgent],
        router: Optional[Callable] = None,
        max_rounds: int = 20,
        **kwargs,
    ):
        if not agents:
            raise ValueError("A swarm requires at least one SwarmAgent.")

        self.agents = {a.name: a for a in agents}
        self.agent_order = [a.name for a in agents]
        self.router = router
        self.max_rounds = max_rounds

        # Shared infrastructure
        config = HardwareDetector.get_optimal_config()
        config.update(kwargs)
        self.config = config

        logger.info(f"Initializing TurboSwarm with {len(agents)} agents on model: {model_id}")

        self.engine = create_engine(model_id, **config)
        self.kv_cache = TurboQuantKVCache(
            bit_mode=config.get("kv_mode", "turbo3"),
            max_context=config.get("context", 131072),
        )

        # Shared conversation ledger (visible to all agents)
        self.history: List[Dict[str, Any]] = []
        self.current_kv_tokens = 0

    # -- Orchestration --------------------------------------------------------

    def run(self, task: str, max_rounds: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Execute the swarm on a task. Each round, the router (or round-robin)
        selects an agent who generates a response visible to all others.

        Args:
            task: The initial task prompt.
            max_rounds: Override the default round cap.

        Returns:
            The full conversation ledger (list of message dicts).
        """
        rounds = max_rounds or self.max_rounds
        self.history.append({"role": "user", "content": task, "agent": "user"})

        for i in range(rounds):
            agent = self._select_next_agent(i)

            # Build agent-specific message list (inject agent's system prompt)
            messages = [{"role": "system", "content": agent.system_prompt}] + self.history

            response, metrics = self.engine.generate_chat(
                messages=messages,
                kv_cache=self.kv_cache,
                tools=agent.tools or None,
            )

            self.history.append({
                "role": "assistant",
                "content": response,
                "agent": agent.name,
                "round": i,
            })
            agent.turn_count += 1
            self.current_kv_tokens = metrics.get(
                "total_tokens_cached", self.current_kv_tokens
            )

            logger.debug(f"Round {i} | Agent: {agent.name} | Tokens cached: {self.current_kv_tokens}")

            # Check for natural termination (agent signals done)
            if self._is_task_complete(response):
                logger.info(f"Swarm completed task in {i + 1} rounds.")
                break

        return self.history

    # -- Agent routing --------------------------------------------------------

    def _select_next_agent(self, round_idx: int) -> SwarmAgent:
        """Pick the next agent via custom router or round-robin."""
        if self.router is not None:
            selected = self.router(self.history, list(self.agents.values()))
            if isinstance(selected, SwarmAgent):
                return selected
            if isinstance(selected, str) and selected in self.agents:
                return self.agents[selected]

        # Default: round-robin through agent_order
        name = self.agent_order[round_idx % len(self.agent_order)]
        return self.agents[name]

    @staticmethod
    def _is_task_complete(response: str) -> bool:
        """Heuristic check for task-complete signals in agent output."""
        markers = ["[DONE]", "[TASK_COMPLETE]", "[FINAL_ANSWER]"]
        return any(m in response for m in markers)

    # -- Management -----------------------------------------------------------

    def add_agent(self, agent: SwarmAgent) -> None:
        """Add an agent to the swarm at runtime."""
        self.agents[agent.name] = agent
        self.agent_order.append(agent.name)

    def remove_agent(self, name: str) -> None:
        """Remove an agent by name."""
        self.agents.pop(name, None)
        self.agent_order = [n for n in self.agent_order if n != name]

    def reset(self) -> None:
        """Clear all shared state."""
        self.history.clear()
        self.kv_cache.clear()
        self.current_kv_tokens = 0
        for agent in self.agents.values():
            agent.turn_count = 0

    def __repr__(self) -> str:
        names = ", ".join(self.agent_order)
        return f"TurboSwarm(agents=[{names}], rounds={self.max_rounds})"
