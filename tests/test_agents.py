"""
Tests for turboagent.agents: TurboAgent, TurboVectorStore, SwarmAgent/TurboSwarm.

Unit tests cover the components that don't require a live LLM engine.
Integration tests (marked @pytest.mark.integration) require a running backend.
"""

import pytest
import torch

from turboagent.agents.rag import TurboVectorStore
from turboagent.agents.swarm import SwarmAgent, TurboSwarm


# ===========================================================================
# TurboVectorStore (RAG)
# ===========================================================================


class TestTurboVectorStore:
    @pytest.fixture
    def store(self):
        return TurboVectorStore(embedding_dim=64, device="cpu")

    def test_empty_store(self, store):
        assert store.size == 0
        assert store.query(torch.randn(64)) == []

    def test_add_documents(self, store):
        texts = ["hello world", "foo bar baz"]
        embeddings = [torch.randn(64), torch.randn(64)]
        count = store.add_documents(texts, embeddings)
        assert count == 2
        assert store.size == 2

    def test_add_documents_length_mismatch_raises(self, store):
        with pytest.raises(ValueError, match="same length"):
            store.add_documents(["a", "b"], [torch.randn(64)])

    def test_query_returns_top_k(self, store):
        texts = [f"doc {i}" for i in range(10)]
        embeddings = [torch.randn(64) for _ in range(10)]
        store.add_documents(texts, embeddings)

        results = store.query(torch.randn(64), top_k=3)
        assert len(results) == 3
        for r in results:
            assert "text" in r
            assert "score" in r
            assert "metadata" in r
            assert "index" in r

    def test_query_top_k_clamped_to_size(self, store):
        store.add_documents(["a", "b"], [torch.randn(64), torch.randn(64)])
        results = store.query(torch.randn(64), top_k=100)
        assert len(results) == 2

    def test_query_ranking_order(self, store):
        """A document whose embedding is identical to the query should rank first."""
        target = torch.randn(64)
        noise = torch.randn(64) * 0.01
        store.add_documents(
            ["noise", "target"],
            [noise, target],
        )
        results = store.query(target, top_k=2)
        assert results[0]["text"] == "target"

    def test_clear(self, store):
        store.add_documents(["x"], [torch.randn(64)])
        store.clear()
        assert store.size == 0

    def test_add_text_with_chunking(self, store):
        long_text = "abcdefgh" * 200  # 1600 chars
        mock_embed = lambda chunks: [torch.randn(64) for _ in chunks]
        count = store.add_text(long_text, embed_fn=mock_embed)
        assert count > 1  # Should produce multiple chunks
        assert store.size == count

    def test_chunk_text_overlap(self):
        store = TurboVectorStore(embedding_dim=64, chunk_size=10, chunk_overlap=3, device="cpu")
        chunks = store._chunk_text("a" * 25)
        # With size=10 and overlap=3, stride=7: positions 0,7,14,21
        assert len(chunks) == 4

    def test_repr(self, store):
        r = repr(store)
        assert "TurboVectorStore" in r
        assert "64" in r


# ===========================================================================
# SwarmAgent
# ===========================================================================


class TestSwarmAgent:
    def test_defaults(self):
        agent = SwarmAgent(name="researcher", role="research")
        assert agent.name == "researcher"
        assert agent.role == "research"
        assert "researcher" in agent.system_prompt
        assert agent.turn_count == 0
        assert agent.tools == []

    def test_custom_system_prompt(self):
        agent = SwarmAgent(name="a", role="b", system_prompt="Custom prompt.")
        assert agent.system_prompt == "Custom prompt."

    def test_repr(self):
        agent = SwarmAgent(name="critic", role="review")
        r = repr(agent)
        assert "critic" in r
        assert "review" in r


# ===========================================================================
# TurboSwarm (unit-level — no engine needed for these)
# ===========================================================================


class TestTurboSwarmValidation:
    def test_empty_agents_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            TurboSwarm(model_id="dummy", agents=[])

    def test_is_task_complete_markers(self):
        assert TurboSwarm._is_task_complete("Here is the answer. [DONE]")
        assert TurboSwarm._is_task_complete("[TASK_COMPLETE]")
        assert TurboSwarm._is_task_complete("Result: [FINAL_ANSWER]")
        assert not TurboSwarm._is_task_complete("Still working on it...")


# ===========================================================================
# TurboAgent (integration — needs live engine)
# ===========================================================================


@pytest.mark.integration
class TestTurboAgentIntegration:
    """These tests require a real model + backend. Skip in CI without GPU."""

    def test_agent_run_returns_string(self):
        from turboagent.agents.base import TurboAgent

        agent = TurboAgent("meta-llama/Llama-3.1-8B-Instruct", context=2048)
        response = agent.run("Say hello.")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_multi_turn_preserves_history(self):
        from turboagent.agents.base import TurboAgent

        agent = TurboAgent("meta-llama/Llama-3.1-8B-Instruct", context=2048)
        agent.run("My name is Alice.")
        response = agent.run("What is my name?")
        assert "Alice" in response

    def test_reset_clears_state(self):
        from turboagent.agents.base import TurboAgent

        agent = TurboAgent("meta-llama/Llama-3.1-8B-Instruct", context=2048)
        agent.run("Remember the code: OMEGA-774.")
        agent.reset_memory()
        assert len(agent.history) == 1  # Only system prompt
        assert agent.current_kv_tokens == 0
