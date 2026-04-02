"""
Multi-Agent Research Swarm — TurboAgent Example

Three specialist agents collaborate on a research task, sharing a single
TurboQuant-compressed KV cache. Memory usage is ~1/5 of what independent
agent contexts would require.

Requirements:
    pip install turboagent-ai[llama,native]
"""

from turboagent.agents.swarm import TurboSwarm, SwarmAgent

# Define specialist agents
researcher = SwarmAgent(
    name="researcher",
    role="deep research and literature review",
    system_prompt=(
        "You are a research scientist. Your job is to provide thorough, "
        "evidence-based analysis. Cite specific findings when possible."
    ),
)

critic = SwarmAgent(
    name="critic",
    role="critical analysis and fact-checking",
    system_prompt=(
        "You are a peer reviewer. Identify weaknesses, missing evidence, "
        "and logical gaps in the research. Be constructive but rigorous."
    ),
)

writer = SwarmAgent(
    name="writer",
    role="clear scientific writing",
    system_prompt=(
        "You are a science writer. Take the research and critique, then "
        "produce a clear, well-structured summary. When the summary is "
        "complete, end with [DONE]."
    ),
)

# Create the swarm — all agents share one compressed KV cache
swarm = TurboSwarm(
    "meta-llama/Llama-3.1-70B-Instruct",
    agents=[researcher, critic, writer],
    max_rounds=6,
)

# Run the collaborative task
print("Starting multi-agent research swarm...\n")
results = swarm.run(
    "Analyze the potential of TurboQuant KV cache compression for enabling "
    "long-context agentic AI on consumer hardware. Cover the technical "
    "approach, practical implications, and limitations."
)

# Print the conversation
for msg in results:
    agent_name = msg.get("agent", "user")
    content = msg["content"][:500]
    print(f"[{agent_name}] {content}")
    print()

# Show swarm stats
print(f"=== Swarm Stats ===")
print(f"Total rounds: {len([m for m in results if m.get('agent') != 'user'])}")
print(f"KV cache: {swarm.kv_cache.memory_usage_gb() * 1000:.1f} MB compressed")
for name, agent in swarm.agents.items():
    print(f"  {name}: {agent.turn_count} turns")
