"""
70B Agent on a Gaming PC — TurboAgent Example

Runs Llama-3.1-70B-Instruct with 128k context on an RTX 4090 (24GB VRAM)
using TurboQuant's 4.9x KV cache compression.

Requirements:
    pip install turboagent-ai[llama,native]

Hardware:
    - RTX 4090 / 5090 (24GB+ VRAM)
    - 64GB+ system RAM
"""

from turboagent import TurboAgent

# Auto-detects hardware and configures:
#   backend=llama.cpp, n_gpu_layers=40, context=131072, kv_mode=turbo3
agent = TurboAgent(
    "meta-llama/Llama-3.1-70B-Instruct",
    system_prompt="You are a helpful research assistant with deep analytical skills.",
)

# Multi-turn conversation with persistent compressed KV cache
print("=== Turn 1: Set context ===")
response = agent.run(
    "I'm working on a research paper about transformer attention mechanisms. "
    "The key finding is that TurboQuant achieves 4.9x KV cache compression "
    "with zero accuracy loss by using PolarQuant + QJL residual coding."
)
print(response)

print("\n=== Turn 2: Build on context ===")
response = agent.run(
    "Based on what I just told you, suggest three follow-up experiments "
    "that could extend this work."
)
print(response)

print("\n=== Turn 3: Verify memory ===")
response = agent.run("What was the compression ratio I mentioned earlier?")
print(response)

# Show compression stats
print(f"\n=== KV Cache Stats ===")
print(f"Compressed: {agent.kv_cache.memory_usage_gb() * 1000:.1f} MB")
print(f"FP16 baseline: {agent.kv_cache.fp16_baseline_gb() * 1000:.1f} MB")
if agent.kv_cache.fp16_baseline_gb() > 0:
    ratio = agent.kv_cache.fp16_baseline_gb() / agent.kv_cache.memory_usage_gb()
    print(f"Compression ratio: {ratio:.1f}x")
