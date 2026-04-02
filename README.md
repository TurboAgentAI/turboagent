# TurboAgent

**TurboQuant-powered agentic AI framework for long-context LLMs on consumer hardware.**

TurboAgent is a pip-installable Python package that brings Google Research's [TurboQuant](https://arxiv.org/abs/2504.19874) KV-cache compression to open-source LLMs for local, consumer-hardware agentic AI. It delivers 6x+ memory reduction and up to 8x attention speedup with zero measurable accuracy loss.

## Features

- **One-line agent creation** with 6x+ KV compression -- 32k-1M+ effective context on a single RTX 4090
- **Hardware-aware auto-tuning** -- detects CUDA/ROCm/Metal/CPU and selects optimal configuration
- **Agentic-first primitives** -- persistent multi-turn memory, RAG with vector-search, multi-agent swarms
- **Multiple backends** -- llama.cpp (consumer GPUs), vLLM (server throughput), PyTorch (research)
- **Zero-calibration, training-free** -- just like the paper guarantees

## Quick Start

```bash
pip install turboagent-ai[llama]
```

```python
from turboagent import TurboAgent

agent = TurboAgent(
    "meta-llama/Llama-3.1-70B-Instruct",
    kv_mode="turbo3",
    context=131072,
)

response = agent.run("Analyze my 50k-token research doc and suggest experiments...")
print(response)  # KV usage <4 GB total
```

## Installation

```bash
# Core + llama.cpp backend (recommended for consumer GPUs)
pip install turboagent-ai[llama]

# With vLLM for server-style throughput
pip install turboagent-ai[vllm]

# With HuggingFace Transformers for research
pip install turboagent-ai[torch]

# With native TurboQuant C++/CUDA kernels (recommended for best performance)
pip install turboagent-ai[native]

# Development
pip install turboagent-ai[dev]
```

## CLI

```bash
# Scaffold a new agent project
turboagent init my_agent

# Detect hardware and show optimal configuration
turboagent info

# Run benchmarks
turboagent benchmark --model-size 70
```

## Multi-Agent Swarms

```python
from turboagent.agents.swarm import TurboSwarm, SwarmAgent

swarm = TurboSwarm(
    "meta-llama/Llama-3.1-70B-Instruct",
    agents=[
        SwarmAgent(name="researcher", role="deep research"),
        SwarmAgent(name="critic", role="critical review"),
        SwarmAgent(name="writer", role="clear writing"),
    ],
)

results = swarm.run("Analyze the latest advances in KV cache compression.")
```

## RAG with TurboVectorStore

```python
from turboagent.agents.rag import TurboVectorStore

store = TurboVectorStore(embedding_dim=768)
store.add_documents(texts=chunks, embeddings=embeddings)
results = store.query(query_embedding, top_k=5)
```

## Architecture

```
turboagent/
├── quant/          # TurboQuantKVCache (PolarQuant + QJL)
├── backends/       # llama.cpp, vLLM, PyTorch engines
├── agents/         # TurboAgent, TurboVectorStore, TurboSwarm
├── hardware/       # Auto-detection and optimal config
├── cli.py          # Project scaffolding and benchmarks
└── utils.py        # Shared helpers
```

## TurboQuant Compression Modes

| Mode    | Bits per Value | Compression | Best For                        |
|---------|---------------|-------------|---------------------------------|
| turbo3  | 3.25 bpv      | 4.9x        | Maximum context on limited VRAM |
| turbo4  | 4.25 bpv      | 3.8x        | Higher quality, ample memory    |

## Requirements

- Python >= 3.10
- PyTorch >= 2.5.0
- One of: llama-cpp-python, vLLM, or HuggingFace Transformers

## Development

```bash
git clone https://github.com/TurboAgentAI/turboagent.git
cd turboagent
pip install -e ".[dev]"
pytest tests/ -v -m "not integration"
```

## Enterprise

The open-source core is **free forever** under the [MIT license](LICENSE).

**TurboAgent Enterprise** adds commercial extensions for teams and organizations:

- **SSO / SAML** authentication
- **Audit logging** and compliance exports (SOC-2, GDPR)
- **Air-gapped** on-premise licensing
- **SecureMultiAgentSwarm** with governance policies and RBAC
- **Multi-node** KV cache sharing
- **Priority kernels** and dedicated support SLAs

```python
# Enterprise features activate with a license key
# export TURBOAGENT_LICENSE_KEY="TA-ENT-your-key-here"

from turboagent.enterprise.swarm import SecureMultiAgentSwarm
from turboagent.enterprise.audit import AuditLogger
```

Learn more: [turboagent.to/enterprise](https://turboagent.to/enterprise) | Contact: enterprise@turboagent.to

## License

[MIT](LICENSE) — the open-source core is free for commercial and personal use.
Commercial extensions are available under a separate license. See [Enterprise](#enterprise).

## Acknowledgments

Built on community TurboQuant implementations:
- [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) (PyTorch reference)
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (llama.cpp fork)
- [0xSero/turboquant](https://github.com/0xSero/turboquant) (vLLM Triton kernels)
- [turboquant-kv](https://pypi.org/project/turboquant-kv/) (C++/CUDA bindings)
