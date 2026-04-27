# TurboAgent

**Long-context agentic AI on consumer hardware. NF4 4-bit weights + streaming compressed KV cache.**

TurboAgent is a pip-installable Python framework that combines NF4 weight quantization (via bitsandbytes) with TurboQuant KV-cache compression and CPU-side streaming. The result: **Qwen2.5-32B runs at 65k context on a single 24 GB GPU**, with 100% retrieval accuracy at every needle depth tested. Built on Google Research's [TurboQuant](https://arxiv.org/abs/2504.19874) algorithms.

## v1.1.0 highlights

- **15/15 NIAH PASS — Qwen2.5-32B-Instruct at 16k / 32k / 65k context on a single RTX 4090 (NF4 + Streaming KV)**
- **50/50 NIAH PASS** across Qwen2.5-7B, Qwen2.5-32B, and Gemma 4 31B at lengths up to 96k tokens (BF16 + Streaming KV)
- Decode throughput on the headline single-4090 NF4 run: **111 tok/s @ 16k → 91 @ 32k → 59 @ 65k**
- 5.22x–5.80x measured KV cache compression with zero intra-turn quantization noise
- Pinned-memory async prefetch and 16-token block-allocated buffers; single-GPU and multi-GPU validated
- Reproducible: see [`examples/niah/`](examples/niah/) for the exact eval script

See [`RELEASE_NOTES.md`](RELEASE_NOTES.md) for the full validation matrix.

## Quick Start — 32B on a single 4090

```bash
pip install "turboagent-ai[torch,bnb,native]==1.1.0"
```

```python
from turboagent.backends.torch import TorchEngine
from turboagent.quant.turboquant import TurboQuantKVCache

engine = TorchEngine(
    model_id="Qwen/Qwen2.5-32B-Instruct",
    quantize_weights="nf4",          # 4-bit weights via bitsandbytes
    kv_storage="cpu_streaming",      # compressed KV on CPU, streamed per-layer
)

flat_dim = engine._n_kv_heads * engine._head_dim
cache = TurboQuantKVCache(
    bit_mode="turbo3", device="cuda",
    head_dim=flat_dim, num_layers=engine._n_layers, max_context=66000,
)

response, metrics = engine.generate_chat(
    [{"role": "user", "content": "Summarize this 60k-token document..."}],
    cache,
)
# Peak VRAM at 65k context: ~21.5 GiB / 24 GiB
# KV cache: ~3.2 GB compressed (vs ~16.7 GB FP16)
```

The hardware detector picks NF4 + `cpu_streaming` automatically when it sees a 30B+ model on a 24 GB GPU; pass the kwargs explicitly to override.

## Reproduce the v1.1.0 numbers

```bash
# NF4 + Streaming KV — single 24 GB GPU, headline result
python examples/niah/run_niah.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --quantize-weights nf4 \
    --kv-storage cpu_streaming \
    --lengths 16000,32000,65000

# BF16 + Streaming KV — multi-GPU workstation
python examples/niah/run_niah.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --kv-storage cpu_streaming \
    --lengths 16000,32000,65000,96000
```

See [`examples/niah/README.md`](examples/niah/README.md) for the full protocol, hardware requirements, and what the benchmark does and doesn't validate.

## Validation matrix

| Model | Hardware | VRAM | Max context | NIAH | Compression |
|---|---|---|---|---|---|
| **Qwen2.5-32B (NF4)** | Single RTX 4090 | 24 GB | **65,000 tokens** | 15/15 (100%) | 5.28x |
| Qwen2.5-32B (BF16) | 2× RTX PRO 6000 | 192 GB | 96,000 tokens | 25/25 (100%) | 5.28x |
| Gemma 4 31B (BF16) | 2× RTX PRO 6000 | 192 GB | 96,000 tokens | 20/20 (100%) | 5.80x |
| Qwen2.5-7B (BF16) | Single RTX 4090 | 24 GB | 16,000 tokens | 5/5 (100%) | 5.22x |

**Total: 65/65 NIAH depths PASS.** NIAH tests retrieval fidelity; downstream tasks (LongBench, RULER full suite, agent-loop coherence) are planned for v1.2 evaluation.

## Installation

```bash
# Core + PyTorch backend + bitsandbytes (NF4) + native bit-pack kernels
pip install "turboagent-ai[torch,bnb,native]==1.1.0"

# Server (FastAPI / OpenAI-compatible)
pip install "turboagent-ai[server,torch]"

# llama.cpp consumer-GPU backend
pip install "turboagent-ai[llama]"

# vLLM server-throughput backend
pip install "turboagent-ai[vllm]"

# Development
pip install "turboagent-ai[dev]"
```

## CLI

```bash
turboagent init my_agent          # scaffold a new agent project
turboagent info                   # detect hardware, show recommended config
turboagent benchmark --model-size 70
```

## Multi-Agent Swarms

```python
from turboagent.agents.swarm import TurboSwarm, SwarmAgent

swarm = TurboSwarm(
    "Qwen/Qwen2.5-32B-Instruct",
    quantize_weights="nf4",
    agents=[
        SwarmAgent(name="researcher", role="deep research"),
        SwarmAgent(name="critic",     role="critical review"),
        SwarmAgent(name="writer",     role="clear writing"),
    ],
)
results = swarm.run("Analyze the latest advances in KV cache compression.")
```

Agents share a single compressed KV pool — ~5x less memory than independent contexts.

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
├── quant/          TurboQuantKVCache + StreamingDynamicCache (PolarQuant + QJL)
├── backends/       PyTorch (TorchEngine, NF4-aware), llama.cpp, vLLM
├── agents/         TurboAgent, TurboVectorStore, TurboSwarm
├── hardware/       Auto-detection + NF4 / streaming-KV recommendation
├── cli.py          Project scaffolding and benchmarks
└── utils.py        Shared helpers
```

## TurboQuant compression modes

| Mode | Bits/value | Compression | 32B KV @ 65k | Best for |
|---|---|---|---|---|
| **turbo3** | 3.25 bpv | 5.22–5.80x measured | ~3.2 GB | Maximum context on limited VRAM |
| turbo4 | 4.25 bpv | 3.8x | ~4.4 GB | Higher fidelity, ample memory |
| FP16 baseline | 16 bpv | 1x | ~16.7 GB | Reference, no compression |

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.5.0
- One of: `transformers` (recommended for NF4), `llama-cpp-python`, or `vLLM`
- For NF4 weight quantization: `bitsandbytes ≥ 0.41.0` (install via `[bnb]` extra)
- For maximum KV-cache compression: `turboquant-kv ≥ 0.2.0` (install via `[native]` extra)

## Development

```bash
git clone https://github.com/TurboAgentAI/turboagent.git
cd turboagent
pip install -e ".[dev,torch,bnb,native]"
pytest tests/ -v -m "not integration"
```

## Roadmap (v1.2)

- BitTorch fine-tuning integration (QLoRA / LoRA / DPO on the same NF4-quantized base)
- FlexTensor automatic GPU↔CPU weight offloading
- Llama-3.1-70B target: 200k+ context on a single 4090, 3–8 tok/s decode
- Tauri/React desktop GUI
- Expanded evaluation: LongBench, RULER full suite, agent-loop coherence, head-to-head throughput vs. vLLM / llama.cpp / ExLlama2

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

Multi-GPU validation performed on Vast.ai (2× RTX PRO 6000 Blackwell, 192 GiB).
