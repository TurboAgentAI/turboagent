#!/bin/bash
# =============================================================================
# TurboAgent Vast.ai Setup Script
#
# Usage:
#   bash vastai/setup.sh               # Gemma-4-31B-it (default, agentic-optimized)
#   bash vastai/setup.sh optionc       # Option C 16k NIAH — Qwen2.5-7B (24 GB+ GPU)
#   bash vastai/setup.sh qwen32        # Qwen2.5-32B
#   bash vastai/setup.sh qwen72        # Qwen2.5-72B (needs >144GB for BF16)
#   bash vastai/setup.sh llama         # Llama-3.1-70B (gated, needs HF approval)
#
# Option C validation (recommended: RTX 4090 24 GB, A10G, L4, or equivalent):
#   bash vastai/setup.sh optionc
#   NIAH_COMPARE_GPU=1 bash vastai/setup.sh optionc  # also run GPU path comparison
# =============================================================================
set -e

MODEL_CHOICE="${1:-gemma4}"
export TURBO_TEST_MODEL="$MODEL_CHOICE"

echo "============================================"
echo "TurboAgent Vast.ai Setup"
echo "Model: $MODEL_CHOICE"
echo "============================================"

# 1. System info
echo ""
echo "[1/6] System info"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 --version

# 2. Check repo
echo ""
echo "[2/6] Checking repo"
cd /workspace/TurboAgent || { echo "ERROR: /workspace/TurboAgent not found"; exit 1; }

# 3. Install dependencies
echo ""
echo "[3/6] Installing dependencies"
pip install -e ".[dev,torch]" --quiet
pip install accelerate --quiet
pip install turboquant-kv --quiet

# Try CUDA-enabled llama-cpp-python, fall back to CPU
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
    --quiet 2>/dev/null || \
pip install llama-cpp-python --quiet 2>/dev/null || \
    echo "WARNING: llama-cpp-python install failed (llama.cpp tests will skip)"

# 4. Download models
echo ""
echo "[4/6] Downloading models"

if [ "$MODEL_CHOICE" = "optionc" ]; then
    # Option C 16k NIAH validation — Qwen2.5-7B-Instruct
    # Hardware: any 24 GB+ GPU (RTX 4090, A10G, L4, RTX 6000 Ada, etc.)
    # Qwen2.5-7B BF16 weights: ~14 GB. 16k KV streaming: <1 GB on GPU.
    python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_id = 'Qwen/Qwen2.5-7B-Instruct'
print(f'Downloading {model_id} tokenizer...')
AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print(f'Pre-downloading {model_id} BF16 weights (~14 GB)...')
AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print('Weights cached.')
"

elif [ "$MODEL_CHOICE" = "llama" ]; then
    [ -z "$HF_TOKEN" ] && echo "WARNING: HF_TOKEN not set. Llama is gated."
    python3 -c "
from transformers import AutoTokenizer
print('Downloading Llama-3.1-70B tokenizer...')
AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-70B-Instruct', trust_remote_code=True)
print('Done.')
"
    python3 -c "
from huggingface_hub import hf_hub_download
print('Downloading Llama-3.1-70B GGUF Q4_K_M...')
p = hf_hub_download('bartowski/Meta-Llama-3.1-70B-Instruct-GGUF', 'Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf')
print(f'GGUF: {p}')
"

elif [ "$MODEL_CHOICE" = "qwen72" ]; then
    python3 -c "
from transformers import AutoTokenizer
print('Downloading Qwen2.5-72B tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-72B-Instruct', trust_remote_code=True)
print('Done.')
"
    python3 -c "
from huggingface_hub import hf_hub_download
print('Downloading Qwen2.5-72B GGUF Q4_K_M...')
p = hf_hub_download('bartowski/Qwen2.5-72B-Instruct-GGUF', 'Qwen2.5-72B-Instruct-Q4_K_M.gguf')
print(f'GGUF: {p}')
"

elif [ "$MODEL_CHOICE" = "qwen32" ]; then
    python3 -c "
from transformers import AutoTokenizer
print('Downloading Qwen2.5-32B tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-32B-Instruct', trust_remote_code=True)
print('Done.')
"
    python3 -c "
from huggingface_hub import hf_hub_download
print('Downloading Qwen2.5-32B GGUF Q4_K_M from bartowski...')
p = hf_hub_download('bartowski/Qwen2.5-32B-Instruct-GGUF', 'Qwen2.5-32B-Instruct-Q4_K_M.gguf')
print(f'GGUF: {p}')
"

else
    # Default: Gemma-4-31B-it BF16 (~62 GB) — agentic-optimized, 256k context
    # Pre-download the FULL BF16 weights, not just the tokenizer.
    # The torch backend uses BF16 — this is what the compression results
    # are measured against.
    python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print('Downloading Gemma-4-31B-it tokenizer...')
AutoTokenizer.from_pretrained('google/gemma-4-31B-it', trust_remote_code=True)
print('Pre-downloading Gemma-4-31B-it BF16 weights (~62 GB)...')
print('(This populates the HF cache so the test does not download mid-run.)')
AutoModelForCausalLM.from_pretrained(
    'google/gemma-4-31B-it',
    dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print('BF16 weights cached.')
"
    # Note: GGUF Q4_K_M download removed. llama.cpp tests fail in Vast.ai
    # containers due to missing libcudart.so.12. Torch backend uses BF16.
fi

# 5. Unit tests
echo ""
echo "[5/N] Running unit tests"
python3 -m pytest tests/ -v -m "not integration" --timeout=60

if [ "$MODEL_CHOICE" = "optionc" ]; then
    # Option C validation: skip integration tests, run the focused 16k NIAH script
    echo ""
    echo "[6/6] Running Option C 16k NIAH validation"
    echo "      Model: Qwen/Qwen2.5-7B-Instruct | Target: 16,000 tokens | 5 depths"
    echo "      NIAH_COMPARE_GPU=${NIAH_COMPARE_GPU:-0} (set to 1 to also run GPU path)"
    echo "      Expected runtime: ~10-20 minutes on a 24 GB GPU"
    python3 vastai/verify_option_c_16k.py

else
    # 6. Integration tests (basic functional, torch backend only)
    # llama.cpp tests are deselected because Vast.ai containers lack libcudart.so.12
    # Timeout is generous (1800s = 30 min) to account for model loading from
    # cache + GPU sharding across multiple devices.
    echo ""
    echo "[6/7] Running integration tests (TURBO_TEST_MODEL=$MODEL_CHOICE)"
    python3 -m pytest vastai/test_70b_integration.py -v -s --timeout=1800 \
        --deselect vastai/test_70b_integration.py::TestLlamaCppLargeModel

    # 7. Long-context NIAH benchmark (the real test)
    echo ""
    echo "[7/7] Running long-context NIAH benchmark (4k -> 96k tokens)"
    echo "      This will take 30-60 minutes."
    echo "      Override lengths via: export NIAH_LENGTHS=4000,16000,32000,65000,96000"
    python3 -m pytest vastai/test_long_context_niah.py -v -s --timeout=3600
fi

echo ""
echo "============================================"
echo "All tests complete!"
echo "============================================"
