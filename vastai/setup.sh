#!/bin/bash
# =============================================================================
# TurboAgent Vast.ai Setup Script
# Target: RTX PRO 6000 (96GB VRAM)
#
# Usage:
#   bash vastai/setup.sh               # Qwen2.5-32B (default, ungated, fits 96GB BF16)
#   bash vastai/setup.sh qwen72        # Qwen2.5-72B (needs >144GB for BF16)
#   bash vastai/setup.sh llama         # Llama-3.1-70B (gated, needs HF approval)
# =============================================================================
set -e

MODEL_CHOICE="${1:-qwen32}"
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

if [ "$MODEL_CHOICE" = "llama" ]; then
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

else
    # Default: Qwen2.5-32B — fits in 96GB BF16, ungated
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
fi

# 5. Unit tests
echo ""
echo "[5/6] Running unit tests"
python3 -m pytest tests/ -v -m "not integration" --timeout=60

# 6. Integration tests
echo ""
echo "[6/6] Running integration tests (TURBO_TEST_MODEL=$MODEL_CHOICE)"
python3 -m pytest vastai/test_70b_integration.py -v -s --timeout=600

echo ""
echo "============================================"
echo "All tests complete!"
echo "============================================"
