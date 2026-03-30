#!/bin/bash
# =============================================================================
# TurboAgent Vast.ai Setup Script
# Target: RTX PRO 6000 (96GB VRAM) — Llama-3.1-70B-Instruct validation
# =============================================================================
set -e

echo "============================================"
echo "TurboAgent Vast.ai Setup"
echo "============================================"

# 1. System info
echo ""
echo "[1/6] System info"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 --version

# 2. Clone or upload the repo (adjust as needed)
echo ""
echo "[2/6] Setting up TurboAgent"
cd /workspace
if [ ! -d "TurboAgent" ]; then
    echo "Please upload/clone TurboAgent to /workspace/TurboAgent"
    echo "e.g.: git clone https://github.com/turboagent/turboagent.git TurboAgent"
    exit 1
fi
cd TurboAgent

# 3. Install dependencies
echo ""
echo "[3/6] Installing dependencies"
pip install -e ".[dev,torch]" --quiet
pip install turboquant-kv --quiet
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --quiet 2>/dev/null || \
    pip install llama-cpp-python --quiet 2>/dev/null || \
    echo "WARNING: llama-cpp-python failed to install (llama.cpp tests will be skipped)"

# 4. Download models
echo ""
echo "[4/6] Downloading models"

# Torch: Llama-3.1-70B-Instruct (will use HF cache, ~140GB download)
python3 -c "
from transformers import AutoTokenizer
print('Pre-downloading Llama-3.1-70B-Instruct tokenizer...')
AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-70B-Instruct', trust_remote_code=True)
print('Tokenizer cached.')
"

# GGUF: Download Q4_K_M quant of 70B (~40GB)
python3 -c "
from huggingface_hub import hf_hub_download
print('Downloading Llama-3.1-70B-Instruct Q4_K_M GGUF (~40GB)...')
path = hf_hub_download(
    repo_id='bartowski/Meta-Llama-3.1-70B-Instruct-GGUF',
    filename='Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf',
)
print(f'GGUF downloaded to: {path}')
"

# 5. Run unit tests first
echo ""
echo "[5/6] Running unit tests"
python3 -m pytest tests/ -v -m "not integration" --timeout=60

# 6. Run the 70B integration suite
echo ""
echo "[6/6] Running 70B integration tests"
python3 -m pytest vastai/test_70b_integration.py -v -s --timeout=600

echo ""
echo "============================================"
echo "Setup and tests complete!"
echo "============================================"
