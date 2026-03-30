# Vast.ai 70B Validation

## Quick Start

1. Rent an RTX PRO 6000 (96GB VRAM) instance on [vast.ai](https://vast.ai)
2. Upload or clone the TurboAgent repo to `/workspace/TurboAgent`
3. Run the setup script:

```bash
cd /workspace/TurboAgent
chmod +x vastai/setup.sh
bash vastai/setup.sh
```

This will:
- Install all dependencies (turboagent, turboquant-kv, llama-cpp-python)
- Download Llama-3.1-70B-Instruct (HuggingFace + GGUF Q4_K_M)
- Run unit tests
- Run the 70B integration test suite

## Running Tests Manually

```bash
# All 70B tests
python -m pytest vastai/test_70b_integration.py -v -s --timeout=600

# Individual test classes
python -m pytest vastai/test_70b_integration.py::TestHardwareDetection -v -s
python -m pytest vastai/test_70b_integration.py::TestTorch70B -v -s --timeout=600
python -m pytest vastai/test_70b_integration.py::TestTorch70BLongContext -v -s --timeout=600
python -m pytest vastai/test_70b_integration.py::TestLlamaCpp70B -v -s --timeout=600
python -m pytest vastai/test_70b_integration.py::TestMemoryProfiling -v -s
python -m pytest vastai/test_70b_integration.py::TestTurboAgentHighLevel -v -s --timeout=600
```

## What the Tests Validate

| Test | What it proves |
|------|---------------|
| `TestHardwareDetection` | 96GB GPU detected, 70B config uses full GPU offload |
| `TestTorch70B::test_single_turn_coherent` | 70B generates correct answer ("Paris") with TurboQuant KV |
| `TestTorch70B::test_multi_turn_coherent` | **Critical**: 70B recalls "PHOENIX-42" after KV compress/decompress cycle |
| `TestTorch70B::test_compression_ratio_70b` | Measures actual compression ratio on 70B-scale KV data |
| `TestTorch70BLongContext::test_needle_in_haystack_8k` | Retrieves hidden fact from 8k-token haystack with compressed KV |
| `TestLlamaCpp70B::test_single_turn_70b_gguf` | 70B GGUF computes 15*17=255 correctly |
| `TestLlamaCpp70B::test_multi_turn_70b_gguf` | Recalls "Biscuit" (golden retriever) across turns |
| `TestMemoryProfiling::test_vram_usage_with_compression` | Measures 80-layer KV compression: expects 4.9x with native |
| `TestTurboAgentHighLevel::test_turboagent_one_liner` | Full TurboAgent API works with 70B + auto-config |

## Expected Results

With `turboquant-kv` installed (native bit-packing):
- **Compression ratio**: ~4.9x (turbo3) matching the paper
- **Multi-turn coherence**: 70B should correctly recall facts after KV injection
- **Needle-in-haystack**: Should retrieve "NEPTUNE-7749" from 8k context
- **Speed**: 70B at ~10-20 tokens/s decode on RTX PRO 6000

## Notes

- The Llama-3.1-70B-Instruct model requires accepting Meta's license on HuggingFace
- Set `HF_TOKEN` environment variable if the model is gated: `export HF_TOKEN=your_token`
- GGUF download is ~40GB, HF model is ~140GB — ensure sufficient disk space
- First run will be slow (model download); subsequent runs use cached models
