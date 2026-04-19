import os
import psutil
import platform
import logging
from typing import Dict, Any

# Delay torch import slightly if needed, but assuming it's a core dependency
import torch

logger = logging.getLogger("turboagent.hardware")

class HardwareDetector:
    """
    Profiles host hardware (VRAM, RAM, Compute Platform) to determine the 
    optimal TurboQuant backend and KV-cache configuration.
    """

    @classmethod
    def get_system_specs(cls) -> Dict[str, Any]:
        """Gathers raw hardware specifications."""
        specs = {
            "platform": platform.system(),
            "ram_gb": psutil.virtual_memory().total / (1024 ** 3),
            "has_cuda": False,
            "has_mps": False,
            "has_rocm": False,
            "vram_gb": 0.0,
            "vram_per_gpu_gb": 0.0,
            "n_gpus": 0,
            "gpu_name": "None"
        }

        # Check for NVIDIA/CUDA
        if torch.cuda.is_available():
            specs["has_cuda"] = True
            # Check if it's actually an AMD card running via ROCm disguised in PyTorch
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                specs["has_cuda"] = False
                specs["has_rocm"] = True

            n_gpus = torch.cuda.device_count()
            specs["n_gpus"] = n_gpus

            # Sum VRAM across all visible GPUs (multi-GPU instances)
            total_vram = 0.0
            per_gpu_vram = 0.0
            for i in range(n_gpus):
                gpu_vram = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                total_vram += gpu_vram
                per_gpu_vram = max(per_gpu_vram, gpu_vram)

            specs["vram_gb"] = total_vram
            specs["vram_per_gpu_gb"] = per_gpu_vram
            specs["gpu_name"] = torch.cuda.get_device_name(0)
            if n_gpus > 1:
                specs["gpu_name"] = f"{n_gpus}x {specs['gpu_name']}"

        # Check for Apple Silicon (Metal Performance Shaders)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            specs["has_mps"] = True
            # Apple Silicon uses unified memory, so VRAM roughly equals system RAM minus OS overhead
            specs["vram_gb"] = specs["ram_gb"] * 0.75 
            specs["gpu_name"] = "Apple Silicon"

        return specs

    @classmethod
    def get_optimal_config(cls, target_model_size_b: float = 70.0) -> Dict[str, Any]:
        """
        Calculates the safest, highest-performance configuration for a given model size.
        
        Args:
            target_model_size_b: Size of the model in billions of parameters (default 70 for Llama-3.1-70B).
        
        Returns:
            Dict containing the backend, kv_mode, n_gpu_layers, and safe context window.
        """
        specs = cls.get_system_specs()
        vram = specs["vram_gb"]
        ram = specs["ram_gb"]
        
        logger.info(f"Detected Hardware: {specs['gpu_name']} | VRAM: {vram:.1f}GB | System RAM: {ram:.1f}GB")

        # Baseline config (Safe CPU fallback)
        config = {
            "backend": "llama.cpp",
            "kv_mode": "turbo3",
            "n_gpu_layers": 0,
            "context": 8192,
            "offload_strategy": "cpu_only",
            "quantize_weights": None,
        }

        # --- HEURISTICS FOR 70B CLASS MODELS ---
        if target_model_size_b >= 60.0:
            if specs["has_cuda"] or specs["has_rocm"]:
                if vram >= 80.0:
                    # RTX PRO 6000 / A100 80GB / H100 (80GB+ Class)
                    # Can fit entire 70B model + massive context entirely on GPU
                    config.update({
                        "n_gpu_layers": -1,  # All layers on GPU
                        "context": 131072,
                        "kv_mode": "turbo3",
                        "offload_strategy": "gpu_only"
                    })
                elif vram >= 40.0:
                    # A6000 / dual-GPU configs (40-80GB Class)
                    # NF4: 70B → ~35 GB, fits on GPU with room for KV
                    config.update({
                        "n_gpu_layers": -1,
                        "context": 131072,
                        "kv_mode": "turbo4",
                        "quantize_weights": "nf4",
                        "offload_strategy": "gpu_only"
                    })
                elif vram >= 23.0:
                    # RTX 3090 / 4090 / 5090 (24GB+ Class)
                    # NF4: 70B → ~35 GB, needs hybrid offload but far more
                    # layers fit on GPU than FP16. Stream KV from CPU.
                    config.update({
                        "n_gpu_layers": -1,
                        "context": 131072,
                        "kv_mode": "turbo3",
                        "quantize_weights": "nf4",
                        "offload_strategy": "hybrid"
                    })
                elif vram >= 15.0:
                    # RTX 4080 (16GB Class)
                    # NF4 critical — without it, only ~25 layers fit
                    config.update({
                        "n_gpu_layers": -1,
                        "context": 65536,
                        "kv_mode": "turbo3",
                        "quantize_weights": "nf4",
                        "offload_strategy": "hybrid"
                    })
                else:
                    # 8GB-12GB VRAM (Heavy CPU offload required)
                    config.update({
                        "n_gpu_layers": 15,
                        "context": 32768,
                        "kv_mode": "turbo3",
                        "quantize_weights": "nf4",
                        "offload_strategy": "heavy_cpu"
                    })
            
            elif specs["has_mps"]:
                if ram >= 60.0:
                    # Mac Studio / Max (64GB+ Unified Memory)
                    # Can fit the entire 70B model + massive context natively
                    config.update({
                        "n_gpu_layers": -1, # Offload all to Metal
                        "context": 131072,
                        "kv_mode": "turbo4", # Plenty of memory, prioritize quality
                        "backend": "llama.cpp", # or 'mlx' if you build the MLX backend
                        "offload_strategy": "metal_unified"
                    })
                elif ram >= 30.0:
                    # Mac M-series Pro (32GB Unified Memory)
                    config.update({
                        "n_gpu_layers": -1,
                        "context": 32768,
                        "kv_mode": "turbo3",
                        "offload_strategy": "metal_unified"
                    })

        # --- HEURISTICS FOR SMALLER MODELS (e.g., 7B - 14B) ---
        else:
            if vram >= 6.0 or (specs["has_mps"] and ram >= 16.0):
                if vram >= 12.0 or (specs["has_mps"] and ram >= 32.0):
                    # 12GB+ VRAM or 32GB+ unified — full GPU, maximum context
                    config.update({
                        "n_gpu_layers": -1,
                        "context": 262144,
                        "kv_mode": "turbo4",
                        "offload_strategy": "gpu_only"
                    })
                else:
                    # 6-11GB VRAM (e.g., RTX 4060/4070) — full GPU, moderate context
                    config.update({
                        "n_gpu_layers": -1,
                        "context": 131072,
                        "kv_mode": "turbo3",
                        "offload_strategy": "gpu_only"
                    })

        # Sanity check: Ensure the system RAM can handle the CPU offload if needed
        if config["offload_strategy"] in ["hybrid", "heavy_cpu"] and ram < 32.0:
            logger.warning("Low system RAM detected for hybrid offloading. Enforcing strict context limit.")
            config["context"] = min(config["context"], 16384)

        return config

if __name__ == "__main__":
    # Quick debug run
    logging.basicConfig(level=logging.INFO)
    print("System Specs:", HardwareDetector.get_system_specs())
    print("\nOptimal 70B Config:", HardwareDetector.get_optimal_config(70.0))
    print("\nOptimal 8B Config:", HardwareDetector.get_optimal_config(8.0))