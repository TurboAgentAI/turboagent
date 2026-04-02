import logging
from typing import Any

logger = logging.getLogger("turboagent.backends")

def create_engine(model_id: str, backend: str = "llama.cpp", **kwargs) -> Any:
    """
    Factory function to instantiate the correct inference engine.
    
    Lazy-loads the backend to respect optional dependencies (e.g., avoiding 
    vLLM imports if the user only installed the llama.cpp extra).
    
    Args:
        model_id: The HuggingFace hub ID or local path to the model.
        backend: "llama.cpp", "vllm", "torch", or "hybrid".
        **kwargs: Hardware config injected by HardwareDetector (n_gpu_layers, context, etc.)
        
    Returns:
        An instantiated Engine object adhering to the BaseEngine interface.
    """
    logger.debug(f"Creating engine for '{model_id}' using backend '{backend}'")
    
    # Map "hybrid" to llama.cpp, as it handles CPU/GPU offloading best
    if backend == "hybrid":
        backend = "llama.cpp"

    if backend == "llama.cpp":
        try:
            from .llama_cpp import LlamaCppEngine
            return LlamaCppEngine(model_id, **kwargs)
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Install it via: pip install turboagent-ai[llama]"
            ) from e

    elif backend == "vllm":
        try:
            from .vllm import VLLMEngine
            return VLLMEngine(model_id, **kwargs)
        except ImportError as e:
            raise ImportError(
                "vLLM is not installed. "
                "Install it via: pip install turboagent-ai[vllm]"
            ) from e

    elif backend == "torch":
        try:
            from .torch import TorchEngine
            return TorchEngine(model_id, **kwargs)
        except ImportError as e:
            raise ImportError(
                "PyTorch is missing. "
                "Install it via: pip install turboagent-ai[torch]"
            ) from e

    elif backend == "mlx":
        raise NotImplementedError("MLX backend for Apple Silicon is planned for v2.0.")

    else:
        raise ValueError(f"Unknown backend '{backend}'. Supported: llama.cpp, vllm, torch.")