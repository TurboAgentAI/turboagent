"""
TurboAgent CLI: project scaffolding and hardware benchmarking.

Entry point: `turboagent` (registered in pyproject.toml [project.scripts]).

Commands:
    turboagent init [project_name]   - Scaffold a new agent project
    turboagent benchmark [model_id]  - Run hardware + quality benchmarks
    turboagent info                  - Print detected hardware and optimal config
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from turboagent.version import __version__

logger = logging.getLogger("turboagent.cli")


def cmd_init(args: argparse.Namespace) -> None:
    """Scaffold a new TurboAgent project directory."""
    project_name = args.project_name
    project_dir = Path.cwd() / project_name

    if project_dir.exists():
        print(f"Error: Directory '{project_name}' already exists.")
        sys.exit(1)

    print(f"Scaffolding TurboAgent project: {project_name}")

    # Create directory structure
    dirs = [
        project_dir,
        project_dir / "agents",
        project_dir / "data",
        project_dir / "notebooks",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Main agent script
    (project_dir / "main.py").write_text(
        f'''"""
{project_name} — powered by TurboAgent.
"""

from turboagent import TurboAgent

agent = TurboAgent(
    "meta-llama/Llama-3.1-70B-Instruct",
    kv_mode="turbo3",
    context=131072,
)

if __name__ == "__main__":
    response = agent.run("Hello! What can you help me with today?")
    print(response)
''',
        encoding="utf-8",
    )

    # Requirements file
    (project_dir / "requirements.txt").write_text(
        "turboagent[llama]>=0.1.0\n",
        encoding="utf-8",
    )

    # Dockerfile for reproducibility
    (project_dir / "Dockerfile").write_text(
        f"""FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "main.py"]
""",
        encoding="utf-8",
    )

    # .gitignore
    (project_dir / ".gitignore").write_text(
        "*.gguf\n*.bin\n__pycache__/\n.env\ndata/\n",
        encoding="utf-8",
    )

    print(f"Project scaffolded at: {project_dir}")
    print(f"  cd {project_name}")
    print(f"  pip install turboagent-ai[llama]")
    print(f"  python main.py")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run hardware detection + TurboQuant quality/speed benchmarks."""
    from turboagent.hardware.detector import HardwareDetector

    print("=" * 60)
    print(f"TurboAgent Benchmark Suite v{__version__}")
    print("=" * 60)

    # 1. Hardware detection
    print("\n[1/3] Detecting hardware...")
    specs = HardwareDetector.get_system_specs()
    for k, v in specs.items():
        print(f"  {k}: {v}")

    config = HardwareDetector.get_optimal_config(
        target_model_size_b=args.model_size
    )
    print(f"\n  Optimal config for {args.model_size}B model:")
    print(f"  {json.dumps(config, indent=4)}")

    # 2. KV cache compression benchmark
    print("\n[2/3] Benchmarking TurboQuant KV compression...")
    try:
        import torch
        from turboagent.quant.turboquant import TurboQuantKVCache

        for mode in ["turbo3", "turbo4"]:
            cache = TurboQuantKVCache(
                bit_mode=mode, num_layers=1, head_dim=128, max_context=4096
            )
            k = torch.randn(4096, 128, dtype=torch.float16)
            v = torch.randn(4096, 128, dtype=torch.float16)

            t0 = time.perf_counter()
            cache.update(0, k, v)
            t_quant = time.perf_counter() - t0

            t0 = time.perf_counter()
            k_d, v_d = cache.get(0, 4096)
            t_dequant = time.perf_counter() - t0

            mse = torch.nn.functional.mse_loss(k.float(), k_d.float()).item()

            print(f"  [{mode}] Quantize: {t_quant*1000:.1f}ms | "
                  f"Dequantize: {t_dequant*1000:.1f}ms | MSE: {mse:.6f} | "
                  f"Cache: {cache.memory_usage_gb()*1000:.1f} MB")
    except Exception as e:
        print(f"  Skipped (torch unavailable): {e}")

    # 3. Validation suite pointer
    print("\n[3/3] Validation suite")
    print("  Run full validation with: pytest turboagent/quant/validators.py -v")

    print("\n" + "=" * 60)
    print("Benchmark complete.")


def cmd_info(args: argparse.Namespace) -> None:
    """Print hardware info and recommended configuration."""
    from turboagent.hardware.detector import HardwareDetector

    print(f"TurboAgent v{__version__}\n")

    specs = HardwareDetector.get_system_specs()
    print("Hardware:")
    for k, v in specs.items():
        print(f"  {k}: {v}")

    print("\nRecommended configs:")
    for size in [7.0, 13.0, 70.0]:
        config = HardwareDetector.get_optimal_config(target_model_size_b=size)
        print(f"  {size:.0f}B: {json.dumps(config)}")

    # Enterprise license status
    print("\nLicense:")
    try:
        from turboagent.enterprise.license import get_license_info
        info = get_license_info()
        print(f"  Tier: {info['tier']}")
        if info["valid"]:
            print(f"  Key: {info['key_prefix']}")
            print(f"  Features: {', '.join(info['features'])}")
        else:
            print("  Enterprise features: https://turboagent.to/enterprise")
    except ImportError:
        print("  Tier: community (open-source core)")
        print("  Enterprise features: https://turboagent.to/enterprise")


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the TurboAgent API server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install via: pip install turboagent-ai[server]")
        sys.exit(1)

    from turboagent.server import create_app

    api_keys = [k.strip() for k in args.api_keys.split(",") if k.strip()] if args.api_keys else []

    print(f"Starting TurboAgent API Server v{__version__}")
    print(f"  Model:    {args.model}")
    print(f"  Backend:  {args.backend or 'auto'}")
    print(f"  KV mode:  {args.kv_mode}")
    print(f"  Context:  {args.context or 'auto'}")
    print(f"  Auth:     {'enabled' if api_keys else 'disabled (open access)'}")
    print(f"  Rate:     {args.rate_limit} req/min")
    print(f"  Endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print()

    app = create_app(
        model_id=args.model,
        backend=args.backend or None,
        kv_mode=args.kv_mode,
        context=args.context,
        api_keys=api_keys,
        rate_limit=args.rate_limit,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="turboagent",
        description="TurboAgent: TurboQuant-powered agentic AI on consumer hardware",
    )
    parser.add_argument(
        "--version", action="version", version=f"turboagent {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command")

    # init
    p_init = subparsers.add_parser("init", help="Scaffold a new agent project")
    p_init.add_argument("project_name", help="Name of the project directory")
    p_init.set_defaults(func=cmd_init)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run hardware + quality benchmarks")
    p_bench.add_argument(
        "--model-size", type=float, default=70.0,
        help="Target model size in billions (default: 70)"
    )
    p_bench.set_defaults(func=cmd_benchmark)

    # info
    p_info = subparsers.add_parser("info", help="Print hardware info and optimal config")
    p_info.set_defaults(func=cmd_info)

    # serve
    p_serve = subparsers.add_parser("serve", help="Start the API server")
    p_serve.add_argument("--model", required=True, help="Model ID (HF hub or local path)")
    p_serve.add_argument("--backend", default="", help="Backend: llama.cpp, torch, vllm")
    p_serve.add_argument("--kv-mode", default="turbo3", help="turbo3 or turbo4")
    p_serve.add_argument("--context", type=int, default=None, help="Max context length")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p_serve.add_argument("--api-keys", default="", help="Comma-separated API keys (empty = no auth)")
    p_serve.add_argument("--rate-limit", type=int, default=60, help="Requests/min per key")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
