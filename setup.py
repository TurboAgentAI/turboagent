"""Legacy setup.py fallback for environments that don't support pyproject.toml."""

from setuptools import setup, find_packages

setup(
    name="turboagent-ai",
    version="1.1.0",
    packages=find_packages(include=["turboagent*"]),
    install_requires=[
        "torch>=2.5.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "llama": ["llama-cpp-python>=0.3.0"],
        "vllm": ["vllm>=0.7.0"],
        "torch": ["transformers>=4.40.0", "huggingface_hub>=0.23.0"],
        "native": ["turboquant-kv>=0.2.0"],
        "server": ["fastapi>=0.115.0", "uvicorn[standard]>=0.32.0"],
        "enterprise": ["turboagent-enterprise>=0.1.0"],
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=5.0",
            "pytest-timeout>=2.3",
            "black>=24.0",
            "ruff>=0.5.0",
            "mypy>=1.10",
        ],
    },
    entry_points={
        "console_scripts": ["turboagent = turboagent.cli:main"],
    },
    python_requires=">=3.10",
    description="TurboQuant-powered agentic AI framework for long-context LLMs on consumer hardware",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TurboAgentAI/turboagent",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
