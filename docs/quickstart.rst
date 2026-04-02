Quick Start
===========

One-Line Agent
--------------

.. code-block:: python

   from turboagent import TurboAgent

   agent = TurboAgent("meta-llama/Llama-3.1-70B-Instruct")
   response = agent.run("What is the capital of France?")
   print(response)

TurboAgent automatically:

1. Detects your hardware (GPU, VRAM, RAM)
2. Selects the optimal backend (llama.cpp, vLLM, or PyTorch)
3. Configures TurboQuant KV cache compression (turbo3 or turbo4)
4. Manages multi-turn conversation memory

Custom Configuration
--------------------

.. code-block:: python

   agent = TurboAgent(
       "meta-llama/Llama-3.1-70B-Instruct",
       backend="llama.cpp",
       kv_mode="turbo3",       # 4.9x compression
       context=131072,          # 128k context window
       n_gpu_layers=40,         # Layers to offload to GPU
       system_prompt="You are a helpful research assistant.",
   )

Multi-Turn Conversations
------------------------

.. code-block:: python

   agent = TurboAgent("meta-llama/Llama-3.1-70B-Instruct")

   # Turn 1
   agent.run("My name is Alice and I'm researching quantum computing.")

   # Turn 2 — agent remembers context via compressed KV cache
   response = agent.run("What field am I researching?")
   # "You're researching quantum computing, Alice."

   # Reset when done
   agent.reset_memory()

Scaffolding a Project
---------------------

.. code-block:: bash

   turboagent init my_agent
   cd my_agent
   pip install turboagent-ai[llama]
   python main.py

This creates a project with ``main.py``, ``Dockerfile``, ``requirements.txt``,
and a ``.gitignore``.

Hardware Benchmarking
---------------------

.. code-block:: bash

   turboagent benchmark --model-size 70

Runs TurboQuant compression benchmarks on your hardware and reports
quantization speed, MSE, and memory usage for turbo3 and turbo4 modes.

Next Steps
----------

* :doc:`guides/gaming_pc_agent` — Run 70B on an RTX 4090
* :doc:`guides/multi_agent_swarm` — Multi-agent collaboration
* :doc:`guides/rag_pipeline` — RAG with TurboVectorStore
* :doc:`guides/cloud_deployment` — Deploy as an API server
