70B Agent on a Gaming PC
========================

This guide walks through running Llama-3.1-70B-Instruct with 128k context
on a single RTX 4090 (24GB VRAM) using TurboQuant's 4.9x KV compression.

Hardware Requirements
---------------------

* NVIDIA RTX 4090 (24GB VRAM) or RTX 5090
* 64GB+ system RAM (for CPU offloading of remaining layers)
* 50GB+ disk space (for model weights)

Setup
-----

.. code-block:: bash

   pip install turboagent-ai[llama,native]

The ``[native]`` extra installs ``turboquant-kv`` for optimal 4.9x compression.
Without it, the fallback PyTorch quantizer achieves ~1.5-2x.

Running the Agent
-----------------

.. code-block:: python

   from turboagent import TurboAgent

   # Auto-detects RTX 4090: n_gpu_layers=40, context=131072, turbo3
   agent = TurboAgent(
       "meta-llama/Llama-3.1-70B-Instruct",
       kv_mode="turbo3",
       context=131072,
   )

   # The KV cache for 128k tokens at 70B (80 layers) would normally
   # require ~20GB in FP16. With turbo3, it uses ~4GB.
   response = agent.run(
       "Analyze the following 50,000 word research paper and "
       "suggest three novel experiments..."
   )
   print(response)

Memory Budget
-------------

With an RTX 4090 (24GB VRAM):

* Model weights (Q4_K_M): ~40GB total, ~40 layers on GPU (~12GB)
* KV cache (turbo3 compressed): ~4GB for 128k tokens
* Remaining VRAM: ~8GB for activations and overhead
* CPU RAM: remaining ~40 layers offloaded

See ``examples/70b_gaming_pc_agent.py`` for a complete runnable script.
