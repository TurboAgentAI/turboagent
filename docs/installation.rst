Installation
============

Requirements
------------

* Python >= 3.10
* PyTorch >= 2.5.0
* NVIDIA GPU with CUDA 12.4+ (recommended), Apple Silicon, or CPU

Install from PyPI
-----------------

Choose the extras for your backend:

.. code-block:: bash

   # Core + llama.cpp (recommended for consumer GPUs)
   pip install turboagent-ai[llama]

   # With HuggingFace Transformers (research / custom models)
   pip install turboagent-ai[torch]

   # With vLLM (high-throughput server)
   pip install turboagent-ai[vllm]

   # With native TurboQuant C++/CUDA kernels (4.9x compression)
   pip install turboagent-ai[native]

   # API server
   pip install turboagent-ai[server]

   # Everything
   pip install turboagent-ai[torch,native,server]

Development Install
-------------------

.. code-block:: bash

   git clone https://github.com/TurboAgentAI/turboagent.git
   cd turboagent
   pip install -e ".[dev,torch,native]"
   pytest tests/ -v -m "not integration"

Verify Installation
-------------------

.. code-block:: bash

   turboagent info

This prints your hardware specs and recommended configuration for 7B, 13B, and
70B models.

Docker
------

.. code-block:: bash

   docker compose up

See :doc:`guides/cloud_deployment` for full Docker configuration.
