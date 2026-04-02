TurboAgent Documentation
========================

**TurboQuant-powered agentic AI framework for long-context LLMs on consumer hardware.**

TurboAgent brings Google Research's TurboQuant KV-cache compression (4.9x memory
reduction, zero accuracy loss) to open-source LLMs. Run 70B+ models with 128k+
context on a single consumer GPU.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/gaming_pc_agent
   guides/multi_agent_swarm
   guides/rag_pipeline
   guides/cloud_deployment

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/turboagent
   api/backends
   api/quant
   api/agents
   api/server

.. toctree::
   :maxdepth: 1
   :caption: Enterprise

   enterprise

Quick Example
-------------

.. code-block:: python

   from turboagent import TurboAgent

   agent = TurboAgent(
       "meta-llama/Llama-3.1-70B-Instruct",
       kv_mode="turbo3",
       context=131072,
   )
   response = agent.run("Analyze my research paper...")
   print(response)

Links
-----

* `GitHub <https://github.com/TurboAgentAI/turboagent>`_
* `PyPI <https://pypi.org/project/turboagent-ai/>`_
* `Enterprise <https://turboagent.to/enterprise>`_
