Cloud Deployment
================

TurboAgent includes an OpenAI-compatible API server for cloud/self-hosted
deployment.

Quick Start
-----------

.. code-block:: bash

   pip install turboagent-ai[server,torch,native]
   turboagent serve --model Qwen/Qwen2.5-32B-Instruct --port 8000

Docker
------

.. code-block:: bash

   docker compose up

The default ``docker-compose.yml`` configures:

* GPU passthrough (all NVIDIA GPUs)
* Persistent HuggingFace model cache
* Environment-based configuration

Configuration
-------------

All settings via environment variables:

.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``TURBOAGENT_MODEL``
     - (required)
     - HuggingFace model ID or local path
   * - ``TURBOAGENT_BACKEND``
     - auto
     - ``llama.cpp``, ``torch``, or ``vllm``
   * - ``TURBOAGENT_KV_MODE``
     - ``turbo3``
     - ``turbo3`` (4.9x) or ``turbo4`` (3.8x)
   * - ``TURBOAGENT_API_KEYS``
     - (empty)
     - Comma-separated API keys (empty = no auth)
   * - ``TURBOAGENT_RATE_LIMIT``
     - ``60``
     - Requests per minute per key

API Endpoints
-------------

.. list-table::
   :header-rows: 1

   * - Method
     - Path
     - Description
   * - POST
     - ``/v1/chat/completions``
     - Chat completion (OpenAI-compatible)
   * - GET
     - ``/v1/models``
     - List available models
   * - GET
     - ``/health``
     - Health check with compression stats
   * - DELETE
     - ``/v1/sessions/{id}``
     - Delete a session and free KV memory

Client Example
--------------

.. code-block:: python

   from openai import OpenAI

   client = OpenAI(
       base_url="http://localhost:8000/v1",
       api_key="sk-your-key",  # or "not-needed" if auth disabled
   )

   response = client.chat.completions.create(
       model="turboagent",
       messages=[{"role": "user", "content": "Hello!"}],
   )
   print(response.choices[0].message.content)

Multi-Turn Sessions
-------------------

Use the ``X-Session-ID`` header to maintain persistent KV cache across requests:

.. code-block:: bash

   # Turn 1
   curl http://localhost:8000/v1/chat/completions \
     -H "X-Session-ID: session-123" \
     -d '{"messages": [{"role": "user", "content": "My name is Alice."}]}'

   # Turn 2 — KV cache is restored from compressed storage
   curl http://localhost:8000/v1/chat/completions \
     -H "X-Session-ID: session-123" \
     -d '{"messages": [..., {"role": "user", "content": "What is my name?"}]}'
