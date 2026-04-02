Enterprise Edition
==================

The open-source core is **free forever** under the MIT license.

TurboAgent Enterprise adds commercial extensions for teams and organizations.

Features
--------

* **SSO / SAML** authentication
* **Audit logging** — structured records with JSON/CSV compliance export
* **Air-gapped** on-premise licensing
* **SecureMultiAgentSwarm** — governance policies, RBAC, content filters
* **Multi-node** KV cache sharing
* **Priority kernel** optimizations
* **Dedicated support** SLAs

License Activation
------------------

.. code-block:: bash

   export TURBOAGENT_LICENSE_KEY="TA-ENT-your-key-here"

Or place the key in ``~/.turboagent/license.key``.

Usage
-----

.. code-block:: python

   from turboagent.enterprise.swarm import SecureMultiAgentSwarm
   from turboagent.enterprise.audit import AuditLogger

   # Enterprise features activate automatically with a valid key
   swarm = SecureMultiAgentSwarm(
       "meta-llama/Llama-3.1-70B-Instruct",
       agents=[...],
       governance_policy={"max_rounds": 10},
       audit_enabled=True,
   )

   results = swarm.run("Analyze the quarterly report.")
   print(swarm.export_audit_log(format="json"))

Contact
-------

* Enterprise inquiries: enterprise@turboagent.to
* Website: `turboagent.to/enterprise <https://turboagent.to/enterprise>`_
