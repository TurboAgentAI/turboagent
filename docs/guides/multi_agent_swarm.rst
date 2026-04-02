Multi-Agent Swarms
==================

TurboSwarm enables multiple specialist agents to collaborate on complex tasks
while sharing a single compressed KV cache — dramatically reducing memory
compared to independent agent contexts.

Basic Swarm
-----------

.. code-block:: python

   from turboagent.agents.swarm import TurboSwarm, SwarmAgent

   swarm = TurboSwarm(
       "meta-llama/Llama-3.1-70B-Instruct",
       agents=[
           SwarmAgent(name="researcher", role="deep research and analysis"),
           SwarmAgent(name="critic", role="critical review and fact-checking"),
           SwarmAgent(name="writer", role="clear, concise writing"),
       ],
       max_rounds=10,
   )

   results = swarm.run("Write a comprehensive report on quantum error correction.")

   for msg in results:
       print(f"[{msg.get('agent', 'user')}]: {msg['content'][:200]}")

Custom Routing
--------------

By default, agents take turns round-robin. You can provide a custom router:

.. code-block:: python

   def priority_router(history, agents):
       last = history[-1] if history else {}
       if "needs review" in last.get("content", "").lower():
           return "critic"
       if "needs rewriting" in last.get("content", "").lower():
           return "writer"
       return "researcher"

   swarm = TurboSwarm(
       "meta-llama/Llama-3.1-70B-Instruct",
       agents=[...],
       router=priority_router,
   )

Task Completion
---------------

Agents can signal completion by including ``[DONE]``, ``[TASK_COMPLETE]``,
or ``[FINAL_ANSWER]`` in their response. The swarm stops automatically.

Enterprise: SecureMultiAgentSwarm
---------------------------------

For teams requiring governance and compliance, TurboAgent Enterprise provides
``SecureMultiAgentSwarm`` with:

* Role-based access control (RBAC)
* Full audit trail (JSON/CSV export)
* Governance policies (max rounds, content filters)
* SOC-2 and GDPR compliance

See `turboagent.to/enterprise <https://turboagent.to/enterprise>`_.
