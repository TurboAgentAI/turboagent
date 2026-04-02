"""
TurboAgent: TurboQuant-powered agentic AI framework for long-context LLMs on consumer hardware.

This open-source core is released under the MIT license and is free forever.

Commercial extensions (TurboAgent Enterprise) are available for teams requiring
SSO, audit logging, compliance exports, advanced swarm governance, and dedicated
support. See https://turboagent.to/enterprise for details.

Public API:
    from turboagent import TurboAgent, create_engine
"""

from turboagent.agents.base import TurboAgent
from turboagent.backends import create_engine
from turboagent.version import __version__

__all__ = ["TurboAgent", "create_engine", "__version__"]
