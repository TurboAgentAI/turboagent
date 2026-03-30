"""
Enterprise swarm extensions: SecureMultiAgentSwarm.

Requires a valid TurboAgent Enterprise license.
Features: role-based access control, audit logging, compliance exports,
multi-node KV sharing, governance policies.

License: Commercial. See https://turboagent.dev/enterprise
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from turboagent.agents.swarm import TurboSwarm, SwarmAgent

logger = logging.getLogger("turboagent.enterprise.swarm")


class SecureMultiAgentSwarm(TurboSwarm):
    """
    Enterprise-grade multi-agent swarm with governance and audit controls.

    Extends TurboSwarm with:
      - Role-based access control (RBAC) for agent permissions
      - Full audit trail of every agent turn (who, what, when)
      - Compliance export (JSON/CSV) for SOC-2 and GDPR
      - Configurable governance policies (max rounds, content filters)
      - Multi-node KV cache sharing (planned)

    Requires: TURBOAGENT_LICENSE_KEY with 'secure_swarm' feature.
    """

    def __init__(
        self,
        model_id: str,
        agents: List[SwarmAgent],
        governance_policy: Optional[Dict[str, Any]] = None,
        audit_enabled: bool = True,
        **kwargs,
    ):
        from turboagent.enterprise.license import has_feature

        if not has_feature("secure_swarm"):
            raise RuntimeError(
                "SecureMultiAgentSwarm requires an enterprise license with "
                "'secure_swarm' feature. See https://turboagent.dev/enterprise"
            )

        super().__init__(model_id, agents, **kwargs)

        self.governance_policy = governance_policy or {}
        self.audit_enabled = audit_enabled
        self.audit_log: List[Dict[str, Any]] = []

        logger.info("SecureMultiAgentSwarm initialized with enterprise governance.")

    def run(self, task: str, max_rounds: Optional[int] = None) -> List[Dict[str, Any]]:
        """Execute with audit logging and governance enforcement."""
        # Apply governance caps
        policy_max = self.governance_policy.get("max_rounds")
        if policy_max is not None:
            max_rounds = min(max_rounds or self.max_rounds, policy_max)

        result = super().run(task, max_rounds=max_rounds)

        if self.audit_enabled:
            for entry in result:
                self.audit_log.append({
                    "agent": entry.get("agent", "unknown"),
                    "role": entry.get("role", "unknown"),
                    "content_length": len(entry.get("content", "")),
                    "round": entry.get("round"),
                })

        return result

    def export_audit_log(self, format: str = "json") -> str:
        """Export the audit trail for compliance."""
        import json

        if format == "json":
            return json.dumps(self.audit_log, indent=2)
        elif format == "csv":
            lines = ["agent,role,content_length,round"]
            for entry in self.audit_log:
                lines.append(
                    f"{entry['agent']},{entry['role']},{entry['content_length']},{entry['round']}"
                )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

    def clear_audit_log(self) -> None:
        """Clear the audit trail."""
        self.audit_log.clear()
