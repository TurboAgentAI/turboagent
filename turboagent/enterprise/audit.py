"""
Enterprise audit logging for compliance (SOC-2, GDPR).

Provides structured audit records for every agent action, with export
capabilities for compliance reviews. Integrates with TurboAgent and
TurboSwarm via the enterprise feature flags.

Requires: TURBOAGENT_LICENSE_KEY with 'audit_logging' feature.
License: Commercial. See https://turboagent.dev/enterprise
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("turboagent.enterprise.audit")


class AuditRecord:
    """A single audit event."""

    __slots__ = ("timestamp", "actor", "action", "detail", "metadata")

    def __init__(
        self,
        actor: str,
        action: str,
        detail: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.actor = actor
        self.action = action
        self.detail = detail
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "detail": self.detail,
            "metadata": self.metadata,
        }


class AuditLogger:
    """
    Structured audit logger for enterprise compliance.

    Records agent actions (run, tool_call, memory_access, swarm_turn)
    with timestamps and actor identity. Supports JSON and CSV export.
    """

    def __init__(self, enabled: bool = True):
        from turboagent.enterprise.license import has_feature

        if not has_feature("audit_logging"):
            raise RuntimeError(
                "AuditLogger requires an enterprise license with "
                "'audit_logging' feature. See https://turboagent.dev/enterprise"
            )

        self.enabled = enabled
        self._records: List[AuditRecord] = []

    def log(
        self,
        actor: str,
        action: str,
        detail: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an audit event."""
        if not self.enabled:
            return
        record = AuditRecord(actor=actor, action=action, detail=detail, metadata=metadata)
        self._records.append(record)
        logger.debug(f"Audit: [{record.timestamp}] {actor} -> {action}")

    def export(self, format: str = "json") -> str:
        """Export all records as JSON or CSV."""
        records = [r.to_dict() for r in self._records]

        if format == "json":
            return json.dumps(records, indent=2)
        elif format == "csv":
            if not records:
                return "timestamp,actor,action,detail"
            lines = ["timestamp,actor,action,detail"]
            for r in records:
                detail = r["detail"].replace(",", ";").replace("\n", " ")
                lines.append(f"{r['timestamp']},{r['actor']},{r['action']},{detail}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @property
    def records(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._records]

    def clear(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)
