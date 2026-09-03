"""
Audit Logger: records every security/tool-execution-relevant event
(auth failures, permission rejections, tool executions, injection
detections) as structured log lines. In production this would ship to a
durable audit sink (e.g. append-only DB table, SIEM); here it logs via the
standard `logging` module through a dedicated "ai_engine.audit" logger so
it can be routed independently of application logs.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.domain.interfaces import IAuditLogger, Principal

audit_log = logging.getLogger("ai_engine.audit")


class StandardAuditLogger(IAuditLogger):
    def log(self, event: str, principal: Principal, details: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "user_id": principal.user_id,
            "tenant_id": principal.tenant_id,
            "details": _json_safe(details),
        }
        audit_log.info(json.dumps(record, default=str))


def _json_safe(details: dict[str, Any]) -> dict[str, Any]:
    """Ensure values are JSON-serializable (Decimal/datetime -> str)."""
    safe = {}
    for k, v in details.items():
        try:
            json.dumps(v)
            safe[k] = v
        except TypeError:
            safe[k] = str(v)
    return safe
