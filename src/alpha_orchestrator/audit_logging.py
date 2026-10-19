"""Single-line JSON logging with recursive secret redaction."""

import json
import logging
from collections.abc import Mapping
from typing import Any

from alpha_orchestrator.domain.errors import redact_text

SECRET_KEYS = frozenset(
    {"api_key", "apikey", "authorization", "cookie", "password", "session", "token"}
)


def sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if str(key).casefold() in SECRET_KEYS else sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [sanitize(item) for item in value]
    return value


class AuditLogger:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("alpha_orchestrator.audit")

    def event(self, event_type: str, **payload: Any) -> None:
        record = {"event_type": event_type, **sanitize(payload)}
        self.logger.info(json.dumps(record, sort_keys=True, separators=(",", ":")))
