"""Error semantics and secret redaction."""

import re

from alpha_orchestrator.domain.enums import ErrorClass

RETRYABLE_ERRORS = frozenset(
    {
        ErrorClass.AUTHENTICATION,
        ErrorClass.HUMAN_VERIFICATION_REQUIRED,
        ErrorClass.RATE_LIMIT,
        ErrorClass.TIMEOUT,
        ErrorClass.NETWORK,
    }
)

REDACTION_PATTERNS = (
    re.compile(
        r"(?i)(api[_-]?key|authorization|password|session|token)\s*[:=]\s*"
        r"(?:bearer\s+)?[^\s,;]+"
    ),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
)


def is_retryable(error_class: ErrorClass) -> bool:
    return error_class in RETRYABLE_ERRORS


def redact_text(value: str | None) -> str | None:
    if value is None:
        return None
    redacted = value
    for pattern in REDACTION_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted
