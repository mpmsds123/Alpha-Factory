"""Error semantics and secret redaction."""

import re
from collections.abc import Mapping
from typing import Any

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
CREDENTIAL_KEYS = frozenset(
    {"api_key", "apikey", "authorization", "cookie", "password", "session", "token"}
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


def contains_sensitive_data(value: Any) -> bool:
    if isinstance(value, str):
        return redact_text(value) != value
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in CREDENTIAL_KEYS or contains_sensitive_data(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple | set):
        return any(contains_sensitive_data(item) for item in value)
    return False


def assert_no_sensitive_data(value: Any) -> None:
    if contains_sensitive_data(value):
        raise ValueError("Credential-like content is forbidden in persisted research data")
