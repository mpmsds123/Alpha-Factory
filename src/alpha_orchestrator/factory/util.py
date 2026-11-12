"""Small shared helpers: time, hashing, canonical JSON, expression tokens."""

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

TOKEN_RE = re.compile(
    r"\"[^\"]*\"|'[^']*'|"  # string literal
    r"[A-Za-z_][A-Za-z0-9_]*|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|"
    r"==|!=|<=|>=|&&|\|\||[-+*/%^<>=(),?:\[\]{}.]"
)


def utc_now() -> datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.now(UTC)


def sha256_text(value: str) -> str:
    """Return the hex SHA-256 of a string."""
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize value as compact, key-sorted ASCII JSON."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def tokenize_expression(expression: str) -> list[str]:
    """Split an expression into tokens, rejecting unsupported characters."""
    tokens = TOKEN_RE.findall(expression)
    if re.sub(r"\s+", "", "".join(tokens)) != re.sub(r"\s+", "", expression):
        raise ValueError("Expression contains unsupported characters")
    return tokens
