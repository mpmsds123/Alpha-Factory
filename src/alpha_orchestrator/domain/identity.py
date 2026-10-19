"""Conservative expression identity without algebraic-equivalence claims."""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

TOKEN_RE = re.compile(
    r"\"[^\"]*\"|'[^']*'|"  # string literal, e.g. bucket(x, range="0.1, 1, 0.1")
    r"[A-Za-z_][A-Za-z0-9_]*|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|"
    r"==|!=|<=|>=|&&|\|\||[-+*/%^<>=(),?:\[\]{}.]"
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def metadata_snapshot_identity(
    kind: str,
    dataset_id: str,
    region: str,
    universe: str,
    delay: int,
    data_type: str,
    payload: Any,
) -> tuple[str, str]:
    serialized = canonical_json(
        {
            "kind": kind,
            "dataset_id": dataset_id,
            "region": region,
            "universe": universe,
            "delay": delay,
            "data_type": data_type,
            "payload": payload,
        }
    )
    content_hash = sha256_text(serialized)
    return f"live-{kind.casefold()}-{content_hash[:20]}", content_hash


def tokenize_expression(expression: str) -> list[str]:
    tokens = TOKEN_RE.findall(expression)
    if re.sub(r"\s+", "", "".join(tokens)) != re.sub(r"\s+", "", expression):
        raise ValueError("Expression contains unsupported characters")
    return tokens


def canonicalize_expression(expression: str, known_operators: Iterable[str]) -> str:
    operator_names = {operator.casefold(): operator.lower() for operator in known_operators}
    tokens = tokenize_expression(expression)
    normalized = [operator_names.get(token.casefold(), token) for token in tokens]
    return "".join(normalized)


def candidate_fingerprint(
    canonical_expression: str,
    settings: Mapping[str, Any],
    region: str,
    universe: str,
    delay: int,
) -> str:
    payload = "\x1f".join(
        (canonical_expression, canonical_json(settings), region, universe, str(delay))
    )
    return sha256_text(payload)


def normalized_content_hash(canonical_expression: str, settings: Mapping[str, Any]) -> str:
    payload = "\x1f".join((canonical_expression, canonical_json(settings)))
    return sha256_text(payload)


def stable_idempotency_key(candidate_id: str, operation: str = "simulation-create") -> str:
    return sha256_text(f"{operation}:{candidate_id}")
