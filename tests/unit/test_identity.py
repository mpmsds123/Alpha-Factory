import pytest

from alpha_orchestrator.domain.identity import (
    candidate_fingerprint,
    canonical_json,
    canonicalize_expression,
    normalized_content_hash,
    stable_idempotency_key,
    tokenize_expression,
)


def test_canonicalization_normalizes_only_whitespace_and_known_operator_case() -> None:
    expression = canonicalize_expression(" RANK ( Close , 5 ) ", {"rank"})
    assert expression == "rank(Close,5)"


def test_tokenizer_rejects_unsupported_characters() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        tokenize_expression("rank(close); drop table candidates")


def test_settings_and_fingerprints_are_deterministic() -> None:
    first = {"b": 2, "a": 1}
    second = {"a": 1, "b": 2}
    assert canonical_json(first) == canonical_json(second) == '{"a":1,"b":2}'
    assert candidate_fingerprint("rank(x)", first, "USA", "TOP1000", 1) == candidate_fingerprint(
        "rank(x)", second, "USA", "TOP1000", 1
    )
    assert normalized_content_hash("rank(x)", first) == normalized_content_hash("rank(x)", second)


def test_identity_preserves_mathematical_order_and_cell() -> None:
    assert normalized_content_hash("a-b", {}) != normalized_content_hash("b-a", {})
    assert candidate_fingerprint("a", {}, "USA", "TOP1000", 1) != candidate_fingerprint(
        "a", {}, "EUR", "TOP1000", 1
    )


def test_idempotency_key_is_stable_and_operation_scoped() -> None:
    assert stable_idempotency_key("candidate") == stable_idempotency_key("candidate")
    assert stable_idempotency_key("candidate") != stable_idempotency_key("candidate", "poll")
