from uuid import uuid4

import pytest
from pydantic import ValidationError

from alpha_orchestrator.domain.enums import CandidateStatus, ErrorClass
from alpha_orchestrator.domain.errors import (
    assert_no_sensitive_data,
    contains_sensitive_data,
    is_retryable,
    redact_text,
)
from alpha_orchestrator.domain.models import CandidateProposal
from alpha_orchestrator.domain.state_machine import InvalidTransition, validate_transition


def test_root_and_child_lineage_contracts() -> None:
    root = CandidateProposal(expression="rank(x)", settings={}, mechanism_claim="Root claim")
    assert root.generation_number == 0
    child = CandidateProposal(
        expression="rank(y)",
        settings={},
        mechanism_claim="Child claim",
        parent_candidate_id=uuid4(),
        generation_number=1,
        changed_dimension="field",
        change_reason="Test one field change",
    )
    assert child.changed_dimension == "field"


@pytest.mark.parametrize(
    "values",
    [
        {"generation_number": 1},
        {"changed_dimension": "field"},
        {"parent_candidate_id": uuid4(), "generation_number": 1},
        {
            "parent_candidate_id": uuid4(),
            "generation_number": 0,
            "changed_dimension": "field",
            "change_reason": "reason",
        },
    ],
)
def test_invalid_lineage_is_rejected(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CandidateProposal(expression="rank(x)", settings={}, mechanism_claim="Claim", **values)


def test_state_machine_accepts_only_declared_edges() -> None:
    validate_transition(CandidateStatus.PROPOSED, CandidateStatus.VALIDATING)
    validate_transition(CandidateStatus.TRIAGED, CandidateStatus.ARCHIVED)
    with pytest.raises(InvalidTransition, match="Invalid candidate transition"):
        validate_transition(CandidateStatus.PROPOSED, CandidateStatus.COMPLETED)


@pytest.mark.parametrize(
    "error_class",
    [
        ErrorClass.AUTHENTICATION,
        ErrorClass.HUMAN_VERIFICATION_REQUIRED,
        ErrorClass.RATE_LIMIT,
        ErrorClass.TIMEOUT,
        ErrorClass.NETWORK,
    ],
)
def test_infrastructure_retry_taxonomy(error_class: ErrorClass) -> None:
    assert is_retryable(error_class)


def test_non_retryable_and_redaction() -> None:
    assert not is_retryable(ErrorClass.AUTHORIZATION)
    assert redact_text(None) is None
    redacted = redact_text("api_key=secret Authorization:Bearer abc token=xyz")
    assert redacted is not None
    assert "secret" not in redacted
    assert "abc" not in redacted
    assert "xyz" not in redacted
    assert contains_sensitive_data({"password": "EXAMPLE"})
    assert contains_sensitive_data(["authorization=EXAMPLE"])
    assert not contains_sensitive_data({"message": "token usage is recorded"})
    with pytest.raises(ValueError, match="forbidden"):
        assert_no_sensitive_data({"api_key": ""})
