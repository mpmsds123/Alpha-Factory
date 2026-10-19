from collections.abc import Callable
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from alpha_orchestrator.domain.enums import (
    AttemptStatus,
    CandidateStatus,
    ErrorClass,
    GateOutcome,
    PlatformCheckStatus,
    RunStatus,
)
from alpha_orchestrator.domain.identity import stable_idempotency_key
from alpha_orchestrator.domain.models import (
    CandidateProposal,
    NormalizedSimulationResult,
    PlatformCheck,
    RunSpecification,
    SimulationMetrics,
)
from alpha_orchestrator.domain.state_machine import InvalidTransition
from alpha_orchestrator.storage.database import Database
from alpha_orchestrator.storage.models import (
    DecisionORM,
    EventORM,
    LineageEdgeORM,
    PlatformCheckORM,
)
from alpha_orchestrator.storage.repository import MissingRecord, Repository


def proposal(expression: str = "rank(close)") -> CandidateProposal:
    return CandidateProposal(
        expression=expression,
        settings={"neutralization": "SUBINDUSTRY", "truncation": 0.08},
        mechanism_claim="A sanitized candidate-specific claim.",
    )


def test_duplicate_insert_1000_times_creates_one_candidate_and_no_attempts(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
) -> None:
    run = make_run()
    first = repository.add_candidate(run.run_id, proposal("RANK ( close )"), {"rank"})
    assert first.created
    for _ in range(999):
        result = repository.add_candidate(run.run_id, proposal("rank(close)"), {"rank"})
        assert not result.created
        assert result.candidate.candidate_id == first.candidate.candidate_id
    assert repository.count_candidates(run.run_id) == 1
    assert repository.count_attempts(run.run_id) == 0
    assert repository.count_duplicate_events(run.run_id) == 999


def test_live_concurrency_learning_is_persisted_and_bounded(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
) -> None:
    run = make_run()
    assert repository.learned_live_concurrency(maximum=10) == 3
    repository.record_live_concurrency(run.run_id, 7, "CLEAN_BATCH")
    assert repository.learned_live_concurrency(maximum=10) == 7
    assert repository.learned_live_concurrency(maximum=4) == 4


def test_lineage_edge_is_complete_and_same_run(
    repository: Repository,
    database: Database,
    make_run: Callable[..., RunSpecification],
) -> None:
    run = make_run()
    parent = repository.add_candidate(run.run_id, proposal(), {"rank"}).candidate
    child_proposal = CandidateProposal(
        expression="rank(volume)",
        settings={"neutralization": "SUBINDUSTRY", "truncation": 0.08},
        mechanism_claim="A one-field repair hypothesis.",
        parent_candidate_id=parent.candidate_id,
        generation_number=1,
        changed_dimension="field",
        change_reason="Replace the input field to test coverage sensitivity.",
    )
    child = repository.add_candidate(run.run_id, child_proposal, {"rank"}).candidate
    assert child.parent_candidate_id == parent.candidate_id
    with database.session_factory() as session:
        edge = session.scalar(select(LineageEdgeORM))
        assert edge is not None
        assert edge.changed_dimension == "field"
    other_run = make_run(region="EUR")
    invalid = child_proposal.model_copy(
        update={"parent_candidate_id": uuid4(), "expression": "rank(other)"}
    )
    with pytest.raises(ValueError, match="same run"):
        repository.add_candidate(other_run.run_id, invalid, {"rank"})


def test_transitions_write_decision_and_event_atomically(
    repository: Repository,
    database: Database,
    make_run: Callable[..., RunSpecification],
) -> None:
    run = make_run()
    candidate = repository.add_candidate(run.run_id, proposal(), {"rank"}).candidate
    repository.transition_candidate(
        candidate.candidate_id,
        CandidateStatus.VALIDATING,
        "G1_STRUCTURAL_VALIDITY",
        "structural.v1",
        GateOutcome.PENDING,
        ["VALIDATION_STARTED"],
    )
    with pytest.raises(InvalidTransition):
        repository.transition_candidate(
            candidate.candidate_id,
            CandidateStatus.COMPLETED,
            "G3",
            "v1",
            GateOutcome.PASS,
            [],
        )
    with database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(DecisionORM)) == 1
        assert session.scalar(select(func.count()).select_from(EventORM)) == 3


def test_attempt_metrics_errors_and_platform_states_are_preserved(
    repository: Repository,
    database: Database,
    make_run: Callable[..., RunSpecification],
) -> None:
    run = make_run()
    candidate = repository.add_candidate(run.run_id, proposal(), {"rank"}).candidate
    key = stable_idempotency_key(candidate.candidate_id)
    attempt = repository.start_attempt(candidate.candidate_id, key, 1)
    repository.attach_receipt(attempt.attempt_id, "sim-1", "alpha-1")
    repository.finish_attempt(
        attempt.attempt_id,
        AttemptStatus.RETRYABLE_ERROR,
        NormalizedSimulationResult(
            completed=False,
            error_class=ErrorClass.AUTHENTICATION,
            error_message="token=must-not-persist",
            metrics=SimulationMetrics(extra={"rows": 1}),
            raw_artifact_ref="safe/ref.json",
        ),
    )
    persisted = repository.latest_attempt(candidate.candidate_id)
    assert persisted is not None
    assert persisted.idempotency_key == key
    assert "must-not-persist" not in (persisted.error_message_redacted or "")
    assert repository.completed_metrics(candidate.candidate_id) is None

    checks = [
        PlatformCheck(name="one", status=PlatformCheckStatus.ERROR, message="token=hidden"),
        PlatformCheck(name="two", status=PlatformCheckStatus.PENDING),
        PlatformCheck(name="three", status=PlatformCheckStatus.FAIL),
    ]
    repository.record_platform_checks(candidate.candidate_id, checks, live=True)
    with database.session_factory() as session:
        statuses = list(session.scalars(select(PlatformCheckORM.status)))
        assert statuses == ["ERROR", "PENDING", "FAIL"]
        message = session.scalar(
            select(PlatformCheckORM.message_redacted).where(PlatformCheckORM.name == "one")
        )
        assert message is not None and "hidden" not in message


def test_repository_missing_records_and_run_status(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
) -> None:
    with pytest.raises(MissingRecord):
        repository.get_run(uuid4())
    with pytest.raises(MissingRecord):
        repository.get_candidate(uuid4())
    with pytest.raises(MissingRecord):
        repository.start_attempt(uuid4(), "key", 1)
    with pytest.raises(MissingRecord):
        repository.attach_receipt(str(uuid4()), "sim", None)
    with pytest.raises(MissingRecord):
        repository.finish_attempt(str(uuid4()), AttemptStatus.CANCELLED)
    with pytest.raises(MissingRecord):
        repository.record_platform_checks(uuid4(), [], live=False)
    run = make_run()
    repository.update_run_status(run.run_id, RunStatus.RUNNING)
    repository.update_run_status(run.run_id, RunStatus.STOPPED, "TEST_STOP")
    persisted = repository.get_run(run.run_id)
    assert persisted.status == RunStatus.STOPPED
    assert persisted.stop_reason == "TEST_STOP"
