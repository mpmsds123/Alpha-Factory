from collections.abc import Callable
from pathlib import Path
from threading import Lock
from time import sleep

import pytest

from alpha_orchestrator.adapters.fake import FakeBrainAdapter
from alpha_orchestrator.adapters.replay import ReplayBrainAdapter
from alpha_orchestrator.config import LimitPolicy, RoutingPolicy
from alpha_orchestrator.domain.enums import (
    AdapterMode,
    AttemptStatus,
    CandidateStatus,
    ErrorClass,
    GateOutcome,
    RunStatus,
)
from alpha_orchestrator.domain.models import (
    CandidateProposal,
    MetadataSnapshot,
    NormalizedSimulationResult,
    RunSpecification,
    SimulationMetrics,
)
from alpha_orchestrator.orchestration.engine import OfflineOrchestrator
from alpha_orchestrator.reporting.report import build_run_report
from alpha_orchestrator.storage.repository import Repository

ROOT = Path(__file__).parents[2]


def candidate(expression: str = "rank(close)") -> CandidateProposal:
    return CandidateProposal(
        expression=expression,
        settings={"neutralization": "SUBINDUSTRY", "truncation": 0.08},
        mechanism_claim="A sanitized candidate claim.",
    )


def completed(
    sharpe: float = 0.7, fitness: float = 0.4, turnover: float = 0.2
) -> NormalizedSimulationResult:
    return NormalizedSimulationResult(
        completed=True,
        metrics=SimulationMetrics(sharpe=sharpe, fitness=fitness, turnover=turnover),
    )


class ConcurrencyTrackingAdapter(FakeBrainAdapter):
    def __init__(self, metadata: MetadataSnapshot, candidates: list[CandidateProposal]) -> None:
        super().__init__(
            metadata,
            candidates,
            results={item.expression: completed() for item in candidates},
        )
        self.active = 0
        self.maximum_active = 0
        self.lock = Lock()

    def create_simulation(self, request):  # type: ignore[no-untyped-def]
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            sleep(0.02)
            return super().create_simulation(request)
        finally:
            with self.lock:
                self.active -= 1


def test_simulation_io_respects_frozen_concurrency_limit(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    limits = limit_policy.model_copy(
        update={"initial_candidates": 10, "maximum_concurrent_simulations": 10}
    )
    run = make_run(_limit_policy=limits)
    proposals = [candidate(f"rank(ts_mean(close, {window}))") for window in range(1, 11)]
    adapter = ConcurrencyTrackingAdapter(metadata, proposals)

    OfflineOrchestrator(repository, routing_policy, limits).execute(run.run_id, adapter)

    assert adapter.maximum_active == 10
    assert repository.count_attempts(run.run_id) == 10


class DyingSessionAdapter(FakeBrainAdapter):
    """A session past the reuse cutoff must stop the run before any new candidate
    spends its budget on a request the session cannot see through -- not fail deep
    inside per-candidate retry handling after a doomed request already went out."""

    def __init__(self, metadata: MetadataSnapshot, candidates: list[CandidateProposal]) -> None:
        super().__init__(
            metadata, candidates, results={item.expression: completed() for item in candidates}
        )

    def session_seconds_remaining(self) -> float | None:
        return 1.0


def test_run_stops_cleanly_when_the_session_is_about_to_expire(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    run = make_run()
    adapter = DyingSessionAdapter(metadata, [candidate("rank(close)")])

    summary = OfflineOrchestrator(repository, routing_policy, limit_policy).execute(
        run.run_id, adapter
    )

    assert summary.status is RunStatus.STOPPED
    assert summary.stop_reason == "SESSION_EXPIRING_SOON"
    # Nothing was spent chasing a dying cookie: no attempt was ever created, and the
    # candidate is left resumable rather than marked as any kind of error.
    assert repository.count_attempts(run.run_id) == 0
    only = repository.list_candidates(run.run_id)[0]
    assert only.status == CandidateStatus.SIMULATING


def test_one_concurrent_failure_does_not_cancel_siblings(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    run = make_run()
    proposals = [candidate("rank(close)"), candidate("rank(volume)")]
    adapter = FakeBrainAdapter(
        metadata,
        proposals,
        results={"rank(volume)": completed()},
        create_failures={"rank(close)": [ErrorClass.AUTHORIZATION]},
    )

    OfflineOrchestrator(repository, routing_policy, limit_policy).execute(run.run_id, adapter)

    states = {item.expression: item.status for item in repository.list_candidates(run.run_id)}
    assert states["rank(close)"] == CandidateStatus.TERMINAL_ERROR
    assert states["rank(volume)"] == CandidateStatus.READY_FOR_PLATFORM_CHECK


def test_concurrent_and_serial_execution_persist_the_same_outcomes(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    proposals = [candidate("rank(close)"), candidate("rank(volume)")]
    serial = limit_policy.model_copy(update={"maximum_concurrent_simulations": 1})
    parallel = limit_policy.model_copy(update={"maximum_concurrent_simulations": 10})
    serial_run = make_run(_limit_policy=serial)
    parallel_run = make_run(_limit_policy=parallel, region="EUR")
    results = {item.expression: completed() for item in proposals}

    OfflineOrchestrator(repository, routing_policy, serial).execute(
        serial_run.run_id, FakeBrainAdapter(metadata, proposals, results=results)
    )
    OfflineOrchestrator(repository, routing_policy, parallel).execute(
        parallel_run.run_id, FakeBrainAdapter(metadata, proposals, results=results)
    )

    serial_states = [
        (item.expression, item.status) for item in repository.list_candidates(serial_run.run_id)
    ]
    parallel_states = [
        (item.expression, item.status) for item in repository.list_candidates(parallel_run.run_id)
    ]
    assert serial_states == parallel_states


def test_basic_replay_is_deterministic_and_report_is_conservative(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
) -> None:
    run = make_run()
    fixture = ROOT / "tests" / "fixtures" / "basic_replay.json"
    first = OfflineOrchestrator(repository, routing_policy, limit_policy).execute(
        run.run_id, ReplayBrainAdapter.from_path(fixture)
    )
    assert first.status is RunStatus.COMPLETED
    assert first.distinct_candidates == 2
    states = {candidate.status for candidate in repository.list_candidates(run.run_id)}
    assert states == {
        CandidateStatus.READY_FOR_PLATFORM_CHECK,
        CandidateStatus.ARCHIVED,
    }
    decisions = repository.list_decisions(run.run_id)
    routing_outcomes = [
        decision.outcome for decision in decisions if decision.gate_name == "G4_EARLY_ROUTING"
    ]
    assert GateOutcome.STRONG in routing_outcomes
    assert GateOutcome.DISCARD in routing_outcomes

    second = OfflineOrchestrator(repository, routing_policy, limit_policy).execute(
        run.run_id, ReplayBrainAdapter.from_path(fixture)
    )
    assert second.simulation_attempts == 2
    assert repository.count_duplicate_events(run.run_id) == 0
    report = build_run_report(repository, run.run_id)
    assert "Automatic submission: unavailable" in report
    assert "platform-all-pass claims: none" in report
    assert "READY_FOR_PLATFORM_CHECK" in report
    assert "submittable" not in report.lower()


def test_retryable_errors_use_one_stable_idempotency_key_and_do_not_fail_research(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    run = make_run()
    proposal = candidate()
    adapter = FakeBrainAdapter(
        metadata,
        [proposal],
        results={proposal.expression: completed()},
        create_failures={
            proposal.expression: [
                ErrorClass.AUTHENTICATION,
                ErrorClass.HUMAN_VERIFICATION_REQUIRED,
            ]
        },
    )
    summary = OfflineOrchestrator(repository, routing_policy, limit_policy).execute(
        run.run_id, adapter
    )
    assert summary.status is RunStatus.COMPLETED
    attempts = repository.list_attempts(run.run_id)
    assert len(attempts) == 3
    assert len({attempt.idempotency_key for attempt in attempts}) == 1
    error_decisions = [
        decision
        for decision in repository.list_decisions(run.run_id)
        if "INFRASTRUCTURE_ERROR_NOT_RESEARCH_FAILURE" in decision.reason_codes
    ]
    assert len(error_decisions) == 2
    assert all(decision.outcome == GateOutcome.ERROR for decision in error_decisions)


@pytest.mark.parametrize(
    "error_class", [ErrorClass.TIMEOUT, ErrorClass.RATE_LIMIT, ErrorClass.NETWORK]
)
def test_each_transient_transport_class_retries(
    error_class: ErrorClass,
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    run = make_run()
    proposal = candidate()
    adapter = FakeBrainAdapter(
        metadata,
        [proposal],
        results={proposal.expression: completed()},
        create_failures={proposal.expression: [error_class]},
    )
    OfflineOrchestrator(repository, routing_policy, limit_policy).execute(run.run_id, adapter)
    assert repository.count_attempts(run.run_id) == 2


def test_retry_exhaustion_and_terminal_error_are_execution_outcomes(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    first_run = make_run()
    proposal = candidate()
    retrying = FakeBrainAdapter(
        metadata,
        [proposal],
        create_failures={proposal.expression: [ErrorClass.TIMEOUT] * 3},
    )
    OfflineOrchestrator(repository, routing_policy, limit_policy).execute(
        first_run.run_id, retrying
    )
    first_candidate = repository.list_candidates(first_run.run_id)[0]
    assert first_candidate.status == CandidateStatus.TERMINAL_ERROR
    assert repository.count_attempts(first_run.run_id) == 3

    second_run = make_run(region="EUR")
    terminal = FakeBrainAdapter(
        metadata,
        [proposal],
        create_failures={proposal.expression: [ErrorClass.AUTHORIZATION]},
    )
    OfflineOrchestrator(repository, routing_policy, limit_policy).execute(
        second_run.run_id, terminal
    )
    second_candidate = repository.list_candidates(second_run.run_id)[0]
    assert second_candidate.status == CandidateStatus.TERMINAL_ERROR
    reasons = [
        reason
        for decision in repository.list_decisions(second_run.run_id)
        for reason in decision.reason_codes
    ]
    assert "EXECUTION_FAILURE_NOT_RESEARCH_VERDICT" in reasons


def test_resume_reconciles_a_candidate_left_simulating_after_an_interrupted_retry(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    """A process killed between finish_attempt(RETRYABLE_ERROR) and the matching
    transition_candidate call in _record_failure (a real run-interruption case:
    SESSION_EXPIRING_SOON, a killed process, a machine sleeping mid-run) can leave
    a candidate in SIMULATING while its latest attempt already reads RETRYABLE_ERROR.
    Resuming that run used to crash with InvalidTransition("SIMULATING -> QUEUED")
    because _process_candidate_batch assumed a RETRYABLE_ERROR attempt always meant
    the candidate itself was already in RETRYABLE_ERROR.

    Reproduce the desync with the same two repository calls _record_failure makes,
    stopping after the first one -- exactly what a kill between them would leave
    behind -- rather than corrupting the row directly, then assert resume
    reconciles it and finishes the retry instead of raising.
    """
    run = make_run()
    proposal = candidate()
    first_adapter = FakeBrainAdapter(
        metadata,
        [proposal],
        results={
            proposal.expression: NormalizedSimulationResult(
                completed=False, error_class=ErrorClass.PENDING
            )
        },
    )
    first = OfflineOrchestrator(repository, routing_policy, limit_policy).execute(
        run.run_id, first_adapter
    )
    assert first.stop_reason == "UNRESOLVED_PENDING_SIMULATION"
    stuck = repository.list_candidates(run.run_id)[0]
    assert stuck.status == CandidateStatus.SIMULATING
    pending_attempt = repository.latest_attempt(stuck.candidate_id)
    assert pending_attempt is not None

    # The next poll would have come back a retryable transport error; record only
    # the attempt half of that outcome, as a kill right after finish_attempt (and
    # before _record_failure's transition_candidate call) would.
    repository.finish_attempt(
        pending_attempt.attempt_id,
        AttemptStatus.RETRYABLE_ERROR,
        NormalizedSimulationResult(completed=False, error_class=ErrorClass.TIMEOUT),
    )
    assert repository.get_candidate(stuck.candidate_id).status == CandidateStatus.SIMULATING

    second_adapter = FakeBrainAdapter(
        metadata, [proposal], results={proposal.expression: completed()}
    )
    resumed = OfflineOrchestrator(repository, routing_policy, limit_policy).execute(
        run.run_id, second_adapter
    )
    assert resumed.status is RunStatus.COMPLETED
    final = repository.get_candidate(stuck.candidate_id)
    assert final.status == CandidateStatus.READY_FOR_PLATFORM_CHECK
    reasons = [
        reason
        for decision in repository.list_decisions(run.run_id)
        for reason in decision.reason_codes
    ]
    assert "RECONCILED_AFTER_INTERRUPTED_RUN" in reasons


def test_invalid_candidate_never_enters_simulation_queue(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    run = make_run()
    adapter = FakeBrainAdapter(metadata, [candidate("rank(unknown_field)")])
    summary = OfflineOrchestrator(repository, routing_policy, limit_policy).execute(
        run.run_id, adapter
    )
    assert summary.status is RunStatus.COMPLETED
    assert repository.list_candidates(run.run_id)[0].status == CandidateStatus.INVALID
    assert repository.count_attempts(run.run_id) == 0


def test_pending_replay_stays_unresolved_and_resume_does_not_create_again(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
) -> None:
    run = make_run(dataset_id="regression-dataset")
    path = ROOT / "tests" / "fixtures" / "historical_regressions.json"
    first_adapter = ReplayBrainAdapter.from_path(path)
    first = OfflineOrchestrator(repository, routing_policy, limit_policy).execute(
        run.run_id, first_adapter
    )
    assert first.status is RunStatus.STOPPED
    assert first.stop_reason == "UNRESOLVED_PENDING_SIMULATION"
    attempt_count = repository.count_attempts(run.run_id)

    second_adapter = ReplayBrainAdapter.from_path(path)
    second = OfflineOrchestrator(repository, routing_policy, limit_policy).execute(
        run.run_id, second_adapter
    )
    assert second.stop_reason == "UNRESOLVED_PENDING_SIMULATION"
    assert repository.count_attempts(run.run_id) == attempt_count
    assert second_adapter.create_call_count == 0
    assert repository.count_duplicate_events(run.run_id) == 1


def test_maximum_simulation_rows_stops_with_persisted_reason(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    constrained = limit_policy.model_copy(update={"maximum_simulation_result_rows": 1})
    run = make_run(_limit_policy=constrained)
    proposals = [candidate("rank(close)"), candidate("rank(volume)")]
    adapter = FakeBrainAdapter(
        metadata,
        proposals,
        results={item.expression: completed() for item in proposals},
    )
    summary = OfflineOrchestrator(repository, routing_policy, constrained).execute(
        run.run_id, adapter
    )
    assert summary.status is RunStatus.STOPPED
    assert summary.stop_reason == "MAXIMUM_SIMULATION_RESULT_ROWS_REACHED"
    assert repository.get_run(run.run_id).stop_reason == summary.stop_reason
    assert repository.count_attempts(run.run_id) == 1


def test_run_contract_rejects_wrong_dataset_and_policy(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    run = make_run()
    wrong_metadata = metadata.model_copy(update={"dataset_id": "other"})
    with pytest.raises(ValueError, match="dataset"):
        OfflineOrchestrator(repository, routing_policy, limit_policy).execute(
            run.run_id, FakeBrainAdapter(wrong_metadata)
        )
    wrong_policy = routing_policy.model_copy(update={"version": "routing.other"})
    with pytest.raises(ValueError, match="Routing policy"):
        OfflineOrchestrator(repository, wrong_policy, limit_policy).execute(
            run.run_id, FakeBrainAdapter(metadata)
        )
    wrong_limits = limit_policy.model_copy(update={"version": "limits.other"})
    with pytest.raises(ValueError, match="Limit policy"):
        OfflineOrchestrator(repository, routing_policy, wrong_limits).execute(
            run.run_id, FakeBrainAdapter(metadata)
        )
    altered_snapshot = limit_policy.model_copy(update={"maximum_pending_candidates": 1})
    with pytest.raises(ValueError, match="content"):
        OfflineOrchestrator(repository, routing_policy, altered_snapshot).execute(
            run.run_id, FakeBrainAdapter(metadata)
        )
    live_run = make_run(adapter_mode=AdapterMode.LIVE)
    with pytest.raises(ValueError, match="Adapter mode"):
        OfflineOrchestrator(repository, routing_policy, limit_policy).execute(
            live_run.run_id, FakeBrainAdapter(metadata)
        )


def test_initial_candidate_and_generation_limits_record_exact_reasons(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    initial_limits = limit_policy.model_copy(update={"initial_candidates": 1})
    initial_run = make_run(_limit_policy=initial_limits)
    initial_adapter = FakeBrainAdapter(
        metadata,
        [candidate("rank(close)"), candidate("rank(volume)")],
        results={"rank(close)": completed(), "rank(volume)": completed()},
    )
    initial_summary = OfflineOrchestrator(repository, routing_policy, initial_limits).execute(
        initial_run.run_id, initial_adapter
    )
    assert initial_summary.stop_reason == "INITIAL_CANDIDATE_LIMIT_REACHED"

    generation_limits = limit_policy.model_copy(update={"maximum_generations": 1})
    generation_run = make_run(_limit_policy=generation_limits, region="EUR")
    parent = repository.add_candidate(
        generation_run.run_id, candidate("rank(close)"), metadata.operators
    ).candidate
    too_deep = CandidateProposal(
        expression="rank(volume)",
        settings={"neutralization": "SUBINDUSTRY", "truncation": 0.08},
        mechanism_claim="A sanitized deep-child proposal.",
        parent_candidate_id=parent.candidate_id,
        generation_number=2,
        changed_dimension="field",
        change_reason="Synthetic generation limit check.",
    )
    generation_adapter = FakeBrainAdapter(metadata, [too_deep])
    generation_summary = OfflineOrchestrator(repository, routing_policy, generation_limits).execute(
        generation_run.run_id, generation_adapter
    )
    assert generation_summary.stop_reason == "MAXIMUM_GENERATIONS_REACHED"


def test_strong_negative_result_spawns_simulated_sign_flip_child(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    run = make_run()
    flipped = "multiply(-1, rank(close))"
    adapter = FakeBrainAdapter(
        metadata.model_copy(update={"operators": metadata.operators | {"multiply"}}),
        [candidate("rank(close)")],
        results={
            "rank(close)": completed(sharpe=-1.2, fitness=-0.6),
            flipped: completed(sharpe=1.2, fitness=0.6),
        },
    )

    OfflineOrchestrator(repository, routing_policy, limit_policy).execute(run.run_id, adapter)

    by_expression = {c.expression: c for c in repository.list_candidates(run.run_id)}
    parent, child = by_expression["rank(close)"], by_expression[flipped]
    assert parent.status == CandidateStatus.ARCHIVED
    assert child.parent_candidate_id == parent.candidate_id
    assert child.changed_dimension == "sign"
    assert child.status == CandidateStatus.READY_FOR_PLATFORM_CHECK
    assert repository.count_attempts(run.run_id) == 2


def test_sign_flip_is_skipped_when_multiply_is_not_an_available_operator(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    run = make_run()
    adapter = FakeBrainAdapter(
        metadata,
        [candidate("rank(close)")],
        results={"rank(close)": completed(sharpe=-1.2, fitness=-0.6)},
    )

    OfflineOrchestrator(repository, routing_policy, limit_policy).execute(run.run_id, adapter)

    assert len(repository.list_candidates(run.run_id)) == 1
