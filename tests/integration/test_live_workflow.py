from collections.abc import Callable

import pytest
from sqlalchemy import func, select

from alpha_orchestrator.adapters.fake import FakeBrainAdapter
from alpha_orchestrator.config import LimitPolicy, RoutingPolicy
from alpha_orchestrator.domain.enums import (
    AdapterMode,
    CandidateStatus,
    DataType,
    ErrorClass,
    GateOutcome,
    MetadataSnapshotKind,
    PlatformCheckStatus,
    RunStatus,
)
from alpha_orchestrator.domain.identity import metadata_snapshot_identity
from alpha_orchestrator.domain.models import (
    AdapterFailure,
    CandidateProposal,
    CapturedMetadata,
    CreateSimulationReceipt,
    CreateSimulationRequest,
    MetadataSnapshot,
    NormalizedSimulationResult,
    PlatformCheck,
    RunSpecification,
    SimulationMetrics,
)
from alpha_orchestrator.orchestration.engine import ResearchOrchestrator
from alpha_orchestrator.orchestration.platform_checks import PlatformCheckService
from alpha_orchestrator.reporting.report import build_run_report
from alpha_orchestrator.storage.database import Database
from alpha_orchestrator.storage.models import MetadataSnapshotORM
from alpha_orchestrator.storage.repository import MissingRecord, Repository


def proposal() -> CandidateProposal:
    return CandidateProposal(
        expression="rank(close)",
        settings={"neutralization": "SUBINDUSTRY", "truncation": 0.08},
        mechanism_claim="A sanitized live workflow claim.",
    )


def strong_result(*, completed: bool = True) -> NormalizedSimulationResult:
    if not completed:
        return NormalizedSimulationResult(completed=False, error_class=ErrorClass.PENDING)
    return NormalizedSimulationResult(
        completed=True,
        external_alpha_id="alpha-live-1",
        metrics=SimulationMetrics(sharpe=0.8, fitness=0.5, turnover=0.2),
    )


class StubLiveAdapter:
    def __init__(
        self,
        metadata: MetadataSnapshot,
        *,
        poll_results: list[NormalizedSimulationResult] | None = None,
        check_responses: list[list[PlatformCheck] | AdapterFailure] | None = None,
    ) -> None:
        self._metadata = metadata
        self._poll_results = poll_results or [strong_result()]
        self._check_responses = check_responses or []
        self._offered_candidates = False
        self.create_calls = 0
        self.check_calls = 0

    @property
    def mode(self) -> AdapterMode:
        return AdapterMode.LIVE

    @property
    def metadata(self) -> MetadataSnapshot:
        return self._metadata

    def initial_candidates(self) -> list[CandidateProposal]:
        if self._offered_candidates:
            return []
        self._offered_candidates = True
        return [proposal()]

    def create_simulation(self, request: CreateSimulationRequest) -> CreateSimulationReceipt:
        del request
        self.create_calls += 1
        return CreateSimulationReceipt(external_simulation_id="sim-live-1")

    def poll_simulation(self, external_simulation_id: str) -> NormalizedSimulationResult:
        assert external_simulation_id == "sim-live-1"
        if len(self._poll_results) > 1:
            return self._poll_results.pop(0)
        return self._poll_results[0]

    def request_platform_checks(self, external_alpha_id: str) -> list[PlatformCheck]:
        assert external_alpha_id == "alpha-live-1"
        self.check_calls += 1
        response = self._check_responses.pop(0)
        if isinstance(response, AdapterFailure):
            raise response
        return response


def metadata_capture(metadata: MetadataSnapshot) -> CapturedMetadata:
    cell = ("USA", "TOP1000", 1, DataType.MATRIX)
    fields = {name: value.value for name, value in sorted(metadata.fields.items())}
    operators = sorted(metadata.operators)
    settings = {
        name: sorted(values, key=repr) for name, values in sorted(metadata.settings.items())
    }
    identifiers = [
        metadata_snapshot_identity(kind, metadata.dataset_id, *cell, payload)[0]
        for kind, payload in (
            (MetadataSnapshotKind.FIELDS, fields),
            (MetadataSnapshotKind.OPERATORS, operators),
            (MetadataSnapshotKind.SETTINGS, settings),
        )
    ]
    return CapturedMetadata(
        metadata=metadata,
        region=cell[0],
        universe=cell[1],
        delay=cell[2],
        data_type=cell[3],
        field_snapshot_id=identifiers[0],
        operator_snapshot_id=identifiers[1],
        setting_snapshot_id=identifiers[2],
        source="intercepted-test",
    )


def create_live_run(
    make_run: Callable[..., RunSpecification],
    capture: CapturedMetadata,
    **overrides: object,
) -> RunSpecification:
    return make_run(
        adapter_mode=AdapterMode.LIVE,
        field_snapshot_id=capture.field_snapshot_id,
        operator_snapshot_id=capture.operator_snapshot_id,
        setting_snapshot_id=capture.setting_snapshot_id,
        **overrides,
    )


def execute_ready_candidate(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
    check_responses: list[list[PlatformCheck] | AdapterFailure],
) -> tuple[RunSpecification, StubLiveAdapter, str]:
    capture = metadata_capture(metadata)
    run = create_live_run(make_run, capture)
    adapter = StubLiveAdapter(metadata, check_responses=check_responses)
    summary = ResearchOrchestrator(repository, routing_policy, limit_policy).execute(
        run.run_id, adapter
    )
    assert summary.status is RunStatus.COMPLETED
    candidate = repository.list_candidates(run.run_id)[0]
    assert candidate.status == CandidateStatus.READY_FOR_PLATFORM_CHECK
    return run, adapter, candidate.candidate_id


def test_metadata_snapshots_are_content_addressed_and_reconstructable(
    repository: Repository,
    database: Database,
    make_run: Callable[..., RunSpecification],
    metadata: MetadataSnapshot,
) -> None:
    capture = metadata_capture(metadata)
    repository.save_metadata_capture(capture)
    repository.save_metadata_capture(capture)
    run = create_live_run(make_run, capture)

    assert repository.load_run_metadata(run.run_id) == metadata
    with database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(MetadataSnapshotORM)) == 3
        sources = set(session.scalars(select(MetadataSnapshotORM.source)))
        assert sources == {"intercepted-test"}

    forged = capture.model_copy(update={"field_snapshot_id": "forged"})
    with pytest.raises(ValueError, match="does not match content"):
        repository.save_metadata_capture(forged)
    mismatched = create_live_run(make_run, capture, region="EUR")
    with pytest.raises(ValueError, match="frozen run cell"):
        repository.load_run_metadata(mismatched.run_id)
    missing = make_run()
    with pytest.raises(MissingRecord, match="incomplete"):
        repository.load_run_metadata(missing.run_id)


def test_live_stop_and_resume_reuses_external_simulation(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    capture = metadata_capture(metadata)
    run = create_live_run(make_run, capture)
    adapter = StubLiveAdapter(
        metadata, poll_results=[strong_result(completed=False), strong_result()]
    )
    orchestrator = ResearchOrchestrator(repository, routing_policy, limit_policy)

    first = orchestrator.execute(run.run_id, adapter)
    assert first.status is RunStatus.STOPPED
    assert first.stop_reason == "UNRESOLVED_PENDING_SIMULATION"
    second = orchestrator.execute(run.run_id, adapter)
    assert second.status is RunStatus.COMPLETED
    assert adapter.create_calls == 1
    attempt = repository.list_attempts(run.run_id)[0]
    assert attempt.external_alpha_id == "alpha-live-1"
    assert attempt.status == "COMPLETED"


def test_platform_all_pass_is_the_only_human_review_promotion(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    checks = [
        PlatformCheck(name="LOW_SHARPE", status=PlatformCheckStatus.PASS),
        PlatformCheck(name="WEIGHT", status=PlatformCheckStatus.PASS),
    ]
    run, adapter, candidate_id = execute_ready_candidate(
        repository, make_run, routing_policy, limit_policy, metadata, [checks]
    )
    summary = PlatformCheckService(repository).check(candidate_id, adapter)

    assert summary.candidate_status is CandidateStatus.READY_FOR_HUMAN_REVIEW
    decisions = repository.list_decisions(run.run_id)
    g6 = [decision for decision in decisions if decision.gate_name == "G6_PLATFORM_READINESS"]
    assert len(g6) == 1
    assert g6[0].outcome == GateOutcome.PASS
    assert g6[0].reason_codes == ["PLATFORM_ALL_PASS_CONFIRMED"]
    assert "Confirmed live platform-all-pass decisions: 1" in build_run_report(
        repository, run.run_id
    )


def test_platform_failure_archives_and_unresolved_statuses_do_not_promote(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    _, failing_adapter, failing_id = execute_ready_candidate(
        repository,
        make_run,
        routing_policy,
        limit_policy,
        metadata,
        [[PlatformCheck(name="WEIGHT", status=PlatformCheckStatus.FAIL)]],
    )
    failed = PlatformCheckService(repository).check(failing_id, failing_adapter)
    assert failed.candidate_status is CandidateStatus.ARCHIVED

    pending_run, pending_adapter, pending_id = execute_ready_candidate(
        repository,
        make_run,
        routing_policy,
        limit_policy,
        metadata,
        [
            [
                PlatformCheck(name="A", status=PlatformCheckStatus.WARNING),
                PlatformCheck(name="B", status=PlatformCheckStatus.PENDING),
            ]
        ],
    )
    pending = PlatformCheckService(repository).check(pending_id, pending_adapter)
    assert pending.candidate_status is CandidateStatus.READY_FOR_PLATFORM_CHECK
    g6 = [
        decision
        for decision in repository.list_decisions(pending_run.run_id)
        if decision.gate_name == "G6_PLATFORM_READINESS"
    ]
    assert g6[0].outcome == GateOutcome.PENDING


def test_platform_check_retries_retryable_failures_within_frozen_limit(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    checks = [PlatformCheck(name="WEIGHT", status=PlatformCheckStatus.PASS)]
    run, adapter, candidate_id = execute_ready_candidate(
        repository,
        make_run,
        routing_policy,
        limit_policy,
        metadata,
        [AdapterFailure(ErrorClass.TIMEOUT, "temporary"), checks],
    )
    summary = PlatformCheckService(repository).check(candidate_id, adapter)
    persisted = repository.list_platform_checks(run.run_id)
    assert summary.request_attempts == 2
    assert summary.candidate_status is CandidateStatus.READY_FOR_HUMAN_REVIEW
    assert [item.status for item in persisted] == ["ERROR", "PASS"]


def test_platform_check_enforces_live_run_and_matching_metadata(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    metadata: MetadataSnapshot,
) -> None:
    _, adapter, candidate_id = execute_ready_candidate(
        repository,
        make_run,
        routing_policy,
        limit_policy,
        metadata,
        [[PlatformCheck(name="WEIGHT", status=PlatformCheckStatus.PASS)]],
    )
    with pytest.raises(ValueError, match="live adapter"):
        PlatformCheckService(repository).check(candidate_id, FakeBrainAdapter(metadata))
    wrong_metadata = metadata.model_copy(update={"dataset_id": "other"})
    with pytest.raises(ValueError, match="frozen run"):
        PlatformCheckService(repository).check(candidate_id, StubLiveAdapter(wrong_metadata))

    proposed_run = create_live_run(make_run, metadata_capture(metadata))
    proposed = repository.add_candidate(
        proposed_run.run_id, proposal(), metadata.operators
    ).candidate
    with pytest.raises(ValueError, match="not ready"):
        PlatformCheckService(repository).check(proposed.candidate_id, adapter)
