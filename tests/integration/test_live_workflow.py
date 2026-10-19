import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from alpha_orchestrator.adapters.fake import FakeBrainAdapter
from alpha_orchestrator.config import CorrelationPolicy, LimitPolicy, RoutingPolicy
from alpha_orchestrator.domain.enums import (
    AdapterMode,
    CandidateStatus,
    DataType,
    ErrorClass,
    EvidenceLevel,
    GateOutcome,
    MetadataSnapshotKind,
    MethodCardStatus,
    PlatformCheckStatus,
    RunStatus,
)
from alpha_orchestrator.domain.identity import metadata_snapshot_identity
from alpha_orchestrator.domain.knowledge import MethodCard, SourceReference
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
from alpha_orchestrator.knowledge.retriever import MethodCardRetriever
from alpha_orchestrator.llm.base import LLMCompletion
from alpha_orchestrator.llm.prompts import register_default_prompts
from alpha_orchestrator.llm.services import CandidateGenerator, RepairPlanner
from alpha_orchestrator.orchestration.engine import ResearchOrchestrator
from alpha_orchestrator.orchestration.platform_checks import PlatformCheckService
from alpha_orchestrator.reporting.report import build_run_report
from alpha_orchestrator.storage.database import Database
from alpha_orchestrator.storage.models import MetadataSnapshotORM
from alpha_orchestrator.storage.repository import MissingRecord, Repository


@dataclass
class StaticLLMClient:
    raw_text: str
    model: str = "intercepted-test-model"
    temperature: float = 0.0
    user_prompts: list[str] = field(default_factory=list)

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, object],
    ) -> LLMCompletion:
        del system_prompt, json_schema
        self.user_prompts.append(user_prompt)
        return LLMCompletion(raw_text=self.raw_text, model=self.model, latency_ms=1)


def add_approved_platform_card(repository: Repository, failure_code: str) -> None:
    repository.upsert_method_card(
        MethodCard(
            card_id=f"MC-TEST-{failure_code}",
            title=f"Test-only platform repair for {failure_code}",
            version="v1",
            status=MethodCardStatus.APPROVED,
            applicable_failure_codes=[failure_code],
            applicable_data_types=[DataType.MATRIX],
            prerequisites=["A test platform failure exists"],
            diagnostic_signals=["The test supplies a failure code"],
            allowed_repair_actions=["Change one test dimension"],
            prohibited_actions=["Do not change limits"],
            stop_condition="Stop after the bounded test repair",
            evidence_level=EvidenceLevel.TRAINING,
            source_references=[SourceReference(source_id="SRC-TEST", section_locator="line:1")],
            reviewer="Test Reviewer",
            approval_timestamp=datetime.now(UTC),
        )
    )


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
        correlations: dict[str, float | None] | AdapterFailure | None = None,
    ) -> None:
        self._metadata = metadata
        self._poll_results = poll_results or [strong_result()]
        self._check_responses = check_responses or []
        # Distinct from anything on the platform by default: tests that only care
        # about the platform-check gate should not be blocked by an unrelated
        # correlation gate they never mentioned.
        self._correlations = {"self": 0.0, "prod": 0.0} if correlations is None else correlations
        self._offered_candidates = False
        self.create_calls = 0
        self.check_calls = 0
        self.correlation_calls = 0

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

    def fetch_correlations(self, external_alpha_id: str) -> dict[str, float | None]:
        assert external_alpha_id == "alpha-live-1"
        self.correlation_calls += 1
        if isinstance(self._correlations, AdapterFailure):
            raise self._correlations
        return self._correlations

    def session_seconds_remaining(self) -> float | None:
        return None


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
    correlations: dict[str, float | None] | AdapterFailure | None = None,
) -> tuple[RunSpecification, StubLiveAdapter, str]:
    capture = metadata_capture(metadata)
    run = create_live_run(make_run, capture)
    adapter = StubLiveAdapter(metadata, check_responses=check_responses, correlations=correlations)
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
    correlation_policy: CorrelationPolicy,
    metadata: MetadataSnapshot,
) -> None:
    checks = [
        PlatformCheck(name="LOW_SHARPE", status=PlatformCheckStatus.PASS),
        PlatformCheck(name="WEIGHT", status=PlatformCheckStatus.PASS),
    ]
    run, adapter, candidate_id = execute_ready_candidate(
        repository, make_run, routing_policy, limit_policy, metadata, [checks]
    )
    summary = PlatformCheckService(repository).check(candidate_id, adapter, correlation_policy)

    assert summary.candidate_status is CandidateStatus.READY_FOR_HUMAN_REVIEW
    decisions = repository.list_decisions(run.run_id)
    g6 = [decision for decision in decisions if decision.gate_name == "G6_PLATFORM_READINESS"]
    assert len(g6) == 1
    assert g6[0].outcome == GateOutcome.PASS
    assert g6[0].reason_codes == ["PLATFORM_ALL_PASS_CONFIRMED"]
    g7 = [decision for decision in decisions if decision.gate_name == "G7_CORRELATION_DEDUP"]
    assert len(g7) == 1
    assert g7[0].outcome == GateOutcome.PASS
    assert g7[0].reason_codes == ["CORRELATION_BELOW_THRESHOLD"]
    assert "Confirmed live platform-all-pass decisions: 1" in build_run_report(
        repository, run.run_id
    )


def test_execute_settles_platform_check_and_correlation_without_a_separate_call(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    correlation_policy: CorrelationPolicy,
    metadata: MetadataSnapshot,
) -> None:
    """Step ⑩ (dedup/check/queue) is supposed to be code's decision, not a human
    remembering to run a separate command for every completed candidate: once a
    correlation policy is configured, one execute() call must reach human review
    on its own for a candidate that passes everything."""
    capture = metadata_capture(metadata)
    run = create_live_run(make_run, capture)
    checks = [PlatformCheck(name="WEIGHT", status=PlatformCheckStatus.PASS)]
    adapter = StubLiveAdapter(metadata, check_responses=[checks])

    summary = ResearchOrchestrator(
        repository, routing_policy, limit_policy, correlation_policy=correlation_policy
    ).execute(run.run_id, adapter)

    assert summary.status is RunStatus.COMPLETED
    candidate = repository.list_candidates(run.run_id)[0]
    assert candidate.status == CandidateStatus.READY_FOR_HUMAN_REVIEW
    assert adapter.check_calls == 1
    assert adapter.correlation_calls == 1


def test_correlation_gate_archives_a_near_duplicate_and_holds_a_pending_one(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    correlation_policy: CorrelationPolicy,
    metadata: MetadataSnapshot,
) -> None:
    """Platform checks passing is not proof of non-redundancy: the platform's own
    correlation numbers still gate the promotion to human review."""
    checks = [PlatformCheck(name="WEIGHT", status=PlatformCheckStatus.PASS)]
    _, duplicate_adapter, duplicate_id = execute_ready_candidate(
        repository,
        make_run,
        routing_policy,
        limit_policy,
        metadata,
        [checks],
        correlations={"self": 0.42, "prod": 0.71},
    )
    duplicate = PlatformCheckService(repository).check(
        duplicate_id, duplicate_adapter, correlation_policy
    )
    assert duplicate.candidate_status is CandidateStatus.ARCHIVED
    decisions = repository.list_decisions(repository.get_candidate(duplicate_id).run_id)
    g7 = [decision for decision in decisions if decision.gate_name == "G7_CORRELATION_DEDUP"]
    assert g7[-1].outcome == GateOutcome.FAIL
    assert g7[-1].reason_codes == ["PROD_CORRELATION_TOO_HIGH"]

    pending_run, pending_adapter, pending_id = execute_ready_candidate(
        repository,
        make_run,
        routing_policy,
        limit_policy,
        metadata,
        [checks],
        correlations={"self": 0.1, "prod": None},
    )
    pending = PlatformCheckService(repository).check(
        pending_id, pending_adapter, correlation_policy
    )
    assert pending.candidate_status is CandidateStatus.READY_FOR_PLATFORM_CHECK
    pending_decisions = repository.list_decisions(pending_run.run_id)
    pending_g7 = [
        decision for decision in pending_decisions if decision.gate_name == "G7_CORRELATION_DEDUP"
    ]
    assert pending_g7[-1].outcome == GateOutcome.PENDING
    assert pending_g7[-1].reason_codes == ["CORRELATION_NOT_YET_COMPUTED"]


def test_platform_failure_archives_and_unresolved_statuses_do_not_promote(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    correlation_policy: CorrelationPolicy,
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
    failed = PlatformCheckService(repository).check(failing_id, failing_adapter, correlation_policy)
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
    pending = PlatformCheckService(repository).check(
        pending_id, pending_adapter, correlation_policy
    )
    assert pending.candidate_status is CandidateStatus.READY_FOR_PLATFORM_CHECK
    g6 = [
        decision
        for decision in repository.list_decisions(pending_run.run_id)
        if decision.gate_name == "G6_PLATFORM_READINESS"
    ]
    assert g6[0].outcome == GateOutcome.PENDING
    assert pending_adapter.correlation_calls == 0  # never reached: checks did not all pass


def test_platform_failures_call_repair_with_check_name_and_bound_children(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    correlation_policy: CorrelationPolicy,
    metadata: MetadataSnapshot,
) -> None:
    register_default_prompts(repository)
    add_approved_platform_card(repository, "CONCENTRATED_WEIGHT")
    add_approved_platform_card(repository, "LOW_SHARPE")
    constrained = limit_policy.model_copy(update={"children_per_parent_per_generation": 1})
    capture = metadata_capture(metadata)
    run = create_live_run(
        make_run,
        capture,
        _limit_policy=constrained,
        generator_prompt_version="generator.v1",
        repair_prompt_version="repair-planner.v1",
    )
    adapter = StubLiveAdapter(
        metadata,
        check_responses=[
            [
                PlatformCheck(name="CONCENTRATED_WEIGHT", status=PlatformCheckStatus.FAIL),
                PlatformCheck(name="LOW_SHARPE", status=PlatformCheckStatus.FAIL),
            ]
        ],
    )
    execution = ResearchOrchestrator(repository, routing_policy, constrained).execute(
        run.run_id, adapter
    )
    assert execution.status is RunStatus.COMPLETED
    parent = repository.list_candidates(run.run_id)[0]
    client = StaticLLMClient(
        json.dumps(
            {
                "action": "REPAIR",
                "reason": "Use one source-supported repair",
                "proposals": [
                    {
                        "changed_dimension": "field",
                        "change_reason": "Replace one sparse input",
                        "expected_effect": "Coverage should change",
                        "revised_expression": "rank(volume)",
                        "revised_settings": parent.settings_json,
                        "source_method_card_ids": [],
                        "falsification_condition": "Stop if coverage does not change",
                    }
                ],
            }
        )
    )
    planner = RepairPlanner(repository, MethodCardRetriever(repository), client)

    summary = PlatformCheckService(repository, planner).check(
        parent.candidate_id, adapter, correlation_policy
    )

    assert summary.candidate_status is CandidateStatus.ARCHIVED
    assert summary.repair_stop_reason == "CHILDREN_PER_PARENT_LIMIT_REACHED"
    assert len(summary.repair_candidate_ids) == 1
    repair_context = json.loads(client.user_prompts[0])
    assert repair_context["failure_code"] == "CONCENTRATED_WEIGHT"
    child = repository.get_candidate(summary.repair_candidate_ids[0])
    assert child.parent_candidate_id == parent.candidate_id
    assert child.generation_number == 1
    assert repository.count_children(parent.candidate_id, 1) == 1


def test_complete_intercepted_daily_cycle_generates_repairs_and_reaches_review(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    correlation_policy: CorrelationPolicy,
    metadata: MetadataSnapshot,
) -> None:
    register_default_prompts(repository)
    add_approved_platform_card(repository, "CONCENTRATED_WEIGHT")
    capture = metadata_capture(metadata)
    run = create_live_run(
        make_run,
        capture,
        generator_prompt_version="generator.v1",
        repair_prompt_version="repair-planner.v1",
    )
    adapter = StubLiveAdapter(
        metadata,
        check_responses=[
            [PlatformCheck(name="CONCENTRATED_WEIGHT", status=PlatformCheckStatus.FAIL)],
            [PlatformCheck(name="CONCENTRATED_WEIGHT", status=PlatformCheckStatus.PASS)],
        ],
    )
    generator_client = StaticLLMClient(
        json.dumps(
            {
                "proposals": [
                    {
                        "expression": "rank(close)",
                        "settings": {
                            "neutralization": "SUBINDUSTRY",
                            "truncation": 0.08,
                        },
                        "mechanism_claim": "A relative price measure may persist.",
                        "source_method_card_ids": [],
                        "uncertainty": "The signal may not persist outside this test.",
                    }
                ]
            }
        )
    )
    repair_client = StaticLLMClient(
        json.dumps(
            {
                "action": "REPAIR",
                "reason": "Change one sparse input",
                "proposals": [
                    {
                        "changed_dimension": "field",
                        "change_reason": "Replace one input",
                        "expected_effect": "The platform weight check should change",
                        "revised_expression": "rank(volume)",
                        "revised_settings": {
                            "neutralization": "SUBINDUSTRY",
                            "truncation": 0.08,
                        },
                        "source_method_card_ids": [],
                        "falsification_condition": "Stop if the weight check still fails",
                    }
                ],
            }
        )
    )
    retriever = MethodCardRetriever(repository)
    generator = CandidateGenerator(repository, retriever, generator_client)
    planner = RepairPlanner(repository, retriever, repair_client)
    orchestrator = ResearchOrchestrator(
        repository,
        routing_policy,
        limit_policy,
        candidate_generator=generator,
        repair_planner=planner,
        research_query="bounded relative price hypothesis",
    )

    first_execution = orchestrator.execute(run.run_id, adapter)
    parent = repository.list_candidates(run.run_id)[0]
    first_check = PlatformCheckService(repository, planner).check(
        parent.candidate_id, adapter, correlation_policy
    )
    second_execution = orchestrator.execute(run.run_id, adapter)
    child = repository.get_candidate(first_check.repair_candidate_ids[0])
    second_check = PlatformCheckService(repository, planner).check(
        child.candidate_id, adapter, correlation_policy
    )

    assert first_execution.status is RunStatus.COMPLETED
    assert first_check.candidate_status is CandidateStatus.ARCHIVED
    assert second_execution.status is RunStatus.COMPLETED
    assert second_check.candidate_status is CandidateStatus.READY_FOR_HUMAN_REVIEW
    assert child.parent_candidate_id == parent.candidate_id
    assert repository.count_attempts(run.run_id) == 2
    assert len(repository.list_llm_requests(run.run_id)) == 2


def test_one_execute_call_settles_a_platform_check_repair_all_the_way_to_review(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    correlation_policy: CorrelationPolicy,
    metadata: MetadataSnapshot,
) -> None:
    """Same repair-then-review scenario as the test above, but with a correlation
    policy configured on the orchestrator: the FAIL, the repair, the child's own
    simulation, and its PASS + correlation check must all settle inside a single
    execute() call -- nobody should have to remember a second command to get here."""
    register_default_prompts(repository)
    add_approved_platform_card(repository, "CONCENTRATED_WEIGHT")
    capture = metadata_capture(metadata)
    run = create_live_run(
        make_run,
        capture,
        generator_prompt_version="generator.v1",
        repair_prompt_version="repair-planner.v1",
    )
    adapter = StubLiveAdapter(
        metadata,
        check_responses=[
            [PlatformCheck(name="CONCENTRATED_WEIGHT", status=PlatformCheckStatus.FAIL)],
            [PlatformCheck(name="CONCENTRATED_WEIGHT", status=PlatformCheckStatus.PASS)],
        ],
    )
    generator_client = StaticLLMClient(
        json.dumps(
            {
                "proposals": [
                    {
                        "expression": "rank(close)",
                        "settings": {"neutralization": "SUBINDUSTRY", "truncation": 0.08},
                        "mechanism_claim": "A relative price measure may persist.",
                        "source_method_card_ids": [],
                        "uncertainty": "The signal may not persist outside this test.",
                    }
                ]
            }
        )
    )
    repair_client = StaticLLMClient(
        json.dumps(
            {
                "action": "REPAIR",
                "reason": "Change one sparse input",
                "proposals": [
                    {
                        "changed_dimension": "field",
                        "change_reason": "Replace one input",
                        "expected_effect": "The platform weight check should change",
                        "revised_expression": "rank(volume)",
                        "revised_settings": {
                            "neutralization": "SUBINDUSTRY",
                            "truncation": 0.08,
                        },
                        "source_method_card_ids": [],
                        "falsification_condition": "Stop if the weight check still fails",
                    }
                ],
            }
        )
    )
    retriever = MethodCardRetriever(repository)
    generator = CandidateGenerator(repository, retriever, generator_client)
    planner = RepairPlanner(repository, retriever, repair_client)
    orchestrator = ResearchOrchestrator(
        repository,
        routing_policy,
        limit_policy,
        candidate_generator=generator,
        repair_planner=planner,
        research_query="bounded relative price hypothesis",
        correlation_policy=correlation_policy,
    )

    execution = orchestrator.execute(run.run_id, adapter)

    candidates = {c.parent_candidate_id: c for c in repository.list_candidates(run.run_id)}
    parent = candidates[None]
    child = candidates[parent.candidate_id]
    assert execution.status is RunStatus.COMPLETED
    assert parent.status == CandidateStatus.ARCHIVED
    assert child.status == CandidateStatus.READY_FOR_HUMAN_REVIEW
    assert repository.count_attempts(run.run_id) == 2
    assert adapter.check_calls == 2
    assert adapter.correlation_calls == 1


def test_platform_check_retries_retryable_failures_within_frozen_limit(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    correlation_policy: CorrelationPolicy,
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
    summary = PlatformCheckService(repository).check(candidate_id, adapter, correlation_policy)
    persisted = repository.list_platform_checks(run.run_id)
    assert summary.request_attempts == 2
    assert summary.candidate_status is CandidateStatus.READY_FOR_HUMAN_REVIEW
    assert [item.status for item in persisted] == ["ERROR", "PASS"]


def test_platform_check_enforces_live_run_and_matching_metadata(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
    correlation_policy: CorrelationPolicy,
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
        PlatformCheckService(repository).check(
            candidate_id, FakeBrainAdapter(metadata), correlation_policy
        )
    wrong_metadata = metadata.model_copy(update={"dataset_id": "other"})
    with pytest.raises(ValueError, match="frozen run"):
        PlatformCheckService(repository).check(
            candidate_id, StubLiveAdapter(wrong_metadata), correlation_policy
        )

    proposed_run = create_live_run(make_run, metadata_capture(metadata))
    proposed = repository.add_candidate(
        proposed_run.run_id, proposal(), metadata.operators
    ).candidate
    with pytest.raises(ValueError, match="not ready"):
        PlatformCheckService(repository).check(proposed.candidate_id, adapter, correlation_policy)
