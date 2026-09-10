import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from alpha_orchestrator.adapters.fake import FakeBrainAdapter
from alpha_orchestrator.config import LimitPolicy, RoutingPolicy
from alpha_orchestrator.domain.enums import (
    CandidateStatus,
    DataType,
    ErrorClass,
    EvidenceLevel,
    LLMRole,
    LLMValidationStatus,
    RepairAction,
)
from alpha_orchestrator.domain.knowledge import MethodCard, SourceReference
from alpha_orchestrator.domain.llm import RepairPlan
from alpha_orchestrator.domain.models import (
    CandidateProposal,
    MetadataSnapshot,
    NormalizedSimulationResult,
    RunSpecification,
    SimulationMetrics,
)
from alpha_orchestrator.knowledge.retriever import MethodCardRetriever
from alpha_orchestrator.llm.base import LLMCompletion
from alpha_orchestrator.llm.prompts import register_default_prompts
from alpha_orchestrator.llm.services import (
    CandidateGenerator,
    LLMOutputRejected,
    RepairPlanner,
    SearchLimitReached,
)
from alpha_orchestrator.orchestration.engine import OfflineOrchestrator
from alpha_orchestrator.reporting.report import build_run_report
from alpha_orchestrator.storage.repository import Repository


@dataclass
class FakeLLMClient:
    raw_text: str
    model: str = "fixture-model"
    temperature: float = 0.0
    raises: Exception | None = None
    calls: int = 0

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, object],
    ) -> LLMCompletion:
        assert "submit" in system_prompt and "Alpha" in system_prompt
        assert json.loads(user_prompt)["run"]["dataset_id"] == "demo-dataset"
        assert json_schema["type"] == "object"
        self.calls += 1
        if self.raises:
            raise self.raises
        return LLMCompletion(
            raw_text=self.raw_text,
            model=self.model,
            latency_ms=7,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )


def generator_json(
    expression: str = "rank(close)",
    source_cards: list[str] | None = None,
    uncertainty: str | None = "Persistence may disappear outside the observed regime.",
) -> str:
    proposal = {
        "expression": expression,
        "settings": {
            "neutralization": "SUBINDUSTRY",
            "truncation": 0.08,
        },
        "mechanism_claim": "A relative price measure may persist briefly.",
        "source_method_card_ids": source_cards or [],
    }
    if uncertainty is not None:
        proposal["uncertainty"] = uncertainty
    return json.dumps({"proposals": [proposal]})


def repair_json(action: RepairAction = RepairAction.REPAIR) -> str:
    if action is RepairAction.STOP:
        return json.dumps(
            {"action": "STOP", "reason": "No evidence-supported repair remains", "proposals": []}
        )
    return json.dumps(
        {
            "action": "REPAIR",
            "reason": "Test one targeted field repair",
            "proposals": [
                {
                    "changed_dimension": "field",
                    "change_reason": "Replace one sparse field",
                    "expected_effect": "Coverage should improve without changing the mechanism",
                    "revised_expression": "rank(volume)",
                    "revised_settings": {
                        "neutralization": "SUBINDUSTRY",
                        "truncation": 0.08,
                    },
                    "source_method_card_ids": [],
                    "falsification_condition": "Stop if coverage does not improve",
                }
            ],
        }
    )


def prepare_run(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    **overrides: object,
) -> RunSpecification:
    register_default_prompts(repository)
    return make_run(
        generator_prompt_version="generator.v1",
        repair_prompt_version="repair-planner.v1",
        **overrides,
    )


def test_valid_generator_output_is_ledgered_then_inserted(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    metadata: MetadataSnapshot,
) -> None:
    run = prepare_run(repository, make_run)
    client = FakeLLMClient(generator_json())
    result = CandidateGenerator(repository, MethodCardRetriever(repository), client).generate(
        run.run_id, metadata, query="price persistence"
    )
    assert len(result.candidates) == 1
    assert result.candidates[0].created
    assert client.calls == 1
    requests = repository.list_llm_requests(run.run_id)
    assert len(requests) == 1
    assert requests[0].validation_status == LLMValidationStatus.ACCEPTED
    assert requests[0].total_tokens == 15
    assert requests[0].output_hash is not None
    assert requests[0].context_manifest_json == {
        "response_schema": "CandidateProposalBatch",
        "field_snapshot_id": "fields.v1",
        "operator_snapshot_id": "operators.v1",
        "setting_snapshot_id": "settings.v1",
        "method_cards": [],
    }
    assert not hasattr(requests[0], "raw_output")
    proposal_event = next(
        event
        for event in repository.list_events(run.run_id)
        if event.event_type == "CANDIDATE_PROPOSED"
    )
    assert proposal_event.payload_json["llm_request_id"] == result.request_id
    assert proposal_event.payload_json["proposal_uncertainty"].startswith("Persistence")
    report = build_run_report(repository, run.run_id)
    assert "- Requests: 1" in report
    assert "- Recorded tokens: 15" in report
    assert "LLM ACCEPTED: 1" in report


def test_generator_ledger_records_exact_approved_method_card_version(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    metadata: MetadataSnapshot,
) -> None:
    run = prepare_run(repository, make_run)
    repository.upsert_method_card(
        MethodCard(
            card_id="MC-GENERATOR-001",
            title="Generate a bounded price hypothesis",
            version="v3",
            status="approved",
            applicable_data_types=[DataType.MATRIX],
            prerequisites=["Use the frozen metadata snapshot"],
            diagnostic_signals=["A rankable matrix field is present"],
            allowed_repair_actions=["Change one field"],
            prohibited_actions=["Do not change limits"],
            stop_condition="Stop when no falsifiable hypothesis remains.",
            evidence_level=EvidenceLevel.TRAINING,
            source_references=[
                SourceReference(source_id="SRC-GENERATOR", section_locator="lines:1-8")
            ],
            reviewer="Research Reviewer",
            approval_timestamp=datetime.now(UTC),
        )
    )
    result = CandidateGenerator(
        repository,
        MethodCardRetriever(repository),
        FakeLLMClient(generator_json(source_cards=["MC-GENERATOR-001"])),
    ).generate(run.run_id, metadata, query="bounded price hypothesis")
    request = repository.list_llm_requests(run.run_id)[0]
    assert request.context_manifest_json["method_cards"] == [
        {"card_id": "MC-GENERATOR-001", "version": "v3"}
    ]
    assert result.candidates[0].candidate.source_method_card_ids == ["MC-GENERATOR-001"]


@pytest.mark.parametrize(
    "raw_text",
    [
        "not-json",
        json.dumps({"proposals": [{"expression": "rank(close)"}]}),
        generator_json(source_cards=["MC-NOT-RETRIEVED"]),
        generator_json(uncertainty=None),
    ],
)
def test_invalid_generator_output_is_rejected_before_candidate_insert(
    raw_text: str,
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    metadata: MetadataSnapshot,
) -> None:
    run = prepare_run(repository, make_run)
    generator = CandidateGenerator(
        repository, MethodCardRetriever(repository), FakeLLMClient(raw_text)
    )
    with pytest.raises(LLMOutputRejected):
        generator.generate(run.run_id, metadata, query="test")
    assert repository.count_candidates(run.run_id) == 0
    request = repository.list_llm_requests(run.run_id)[0]
    assert request.validation_status == LLMValidationStatus.REJECTED
    assert request.output_hash is not None


def test_generator_may_return_no_candidate_when_no_hypothesis_is_defensible(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    metadata: MetadataSnapshot,
) -> None:
    run = prepare_run(repository, make_run)
    result = CandidateGenerator(
        repository,
        MethodCardRetriever(repository),
        FakeLLMClient(json.dumps({"proposals": []})),
    ).generate(run.run_id, metadata, query="no defensible hypothesis")
    assert result.candidates == ()
    assert repository.list_llm_requests(run.run_id)[0].validation_status == "ACCEPTED"


def test_credential_like_generator_output_is_rejected_and_never_persisted(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    metadata: MetadataSnapshot,
) -> None:
    run = prepare_run(repository, make_run)
    payload = json.loads(generator_json())
    payload["proposals"][0]["settings"]["api_key"] = "EXAMPLE_NOT_REAL"
    generator = CandidateGenerator(
        repository,
        MethodCardRetriever(repository),
        FakeLLMClient(json.dumps(payload)),
    )
    with pytest.raises(LLMOutputRejected):
        generator.generate(run.run_id, metadata, query="test")
    assert repository.count_candidates(run.run_id) == 0
    assert repository.list_llm_requests(run.run_id)[0].validation_status == "REJECTED"


def test_llm_transport_error_is_redacted_and_does_not_insert(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    metadata: MetadataSnapshot,
) -> None:
    run = prepare_run(repository, make_run)
    client = FakeLLMClient("", raises=RuntimeError("token=EXAMPLE_DO_NOT_STORE"))
    generator = CandidateGenerator(repository, MethodCardRetriever(repository), client)
    with pytest.raises(RuntimeError):
        generator.generate(run.run_id, metadata, query="test")
    request = repository.list_llm_requests(run.run_id)[0]
    assert request.validation_status == LLMValidationStatus.ERROR
    assert "EXAMPLE_DO_NOT_STORE" not in (request.error_message_redacted or "")
    assert request.error_message_redacted == "LLM client error: RuntimeError"
    assert repository.count_candidates(run.run_id) == 0


def test_repair_planner_creates_only_single_parent_children(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    metadata: MetadataSnapshot,
) -> None:
    run = prepare_run(repository, make_run)
    parent = repository.add_candidate(
        run.run_id,
        CandidateProposal(
            expression="rank(close)",
            settings={"neutralization": "SUBINDUSTRY", "truncation": 0.08},
            mechanism_claim="A relative price measure may persist briefly.",
        ),
        metadata.operators,
    ).candidate
    planner = RepairPlanner(
        repository, MethodCardRetriever(repository), FakeLLMClient(repair_json())
    )
    result = planner.plan(
        parent.candidate_id,
        failure_code=ErrorClass.INVALID_FIELD,
        evidence_snapshot={"validator_code": "INVALID_FIELD"},
        metadata=metadata,
    )
    assert result.plan.action is RepairAction.REPAIR
    assert len(result.candidates) == 1
    child = result.candidates[0].candidate
    assert child.parent_candidate_id == parent.candidate_id
    assert child.generation_number == 1
    assert child.changed_dimension == "field"

    with pytest.raises(SearchLimitReached, match="STRUCTURAL_REPAIR_LIMIT_REACHED"):
        planner.plan(
            parent.candidate_id,
            failure_code=ErrorClass.INVALID_FIELD,
            evidence_snapshot={},
            metadata=metadata,
        )
    assert repository.get_run(run.run_id).stop_reason == "STRUCTURAL_REPAIR_LIMIT_REACHED"


def test_repair_planner_can_stop_without_creating_a_child(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    metadata: MetadataSnapshot,
) -> None:
    run = prepare_run(repository, make_run, region="EUR")
    parent = repository.add_candidate(
        run.run_id,
        CandidateProposal(expression="rank(close)", settings={}, mechanism_claim="Test claim"),
        metadata.operators,
    ).candidate
    result = RepairPlanner(
        repository,
        MethodCardRetriever(repository),
        FakeLLMClient(repair_json(RepairAction.STOP)),
    ).plan(
        parent.candidate_id,
        failure_code="LOW_COVERAGE",
        evidence_snapshot={},
        metadata=metadata,
    )
    assert result.plan.action is RepairAction.STOP
    assert result.candidates == ()
    assert repository.count_candidates(run.run_id) == 1


def test_generated_candidate_can_be_repaired_and_replayed_after_run_completion(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    metadata: MetadataSnapshot,
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
) -> None:
    run = prepare_run(repository, make_run)
    generated = CandidateGenerator(
        repository,
        MethodCardRetriever(repository),
        FakeLLMClient(generator_json()),
    ).generate(run.run_id, metadata, query="price persistence")
    parent = generated.candidates[0].candidate
    first_adapter = FakeBrainAdapter(
        metadata,
        results={
            "rank(close)": NormalizedSimulationResult(
                completed=True,
                metrics=SimulationMetrics(sharpe=0.6, fitness=0.4, turnover=0.2),
            )
        },
    )
    first_summary = OfflineOrchestrator(repository, routing_policy, limit_policy).execute(
        run.run_id, first_adapter
    )
    assert first_summary.status == "COMPLETED"
    assert repository.get_candidate(parent.candidate_id).status == "READY_FOR_PLATFORM_CHECK"

    repaired = RepairPlanner(
        repository,
        MethodCardRetriever(repository),
        FakeLLMClient(repair_json()),
    ).plan(
        parent.candidate_id,
        failure_code="LOW_COVERAGE",
        evidence_snapshot={"platform_code": "LOW_COVERAGE"},
        metadata=metadata,
    )
    child = repaired.candidates[0].candidate
    assert child.status == CandidateStatus.PROPOSED

    second_adapter = FakeBrainAdapter(
        metadata,
        results={
            "rank(volume)": NormalizedSimulationResult(
                completed=True,
                metrics=SimulationMetrics(sharpe=0.55, fitness=0.35, turnover=0.2),
            )
        },
    )
    second_summary = OfflineOrchestrator(repository, routing_policy, limit_policy).execute(
        run.run_id, second_adapter
    )
    assert second_summary.status == "COMPLETED"
    assert second_summary.simulation_attempts == 2
    assert repository.get_candidate(child.candidate_id).status == "READY_FOR_PLATFORM_CHECK"


def test_llm_request_ceiling_stops_before_another_call(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    metadata: MetadataSnapshot,
    limit_policy: LimitPolicy,
) -> None:
    constrained = limit_policy.model_copy(update={"maximum_llm_requests": 1})
    run = prepare_run(repository, make_run, _limit_policy=constrained, region="ASI")
    client = FakeLLMClient(generator_json())
    generator = CandidateGenerator(repository, MethodCardRetriever(repository), client)
    generator.generate(run.run_id, metadata, query="first")
    with pytest.raises(SearchLimitReached, match="MAXIMUM_LLM_REQUESTS_REACHED"):
        generator.generate(run.run_id, metadata, query="second")
    assert client.calls == 1
    assert repository.get_run(run.run_id).status == "STOPPED"


def test_prompt_versions_are_immutable_and_role_checked(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    metadata: MetadataSnapshot,
) -> None:
    register_default_prompts(repository)
    digest = repository.register_prompt_version(
        "generator.v1", LLMRole.GENERATOR, repository.get_prompt_version("generator.v1").content
    )
    assert len(digest) == 64
    with pytest.raises(ValueError, match="immutable"):
        repository.register_prompt_version("generator.v1", LLMRole.GENERATOR, "changed")
    with pytest.raises(ValueError, match="Credential-like"):
        repository.register_prompt_version(
            "unsafe.v1", LLMRole.GENERATOR, "api_key=EXAMPLE_NOT_REAL"
        )
    run = make_run(
        generator_prompt_version="repair-planner.v1",
        repair_prompt_version="repair-planner.v1",
    )
    with pytest.raises(ValueError, match="role"):
        CandidateGenerator(
            repository, MethodCardRetriever(repository), FakeLLMClient(generator_json())
        ).generate(run.run_id, metadata, query="test")


def test_generator_rejects_metadata_from_another_dataset_before_call(
    repository: Repository,
    make_run: Callable[..., RunSpecification],
    metadata: MetadataSnapshot,
) -> None:
    run = prepare_run(repository, make_run)
    client = FakeLLMClient(generator_json())
    with pytest.raises(ValueError, match="Metadata dataset"):
        CandidateGenerator(repository, MethodCardRetriever(repository), client).generate(
            run.run_id,
            metadata.model_copy(update={"dataset_id": "other-dataset"}),
            query="test",
        )
    assert client.calls == 0
    assert repository.list_llm_requests(run.run_id) == []


def test_repair_plan_schema_rejects_mismatched_action() -> None:
    payload = json.loads(repair_json())
    payload["action"] = "STOP"
    with pytest.raises(ValueError, match="STOP"):
        RepairPlan.model_validate(payload)
