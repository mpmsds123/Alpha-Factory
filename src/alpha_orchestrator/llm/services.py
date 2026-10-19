"""Ledgered candidate generation and single-parent repair planning."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import TypeVar
from uuid import UUID

from pydantic import BaseModel, ValidationError

from alpha_orchestrator.config import LimitPolicy
from alpha_orchestrator.domain.enums import (
    DataType,
    ErrorClass,
    LLMRole,
    LLMValidationStatus,
    RepairAction,
    RunStatus,
)
from alpha_orchestrator.domain.errors import assert_no_sensitive_data
from alpha_orchestrator.domain.identity import canonicalize_expression, sha256_text
from alpha_orchestrator.domain.knowledge import MethodCard
from alpha_orchestrator.domain.llm import CandidateProposalBatch, RepairPlan
from alpha_orchestrator.domain.models import CandidateProposal, MetadataSnapshot
from alpha_orchestrator.knowledge.retriever import MethodCardRetriever
from alpha_orchestrator.llm.base import LLMClient
from alpha_orchestrator.storage.models import CandidateORM, RunORM
from alpha_orchestrator.storage.repository import CandidateInsertResult, Repository

TModel = TypeVar("TModel", bound=BaseModel)
STRUCTURAL_FAILURES = frozenset(
    str(error_class)
    for error_class in {
        ErrorClass.INVALID_EXPRESSION,
        ErrorClass.INVALID_FIELD,
        ErrorClass.INVALID_OPERATOR,
        ErrorClass.INVALID_SETTING,
    }
)


class LLMOutputRejected(ValueError):
    pass


class SearchLimitReached(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class CandidateGenerationResult:
    request_id: str
    candidates: tuple[CandidateInsertResult, ...]


@dataclass(frozen=True)
class RepairPlanningResult:
    request_id: str | None
    plan: RepairPlan
    candidates: tuple[CandidateInsertResult, ...]


class LedgeredLLMInvoker:
    def __init__(self, repository: Repository, client: LLMClient) -> None:
        self.repository = repository
        self.client = client

    def invoke(
        self,
        *,
        run: RunORM,
        role: LLMRole,
        prompt_version: str,
        related_candidate_id: str | None,
        context: dict[str, object],
        method_cards: Sequence[MethodCard],
        response_model: type[TModel],
        deterministic_validation: Callable[[TModel], None],
    ) -> tuple[str, TModel]:
        prompt = self.repository.get_prompt_version(prompt_version)
        if prompt.role != role:
            raise ValueError("Prompt role does not match the requested LLM role")
        assert_no_sensitive_data(context)
        user_prompt = json.dumps(context, sort_keys=True, separators=(",", ":"))
        schema = response_model.model_json_schema()
        input_hash = sha256_text(
            "\x1f".join(
                (
                    prompt.content_hash,
                    user_prompt,
                    json.dumps(schema, sort_keys=True, separators=(",", ":")),
                )
            )
        )
        request = self.repository.start_llm_request(
            prompt_version=prompt_version,
            role=role,
            model=self.client.model,
            temperature=self.client.temperature,
            input_hash=input_hash,
            context_manifest=_context_manifest(run, response_model, method_cards),
            related_run_id=run.run_id,
            related_candidate_id=related_candidate_id,
        )
        started = monotonic()
        try:
            completion = self.client.complete(
                system_prompt=prompt.content,
                user_prompt=user_prompt,
                json_schema=schema,
            )
        except Exception as exc:
            self.repository.finish_llm_request(
                request.request_id,
                validation_status=LLMValidationStatus.ERROR,
                output_hash=None,
                latency_ms=max(0, int((monotonic() - started) * 1_000)),
                error_message=f"LLM client error: {type(exc).__name__}",
            )
            raise
        output_hash = sha256_text(completion.raw_text)
        try:
            parsed = response_model.model_validate_json(completion.raw_text)
            deterministic_validation(parsed)
        except (ValidationError, ValueError) as exc:
            self.repository.finish_llm_request(
                request.request_id,
                validation_status=LLMValidationStatus.REJECTED,
                output_hash=output_hash,
                latency_ms=completion.latency_ms,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                total_tokens=completion.total_tokens,
                actual_model=completion.model,
                error_message="Schema or deterministic validation failed",
            )
            raise LLMOutputRejected("LLM output failed deterministic validation") from exc
        self.repository.finish_llm_request(
            request.request_id,
            validation_status=LLMValidationStatus.ACCEPTED,
            output_hash=output_hash,
            latency_ms=completion.latency_ms,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.total_tokens,
            actual_model=completion.model,
        )
        return request.request_id, parsed


class CandidateGenerator:
    def __init__(
        self,
        repository: Repository,
        retriever: MethodCardRetriever,
        client: LLMClient,
        base_settings: dict[str, object] | None = None,
    ) -> None:
        self.repository = repository
        self.retriever = retriever
        self.invoker = LedgeredLLMInvoker(repository, client)
        self.base_settings = dict(base_settings or {})

    def generate(
        self,
        run_id: str | UUID,
        metadata: MetadataSnapshot,
        *,
        query: str,
        retrieval_limit: int = 5,
    ) -> CandidateGenerationResult:
        run = self.repository.get_run(run_id)
        if metadata.dataset_id != run.dataset_id:
            raise ValueError("Metadata dataset does not match the frozen run")
        limits = LimitPolicy.model_validate(run.limit_policy_json)
        self._guard(run, limits)
        cards = [
            result.card
            for result in self.retriever.retrieve(
                query=query,
                data_type=DataType(run.data_type),
                limit=retrieval_limit,
            )
        ]
        capacity = min(
            limits.initial_candidates,
            limits.maximum_distinct_candidates - self.repository.count_candidates(run.run_id),
        )
        context = {
            "run": _run_context(run),
            "metadata": _metadata_context(metadata),
            "approved_method_cards": [card.prompt_summary() for card in cards],
            "maximum_proposals": capacity,
            "research_query": query,
        }
        allowed_cards = {card.card_id for card in cards}

        def validate(batch: CandidateProposalBatch) -> None:
            assert_no_sensitive_data(batch.model_dump(mode="python"))
            if len(batch.proposals) > capacity:
                raise ValueError("Generator exceeded the frozen candidate capacity")
            _validate_card_references(batch.proposals, allowed_cards)
            for proposal in batch.proposals:
                canonicalize_expression(proposal.expression, metadata.operators)

        request_id, batch = self.invoker.invoke(
            run=run,
            role=LLMRole.GENERATOR,
            prompt_version=run.generator_prompt_version,
            related_candidate_id=None,
            context=context,
            method_cards=cards,
            response_model=CandidateProposalBatch,
            deterministic_validation=validate,
        )
        normalized_proposals = (
            proposal.model_copy(update={"settings": {**proposal.settings, **self.base_settings}})
            for proposal in batch.proposals
        )
        inserted = tuple(
            self.repository.add_candidate(
                run.run_id,
                proposal,
                metadata.operators,
                llm_request_id=request_id,
            )
            for proposal in normalized_proposals
        )
        return CandidateGenerationResult(request_id=request_id, candidates=inserted)

    def _guard(self, run: RunORM, limits: LimitPolicy) -> None:
        reasons = (
            (
                self.repository.count_llm_requests(run.run_id) >= limits.maximum_llm_requests,
                "MAXIMUM_LLM_REQUESTS_REACHED",
            ),
            (
                self.repository.count_candidates(run.run_id) >= limits.maximum_distinct_candidates,
                "MAXIMUM_DISTINCT_CANDIDATES_REACHED",
            ),
            (
                self.repository.count_pending(run.run_id) >= limits.maximum_pending_candidates,
                "MAXIMUM_PENDING_CANDIDATES_REACHED",
            ),
        )
        for reached, reason in reasons:
            if reached:
                self.repository.update_run_status(run.run_id, RunStatus.STOPPED, reason)
                raise SearchLimitReached(reason)


class RepairPlanner:
    def __init__(
        self,
        repository: Repository,
        retriever: MethodCardRetriever,
        client: LLMClient,
    ) -> None:
        self.repository = repository
        self.retriever = retriever
        self.invoker = LedgeredLLMInvoker(repository, client)

    def plan(
        self,
        candidate_id: str | UUID,
        *,
        failure_code: str | ErrorClass,
        evidence_snapshot: dict[str, object],
        metadata: MetadataSnapshot,
        retrieval_limit: int = 5,
    ) -> RepairPlanningResult:
        parent = self.repository.get_candidate(candidate_id)
        run = self.repository.get_run(parent.run_id)
        if metadata.dataset_id != run.dataset_id:
            raise ValueError("Metadata dataset does not match the frozen run")
        limits = LimitPolicy.model_validate(run.limit_policy_json)
        failure_code_value = str(failure_code)
        self._guard(run, parent, limits, failure_code_value)
        cards = [
            result.card
            for result in self.retriever.retrieve(
                query=f"{failure_code_value} {parent.mechanism_claim}",
                data_type=DataType(run.data_type),
                failure_codes={failure_code_value},
                limit=retrieval_limit,
            )
        ]
        if not cards:
            return RepairPlanningResult(
                request_id=None,
                plan=RepairPlan(
                    action=RepairAction.STOP,
                    reason=(
                        "No approved method card matches this failure code; "
                        "repair planning must not improvise."
                    ),
                    proposals=[],
                ),
                candidates=(),
            )
        existing_children = self.repository.count_children(
            parent.candidate_id, parent.generation_number + 1
        )
        child_capacity = min(
            2,
            limits.children_per_parent_per_generation - existing_children,
            limits.maximum_distinct_candidates - self.repository.count_candidates(run.run_id),
        )
        context = {
            "run": _run_context(run),
            "parent_candidate": {
                "candidate_id": parent.candidate_id,
                "generation_number": parent.generation_number,
                "expression": parent.expression,
                "settings": parent.settings_json,
                "mechanism_claim": parent.mechanism_claim,
                "previous_changed_dimensions": _lineage_dimensions(self.repository, parent),
            },
            "failure_code": failure_code_value,
            "evidence_snapshot": evidence_snapshot,
            "approved_method_cards": [card.prompt_summary() for card in cards],
            "remaining_child_capacity": child_capacity,
        }
        allowed_cards = {card.card_id for card in cards}

        def validate(plan: RepairPlan) -> None:
            assert_no_sensitive_data(plan.model_dump(mode="python"))
            if len(plan.proposals) > child_capacity:
                raise ValueError("Repair planner exceeded the frozen child capacity")
            _validate_repair_card_references(plan, allowed_cards)
            for proposal in plan.proposals:
                canonicalize_expression(proposal.revised_expression, metadata.operators)

        request_id, plan = self.invoker.invoke(
            run=run,
            role=LLMRole.REPAIR_PLANNER,
            prompt_version=run.repair_prompt_version,
            related_candidate_id=parent.candidate_id,
            context=context,
            method_cards=cards,
            response_model=RepairPlan,
            deterministic_validation=validate,
        )
        inserted: tuple[CandidateInsertResult, ...] = ()
        if plan.action is RepairAction.REPAIR:
            inserted = tuple(
                self.repository.add_candidate(
                    run.run_id,
                    CandidateProposal(
                        expression=proposal.revised_expression,
                        settings=proposal.revised_settings,
                        mechanism_claim=proposal.expected_effect,
                        source_method_card_ids=proposal.source_method_card_ids,
                        parent_candidate_id=UUID(parent.candidate_id),
                        generation_number=parent.generation_number + 1,
                        changed_dimension=proposal.changed_dimension,
                        change_reason=proposal.change_reason,
                    ),
                    metadata.operators,
                    llm_request_id=request_id,
                )
                for proposal in plan.proposals
            )
        return RepairPlanningResult(request_id=request_id, plan=plan, candidates=inserted)

    def _guard(
        self,
        run: RunORM,
        parent: CandidateORM,
        limits: LimitPolicy,
        failure_code: str,
    ) -> None:
        next_generation = parent.generation_number + 1
        reasons = [
            (
                self.repository.count_llm_requests(run.run_id) >= limits.maximum_llm_requests,
                "MAXIMUM_LLM_REQUESTS_REACHED",
            ),
            (
                next_generation > limits.maximum_generations,
                "MAXIMUM_GENERATIONS_REACHED",
            ),
            (
                self.repository.count_children(parent.candidate_id, next_generation)
                >= limits.children_per_parent_per_generation,
                "CHILDREN_PER_PARENT_LIMIT_REACHED",
            ),
            (
                self.repository.count_candidates(run.run_id) >= limits.maximum_distinct_candidates,
                "MAXIMUM_DISTINCT_CANDIDATES_REACHED",
            ),
            (
                self.repository.count_pending(run.run_id) >= limits.maximum_pending_candidates,
                "MAXIMUM_PENDING_CANDIDATES_REACHED",
            ),
        ]
        if failure_code in STRUCTURAL_FAILURES:
            reasons.append(
                (
                    self.repository.count_llm_requests(
                        run.run_id,
                        LLMRole.REPAIR_PLANNER,
                        parent.candidate_id,
                    )
                    >= limits.structural_repairs_per_candidate,
                    "STRUCTURAL_REPAIR_LIMIT_REACHED",
                )
            )
        for reached, reason in reasons:
            if reached:
                self.repository.update_run_status(run.run_id, RunStatus.STOPPED, reason)
                raise SearchLimitReached(reason)


def _validate_card_references(
    proposals: Sequence[CandidateProposal], allowed_cards: set[str]
) -> None:
    referenced = {card_id for proposal in proposals for card_id in proposal.source_method_card_ids}
    if not referenced <= allowed_cards:
        raise ValueError("LLM cited a method card that was not approved and retrieved")


def _validate_repair_card_references(plan: RepairPlan, allowed_cards: set[str]) -> None:
    referenced = {
        card_id for proposal in plan.proposals for card_id in proposal.source_method_card_ids
    }
    if not referenced <= allowed_cards:
        raise ValueError("LLM cited a method card that was not approved and retrieved")


def _run_context(run: RunORM) -> dict[str, object]:
    return {
        "run_id": run.run_id,
        "dataset_id": run.dataset_id,
        "region": run.region,
        "universe": run.universe,
        "delay": run.delay,
        "data_type": run.data_type,
        "field_snapshot_id": run.field_snapshot_id,
        "operator_snapshot_id": run.operator_snapshot_id,
        "setting_snapshot_id": run.setting_snapshot_id,
        "routing_policy_version": run.routing_policy_version,
        "limit_policy_version": run.limit_policy_version,
    }


def _context_manifest(
    run: RunORM,
    response_model: type[BaseModel],
    method_cards: Sequence[MethodCard],
) -> dict[str, object]:
    return {
        "response_schema": response_model.__name__,
        "field_snapshot_id": run.field_snapshot_id,
        "operator_snapshot_id": run.operator_snapshot_id,
        "setting_snapshot_id": run.setting_snapshot_id,
        "method_cards": [
            {"card_id": card.card_id, "version": card.version}
            for card in sorted(method_cards, key=lambda item: (item.card_id, item.version))
        ],
    }


def _metadata_context(metadata: MetadataSnapshot) -> dict[str, object]:
    return {
        "dataset_id": metadata.dataset_id,
        "fields": {name: value for name, value in sorted(metadata.fields.items())},
        "operators": sorted(metadata.operators),
        # BRAIN's own call signature per operator (e.g. "group_rank(x, group)"),
        # not just its name -- without this the model has to guess argument
        # counts from its own training data. A name absent here has no signature
        # on record; that is not a green light to guess it either.
        "operator_definitions": dict(sorted(metadata.operator_definitions.items())),
        "settings": {
            name: sorted(values, key=repr) for name, values in sorted(metadata.settings.items())
        },
    }


def _lineage_dimensions(repository: Repository, candidate: CandidateORM) -> list[str]:
    dimensions: list[str] = []
    current = candidate
    while current.parent_candidate_id is not None:
        if current.changed_dimension:
            dimensions.append(current.changed_dimension)
        current = repository.get_candidate(current.parent_candidate_id)
    return list(reversed(dimensions))
