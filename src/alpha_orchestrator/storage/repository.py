"""Transactional source-of-truth operations."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from alpha_orchestrator.config import LimitPolicy, RoutingPolicy
from alpha_orchestrator.domain.enums import (
    AttemptStatus,
    CandidateStatus,
    DataType,
    EvidenceLevel,
    GateOutcome,
    LLMRole,
    LLMValidationStatus,
    MetadataSnapshotKind,
    MethodCardStatus,
    RunStatus,
)
from alpha_orchestrator.domain.errors import assert_no_sensitive_data, redact_text
from alpha_orchestrator.domain.identity import (
    candidate_fingerprint,
    canonicalize_expression,
    metadata_snapshot_identity,
    normalized_content_hash,
)
from alpha_orchestrator.domain.knowledge import (
    MethodCard,
    SourceDocument,
    SourceReference,
    SourceSection,
)
from alpha_orchestrator.domain.models import (
    CandidateProposal,
    CapturedMetadata,
    MetadataSnapshot,
    NormalizedSimulationResult,
    PlatformCheck,
    RunSpecification,
    SimulationMetrics,
    utc_now,
)
from alpha_orchestrator.domain.state_machine import validate_transition
from alpha_orchestrator.storage.models import (
    CandidateORM,
    DecisionORM,
    EventORM,
    LineageEdgeORM,
    LLMRequestORM,
    MetadataSnapshotORM,
    MethodCardORM,
    PlatformCheckORM,
    PromptVersionORM,
    RunORM,
    SimulationAttemptORM,
    SourceDocumentORM,
    SourceSectionORM,
)


class MissingRecord(LookupError):
    pass


@dataclass(frozen=True)
class CandidateInsertResult:
    candidate: CandidateORM
    created: bool


class Repository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._sessions = session_factory

    def create_run(
        self,
        specification: RunSpecification,
        routing_policy: RoutingPolicy,
        limit_policy: LimitPolicy,
    ) -> RunORM:
        if specification.routing_policy_version != routing_policy.version:
            raise ValueError("Run routing-policy version does not match the snapshot")
        if specification.limit_policy_version != limit_policy.version:
            raise ValueError("Run limit-policy version does not match the snapshot")
        run = RunORM(
            **specification.model_dump(mode="python", exclude={"run_id"}),
            run_id=str(specification.run_id),
            routing_policy_json=routing_policy.model_dump(mode="json"),
            limit_policy_json=limit_policy.model_dump(mode="json"),
            status=RunStatus.CREATED,
        )
        with self._sessions.begin() as session:
            session.add(run)
            session.flush()
            session.add(
                EventORM(
                    event_id=str(uuid4()),
                    run_id=run.run_id,
                    event_type="RUN_CREATED",
                    payload_json={
                        "routing_policy_version": routing_policy.version,
                        "limit_policy_version": limit_policy.version,
                    },
                )
            )
        return run

    def get_run(self, run_id: str | UUID) -> RunORM:
        with self._sessions() as session:
            run = session.get(RunORM, str(run_id))
            if run is None:
                raise MissingRecord(f"Run not found: {run_id}")
            session.expunge(run)
            return run

    def update_run_status(
        self, run_id: str | UUID, status: RunStatus, stop_reason: str | None = None
    ) -> None:
        with self._sessions.begin() as session:
            run = session.get(RunORM, str(run_id))
            if run is None:
                raise MissingRecord(f"Run not found: {run_id}")
            run.status = status
            run.stop_reason = stop_reason
            # A run may be resumed hours or days later. The wall-clock budget applies
            # to this active worker invocation, not time spent safely stopped.
            if status is RunStatus.RUNNING:
                run.started_at = utc_now()
            if status is RunStatus.RUNNING:
                run.completed_at = None
            if status in {RunStatus.COMPLETED, RunStatus.STOPPED, RunStatus.FAILED}:
                run.completed_at = utc_now()
            session.add(
                EventORM(
                    event_id=str(uuid4()),
                    run_id=run.run_id,
                    event_type=f"RUN_{status}",
                    payload_json={"stop_reason": stop_reason},
                )
            )

    def add_candidate(
        self,
        run_id: str | UUID,
        proposal: CandidateProposal,
        known_operators: set[str],
        llm_request_id: str | None = None,
    ) -> CandidateInsertResult:
        assert_no_sensitive_data(proposal.model_dump(mode="python"))
        run = self.get_run(run_id)
        canonical_expression = canonicalize_expression(proposal.expression, known_operators)
        fingerprint = candidate_fingerprint(
            canonical_expression,
            proposal.settings,
            run.region,
            run.universe,
            run.delay,
        )
        content_hash = normalized_content_hash(canonical_expression, proposal.settings)
        with self._sessions.begin() as session:
            llm_request = None
            if llm_request_id is not None:
                llm_request = session.get(LLMRequestORM, llm_request_id)
                if llm_request is None or llm_request.related_run_id != str(run_id):
                    raise ValueError("LLM request must exist in the same run as the candidate")
            duplicate = session.scalar(
                select(CandidateORM).where(
                    CandidateORM.run_id == str(run_id),
                    (CandidateORM.candidate_fingerprint == fingerprint)
                    | (CandidateORM.normalized_content_hash == content_hash),
                )
            )
            if duplicate is not None:
                self._add_decision(
                    session,
                    duplicate.candidate_id,
                    "G2_DEDUPLICATION",
                    "identity.v1",
                    GateOutcome.REUSED,
                    ["DUPLICATE_REUSED"],
                    {},
                    llm_request_id,
                )
                duplicate_payload: dict[str, Any] = {"candidate_fingerprint": fingerprint}
                if proposal.uncertainty is not None:
                    duplicate_payload["proposal_uncertainty"] = proposal.uncertainty
                if llm_request is not None:
                    duplicate_payload["llm_request_id"] = llm_request.request_id
                self._add_event(
                    session,
                    str(run_id),
                    duplicate.candidate_id,
                    "DUPLICATE_REUSED",
                    duplicate_payload,
                )
                return CandidateInsertResult(candidate=duplicate, created=False)

            parent_id = str(proposal.parent_candidate_id) if proposal.parent_candidate_id else None
            if parent_id is not None:
                parent = session.get(CandidateORM, parent_id)
                if parent is None or parent.run_id != str(run_id):
                    raise ValueError("Child parent must exist in the same run")
            candidate = CandidateORM(
                candidate_id=str(uuid4()),
                run_id=str(run_id),
                parent_candidate_id=parent_id,
                generation_number=proposal.generation_number,
                expression=proposal.expression,
                canonical_expression=canonical_expression,
                settings_json=proposal.settings,
                candidate_fingerprint=fingerprint,
                normalized_content_hash=content_hash,
                mechanism_claim=proposal.mechanism_claim,
                source_method_card_ids=proposal.source_method_card_ids,
                changed_dimension=proposal.changed_dimension,
                change_reason=proposal.change_reason,
                status=CandidateStatus.PROPOSED,
            )
            session.add(candidate)
            session.flush()
            if parent_id is not None:
                session.add(
                    LineageEdgeORM(
                        run_id=str(run_id),
                        parent_candidate_id=parent_id,
                        child_candidate_id=candidate.candidate_id,
                        changed_dimension=proposal.changed_dimension or "",
                        change_reason=proposal.change_reason or "",
                    )
                )
            proposal_payload: dict[str, Any] = {"candidate_fingerprint": fingerprint}
            if proposal.uncertainty is not None:
                proposal_payload["proposal_uncertainty"] = proposal.uncertainty
            if llm_request is not None:
                proposal_payload["llm_request_id"] = llm_request.request_id
            self._add_event(
                session,
                str(run_id),
                candidate.candidate_id,
                "CANDIDATE_PROPOSED",
                proposal_payload,
            )
            return CandidateInsertResult(candidate=candidate, created=True)

    def get_candidate(self, candidate_id: str | UUID) -> CandidateORM:
        with self._sessions() as session:
            candidate = session.get(CandidateORM, str(candidate_id))
            if candidate is None:
                raise MissingRecord(f"Candidate not found: {candidate_id}")
            session.expunge(candidate)
            return candidate

    def list_candidates(
        self, run_id: str | UUID, statuses: set[CandidateStatus] | None = None
    ) -> list[CandidateORM]:
        with self._sessions() as session:
            statement = select(CandidateORM).where(CandidateORM.run_id == str(run_id))
            if statuses:
                statement = statement.where(CandidateORM.status.in_(statuses))
            candidates = list(session.scalars(statement.order_by(CandidateORM.created_at)))
            for candidate in candidates:
                session.expunge(candidate)
            return candidates

    def transition_candidate(
        self,
        candidate_id: str | UUID,
        target: CandidateStatus,
        gate_name: str,
        gate_version: str,
        outcome: GateOutcome,
        reason_codes: list[str],
        metric_snapshot: dict[str, Any] | None = None,
    ) -> CandidateORM:
        with self._sessions.begin() as session:
            candidate = session.get(CandidateORM, str(candidate_id))
            if candidate is None:
                raise MissingRecord(f"Candidate not found: {candidate_id}")
            source = CandidateStatus(candidate.status)
            validate_transition(source, target)
            candidate.status = target
            candidate.updated_at = utc_now()
            self._add_decision(
                session,
                candidate.candidate_id,
                gate_name,
                gate_version,
                outcome,
                reason_codes,
                metric_snapshot or {},
            )
            self._add_event(
                session,
                candidate.run_id,
                candidate.candidate_id,
                "CANDIDATE_TRANSITION",
                {"source": source, "target": target, "reason_codes": reason_codes},
            )
            session.flush()
            session.expunge(candidate)
            return candidate

    def start_attempt(
        self, candidate_id: str | UUID, idempotency_key: str, attempt_number: int
    ) -> SimulationAttemptORM:
        with self._sessions.begin() as session:
            candidate = session.get(CandidateORM, str(candidate_id))
            if candidate is None:
                raise MissingRecord(f"Candidate not found: {candidate_id}")
            attempt = SimulationAttemptORM(
                attempt_id=str(uuid4()),
                candidate_id=str(candidate_id),
                idempotency_key=idempotency_key,
                attempt_number=attempt_number,
                status=AttemptStatus.PENDING,
                metrics_json={},
            )
            session.add(attempt)
            self._add_event(
                session,
                candidate.run_id,
                candidate.candidate_id,
                "SIMULATION_ATTEMPT_STARTED",
                {"attempt_number": attempt_number, "idempotency_key": idempotency_key},
            )
        return attempt

    def attach_receipt(
        self,
        attempt_id: str,
        external_simulation_id: str,
        external_alpha_id: str | None,
    ) -> None:
        with self._sessions.begin() as session:
            attempt = session.get(SimulationAttemptORM, attempt_id)
            if attempt is None:
                raise MissingRecord(f"Attempt not found: {attempt_id}")
            attempt.external_simulation_id = external_simulation_id
            attempt.external_alpha_id = external_alpha_id

    def finish_attempt(
        self,
        attempt_id: str,
        status: AttemptStatus,
        result: NormalizedSimulationResult | None = None,
    ) -> None:
        with self._sessions.begin() as session:
            attempt = session.get(SimulationAttemptORM, attempt_id)
            if attempt is None:
                raise MissingRecord(f"Attempt not found: {attempt_id}")
            attempt.status = status
            attempt.completed_at = utc_now()
            if result is not None:
                attempt.error_class = result.error_class
                attempt.error_message_redacted = redact_text(result.error_message)
                attempt.sharpe = result.metrics.sharpe
                attempt.fitness = result.metrics.fitness
                attempt.turnover = result.metrics.turnover
                attempt.metrics_json = result.metrics.model_dump(mode="json")
                attempt.raw_response_artifact_ref = redact_text(result.raw_artifact_ref)

    def latest_attempt(self, candidate_id: str | UUID) -> SimulationAttemptORM | None:
        with self._sessions() as session:
            attempt = session.scalar(
                select(SimulationAttemptORM)
                .where(SimulationAttemptORM.candidate_id == str(candidate_id))
                .order_by(SimulationAttemptORM.attempt_number.desc())
                .limit(1)
            )
            if attempt is not None:
                session.expunge(attempt)
            return attempt

    def completed_metrics(self, candidate_id: str | UUID) -> SimulationMetrics | None:
        with self._sessions() as session:
            attempt = session.scalar(
                select(SimulationAttemptORM)
                .where(
                    SimulationAttemptORM.candidate_id == str(candidate_id),
                    SimulationAttemptORM.status == AttemptStatus.COMPLETED,
                )
                .order_by(SimulationAttemptORM.attempt_number.desc())
                .limit(1)
            )
            if attempt is None:
                return None
            return SimulationMetrics(
                sharpe=attempt.sharpe,
                fitness=attempt.fitness,
                turnover=attempt.turnover,
                extra=attempt.metrics_json.get("extra", {}),
            )

    def record_platform_checks(
        self, candidate_id: str | UUID, checks: list[PlatformCheck], *, live: bool
    ) -> None:
        with self._sessions.begin() as session:
            candidate = session.get(CandidateORM, str(candidate_id))
            if candidate is None:
                raise MissingRecord(f"Candidate not found: {candidate_id}")
            for check in checks:
                session.add(
                    PlatformCheckORM(
                        check_id=str(uuid4()),
                        candidate_id=candidate.candidate_id,
                        name=check.name,
                        status=check.status,
                        message_redacted=redact_text(check.message),
                        is_live_response=live,
                    )
                )
            self._add_event(
                session,
                candidate.run_id,
                candidate.candidate_id,
                "PLATFORM_CHECKS_RECORDED",
                {"live": live, "statuses": [check.status for check in checks]},
            )

    def record_candidate_decision(
        self,
        candidate_id: str | UUID,
        gate_name: str,
        gate_version: str,
        outcome: GateOutcome,
        reason_codes: list[str],
        metric_snapshot: dict[str, Any] | None = None,
    ) -> None:
        assert_no_sensitive_data(metric_snapshot or {})
        with self._sessions.begin() as session:
            candidate = session.get(CandidateORM, str(candidate_id))
            if candidate is None:
                raise MissingRecord(f"Candidate not found: {candidate_id}")
            self._add_decision(
                session,
                candidate.candidate_id,
                gate_name,
                gate_version,
                outcome,
                reason_codes,
                metric_snapshot or {},
            )
            self._add_event(
                session,
                candidate.run_id,
                candidate.candidate_id,
                "CANDIDATE_DECISION",
                {
                    "gate_name": gate_name,
                    "gate_version": gate_version,
                    "outcome": outcome,
                    "reason_codes": reason_codes,
                },
            )

    def count_candidates(self, run_id: str | UUID) -> int:
        with self._sessions() as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(CandidateORM)
                    .where(CandidateORM.run_id == str(run_id))
                )
                or 0
            )

    def count_pending(self, run_id: str | UUID) -> int:
        statuses = {CandidateStatus.QUEUED, CandidateStatus.SIMULATING}
        with self._sessions() as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(CandidateORM)
                    .where(CandidateORM.run_id == str(run_id), CandidateORM.status.in_(statuses))
                )
                or 0
            )

    def count_attempts(self, run_id: str | UUID) -> int:
        with self._sessions() as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(SimulationAttemptORM)
                    .join(CandidateORM)
                    .where(CandidateORM.run_id == str(run_id))
                )
                or 0
            )

    def learned_live_concurrency(self, maximum: int, default: int = 3) -> int:
        """Return the last account-wide learned slot count stored in the ledger."""
        with self._sessions() as session:
            event = session.scalar(
                select(EventORM)
                .where(EventORM.event_type == "LIVE_CONCURRENCY_ADAPTED")
                .order_by(EventORM.created_at.desc())
                .limit(1)
            )
            value = event.payload_json.get("effective_concurrency") if event else default
            learned = value if isinstance(value, int) else default
            return max(1, min(maximum, learned))

    def record_live_concurrency(self, run_id: str | UUID, value: int, reason: str) -> None:
        with self._sessions.begin() as session:
            session.add(
                EventORM(
                    event_id=str(uuid4()),
                    run_id=str(run_id),
                    event_type="LIVE_CONCURRENCY_ADAPTED",
                    payload_json={"effective_concurrency": value, "reason": reason},
                )
            )

    def count_duplicate_events(self, run_id: str | UUID) -> int:
        with self._sessions() as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(EventORM)
                    .where(
                        EventORM.run_id == str(run_id),
                        EventORM.event_type == "DUPLICATE_REUSED",
                    )
                )
                or 0
            )

    def count_children(self, parent_candidate_id: str | UUID, generation_number: int) -> int:
        with self._sessions() as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(CandidateORM)
                    .where(
                        CandidateORM.parent_candidate_id == str(parent_candidate_id),
                        CandidateORM.generation_number == generation_number,
                    )
                )
                or 0
            )

    def list_attempts(self, run_id: str | UUID) -> list[SimulationAttemptORM]:
        with self._sessions() as session:
            attempts = list(
                session.scalars(
                    select(SimulationAttemptORM)
                    .join(CandidateORM)
                    .where(CandidateORM.run_id == str(run_id))
                    .order_by(SimulationAttemptORM.created_at)
                )
            )
            for attempt in attempts:
                session.expunge(attempt)
            return attempts

    def list_decisions(self, run_id: str | UUID) -> list[DecisionORM]:
        with self._sessions() as session:
            decisions = list(
                session.scalars(
                    select(DecisionORM)
                    .join(CandidateORM)
                    .where(CandidateORM.run_id == str(run_id))
                    .order_by(DecisionORM.created_at)
                )
            )
            for decision in decisions:
                session.expunge(decision)
            return decisions

    def list_events(self, run_id: str | UUID) -> list[EventORM]:
        with self._sessions() as session:
            events = list(
                session.scalars(
                    select(EventORM)
                    .where(EventORM.run_id == str(run_id))
                    .order_by(EventORM.created_at)
                )
            )
            for event in events:
                session.expunge(event)
            return events

    def list_platform_checks(self, run_id: str | UUID) -> list[PlatformCheckORM]:
        with self._sessions() as session:
            checks = list(
                session.scalars(
                    select(PlatformCheckORM)
                    .join(CandidateORM)
                    .where(CandidateORM.run_id == str(run_id))
                    .order_by(PlatformCheckORM.created_at)
                )
            )
            for check in checks:
                session.expunge(check)
            return checks

    def upsert_source_document(self, document: SourceDocument) -> None:
        with self._sessions.begin() as session:
            persisted = session.get(SourceDocumentORM, document.source_id)
            values = document.model_dump(mode="json", exclude={"sections"})
            if persisted is None:
                persisted = SourceDocumentORM(**values)
                session.add(persisted)
                session.flush()
            else:
                for key, value in values.items():
                    setattr(persisted, key, value)
                persisted.inventoried_at = utc_now()
                session.execute(
                    delete(SourceSectionORM).where(SourceSectionORM.source_id == document.source_id)
                )
            for section in document.sections:
                session.add(
                    SourceSectionORM(
                        **section.model_dump(mode="json"), source_id=document.source_id
                    )
                )

    def get_source_document(self, source_id: str) -> SourceDocument:
        with self._sessions() as session:
            document = session.get(SourceDocumentORM, source_id)
            if document is None:
                raise MissingRecord(f"Source document not found: {source_id}")
            sections = list(
                session.scalars(
                    select(SourceSectionORM)
                    .where(SourceSectionORM.source_id == source_id)
                    .order_by(SourceSectionORM.section_locator)
                )
            )
            return SourceDocument(
                source_id=document.source_id,
                corpus_root_locator=document.corpus_root_locator,
                relative_locator=document.relative_locator,
                title=document.title,
                source_type=EvidenceLevel(document.source_type),
                content_hash=document.content_hash,
                excluded=document.excluded,
                exclusion_reason=document.exclusion_reason,
                sections=[
                    SourceSection(
                        section_id=section.section_id,
                        heading=section.heading,
                        section_locator=section.section_locator,
                        content_hash=section.content_hash,
                    )
                    for section in sections
                ],
            )

    def list_source_documents(self) -> list[SourceDocument]:
        with self._sessions() as session:
            source_ids = list(
                session.scalars(
                    select(SourceDocumentORM.source_id).order_by(SourceDocumentORM.relative_locator)
                )
            )
        return [self.get_source_document(source_id) for source_id in source_ids]

    def upsert_method_card(self, card: MethodCard) -> MethodCard:
        assert_no_sensitive_data(card.model_dump(mode="python"))
        with self._sessions.begin() as session:
            persisted = session.scalar(
                select(MethodCardORM).where(
                    MethodCardORM.card_id == card.card_id,
                    MethodCardORM.version == card.version,
                )
            )
            values = card.model_dump(mode="python")
            if persisted is not None and persisted.status == MethodCardStatus.APPROVED:
                immutable_existing = self._method_card_from_orm(persisted).model_dump(
                    mode="json", exclude={"status", "reviewer", "approval_timestamp"}
                )
                immutable_incoming = card.model_dump(
                    mode="json", exclude={"status", "reviewer", "approval_timestamp"}
                )
                if immutable_existing != immutable_incoming:
                    raise ValueError("Approved method-card versions are immutable")
                return self._method_card_from_orm(persisted)
            if persisted is None:
                session.add(MethodCardORM(row_id=str(uuid4()), **values))
                return card
            for key, value in values.items():
                setattr(persisted, key, value)
            persisted.updated_at = utc_now()
            return card

    def approve_method_card(self, card_id: str, reviewer: str) -> MethodCard:
        if not reviewer.strip():
            raise ValueError("Reviewer is required")
        with self._sessions.begin() as session:
            persisted = session.scalar(
                select(MethodCardORM)
                .where(MethodCardORM.card_id == card_id)
                .order_by(MethodCardORM.version.desc())
                .limit(1)
            )
            if persisted is None:
                raise MissingRecord(f"Method card not found: {card_id}")
            persisted.status = MethodCardStatus.APPROVED
            persisted.reviewer = reviewer.strip()
            persisted.approval_timestamp = utc_now()
            persisted.updated_at = utc_now()
            session.flush()
            return self._method_card_from_orm(persisted)

    def retire_method_card(self, card_id: str, version: str, reviewer: str) -> MethodCard:
        """Retire one specific approved version -- typically the one a newly
        approved replacement supersedes. `list_method_cards(APPROVED)` (and so
        MethodCardRetriever) returns every row with that status; leaving a
        superseded version approved would hand the LLM both versions at once,
        including whatever the replacement was written to remove.
        """
        if not reviewer.strip():
            raise ValueError("Reviewer is required")
        with self._sessions.begin() as session:
            persisted = session.scalar(
                select(MethodCardORM).where(
                    MethodCardORM.card_id == card_id,
                    MethodCardORM.version == version,
                )
            )
            if persisted is None:
                raise MissingRecord(f"Method card version not found: {card_id} {version}")
            if persisted.status != MethodCardStatus.APPROVED:
                raise ValueError(
                    f"Only an approved card can be retired (current status: {persisted.status})"
                )
            persisted.status = MethodCardStatus.RETIRED
            persisted.reviewer = reviewer.strip()
            persisted.approval_timestamp = utc_now()
            persisted.updated_at = utc_now()
            session.flush()
            return self._method_card_from_orm(persisted)

    def list_method_cards(self, status: MethodCardStatus | None = None) -> list[MethodCard]:
        with self._sessions() as session:
            statement = select(MethodCardORM)
            if status is not None:
                statement = statement.where(MethodCardORM.status == status)
            cards = list(
                session.scalars(statement.order_by(MethodCardORM.card_id, MethodCardORM.version))
            )
            return [self._method_card_from_orm(card) for card in cards]

    def register_prompt_version(self, version: str, role: LLMRole, content: str) -> str:
        from alpha_orchestrator.domain.identity import sha256_text

        assert_no_sensitive_data(content)
        content_hash = sha256_text(content)
        with self._sessions.begin() as session:
            persisted = session.get(PromptVersionORM, version)
            if persisted is not None:
                if persisted.content_hash != content_hash or persisted.role != role:
                    raise ValueError(f"Prompt version is immutable: {version}")
                return content_hash
            session.add(
                PromptVersionORM(
                    version=version,
                    role=role,
                    content_hash=content_hash,
                    content=content,
                )
            )
        return content_hash

    def get_prompt_version(self, version: str) -> PromptVersionORM:
        with self._sessions() as session:
            prompt = session.get(PromptVersionORM, version)
            if prompt is None:
                raise MissingRecord(f"Prompt version not found: {version}")
            session.expunge(prompt)
            return prompt

    def start_llm_request(
        self,
        *,
        prompt_version: str,
        role: LLMRole,
        model: str,
        temperature: float,
        input_hash: str,
        context_manifest: dict[str, Any],
        related_run_id: str | UUID,
        related_candidate_id: str | UUID | None = None,
    ) -> LLMRequestORM:
        assert_no_sensitive_data(context_manifest)
        request = LLMRequestORM(
            request_id=str(uuid4()),
            prompt_version=prompt_version,
            role=role,
            model=model,
            temperature=temperature,
            input_hash=input_hash,
            context_manifest_json=context_manifest,
            validation_status=LLMValidationStatus.PENDING,
            related_run_id=str(related_run_id),
            related_candidate_id=(
                str(related_candidate_id) if related_candidate_id is not None else None
            ),
        )
        with self._sessions.begin() as session:
            if session.get(RunORM, str(related_run_id)) is None:
                raise MissingRecord(f"Run not found: {related_run_id}")
            if session.get(PromptVersionORM, prompt_version) is None:
                raise MissingRecord(f"Prompt version not found: {prompt_version}")
            if (
                related_candidate_id is not None
                and session.get(CandidateORM, str(related_candidate_id)) is None
            ):
                raise MissingRecord(f"Candidate not found: {related_candidate_id}")
            session.add(request)
        return request

    def finish_llm_request(
        self,
        request_id: str,
        *,
        validation_status: LLMValidationStatus,
        output_hash: str | None,
        latency_ms: int,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        actual_model: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._sessions.begin() as session:
            request = session.get(LLMRequestORM, request_id)
            if request is None:
                raise MissingRecord(f"LLM request not found: {request_id}")
            request.validation_status = validation_status
            request.output_hash = output_hash
            request.latency_ms = latency_ms
            request.prompt_tokens = prompt_tokens
            request.completion_tokens = completion_tokens
            request.total_tokens = total_tokens
            if actual_model:
                request.model = actual_model
            request.error_message_redacted = redact_text(error_message)
            request.completed_at = utc_now()

    def list_llm_requests(self, run_id: str | UUID) -> list[LLMRequestORM]:
        with self._sessions() as session:
            requests = list(
                session.scalars(
                    select(LLMRequestORM)
                    .where(LLMRequestORM.related_run_id == str(run_id))
                    .order_by(LLMRequestORM.created_at)
                )
            )
            for request in requests:
                session.expunge(request)
            return requests

    def count_llm_requests(
        self,
        run_id: str | UUID,
        role: LLMRole | None = None,
        candidate_id: str | UUID | None = None,
    ) -> int:
        with self._sessions() as session:
            statement = (
                select(func.count())
                .select_from(LLMRequestORM)
                .where(LLMRequestORM.related_run_id == str(run_id))
            )
            if role is not None:
                statement = statement.where(LLMRequestORM.role == role)
            if candidate_id is not None:
                statement = statement.where(LLMRequestORM.related_candidate_id == str(candidate_id))
            return int(session.scalar(statement) or 0)

    def save_metadata_capture(self, capture: CapturedMetadata) -> None:
        fields_payload = {
            name: data_type.value for name, data_type in sorted(capture.metadata.fields.items())
        }
        operators_payload = sorted(capture.metadata.operators)
        settings_payload = {
            name: sorted(values, key=repr)
            for name, values in sorted(capture.metadata.settings.items())
        }
        operator_definitions = dict(sorted(capture.metadata.operator_definitions.items()))
        records = (
            (MetadataSnapshotKind.FIELDS, capture.field_snapshot_id, fields_payload),
            (MetadataSnapshotKind.OPERATORS, capture.operator_snapshot_id, operators_payload),
            (MetadataSnapshotKind.SETTINGS, capture.setting_snapshot_id, settings_payload),
        )
        assert_no_sensitive_data([record[2] for record in records])
        assert_no_sensitive_data(operator_definitions)
        with self._sessions.begin() as session:
            for kind, snapshot_id, payload in records:
                expected_id, content_hash = metadata_snapshot_identity(
                    kind,
                    capture.metadata.dataset_id,
                    capture.region,
                    capture.universe,
                    capture.delay,
                    capture.data_type,
                    payload,
                )
                if snapshot_id != expected_id:
                    raise ValueError(f"Metadata snapshot ID does not match content: {kind}")
                # Enrichment, not identity: BRAIN can add or reword a `definition`
                # string between captures of the exact same name set without that
                # being a change to what this frozen snapshot IS, so it is not
                # covered by the immutability check below.
                definitions = (
                    operator_definitions if kind is MetadataSnapshotKind.OPERATORS else None
                )
                persisted = session.get(MetadataSnapshotORM, snapshot_id)
                if persisted is not None:
                    if persisted.content_hash != content_hash or persisted.payload_json != payload:
                        raise ValueError(f"Metadata snapshot is immutable: {snapshot_id}")
                    if definitions is not None and not persisted.operator_definitions_json:
                        persisted.operator_definitions_json = definitions
                    continue
                session.add(
                    MetadataSnapshotORM(
                        snapshot_id=snapshot_id,
                        kind=kind,
                        dataset_id=capture.metadata.dataset_id,
                        region=capture.region,
                        universe=capture.universe,
                        delay=capture.delay,
                        data_type=capture.data_type,
                        content_hash=content_hash,
                        payload_json=payload,
                        operator_definitions_json=definitions,
                        source=capture.source,
                        captured_at=capture.captured_at,
                    )
                )

    def load_run_metadata(self, run_id: str | UUID) -> MetadataSnapshot:
        run = self.get_run(run_id)
        requested = {
            MetadataSnapshotKind.FIELDS: run.field_snapshot_id,
            MetadataSnapshotKind.OPERATORS: run.operator_snapshot_id,
            MetadataSnapshotKind.SETTINGS: run.setting_snapshot_id,
        }
        with self._sessions() as session:
            records = {
                kind: session.get(MetadataSnapshotORM, snapshot_id)
                for kind, snapshot_id in requested.items()
            }
            if any(record is None for record in records.values()):
                raise MissingRecord("The frozen run metadata snapshots are incomplete")
            for kind, record in records.items():
                assert record is not None
                if (
                    record.kind != kind
                    or record.dataset_id != run.dataset_id
                    or record.region != run.region
                    or record.universe != run.universe
                    or record.delay != run.delay
                    or record.data_type != run.data_type
                ):
                    raise ValueError("Metadata snapshot does not match the frozen run cell")
            fields = records[MetadataSnapshotKind.FIELDS]
            operators = records[MetadataSnapshotKind.OPERATORS]
            settings = records[MetadataSnapshotKind.SETTINGS]
            assert fields is not None and operators is not None and settings is not None
            return MetadataSnapshot(
                dataset_id=run.dataset_id,
                fields={name: DataType(value) for name, value in dict(fields.payload_json).items()},
                operators=set(operators.payload_json),
                operator_definitions=dict(operators.operator_definitions_json or {}),
                settings={
                    name: set(values) for name, values in dict(settings.payload_json).items()
                },
            )

    @staticmethod
    def _method_card_from_orm(card: MethodCardORM) -> MethodCard:
        return MethodCard(
            card_id=card.card_id,
            title=card.title,
            version=card.version,
            status=MethodCardStatus(card.status),
            applicable_failure_codes=card.applicable_failure_codes,
            applicable_data_types=[DataType(value) for value in card.applicable_data_types],
            prerequisites=card.prerequisites,
            diagnostic_signals=card.diagnostic_signals,
            allowed_repair_actions=card.allowed_repair_actions,
            prohibited_actions=card.prohibited_actions,
            stop_condition=card.stop_condition,
            evidence_level=EvidenceLevel(card.evidence_level),
            source_references=[
                SourceReference.model_validate(reference) for reference in card.source_references
            ],
            reviewer=card.reviewer,
            approval_timestamp=card.approval_timestamp,
        )

    @staticmethod
    def _add_decision(
        session: Session,
        candidate_id: str,
        gate_name: str,
        gate_version: str,
        outcome: GateOutcome,
        reason_codes: list[str],
        metric_snapshot: dict[str, Any],
        llm_request_id: str | None = None,
    ) -> None:
        session.add(
            DecisionORM(
                decision_id=str(uuid4()),
                candidate_id=candidate_id,
                gate_name=gate_name,
                gate_version=gate_version,
                outcome=outcome,
                reason_codes=reason_codes,
                metric_snapshot=metric_snapshot,
                llm_request_id=llm_request_id,
            )
        )

    @staticmethod
    def _add_event(
        session: Session,
        run_id: str,
        candidate_id: str | None,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        session.add(
            EventORM(
                event_id=str(uuid4()),
                run_id=run_id,
                candidate_id=candidate_id,
                event_type=event_type,
                payload_json=payload,
            )
        )

    def age_seconds(self, run_id: str | UUID, now: datetime | None = None) -> float:
        run = self.get_run(run_id)
        reference = now or run.completed_at or utc_now()
        created = run.started_at or run.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=reference.tzinfo)
        return max(0.0, (reference - created).total_seconds())
