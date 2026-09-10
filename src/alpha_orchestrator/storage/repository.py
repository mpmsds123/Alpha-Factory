"""Transactional source-of-truth operations."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from alpha_orchestrator.config import LimitPolicy, RoutingPolicy
from alpha_orchestrator.domain.enums import (
    AttemptStatus,
    CandidateStatus,
    GateOutcome,
    RunStatus,
)
from alpha_orchestrator.domain.errors import redact_text
from alpha_orchestrator.domain.identity import (
    candidate_fingerprint,
    canonicalize_expression,
    normalized_content_hash,
)
from alpha_orchestrator.domain.models import (
    CandidateProposal,
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
    PlatformCheckORM,
    RunORM,
    SimulationAttemptORM,
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
            if status is RunStatus.RUNNING and run.started_at is None:
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
    ) -> CandidateInsertResult:
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
                )
                self._add_event(
                    session,
                    str(run_id),
                    duplicate.candidate_id,
                    "DUPLICATE_REUSED",
                    {"candidate_fingerprint": fingerprint},
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
            self._add_event(
                session,
                str(run_id),
                candidate.candidate_id,
                "CANDIDATE_PROPOSED",
                {"candidate_fingerprint": fingerprint},
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

    @staticmethod
    def _add_decision(
        session: Session,
        candidate_id: str,
        gate_name: str,
        gate_version: str,
        outcome: GateOutcome,
        reason_codes: list[str],
        metric_snapshot: dict[str, Any],
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
