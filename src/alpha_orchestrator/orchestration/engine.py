"""Milestone 1 offline execution engine."""

from dataclasses import dataclass
from uuid import UUID

from alpha_orchestrator.adapters.base import BrainAdapter
from alpha_orchestrator.audit_logging import AuditLogger
from alpha_orchestrator.config import LimitPolicy, RoutingPolicy
from alpha_orchestrator.domain.enums import (
    AttemptStatus,
    CandidateStatus,
    DataType,
    ErrorClass,
    GateOutcome,
    RunStatus,
)
from alpha_orchestrator.domain.errors import is_retryable
from alpha_orchestrator.domain.identity import stable_idempotency_key
from alpha_orchestrator.domain.models import (
    AdapterFailure,
    CreateSimulationRequest,
    NormalizedSimulationResult,
)
from alpha_orchestrator.gates.limits import LimitController, UsageSnapshot
from alpha_orchestrator.gates.routing import route_completed_result
from alpha_orchestrator.gates.structural import StructuralValidator
from alpha_orchestrator.storage.models import CandidateORM, RunORM, SimulationAttemptORM
from alpha_orchestrator.storage.repository import Repository


@dataclass(frozen=True)
class ExecutionSummary:
    run_id: str
    status: RunStatus
    stop_reason: str | None
    distinct_candidates: int
    simulation_attempts: int


class ResearchOrchestrator:
    """Runs bounded experiments through an explicitly selected adapter."""

    def __init__(
        self,
        repository: Repository,
        routing_policy: RoutingPolicy,
        limit_policy: LimitPolicy,
        validator: StructuralValidator | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.repository = repository
        self.routing_policy = routing_policy
        self.limit_policy = limit_policy
        self.validator = validator or StructuralValidator()
        self.audit = audit_logger or AuditLogger()
        self.limits = LimitController(limit_policy)

    def execute(self, run_id: str | UUID, adapter: BrainAdapter) -> ExecutionSummary:
        run = self.repository.get_run(run_id)
        unfinished = self.repository.list_candidates(
            run_id,
            statuses={
                CandidateStatus.PROPOSED,
                CandidateStatus.VALIDATING,
                CandidateStatus.READY,
                CandidateStatus.QUEUED,
                CandidateStatus.SIMULATING,
                CandidateStatus.RETRYABLE_ERROR,
            },
        )
        if run.status == RunStatus.COMPLETED and not unfinished:
            return ExecutionSummary(
                run_id=str(run_id),
                status=RunStatus.COMPLETED,
                stop_reason=None,
                distinct_candidates=self.repository.count_candidates(run_id),
                simulation_attempts=self.repository.count_attempts(run_id),
            )
        self._validate_run_contract(run, adapter)
        self.repository.update_run_status(run_id, RunStatus.RUNNING)
        self.audit.event("run_started", run_id=str(run_id), adapter=adapter.mode)

        stop_reason = self._resume_existing(run_id, adapter)
        if stop_reason is None:
            all_proposals = adapter.initial_candidates()
            proposals = all_proposals[: self.limit_policy.initial_candidates]
            for proposal in proposals:
                stop_reason = self._current_limit_reason(run_id)
                if stop_reason is not None:
                    break
                if proposal.generation_number > self.limit_policy.maximum_generations:
                    stop_reason = "MAXIMUM_GENERATIONS_REACHED"
                    break
                if (
                    proposal.parent_candidate_id is not None
                    and self.repository.count_children(
                        proposal.parent_candidate_id, proposal.generation_number
                    )
                    >= self.limit_policy.children_per_parent_per_generation
                ):
                    stop_reason = "CHILDREN_PER_PARENT_LIMIT_REACHED"
                    break
                insert = self.repository.add_candidate(run_id, proposal, adapter.metadata.operators)
                if not insert.created:
                    continue
                candidate = self._validate_and_queue(insert.candidate, adapter)
                if CandidateStatus(candidate.status) is CandidateStatus.QUEUED:
                    stop_reason = self._process_candidate(candidate, adapter)
                    if stop_reason is not None:
                        break
            if stop_reason is None and len(all_proposals) > len(proposals):
                stop_reason = "INITIAL_CANDIDATE_LIMIT_REACHED"

        if stop_reason is None:
            stop_reason = self._resume_existing(run_id, adapter)

        unfinished = self.repository.list_candidates(
            run_id,
            {
                CandidateStatus.PROPOSED,
                CandidateStatus.VALIDATING,
                CandidateStatus.READY,
                CandidateStatus.QUEUED,
                CandidateStatus.SIMULATING,
                CandidateStatus.RETRYABLE_ERROR,
            },
        )
        if stop_reason is not None or unfinished:
            reason = stop_reason or "UNFINISHED_CANDIDATES_REMAIN"
            self.repository.update_run_status(run_id, RunStatus.STOPPED, reason)
            status = RunStatus.STOPPED
        else:
            self.repository.update_run_status(run_id, RunStatus.COMPLETED)
            status = RunStatus.COMPLETED
        self.audit.event("run_finished", run_id=str(run_id), status=status, reason=stop_reason)
        return ExecutionSummary(
            run_id=str(run_id),
            status=status,
            stop_reason=stop_reason,
            distinct_candidates=self.repository.count_candidates(run_id),
            simulation_attempts=self.repository.count_attempts(run_id),
        )

    def _validate_run_contract(self, run: RunORM, adapter: BrainAdapter) -> None:
        required = (
            "dataset_id",
            "region",
            "universe",
            "field_snapshot_id",
            "operator_snapshot_id",
            "setting_snapshot_id",
            "routing_policy_version",
            "limit_policy_version",
            "generator_prompt_version",
            "repair_prompt_version",
        )
        if any(not getattr(run, name, None) for name in required):
            raise ValueError("G0 run contract is not frozen")
        if run.dataset_id != adapter.metadata.dataset_id:
            raise ValueError("Adapter metadata does not match the frozen dataset")
        if run.adapter_mode != adapter.mode:
            raise ValueError("Adapter mode does not match the frozen run")
        if run.routing_policy_version != self.routing_policy.version:
            raise ValueError("Routing policy does not match the frozen run")
        if run.limit_policy_version != self.limit_policy.version:
            raise ValueError("Limit policy does not match the frozen run")
        if run.routing_policy_json != self.routing_policy.model_dump(mode="json"):
            raise ValueError("Routing policy content does not match the frozen snapshot")
        if run.limit_policy_json != self.limit_policy.model_dump(mode="json"):
            raise ValueError("Limit policy content does not match the frozen snapshot")

    def _resume_existing(self, run_id: str | UUID, adapter: BrainAdapter) -> str | None:
        resumable = self.repository.list_candidates(
            run_id,
            {
                CandidateStatus.PROPOSED,
                CandidateStatus.VALIDATING,
                CandidateStatus.READY,
                CandidateStatus.QUEUED,
                CandidateStatus.SIMULATING,
                CandidateStatus.RETRYABLE_ERROR,
            },
        )
        for candidate in resumable:
            status = CandidateStatus(candidate.status)
            if status in {CandidateStatus.PROPOSED, CandidateStatus.VALIDATING}:
                candidate = self._validate_and_queue(candidate, adapter)
                status = CandidateStatus(candidate.status)
            elif status is CandidateStatus.READY:
                candidate = self._queue(candidate)
                status = CandidateStatus(candidate.status)
            elif status is CandidateStatus.RETRYABLE_ERROR:
                candidate = self.repository.transition_candidate(
                    candidate.candidate_id,
                    CandidateStatus.QUEUED,
                    "G3_EXECUTION_VALIDITY",
                    "error-taxonomy.v1",
                    GateOutcome.PENDING,
                    ["RESUME_RETRY"],
                )
                status = CandidateStatus.QUEUED
            if status in {CandidateStatus.QUEUED, CandidateStatus.SIMULATING}:
                reason = self._process_candidate(candidate, adapter)
                if reason is not None:
                    return reason
        return None

    def _validate_and_queue(self, candidate: CandidateORM, adapter: BrainAdapter) -> CandidateORM:
        status = CandidateStatus(candidate.status)
        if status is CandidateStatus.PROPOSED:
            candidate = self.repository.transition_candidate(
                candidate.candidate_id,
                CandidateStatus.VALIDATING,
                "G1_STRUCTURAL_VALIDITY",
                "structural.v1",
                GateOutcome.PENDING,
                ["VALIDATION_STARTED"],
            )
        result = self.validator.validate(
            candidate.expression,
            candidate.settings_json,
            adapter.metadata,
            DataType(self.repository.get_run(candidate.run_id).data_type),
        )
        if not result.valid:
            return self.repository.transition_candidate(
                candidate.candidate_id,
                CandidateStatus.INVALID,
                "G1_STRUCTURAL_VALIDITY",
                "structural.v1",
                GateOutcome.FAIL,
                [issue.code for issue in result.issues],
            )
        candidate = self.repository.transition_candidate(
            candidate.candidate_id,
            CandidateStatus.READY,
            "G1_STRUCTURAL_VALIDITY",
            "structural.v1",
            GateOutcome.PASS,
            ["STRUCTURALLY_VALID"],
        )
        return self._queue(candidate)

    def _queue(self, candidate: CandidateORM) -> CandidateORM:
        pending = self.repository.count_pending(candidate.run_id)
        if not self.limits.can_enqueue(pending):
            return candidate
        return self.repository.transition_candidate(
            candidate.candidate_id,
            CandidateStatus.QUEUED,
            "G5_SEARCH_CONTROL",
            self.limit_policy.version,
            GateOutcome.PASS,
            ["QUEUE_CAPACITY_AVAILABLE"],
        )

    def _process_candidate(self, candidate: CandidateORM, adapter: BrainAdapter) -> str | None:
        if CandidateStatus(candidate.status) is CandidateStatus.QUEUED:
            candidate = self.repository.transition_candidate(
                candidate.candidate_id,
                CandidateStatus.SIMULATING,
                "G3_EXECUTION_VALIDITY",
                "error-taxonomy.v1",
                GateOutcome.PENDING,
                ["SIMULATION_STARTED"],
            )

        latest = self.repository.latest_attempt(candidate.candidate_id)
        if latest is not None and latest.status == AttemptStatus.PENDING:
            if latest.external_simulation_id:
                return self._poll_and_finish(candidate, latest, adapter)
            return self._create_and_finish(candidate, latest, adapter)

        next_attempt = 1 if latest is None else latest.attempt_number + 1
        maximum_attempts = self.limit_policy.transient_retries_per_external_request + 1
        while next_attempt <= maximum_attempts:
            limit_reason = self._execution_limit_reason(candidate.run_id)
            if limit_reason is not None:
                return limit_reason
            key = stable_idempotency_key(candidate.candidate_id)
            attempt = self.repository.start_attempt(candidate.candidate_id, key, next_attempt)
            reason = self._create_and_finish(candidate, attempt, adapter)
            refreshed = self.repository.get_candidate(candidate.candidate_id)
            if CandidateStatus(refreshed.status) is CandidateStatus.RETRYABLE_ERROR:
                if next_attempt >= maximum_attempts:
                    self.repository.transition_candidate(
                        candidate.candidate_id,
                        CandidateStatus.TERMINAL_ERROR,
                        "G3_EXECUTION_VALIDITY",
                        "error-taxonomy.v1",
                        GateOutcome.ERROR,
                        ["INFRASTRUCTURE_RETRIES_EXHAUSTED"],
                    )
                    return None
                candidate = self.repository.transition_candidate(
                    candidate.candidate_id,
                    CandidateStatus.QUEUED,
                    "G3_EXECUTION_VALIDITY",
                    "error-taxonomy.v1",
                    GateOutcome.PENDING,
                    ["BOUNDED_RETRY_SCHEDULED"],
                )
                candidate = self.repository.transition_candidate(
                    candidate.candidate_id,
                    CandidateStatus.SIMULATING,
                    "G3_EXECUTION_VALIDITY",
                    "error-taxonomy.v1",
                    GateOutcome.PENDING,
                    ["BOUNDED_RETRY_STARTED"],
                )
                next_attempt += 1
                continue
            return reason
        return None

    def _create_and_finish(
        self,
        candidate: CandidateORM,
        attempt: SimulationAttemptORM,
        adapter: BrainAdapter,
    ) -> str | None:
        run = self.repository.get_run(candidate.run_id)
        request = CreateSimulationRequest(
            candidate_id=UUID(candidate.candidate_id),
            expression=candidate.expression,
            settings=candidate.settings_json,
            region=run.region,
            universe=run.universe,
            delay=run.delay,
            idempotency_key=attempt.idempotency_key,
        )
        try:
            receipt = adapter.create_simulation(request)
        except AdapterFailure as exc:
            return self._record_failure(candidate, attempt, exc.error_class, exc.message)
        self.repository.attach_receipt(
            attempt.attempt_id,
            receipt.external_simulation_id,
            receipt.external_alpha_id,
        )
        attempt.external_simulation_id = receipt.external_simulation_id
        return self._poll_and_finish(candidate, attempt, adapter)

    def _poll_and_finish(
        self,
        candidate: CandidateORM,
        attempt: SimulationAttemptORM,
        adapter: BrainAdapter,
    ) -> str | None:
        if not attempt.external_simulation_id:
            raise ValueError("Cannot poll without an external simulation ID")
        try:
            result = adapter.poll_simulation(attempt.external_simulation_id)
        except AdapterFailure as exc:
            return self._record_failure(candidate, attempt, exc.error_class, exc.message)
        if not result.completed:
            error_class = result.error_class or ErrorClass.UNKNOWN
            if error_class in {ErrorClass.PENDING, ErrorClass.UNKNOWN}:
                self.repository.finish_attempt(attempt.attempt_id, AttemptStatus.PENDING, result)
                return f"UNRESOLVED_{error_class}_SIMULATION"
            return self._record_failure(
                candidate, attempt, error_class, result.error_message or "Simulation failed"
            )

        if result.external_alpha_id and not attempt.external_alpha_id:
            self.repository.attach_receipt(
                attempt.attempt_id,
                attempt.external_simulation_id,
                result.external_alpha_id,
            )
        self.repository.finish_attempt(attempt.attempt_id, AttemptStatus.COMPLETED, result)
        candidate = self.repository.transition_candidate(
            candidate.candidate_id,
            CandidateStatus.COMPLETED,
            "G3_EXECUTION_VALIDITY",
            "error-taxonomy.v1",
            GateOutcome.PASS,
            ["SIMULATION_COMPLETED"],
            result.metrics.model_dump(mode="json"),
        )
        candidate = self.repository.transition_candidate(
            candidate.candidate_id,
            CandidateStatus.TRIAGED,
            "G4_EARLY_ROUTING",
            self.routing_policy.version,
            GateOutcome.PENDING,
            ["ROUTING_STARTED"],
        )
        outcome, reasons = route_completed_result(result.metrics, self.routing_policy)
        target = (
            CandidateStatus.READY_FOR_PLATFORM_CHECK
            if outcome is GateOutcome.STRONG
            else CandidateStatus.ARCHIVED
        )
        self.repository.transition_candidate(
            candidate.candidate_id,
            target,
            "G4_EARLY_ROUTING",
            self.routing_policy.version,
            outcome,
            list(reasons),
            result.metrics.model_dump(mode="json"),
        )
        return None

    def _record_failure(
        self,
        candidate: CandidateORM,
        attempt: SimulationAttemptORM,
        error_class: ErrorClass,
        message: str,
    ) -> str | None:
        result = NormalizedSimulationResult(
            completed=False,
            error_class=error_class,
            error_message=message,
        )
        if is_retryable(error_class):
            self.repository.finish_attempt(
                attempt.attempt_id, AttemptStatus.RETRYABLE_ERROR, result
            )
            self.repository.transition_candidate(
                candidate.candidate_id,
                CandidateStatus.RETRYABLE_ERROR,
                "G3_EXECUTION_VALIDITY",
                "error-taxonomy.v1",
                GateOutcome.ERROR,
                [error_class, "INFRASTRUCTURE_ERROR_NOT_RESEARCH_FAILURE"],
            )
            return None
        if error_class in {ErrorClass.PENDING, ErrorClass.UNKNOWN}:
            self.repository.finish_attempt(attempt.attempt_id, AttemptStatus.PENDING, result)
            return f"UNRESOLVED_{error_class}_SIMULATION"
        attempt_status = (
            AttemptStatus.CANCELLED
            if error_class is ErrorClass.CANCELLED
            else AttemptStatus.TERMINAL_ERROR
        )
        self.repository.finish_attempt(attempt.attempt_id, attempt_status, result)
        self.repository.transition_candidate(
            candidate.candidate_id,
            CandidateStatus.TERMINAL_ERROR,
            "G3_EXECUTION_VALIDITY",
            "error-taxonomy.v1",
            GateOutcome.ERROR,
            [error_class, "EXECUTION_FAILURE_NOT_RESEARCH_VERDICT"],
        )
        return None

    def _current_limit_reason(self, run_id: str | UUID) -> str | None:
        return self.limits.stop_reason(
            UsageSnapshot(
                distinct_candidates=self.repository.count_candidates(run_id),
                simulation_result_rows=self.repository.count_attempts(run_id),
                pending_candidates=self.repository.count_pending(run_id),
                llm_requests=self.repository.count_llm_requests(run_id),
                elapsed_seconds=self.repository.age_seconds(run_id),
            )
        )

    def _execution_limit_reason(self, run_id: str | UUID) -> str | None:
        return self.limits.stop_reason(
            UsageSnapshot(
                distinct_candidates=0,
                simulation_result_rows=self.repository.count_attempts(run_id),
                pending_candidates=0,
                llm_requests=0,
                elapsed_seconds=self.repository.age_seconds(run_id),
            )
        )


OfflineOrchestrator = ResearchOrchestrator
