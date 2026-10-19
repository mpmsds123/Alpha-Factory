"""Bounded research execution engine."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from uuid import UUID

from alpha_orchestrator.adapters.base import MINIMUM_SESSION_REMAINING_SECONDS, BrainAdapter
from alpha_orchestrator.audit_logging import AuditLogger
from alpha_orchestrator.config import CorrelationPolicy, LimitPolicy, RoutingPolicy
from alpha_orchestrator.domain.enums import (
    AdapterMode,
    AttemptStatus,
    CandidateStatus,
    DataType,
    ErrorClass,
    GateOutcome,
    LLMRole,
    RunStatus,
)
from alpha_orchestrator.domain.errors import is_retryable
from alpha_orchestrator.domain.identity import stable_idempotency_key
from alpha_orchestrator.domain.models import (
    AdapterFailure,
    CandidateProposal,
    CreateSimulationRequest,
    NormalizedSimulationResult,
)
from alpha_orchestrator.gates.limits import LimitController, UsageSnapshot
from alpha_orchestrator.gates.routing import route_completed_result
from alpha_orchestrator.gates.sign_flip import SIGN_FLIP_DIMENSION, negate, should_flip
from alpha_orchestrator.gates.structural import StructuralValidator
from alpha_orchestrator.llm.services import (
    STRUCTURAL_FAILURES,
    CandidateGenerator,
    RepairPlanner,
    SearchLimitReached,
)
from alpha_orchestrator.orchestration.platform_checks import PlatformCheckService
from alpha_orchestrator.storage.models import CandidateORM, RunORM, SimulationAttemptORM
from alpha_orchestrator.storage.repository import Repository


@dataclass(frozen=True)
class ExecutionSummary:
    run_id: str
    status: RunStatus
    stop_reason: str | None
    distinct_candidates: int
    simulation_attempts: int


@dataclass(frozen=True)
class _ExternalOutcome:
    simulation_id: str | None = None
    alpha_id: str | None = None
    result: NormalizedSimulationResult | None = None
    failure: AdapterFailure | None = None


class ResearchOrchestrator:
    """Runs bounded experiments through an explicitly selected adapter."""

    def __init__(
        self,
        repository: Repository,
        routing_policy: RoutingPolicy,
        limit_policy: LimitPolicy,
        validator: StructuralValidator | None = None,
        audit_logger: AuditLogger | None = None,
        candidate_generator: CandidateGenerator | None = None,
        repair_planner: RepairPlanner | None = None,
        research_query: str | None = None,
        correlation_policy: CorrelationPolicy | None = None,
    ) -> None:
        self.repository = repository
        self.routing_policy = routing_policy
        self.limit_policy = limit_policy
        self.validator = validator or StructuralValidator()
        self.audit = audit_logger or AuditLogger()
        self.limits = LimitController(limit_policy)
        self.candidate_generator = candidate_generator
        self.repair_planner = repair_planner
        self.research_query = research_query
        # Only live runs can reach G6/G7 (BRAIN's own platform checks and correlation
        # numbers do not exist for a replay fixture or the fake adapter), so a missing
        # policy is harmless there; _settle skips the platform-check phase entirely
        # for any adapter whose mode is not LIVE.
        self.correlation_policy = correlation_policy
        if candidate_generator is not None and not research_query:
            raise ValueError("research_query is required when CandidateGenerator is enabled")

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

        stop_reason = self._settle(run_id, adapter)
        if stop_reason is None:
            if self.candidate_generator is not None:
                stop_reason = self._generate_roots(run_id, adapter)
            else:
                stop_reason = self._ingest_adapter_candidates(run_id, adapter)

        if stop_reason is None:
            stop_reason = self._settle(run_id, adapter)

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
        queued: list[CandidateORM] = []
        for candidate in resumable:
            status = CandidateStatus(candidate.status)
            if status in {CandidateStatus.PROPOSED, CandidateStatus.VALIDATING}:
                candidate, reason = self._validate_and_queue(candidate, adapter)
                if reason is not None:
                    return reason
                status = CandidateStatus(candidate.status)
            elif status is CandidateStatus.READY:
                candidate = self._queue(candidate)
                status = CandidateStatus(candidate.status)
            elif status is CandidateStatus.RETRYABLE_ERROR:
                pass
            if status in {
                CandidateStatus.QUEUED,
                CandidateStatus.SIMULATING,
                CandidateStatus.RETRYABLE_ERROR,
            }:
                queued.append(candidate)
        return self._process_candidate_batch(queued, adapter)

    def _candidate_snapshot(self, run_id: str | UUID) -> tuple[list[tuple[str, str]], int]:
        return (
            [
                (candidate.candidate_id, candidate.status)
                for candidate in self.repository.list_candidates(run_id)
            ],
            self.repository.count_attempts(run_id),
        )

    def _drain_existing(self, run_id: str | UUID, adapter: BrainAdapter) -> str | None:
        """Process descendants added by repair planning in the same bounded execute call."""
        deferred_reason: str | None = None
        for _ in range(self.limit_policy.maximum_generations + 2):
            before = self._candidate_snapshot(run_id)
            reason = self._resume_existing(run_id, adapter)
            if reason is not None:
                deferred_reason = deferred_reason or reason
                retryable = self.repository.list_candidates(
                    run_id, {CandidateStatus.RETRYABLE_ERROR}
                )
                if not retryable:
                    return deferred_reason
            after = self._candidate_snapshot(run_id)
            if after == before:
                return deferred_reason
        return deferred_reason

    def _settle(self, run_id: str | UUID, adapter: BrainAdapter) -> str | None:
        """Drain simulation work, then (live runs only) push anything that reached
        READY_FOR_PLATFORM_CHECK through BRAIN's own checks and correlation numbers --
        repeating as long as that produces new repair candidates to drain. Step ⑩
        (dedup/check/queue) is supposed to be code's decision, not something a human
        has to remember to trigger separately for every completed candidate.
        """
        # A survivor is more valuable than clearing the tail of the queue. Check any
        # already-ready candidate first, so stale pending work or a time limit cannot
        # strand it before G6/G7.
        if adapter.mode is AdapterMode.LIVE:
            stop_reason = self._check_platform_readiness(run_id, adapter)
            if stop_reason is not None:
                return stop_reason
        stop_reason = self._drain_existing(run_id, adapter)
        if adapter.mode is not AdapterMode.LIVE:
            return stop_reason
        # A batch may have produced a survivor and a deferred pending/error reason at
        # the same time. Preserve the reason, but never skip the survivor's check.
        platform_reason = self._check_platform_readiness(run_id, adapter)
        if platform_reason is not None:
            return platform_reason
        if stop_reason is not None:
            return stop_reason
        for _ in range(self.limit_policy.maximum_generations + 2):
            before = self._candidate_snapshot(run_id)
            stop_reason = self._check_platform_readiness(run_id, adapter)
            if stop_reason is not None:
                return stop_reason
            stop_reason = self._drain_existing(run_id, adapter)
            if stop_reason is not None:
                return stop_reason
            if self._candidate_snapshot(run_id) == before:
                return None
        return None

    def _check_platform_readiness(self, run_id: str | UUID, adapter: BrainAdapter) -> str | None:
        if self.correlation_policy is None:
            return None
        service = PlatformCheckService(self.repository, self.repair_planner)
        ready = self.repository.list_candidates(run_id, {CandidateStatus.READY_FOR_PLATFORM_CHECK})
        for candidate in ready:
            summary = service.check(candidate.candidate_id, adapter, self.correlation_policy)
            if summary.repair_stop_reason is not None:
                return summary.repair_stop_reason
        return None

    def _generate_roots(self, run_id: str | UUID, adapter: BrainAdapter) -> str | None:
        generator = self.candidate_generator
        if generator is None:
            return None
        if self.repository.count_candidates(run_id) > 0:
            return None
        if self.repository.count_llm_requests(run_id, LLMRole.GENERATOR) > 0:
            return None
        try:
            generator.generate(
                run_id,
                adapter.metadata,
                query=self.research_query or "",
            )
        except SearchLimitReached as exc:
            return exc.reason
        return None

    def _ingest_adapter_candidates(self, run_id: str | UUID, adapter: BrainAdapter) -> str | None:
        all_proposals = adapter.initial_candidates()
        proposals = all_proposals[: self.limit_policy.initial_candidates]
        for proposal in proposals:
            stop_reason = self._current_limit_reason(run_id)
            if stop_reason is not None:
                return stop_reason
            if proposal.generation_number > self.limit_policy.maximum_generations:
                return "MAXIMUM_GENERATIONS_REACHED"
            if (
                proposal.parent_candidate_id is not None
                and self.repository.count_children(
                    proposal.parent_candidate_id, proposal.generation_number
                )
                >= self.limit_policy.children_per_parent_per_generation
            ):
                return "CHILDREN_PER_PARENT_LIMIT_REACHED"
            insert = self.repository.add_candidate(run_id, proposal, adapter.metadata.operators)
            if insert.created:
                _, reason = self._validate_and_queue(insert.candidate, adapter)
                if reason is not None:
                    return reason
        if len(all_proposals) > len(proposals):
            return "INITIAL_CANDIDATE_LIMIT_REACHED"
        return None

    def _validate_and_queue(
        self, candidate: CandidateORM, adapter: BrainAdapter
    ) -> tuple[CandidateORM, str | None]:
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
            invalid = self.repository.transition_candidate(
                candidate.candidate_id,
                CandidateStatus.INVALID,
                "G1_STRUCTURAL_VALIDITY",
                "structural.v1",
                GateOutcome.FAIL,
                [issue.code for issue in result.issues],
            )
            failure = next(
                (issue for issue in result.issues if str(issue.code) in STRUCTURAL_FAILURES),
                None,
            )
            if failure is None:
                return invalid, None
            reason = self._plan_repair(
                invalid,
                failure_code=str(failure.code),
                evidence_snapshot={
                    "validation_issues": [
                        {"code": str(issue.code), "message": issue.message}
                        for issue in result.issues
                    ]
                },
                adapter=adapter,
            )
            return invalid, reason
        candidate = self.repository.transition_candidate(
            candidate.candidate_id,
            CandidateStatus.READY,
            "G1_STRUCTURAL_VALIDITY",
            "structural.v1",
            GateOutcome.PASS,
            ["STRUCTURALLY_VALID"],
        )
        return self._queue(candidate), None

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

    def _process_candidate_batch(
        self, candidates: list[CandidateORM], adapter: BrainAdapter
    ) -> str | None:
        """Run create/poll concurrently, then persist every outcome in stable order."""
        prepared: list[tuple[CandidateORM, SimulationAttemptORM, CreateSimulationRequest]] = []
        maximum_attempts = self.limit_policy.transient_retries_per_external_request + 1
        for original in candidates:
            candidate = self.repository.get_candidate(original.candidate_id)
            latest = self.repository.latest_attempt(candidate.candidate_id)
            retry_attempt_number: int | None = None
            current_status = CandidateStatus(candidate.status)
            if (
                latest is not None
                and latest.status == AttemptStatus.RETRYABLE_ERROR
                and (
                    current_status is CandidateStatus.RETRYABLE_ERROR
                    or current_status is CandidateStatus.SIMULATING
                )
            ):
                if current_status is CandidateStatus.SIMULATING:
                    # A prior run was interrupted (killed process, SESSION_EXPIRING_SOON)
                    # between finish_attempt(RETRYABLE_ERROR) and the matching
                    # transition_candidate call in _record_failure, leaving the
                    # candidate stuck in SIMULATING while its latest attempt already
                    # says RETRYABLE_ERROR -- SIMULATING -> QUEUED is not a valid
                    # transition, so resuming this candidate crashed here before this
                    # fix. Reconcile to the true last-known state first.
                    candidate = self.repository.transition_candidate(
                        candidate.candidate_id,
                        CandidateStatus.RETRYABLE_ERROR,
                        "G3_EXECUTION_VALIDITY",
                        "error-taxonomy.v1",
                        GateOutcome.ERROR,
                        ["RECONCILED_AFTER_INTERRUPTED_RUN"],
                    )
                if latest.attempt_number >= maximum_attempts:
                    self.repository.transition_candidate(
                        candidate.candidate_id,
                        CandidateStatus.TERMINAL_ERROR,
                        "G3_EXECUTION_VALIDITY",
                        "error-taxonomy.v1",
                        GateOutcome.ERROR,
                        ["INFRASTRUCTURE_RETRIES_EXHAUSTED"],
                    )
                    continue
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
                retry_attempt_number = latest.attempt_number + 1
                latest = None
            elif CandidateStatus(candidate.status) is CandidateStatus.QUEUED:
                candidate = self.repository.transition_candidate(
                    candidate.candidate_id,
                    CandidateStatus.SIMULATING,
                    "G3_EXECUTION_VALIDITY",
                    "error-taxonomy.v1",
                    GateOutcome.PENDING,
                    ["SIMULATION_STARTED"],
                )
            if latest is None or latest.status != AttemptStatus.PENDING:
                limit_reason = self._execution_limit_reason(candidate.run_id)
                if limit_reason is not None:
                    return limit_reason
                session_reason = self._session_freshness_reason(adapter)
                if session_reason is not None:
                    return session_reason
                attempt_number = retry_attempt_number or (
                    1 if latest is None else latest.attempt_number + 1
                )
                latest = self.repository.start_attempt(
                    candidate.candidate_id,
                    stable_idempotency_key(candidate.candidate_id),
                    attempt_number,
                )
            run = self.repository.get_run(candidate.run_id)
            request = CreateSimulationRequest(
                candidate_id=UUID(candidate.candidate_id),
                expression=candidate.expression,
                settings=candidate.settings_json,
                region=run.region,
                universe=run.universe,
                delay=run.delay,
                idempotency_key=latest.idempotency_key,
            )
            prepared.append((candidate, latest, request))

        if not prepared:
            return None
        first_reason: str | None = None
        cursor = 0
        while cursor < len(prepared):
            worker_count = min(
                self.limit_policy.maximum_concurrent_simulations,
                len(prepared) - cursor,
            )
            if adapter.mode is AdapterMode.LIVE:
                worker_count = min(
                    worker_count,
                    self.repository.learned_live_concurrency(
                        self.limit_policy.maximum_concurrent_simulations
                    ),
                )
            batch = prepared[cursor : cursor + worker_count]
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                outcomes = list(
                    pool.map(
                        lambda item: self._external_create_and_poll(item[1], item[2], adapter),
                        batch,
                    )
                )
            for (candidate, attempt, _), outcome in zip(batch, outcomes, strict=True):
                reason = self._persist_external_outcome(candidate, attempt, outcome, adapter)
                if first_reason is None and reason is not None:
                    first_reason = reason
            if adapter.mode is AdapterMode.LIVE:
                rate_limited = any(
                    outcome.failure is not None
                    and outcome.failure.error_class is ErrorClass.RATE_LIMIT
                    for outcome in outcomes
                )
                if rate_limited:
                    learned = max(1, worker_count - 1)
                    adaptation_reason = "RATE_LIMIT"
                else:
                    learned = min(
                        self.limit_policy.maximum_concurrent_simulations,
                        worker_count + 1,
                    )
                    adaptation_reason = "CLEAN_BATCH"
                self.repository.record_live_concurrency(
                    batch[0][0].run_id, learned, adaptation_reason
                )
            cursor += len(batch)
        return first_reason

    def _external_create_and_poll(
        self,
        attempt: SimulationAttemptORM,
        request: CreateSimulationRequest,
        adapter: BrainAdapter,
    ) -> _ExternalOutcome:
        simulation_id = attempt.external_simulation_id
        alpha_id = attempt.external_alpha_id
        try:
            if not simulation_id:
                receipt = adapter.create_simulation(request)
                simulation_id = receipt.external_simulation_id
                alpha_id = receipt.external_alpha_id
            result = adapter.poll_simulation(simulation_id)
            return _ExternalOutcome(simulation_id, alpha_id, result)
        except AdapterFailure as exc:
            return _ExternalOutcome(simulation_id, alpha_id, failure=exc)

    def _persist_external_outcome(
        self,
        candidate: CandidateORM,
        attempt: SimulationAttemptORM,
        outcome: _ExternalOutcome,
        adapter: BrainAdapter,
    ) -> str | None:
        if outcome.simulation_id and not attempt.external_simulation_id:
            self.repository.attach_receipt(
                attempt.attempt_id, outcome.simulation_id, outcome.alpha_id
            )
            attempt.external_simulation_id = outcome.simulation_id
            attempt.external_alpha_id = outcome.alpha_id
        if outcome.failure is not None:
            return self._record_failure(
                candidate,
                attempt,
                outcome.failure.error_class,
                outcome.failure.message,
                adapter,
            )
        if outcome.result is None:
            raise ValueError("Concurrent simulation returned no result")
        return self._persist_result(candidate, attempt, outcome.result, adapter)

    def _persist_result(
        self,
        candidate: CandidateORM,
        attempt: SimulationAttemptORM,
        result: NormalizedSimulationResult,
        adapter: BrainAdapter,
    ) -> str | None:
        if not result.completed:
            error_class = result.error_class or ErrorClass.UNKNOWN
            if error_class in {ErrorClass.PENDING, ErrorClass.UNKNOWN}:
                self.repository.finish_attempt(attempt.attempt_id, AttemptStatus.PENDING, result)
                return f"UNRESOLVED_{error_class}_SIMULATION"
            return self._record_failure(
                candidate,
                attempt,
                error_class,
                result.error_message or "Simulation failed",
                adapter,
            )
        if result.external_alpha_id and not attempt.external_alpha_id:
            self.repository.attach_receipt(
                attempt.attempt_id,
                attempt.external_simulation_id or "",
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
        gate_outcome, reasons = route_completed_result(result.metrics, self.routing_policy)
        target = (
            CandidateStatus.READY_FOR_PLATFORM_CHECK
            if gate_outcome is GateOutcome.STRONG
            else CandidateStatus.ARCHIVED
        )
        self.repository.transition_candidate(
            candidate.candidate_id,
            target,
            "G4_EARLY_ROUTING",
            self.routing_policy.version,
            gate_outcome,
            list(reasons),
            result.metrics.model_dump(mode="json"),
        )
        if target is CandidateStatus.ARCHIVED:
            self._propose_sign_flip(candidate, result, adapter)
        return None

    def _propose_sign_flip(
        self,
        candidate: CandidateORM,
        result: NormalizedSimulationResult,
        adapter: BrainAdapter,
    ) -> None:
        """Queue `multiply(-1, parent)` when the mirror image clears the strong floor.

        The child is an ordinary candidate: it is deduplicated, structurally validated and
        simulated like any other, and `_drain_existing` picks it up in the same execute call.
        """
        run = self.repository.get_run(candidate.run_id)
        if not should_flip(result.metrics, self.routing_policy, run.region):
            return
        expression = negate(candidate.expression)
        if expression is None or "multiply" not in adapter.metadata.operators:
            return
        if candidate.generation_number + 1 > self.limit_policy.maximum_generations:
            return
        if (
            self.repository.count_candidates(run.run_id)
            >= self.limit_policy.maximum_distinct_candidates
        ):
            return
        sharpe = result.metrics.sharpe or 0.0
        self.repository.add_candidate(
            run.run_id,
            CandidateProposal(
                expression=expression,
                settings=candidate.settings_json,
                mechanism_claim=candidate.mechanism_claim,
                source_method_card_ids=list(candidate.source_method_card_ids),
                parent_candidate_id=UUID(candidate.candidate_id),
                generation_number=candidate.generation_number + 1,
                changed_dimension=SIGN_FLIP_DIMENSION,
                change_reason=f"Parent sharpe {sharpe:.2f}; negation mirrors the measured signal",
            ),
            adapter.metadata.operators,
        )

    def _record_failure(
        self,
        candidate: CandidateORM,
        attempt: SimulationAttemptORM,
        error_class: ErrorClass,
        message: str,
        adapter: BrainAdapter,
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
        terminal = self.repository.transition_candidate(
            candidate.candidate_id,
            CandidateStatus.TERMINAL_ERROR,
            "G3_EXECUTION_VALIDITY",
            "error-taxonomy.v1",
            GateOutcome.ERROR,
            [error_class, "EXECUTION_FAILURE_NOT_RESEARCH_VERDICT"],
        )
        if str(error_class) in STRUCTURAL_FAILURES:
            return self._plan_repair(
                terminal,
                failure_code=str(error_class),
                evidence_snapshot={
                    "error_class": str(error_class),
                    "error_message": message,
                },
                adapter=adapter,
            )
        return None

    def _plan_repair(
        self,
        candidate: CandidateORM,
        *,
        failure_code: str,
        evidence_snapshot: dict[str, object],
        adapter: BrainAdapter,
    ) -> str | None:
        if self.repair_planner is None:
            return None
        try:
            self.repair_planner.plan(
                candidate.candidate_id,
                failure_code=failure_code,
                evidence_snapshot=evidence_snapshot,
                metadata=adapter.metadata,
            )
        except SearchLimitReached as exc:
            return exc.reason
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

    def _session_freshness_reason(self, adapter: BrainAdapter) -> str | None:
        """Checked purely from the adapter's local clock, right before a candidate
        would spend the run's budget on a new external request. Replay/fake adapters
        report None (no session concept) and never trip this; only a live adapter's
        estimate, ticking down since the last real `GET /authentication`, can.
        """
        remaining = adapter.session_seconds_remaining()
        if remaining is not None and remaining < MINIMUM_SESSION_REMAINING_SECONDS:
            return "SESSION_EXPIRING_SOON"
        return None


OfflineOrchestrator = ResearchOrchestrator
