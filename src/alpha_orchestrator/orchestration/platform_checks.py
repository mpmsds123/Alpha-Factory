"""Conservative live platform-check recording and promotion."""

from dataclasses import dataclass
from uuid import UUID

from alpha_orchestrator.adapters.base import BrainAdapter
from alpha_orchestrator.config import CorrelationPolicy, LimitPolicy
from alpha_orchestrator.domain.enums import (
    AdapterMode,
    AttemptStatus,
    CandidateStatus,
    GateOutcome,
    PlatformCheckStatus,
)
from alpha_orchestrator.domain.errors import is_retryable
from alpha_orchestrator.domain.models import AdapterFailure, PlatformCheck
from alpha_orchestrator.llm.services import RepairPlanner, SearchLimitReached
from alpha_orchestrator.storage.models import CandidateORM
from alpha_orchestrator.storage.repository import Repository

PLATFORM_GATE_VERSION = "platform-check.live.v1"
CORRELATION_GATE_NAME = "G7_CORRELATION_DEDUP"


@dataclass(frozen=True)
class PlatformCheckSummary:
    candidate_id: str
    candidate_status: CandidateStatus
    request_attempts: int
    checks: tuple[PlatformCheck, ...]
    repair_request_ids: tuple[str, ...] = ()
    repair_candidate_ids: tuple[str, ...] = ()
    repair_stop_reason: str | None = None


class PlatformCheckService:
    def __init__(
        self,
        repository: Repository,
        repair_planner: RepairPlanner | None = None,
    ) -> None:
        self.repository = repository
        self.repair_planner = repair_planner

    def check(
        self,
        candidate_id: str | UUID,
        adapter: BrainAdapter,
        correlation_policy: CorrelationPolicy,
    ) -> PlatformCheckSummary:
        candidate = self.repository.get_candidate(candidate_id)
        if CandidateStatus(candidate.status) is not CandidateStatus.READY_FOR_PLATFORM_CHECK:
            raise ValueError("Candidate is not ready for a live platform check")
        if adapter.mode is not AdapterMode.LIVE:
            raise ValueError("Platform readiness requires the live adapter")
        run = self.repository.get_run(candidate.run_id)
        if run.adapter_mode != AdapterMode.LIVE:
            raise ValueError("Candidate does not belong to a live run")
        if run.dataset_id != adapter.metadata.dataset_id:
            raise ValueError("Live adapter metadata does not match the frozen run")
        attempt = self.repository.latest_attempt(candidate.candidate_id)
        if (
            attempt is None
            or attempt.status != AttemptStatus.COMPLETED
            or not attempt.external_alpha_id
        ):
            raise ValueError("Candidate has no completed live simulation Alpha ID")
        limits = LimitPolicy.model_validate(run.limit_policy_json)
        maximum_attempts = limits.transient_retries_per_external_request + 1
        checks: list[PlatformCheck] = []
        request_attempts = 0
        for request_attempts in range(1, maximum_attempts + 1):
            try:
                checks = adapter.request_platform_checks(attempt.external_alpha_id)
                if not checks:
                    checks = [
                        PlatformCheck(
                            name="platform-check-response",
                            status=PlatformCheckStatus.PENDING,
                            message="BRAIN returned no platform checks",
                        )
                    ]
                self.repository.record_platform_checks(candidate.candidate_id, checks, live=True)
                break
            except AdapterFailure as exc:
                failure = PlatformCheck(
                    name=f"platform-check-request-{request_attempts}-{exc.error_class}",
                    status=PlatformCheckStatus.ERROR,
                    message=exc.message,
                )
                self.repository.record_platform_checks(candidate.candidate_id, [failure], live=True)
                checks = [failure]
                if not is_retryable(exc.error_class) or request_attempts >= maximum_attempts:
                    break

        snapshot = {check.name: check.status for check in checks}
        statuses = {check.status for check in checks}
        repair_request_ids: list[str] = []
        repair_candidate_ids: list[str] = []
        repair_stop_reason: str | None = None
        if statuses == {PlatformCheckStatus.PASS}:
            # Platform checks alone are not proof of non-redundancy: a candidate can
            # pass every check and still be a near-duplicate of an alpha that already
            # exists. The transition to human review waits on the platform's own
            # correlation numbers too -- READY_FOR_PLATFORM_CHECK has no way back once
            # left, so nothing is promoted until both gates have actually cleared.
            self.repository.record_candidate_decision(
                candidate.candidate_id,
                "G6_PLATFORM_READINESS",
                PLATFORM_GATE_VERSION,
                GateOutcome.PASS,
                ["PLATFORM_ALL_PASS_CONFIRMED"],
                snapshot,
            )
            candidate = self._apply_correlation_gate(
                candidate.candidate_id, attempt.external_alpha_id, adapter, correlation_policy
            )
        elif PlatformCheckStatus.FAIL in statuses:
            candidate = self.repository.transition_candidate(
                candidate.candidate_id,
                CandidateStatus.ARCHIVED,
                "G6_PLATFORM_READINESS",
                PLATFORM_GATE_VERSION,
                GateOutcome.FAIL,
                ["PLATFORM_CHECK_FAILED"],
                snapshot,
            )
            if self.repair_planner is not None:
                evidence_snapshot: dict[str, object] = {
                    "platform_checks": [
                        {
                            "name": check.name,
                            "status": str(check.status),
                            "message": check.message,
                        }
                        for check in checks
                    ]
                }
                for failed_check in sorted(
                    (check for check in checks if check.status is PlatformCheckStatus.FAIL),
                    key=lambda check: check.name,
                ):
                    try:
                        result = self.repair_planner.plan(
                            candidate.candidate_id,
                            failure_code=failed_check.name,
                            evidence_snapshot=evidence_snapshot,
                            metadata=adapter.metadata,
                        )
                    except SearchLimitReached as exc:
                        repair_stop_reason = exc.reason
                        break
                    if result.request_id is not None:
                        repair_request_ids.append(result.request_id)
                    repair_candidate_ids.extend(
                        insert.candidate.candidate_id
                        for insert in result.candidates
                        if insert.created
                    )
        else:
            outcome = _unresolved_outcome(statuses)
            self.repository.record_candidate_decision(
                candidate.candidate_id,
                "G6_PLATFORM_READINESS",
                PLATFORM_GATE_VERSION,
                outcome,
                ["PLATFORM_CHECK_UNRESOLVED"],
                snapshot,
            )
        return PlatformCheckSummary(
            candidate_id=candidate.candidate_id,
            candidate_status=CandidateStatus(candidate.status),
            request_attempts=request_attempts,
            checks=tuple(checks),
            repair_request_ids=tuple(repair_request_ids),
            repair_candidate_ids=tuple(repair_candidate_ids),
            repair_stop_reason=repair_stop_reason,
        )

    def _apply_correlation_gate(
        self,
        candidate_id: str,
        external_alpha_id: str,
        adapter: BrainAdapter,
        policy: CorrelationPolicy,
    ) -> CandidateORM:
        try:
            correlations = adapter.fetch_correlations(external_alpha_id)
        except AdapterFailure as exc:
            self.repository.record_candidate_decision(
                candidate_id,
                CORRELATION_GATE_NAME,
                policy.version,
                GateOutcome.ERROR,
                [f"CORRELATION_REQUEST_{exc.error_class}"],
                {},
            )
            return self.repository.get_candidate(candidate_id)

        snapshot: dict[str, object] = dict(correlations)
        breached = sorted(
            name
            for name, value in correlations.items()
            if value is not None and value >= policy.maximum_correlation
        )
        if breached:
            return self.repository.transition_candidate(
                candidate_id,
                CandidateStatus.ARCHIVED,
                CORRELATION_GATE_NAME,
                policy.version,
                GateOutcome.FAIL,
                [f"{name.upper()}_CORRELATION_TOO_HIGH" for name in breached],
                snapshot,
            )
        # An absent correlation means BRAIN has not finished computing it yet, not
        # that the candidate is safe (see fetch_correlations' own docstring) -- stay
        # in READY_FOR_PLATFORM_CHECK so a later re-run of this same check can see it.
        if not correlations or any(value is None for value in correlations.values()):
            self.repository.record_candidate_decision(
                candidate_id,
                CORRELATION_GATE_NAME,
                policy.version,
                GateOutcome.PENDING,
                ["CORRELATION_NOT_YET_COMPUTED"],
                snapshot,
            )
            return self.repository.get_candidate(candidate_id)
        return self.repository.transition_candidate(
            candidate_id,
            CandidateStatus.READY_FOR_HUMAN_REVIEW,
            CORRELATION_GATE_NAME,
            policy.version,
            GateOutcome.PASS,
            ["CORRELATION_BELOW_THRESHOLD"],
            snapshot,
        )


def _unresolved_outcome(statuses: set[PlatformCheckStatus]) -> GateOutcome:
    if PlatformCheckStatus.ERROR in statuses:
        return GateOutcome.ERROR
    if PlatformCheckStatus.PENDING in statuses:
        return GateOutcome.PENDING
    return GateOutcome.WARNING
