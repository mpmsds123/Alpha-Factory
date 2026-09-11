"""Conservative live platform-check recording and promotion."""

from dataclasses import dataclass
from uuid import UUID

from alpha_orchestrator.adapters.base import BrainAdapter
from alpha_orchestrator.config import LimitPolicy
from alpha_orchestrator.domain.enums import (
    AdapterMode,
    AttemptStatus,
    CandidateStatus,
    GateOutcome,
    PlatformCheckStatus,
)
from alpha_orchestrator.domain.errors import is_retryable
from alpha_orchestrator.domain.models import AdapterFailure, PlatformCheck
from alpha_orchestrator.storage.repository import Repository

PLATFORM_GATE_VERSION = "platform-check.live.v1"


@dataclass(frozen=True)
class PlatformCheckSummary:
    candidate_id: str
    candidate_status: CandidateStatus
    request_attempts: int
    checks: tuple[PlatformCheck, ...]


class PlatformCheckService:
    def __init__(self, repository: Repository) -> None:
        self.repository = repository

    def check(
        self,
        candidate_id: str | UUID,
        adapter: BrainAdapter,
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
        if statuses == {PlatformCheckStatus.PASS}:
            candidate = self.repository.transition_candidate(
                candidate.candidate_id,
                CandidateStatus.READY_FOR_HUMAN_REVIEW,
                "G6_PLATFORM_READINESS",
                PLATFORM_GATE_VERSION,
                GateOutcome.PASS,
                ["PLATFORM_ALL_PASS_CONFIRMED"],
                snapshot,
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
        )


def _unresolved_outcome(statuses: set[PlatformCheckStatus]) -> GateOutcome:
    if PlatformCheckStatus.ERROR in statuses:
        return GateOutcome.ERROR
    if PlatformCheckStatus.PENDING in statuses:
        return GateOutcome.PENDING
    return GateOutcome.WARNING
