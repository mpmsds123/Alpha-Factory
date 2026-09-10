"""Candidate transition policy. Storage applies it transactionally."""

from alpha_orchestrator.domain.enums import CandidateStatus


class InvalidTransition(ValueError):
    pass


ALLOWED_TRANSITIONS: dict[CandidateStatus, frozenset[CandidateStatus]] = {
    CandidateStatus.PROPOSED: frozenset({CandidateStatus.VALIDATING}),
    CandidateStatus.VALIDATING: frozenset({CandidateStatus.READY, CandidateStatus.INVALID}),
    CandidateStatus.READY: frozenset({CandidateStatus.QUEUED, CandidateStatus.ARCHIVED}),
    CandidateStatus.QUEUED: frozenset({CandidateStatus.SIMULATING, CandidateStatus.ARCHIVED}),
    CandidateStatus.SIMULATING: frozenset(
        {
            CandidateStatus.COMPLETED,
            CandidateStatus.RETRYABLE_ERROR,
            CandidateStatus.TERMINAL_ERROR,
        }
    ),
    CandidateStatus.RETRYABLE_ERROR: frozenset(
        {CandidateStatus.QUEUED, CandidateStatus.TERMINAL_ERROR}
    ),
    CandidateStatus.COMPLETED: frozenset({CandidateStatus.TRIAGED}),
    CandidateStatus.TRIAGED: frozenset(
        {CandidateStatus.ARCHIVED, CandidateStatus.READY_FOR_PLATFORM_CHECK}
    ),
    CandidateStatus.READY_FOR_PLATFORM_CHECK: frozenset(
        {CandidateStatus.READY_FOR_HUMAN_REVIEW, CandidateStatus.ARCHIVED}
    ),
    CandidateStatus.INVALID: frozenset(),
    CandidateStatus.DUPLICATE: frozenset(),
    CandidateStatus.TERMINAL_ERROR: frozenset(),
    CandidateStatus.ARCHIVED: frozenset(),
    CandidateStatus.READY_FOR_HUMAN_REVIEW: frozenset(),
}


def validate_transition(source: CandidateStatus, target: CandidateStatus) -> None:
    if target not in ALLOWED_TRANSITIONS[source]:
        raise InvalidTransition(f"Invalid candidate transition: {source} -> {target}")
