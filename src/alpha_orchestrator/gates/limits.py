"""Deterministic search ceilings and queue backpressure."""

from dataclasses import dataclass

from alpha_orchestrator.config import LimitPolicy


@dataclass(frozen=True)
class UsageSnapshot:
    distinct_candidates: int = 0
    simulation_result_rows: int = 0
    pending_candidates: int = 0
    llm_requests: int = 0
    elapsed_seconds: float = 0


class LimitController:
    def __init__(self, policy: LimitPolicy) -> None:
        self.policy = policy

    def stop_reason(self, usage: UsageSnapshot) -> str | None:
        checks = (
            (
                usage.distinct_candidates >= self.policy.maximum_distinct_candidates,
                "MAXIMUM_DISTINCT_CANDIDATES_REACHED",
            ),
            (
                usage.simulation_result_rows >= self.policy.maximum_simulation_result_rows,
                "MAXIMUM_SIMULATION_RESULT_ROWS_REACHED",
            ),
            (
                usage.pending_candidates >= self.policy.maximum_pending_candidates,
                "MAXIMUM_PENDING_CANDIDATES_REACHED",
            ),
            (
                usage.llm_requests >= self.policy.maximum_llm_requests
                and self.policy.maximum_llm_requests > 0,
                "MAXIMUM_LLM_REQUESTS_REACHED",
            ),
            (
                usage.elapsed_seconds >= self.policy.maximum_wall_clock_seconds,
                "MAXIMUM_WALL_CLOCK_REACHED",
            ),
        )
        return next((reason for reached, reason in checks if reached), None)

    def can_enqueue(self, pending_candidates: int) -> bool:
        return pending_candidates < self.policy.maximum_pending_candidates
