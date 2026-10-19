"""Early routing allocates research effort; it is not evidence of tradability."""

from alpha_orchestrator.config import RoutingPolicy
from alpha_orchestrator.domain.enums import GateOutcome
from alpha_orchestrator.domain.models import SimulationMetrics


def route_completed_result(
    metrics: SimulationMetrics | None, policy: RoutingPolicy
) -> tuple[GateOutcome, tuple[str, ...]]:
    # No turnover gate here: routing.v1 discarded on turnover before sharpe/fitness
    # were even read, using a ceiling (0.90) nobody had validated against real
    # data. Turnover is still recorded on every result -- it is just not this
    # gate's job to act on it. High turnover is diagnosed and repaired downstream,
    # where this project has real, cited method cards for it, once a live
    # platform-check actually reports HIGH_TURNOVER.
    if metrics is None:
        return GateOutcome.DISCARD, ("NO_COMPLETED_RESULT",)
    sharpe = metrics.sharpe
    fitness = metrics.fitness
    if (
        sharpe is not None
        and fitness is not None
        and sharpe >= policy.strong.minimum_sharpe
        and fitness >= policy.strong.minimum_fitness
    ):
        return GateOutcome.STRONG, ("ROUTING_STRONG",)
    if (sharpe is not None and sharpe >= policy.weak.minimum_sharpe) or (
        fitness is not None and fitness >= policy.weak.minimum_fitness
    ):
        return GateOutcome.WEAK, ("ROUTING_WEAK",)
    return GateOutcome.DISCARD, ("METRICS_BELOW_ROUTING_FLOOR",)
