"""Early routing allocates research effort; it is not evidence of tradability."""

from alpha_orchestrator.config import RoutingPolicy
from alpha_orchestrator.domain.enums import GateOutcome
from alpha_orchestrator.domain.models import SimulationMetrics


def route_completed_result(
    metrics: SimulationMetrics | None, policy: RoutingPolicy
) -> tuple[GateOutcome, tuple[str, ...]]:
    if metrics is None:
        return GateOutcome.DISCARD, ("NO_COMPLETED_RESULT",)
    if metrics.turnover is not None and metrics.turnover > policy.discard.maximum_turnover:
        return GateOutcome.DISCARD, ("TURNOVER_ABOVE_ROUTING_CEILING",)
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
