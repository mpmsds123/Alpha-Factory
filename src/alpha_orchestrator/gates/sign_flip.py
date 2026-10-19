"""Deterministic sign flip for strongly negative results.

A cross-sectional alpha with sharpe -1.2 carries the same information as one with
sharpe +1.2: BRAIN fitness is sharpe * sqrt(|returns| / max(turnover, 0.125)), so
negating the expression flips the sign of sharpe, returns and fitness while turnover
is unchanged. Archiving it would discard a signal the platform already measured.

This is the one repair that needs no model: the child is `multiply(-1, parent)`, its
expected metrics are the mirror of the parent's, and it is still simulated before
anything trusts it. CHN is excluded because BRAIN documents asymmetric short-side
constraints there, so the mirror argument does not hold.
"""

from alpha_orchestrator.config import RoutingPolicy
from alpha_orchestrator.domain.models import SimulationMetrics

SIGN_FLIP_DIMENSION = "sign"
NO_FLIP_REGIONS = frozenset({"CHN"})


def should_flip(metrics: SimulationMetrics | None, policy: RoutingPolicy, region: str) -> bool:
    """True when the negated result would clear the strong routing floor."""
    if metrics is None or region in NO_FLIP_REGIONS:
        return False
    if metrics.sharpe is None or metrics.fitness is None:
        return False
    return (
        -metrics.sharpe >= policy.strong.minimum_sharpe
        and -metrics.fitness >= policy.strong.minimum_fitness
    )


def negate(expression: str) -> str | None:
    """`multiply(-1, x)`; None when x is already a negation (flipping back is pointless)."""
    stripped = expression.strip()
    if stripped.startswith("multiply(-1,"):
        return None
    return f"multiply(-1, {stripped})"
