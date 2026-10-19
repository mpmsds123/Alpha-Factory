from alpha_orchestrator.config import RoutingPolicy
from alpha_orchestrator.domain.models import SimulationMetrics
from alpha_orchestrator.gates.sign_flip import negate, should_flip


def test_strong_negative_result_flips(routing_policy: RoutingPolicy) -> None:
    metrics = SimulationMetrics(sharpe=-1.2, fitness=-0.6, turnover=0.1)
    assert should_flip(metrics, routing_policy, "USA")


def test_weak_or_positive_results_do_not_flip(routing_policy: RoutingPolicy) -> None:
    assert not should_flip(SimulationMetrics(sharpe=-0.1, fitness=-0.05), routing_policy, "USA")
    assert not should_flip(SimulationMetrics(sharpe=1.2, fitness=0.6), routing_policy, "USA")
    assert not should_flip(SimulationMetrics(sharpe=None, fitness=-0.6), routing_policy, "USA")
    assert not should_flip(None, routing_policy, "USA")


def test_chn_never_flips(routing_policy: RoutingPolicy) -> None:
    metrics = SimulationMetrics(sharpe=-2.0, fitness=-1.0)
    assert not should_flip(metrics, routing_policy, "CHN")


def test_negate_wraps_once() -> None:
    assert negate("rank(close)") == "multiply(-1, rank(close))"
    assert negate("multiply(-1, rank(close))") is None
