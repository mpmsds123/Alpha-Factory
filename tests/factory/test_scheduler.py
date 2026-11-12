from typing import Any

import pytest

from alpha_orchestrator.brain.client import AmbiguousRequest
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger, fingerprint
from alpha_orchestrator.factory.models import (
    BatchORM,
    ExpressionORM,
    HypothesisORM,
    SimStatus,
    SimulationORM,
)
from alpha_orchestrator.factory.scheduler import (
    Simulator,
    metrics_from_alpha,
    rejected_children,
    same_request,
)
from tests.factory.platform import FakeAPI

SETTINGS = {"region": "R1", "delay": 1, "universe": "U", "decay": 0, "testPeriod": "P2Y"}


def _queue(ledger: Ledger, exprs: list[str]) -> list[int]:
    ids = []
    with ledger.session() as s:
        h = HypothesisORM(region="R1", dataset_id="d")
        s.add(h)
        s.flush()
        for x in exprs:
            e = ExpressionORM(hypothesis_id=h.id, expression=x)
            s.add(e)
            s.flush()
            sim = SimulationORM(
                expression_id=e.id,
                hypothesis_id=h.id,
                region="R1",
                delay=1,
                expression=x,
                settings=SETTINGS,
                fingerprint=fingerprint(x, SETTINGS),
            )
            s.add(sim)
            s.flush()
            ids.append(sim.id)
    return ids


def _sim(cfg: FactoryConfig | None = None) -> tuple[Simulator, Ledger, FakeAPI]:
    cfg = cfg or FactoryConfig()
    cfg.budget.min_batch_interval_s = 0
    ledger = Ledger("sqlite://")
    api = FakeAPI(
        lambda e, st: {"sharpe": 1.2, "fitness": 0.8, "turnover": 0.1, "test_sharpe": 0.7}
    )
    return Simulator(api, ledger, cfg, sleep=lambda _: None), ledger, api  # type: ignore[arg-type]


def test_runs_queue_and_splits_train_test_is() -> None:
    sim, ledger, api = _sim()
    ids = _queue(ledger, [f"op(f{i})" for i in range(12)])
    stats = sim.run(100)
    assert (stats.batches, stats.completed, stats.stopped_because) == (2, 12, "queue empty")
    with ledger.session() as s:
        row = s.get(SimulationORM, ids[0])
        assert row is not None and row.status == SimStatus.DONE
        assert (row.train_sharpe, row.test_sharpe, row.is_sharpe) == (1.2, 0.7, 1.2)
        assert ledger.sims_reserved(s, FactoryConfig().budget.window_start()) == 12
    assert len(api.batches["/simulations/b0"]) == 10
    assert api.batches["/simulations/b0"][0]["settings"]["testPeriod"] == "P2Y"


def test_daily_cap_and_budget_stop() -> None:
    cfg = FactoryConfig()
    cfg.budget.daily_sim_cap = 5
    sim, ledger, _ = _sim(cfg)
    _queue(ledger, [f"op(f{i})" for i in range(8)])
    stats = sim.run(100)
    assert (stats.completed, stats.stopped_because) == (5, "budget")
    assert sim.remaining_daily() == 0


def test_rejected_children_fail_and_rest_requeue() -> None:
    sim, ledger, api = _sim()
    ids = _queue(ledger, ["a(f)", "b(f)"])
    api.reject_next = True
    sim.run(100)
    with ledger.session() as s:
        first, second = (s.get(SimulationORM, i) for i in ids)
        assert first is not None and first.status == SimStatus.FAILED
        assert second is not None and second.status == SimStatus.DONE
    assert rejected_children("x (400): {}", 1) == set()
    assert rejected_children("no", 1) == set()
    assert rejected_children('x (400): {"a": 1}', 1) == {0}


def test_uncertain_batch_blocks_running() -> None:
    sim, ledger, _ = _sim()
    with ledger.session() as s:
        s.add(BatchORM(region="R1", delay=1, size=1, status="UNKNOWN"))
    with pytest.raises(AmbiguousRequest):
        sim.run(10)


def test_learned_concurrency_is_reused() -> None:
    ledger = Ledger("sqlite://")
    ledger.log("concurrency_limit", batches=2)
    assert Simulator(FakeAPI(), ledger, FactoryConfig()).slots == 2  # type: ignore[arg-type]


def test_metrics_without_test_period_use_is_as_train() -> None:
    alpha: dict[str, Any] = {
        "is": {"sharpe": 1.0, "checks": [{"name": "X", "result": "FAIL"}]},
        "train": None,
    }
    m = metrics_from_alpha(alpha)
    assert (m["train_sharpe"], m["test_sharpe"], m["fails"]) == (1.0, None, "X")


def test_same_request_ignores_unechoed_settings() -> None:
    row = SimulationORM(expression="a( b )", settings={"decay": 0, "testPeriod": "P2Y"})
    assert same_request("a(b)", {"decay": 0}, row)
    assert not same_request("a(b)", {"decay": 4}, row)
    assert not same_request("a(c)", {"decay": 0}, row)
