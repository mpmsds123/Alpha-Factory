from typing import Any

from alpha_orchestrator.factory import gates
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import (
    ExpressionORM,
    HypothesisORM,
    Review,
    SimStatus,
    SimulationORM,
)
from tests.factory.platform import FakeAPI

CFG = FactoryConfig()
GOOD = {"train_sharpe": 1.8, "train_fitness": 1.2, "train_turnover": 0.2, "test_sharpe": 0.9}


def _add(ledger: Ledger, n: int = 0, region: str = "R1", **kw: Any) -> int:
    with ledger.session() as s:
        h = HypothesisORM(region=region, dataset_id="d", claim="a claim")
        s.add(h)
        s.flush()
        e = ExpressionORM(hypothesis_id=h.id, expression="x")
        s.add(e)
        s.flush()
        sim = SimulationORM(
            expression_id=e.id,
            hypothesis_id=h.id,
            region=region,
            delay=1,
            settings={},
            fingerprint=f"fp{n}",
            status=SimStatus.DONE,
            alpha_id=f"A{n}",
            **{**GOOD, **kw},
        )
        s.add(sim)
        s.flush()
        return sim.id


def _get(ledger: Ledger, sid: int) -> SimulationORM:
    with ledger.session() as s:
        row = s.get(SimulationORM, sid)
        assert row is not None
        return row


def test_train_gate_uses_official_numbers() -> None:
    sim = SimulationORM(**GOOD)
    assert gates.train_pass(sim, "R1", CFG.gates)
    assert not gates.train_pass(SimulationORM(**{**GOOD, "train_sharpe": 1.5}), "R1", CFG.gates)
    assert not gates.train_pass(sim, "CHN", CFG.gates)  # 2.08 line
    chn = SimulationORM(**{**GOOD, "train_sharpe": 2.2, "train_returns": 0.05})
    assert not gates.train_pass(chn, "CHN", CFG.gates)  # returns below 8%
    assert not gates.train_pass(SimulationORM(**{**GOOD, "train_turnover": 0.005}), "R", CFG.gates)
    assert not gates.train_pass(SimulationORM(), "R1", CFG.gates)


def test_oos_and_suspicious() -> None:
    assert gates.oos_pass(SimulationORM(**GOOD), CFG.search)
    assert not gates.oos_pass(SimulationORM(**{**GOOD, "test_sharpe": 0.3}), CFG.search)
    assert not gates.oos_pass(SimulationORM(**{**GOOD, "test_sharpe": None}), CFG.search)
    neg = SimulationORM(train_sharpe=-1.8, test_sharpe=-0.9)
    assert gates.oos_pass(neg, CFG.search)
    assert gates.suspicious(SimulationORM(train_sharpe=3.5), CFG.search)
    assert gates.suspicious(SimulationORM(train_turnover=0.9), CFG.search)


def test_evaluate_runs_the_gate_sequence() -> None:
    ledger = Ledger("sqlite://")
    api = FakeAPI()
    good = _add(ledger, 0)
    weak = _add(ledger, 1, train_sharpe=1.0)
    overfit = _add(ledger, 2, test_sharpe=0.1)
    crazy = _add(ledger, 3, train_sharpe=4.0)
    corr = _add(ledger, 4)
    api.checks["A4"] = [{"name": "SELF_CORRELATION", "result": "FAIL", "value": 0.9}]
    pending = _add(ledger, 5)
    api.checks["A5"] = [{"name": "PROD_CORRELATION", "result": "PENDING"}]
    quota = _add(ledger, 6)
    api.checks["A6"] = [{"name": "REGULAR_SUBMISSION", "result": "FAIL"}]
    assert gates.evaluate(api, ledger, CFG) == ["A0"]  # type: ignore[arg-type]
    notes = {sid: _get(ledger, sid).gate_note for sid in (good, weak, overfit, crazy, corr)}
    assert notes == {
        good: "stored",
        weak: "train",
        overfit: "oos",
        crazy: "suspicious",
        corr: "platform: SELF_CORRELATION,pool: no pool test",
    }
    assert _get(ledger, good).review == Review.PENDING
    assert _get(ledger, pending).checks_pass is None  # retried next time
    assert _get(ledger, quota).checks_pass is None
    api.checks["A5"] = api.check("A0")
    assert gates.evaluate(api, ledger, CFG) == ["A5"]  # type: ignore[arg-type]


def test_approve_is_the_only_submit_path() -> None:
    ledger = Ledger("sqlite://")
    api = FakeAPI()
    sid = _add(ledger, 0)
    assert not gates.approve(api, ledger, sid, CFG)  # type: ignore[arg-type]  # not pending yet
    gates.evaluate(api, ledger, CFG)  # type: ignore[arg-type]
    api.checks["A0"] = [{"name": "X", "result": "FAIL"}]
    assert not gates.approve(api, ledger, sid, CFG)  # type: ignore[arg-type]
    del api.checks["A0"]
    assert gates.approve(api, ledger, sid, CFG)  # type: ignore[arg-type]
    assert api.submitted == ["A0"] and _get(ledger, sid).review == Review.SUBMITTED
    other = _add(ledger, 1)
    gates.evaluate(api, ledger, CFG)  # type: ignore[arg-type]
    assert gates.reject(ledger, other, "no") and not gates.reject(ledger, other)
    assert gates.platform_pass([], 0.7) is False


POOL_OK = [
    {"name": "LOW_SHARPE", "result": "PASS"},
    {"name": "PROD_CORRELATION", "result": "FAIL", "value": 0.95},
    {"name": "POWER_POOL_CORRELATION", "result": "PASS"},
    {"name": "MATCHES_THEMES", "result": "PASS", "themes": [{"name": "R1/D1 Power Pool"}]},
    {"name": "MATCHES_COMPETITION", "result": "WARNING"},
]


def test_pool_route_verdicts() -> None:
    from alpha_orchestrator.factory import pool

    assert pool.verdict(POOL_OK) == (True, "pure pool (daily pool quota)")
    off = [*POOL_OK[:3], {**POOL_OK[3], "result": "WARNING"}]
    assert pool.verdict(off) == (False, "off theme")
    waiting = [*POOL_OK[:2], {"name": "POWER_POOL_CORRELATION", "result": "PENDING"}]
    assert pool.verdict(waiting)[0] is None
    both = [{"name": "LOW_SHARPE", "result": "PASS"}, POOL_OK[2]]
    assert pool.verdict(both) == (True, "pool and regular")
    broken = [*POOL_OK, {"name": "CONCENTRATED_WEIGHT", "result": "FAIL"}]
    assert pool.verdict(broken) == (False, "CONCENTRATED_WEIGHT")
    assert pool.verdict([]) == (False, "no pool test")
    text = pool.description("Late filers drift down.", "Filing delay in days.", "", 5)
    assert text.startswith("Idea: Late filers") and len(text) >= pool.MIN_DESCRIPTION
    assert "Rationale for data used:" in text and "Rationale for operators used:" in text


def test_pool_slot_alpha_is_stored_by_the_pool_route_and_described() -> None:
    ledger = Ledger("sqlite://")
    api = FakeAPI()
    sid = _add(ledger, 0)
    api.checks["A0"] = POOL_OK
    assert gates.evaluate(api, ledger, CFG) == ["A0"]  # type: ignore[arg-type]
    assert _get(ledger, sid).gate_note == "stored: pure pool (daily pool quota)"
    assert any("Idea:" in str(p.get("description")) for p in api.properties.values())
    assert gates.approve(api, ledger, sid, CFG)  # type: ignore[arg-type]


def test_warning_does_not_block_the_regular_route() -> None:
    ok = [{"name": "LOW_SHARPE", "result": "PASS"}, {"name": "MATCHES_THEMES", "result": "WARNING"}]
    ok += [
        {"name": "SELF_CORRELATION", "result": "PASS", "value": 0.2},
        {"name": "PROD_CORRELATION", "result": "PASS", "value": 0.3},
    ]
    assert gates.platform_pass(ok, 0.7)
    assert not gates.platform_pass([*ok, {"name": "X", "result": "PENDING"}], 0.7)
