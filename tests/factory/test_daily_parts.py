"""Daily slots, the fix table, local checks, the platform retry, the memory files and the
submit switch."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from alpha_orchestrator.factory import fixes, gates, guards, notes, slots
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import (
    ExpressionORM,
    HStatus,
    HypothesisORM,
    Review,
    SimStatus,
    SimulationORM,
)
from alpha_orchestrator.factory.validate import check, signature
from tests.factory.fakes import OPERATORS, VOCAB
from tests.factory.platform import FakeAPI
from tests.factory.test_loop import HYP, LOW, MID, STEP, _expr, _factory, _field

SIGS = {k: s for k, d in OPERATORS.items() if (s := signature(d))}


def _cfg(tmp_path: Path) -> FactoryConfig:
    cfg = FactoryConfig()
    cfg.private_dir = str(tmp_path / "private")
    cfg.memory_dir = str(tmp_path / "memory")
    cfg.checks.pnl_cache = str(tmp_path / "pnl")
    return cfg


def _stored(ledger: Ledger, slot: str, n: int, review: str = Review.PENDING) -> int:
    with ledger.session() as s:
        h = HypothesisORM(region="R1", dataset_id="d", claim="c", state={"slot": slot})
        s.add(h)
        s.flush()
        e = ExpressionORM(hypothesis_id=h.id, expression="x", fields=["sig_a"])
        s.add(e)
        s.flush()
        sim = SimulationORM(
            expression_id=e.id,
            hypothesis_id=h.id,
            region="R1",
            delay=1,
            settings={},
            fingerprint=f"fp{n}",
            status=SimStatus.DONE,
            alpha_id=f"A{n}",
            review=review,
            completed_at=datetime.now(UTC),
            train_sharpe=1.8,
            train_fitness=1.2,
            train_turnover=0.2,
            test_sharpe=0.9,
            expression="x",
        )
        s.add(sim)
        s.flush()
        return sim.id


# ---------- slots ----------
def test_slots_fill_empty_first_then_spare_then_stop() -> None:
    cfg = FactoryConfig()
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        first = slots.next_slot(s, cfg)
        assert first is not None and first.name == "pool"
    _stored(ledger, "pool", 1)
    with ledger.session() as s:
        nxt = slots.next_slot(s, cfg)
        assert nxt is not None and nxt.name == "single_dataset"
        only = slots.next_slot(s, cfg, skip=frozenset({"single_dataset", "multi_dataset"}))
        assert only is not None and only.name == "pool"  # the others cannot open today
    _stored(ledger, "single_dataset", 2)
    _stored(ledger, "multi_dataset", 3)
    with ledger.session() as s:
        spare = slots.next_slot(s, cfg)  # every slot has one: the 4th place is open to all
        assert spare is not None
    _stored(ledger, "pool", 4)
    with ledger.session() as s:
        assert slots.next_slot(s, cfg) is None  # 4 a day
        assert slots.stored_today(s, cfg) == {"pool": 2, "single_dataset": 1, "multi_dataset": 1}


def test_outside_pool_slot_needs_more_operators_or_fields() -> None:
    small = "order(sig_a)"
    kw: dict[str, Any] = {
        "operators": set(OPERATORS),
        "signatures": SIGS,
        "fields": {"sig_a": "MATRIX"},
        "vocab": VOCAB,
        "windows": (5, 21),
        "max_operators": 64,
        "max_fields": 6,
    }
    assert check(small, **kw).ok
    v = check(small, outside_pool=(8, 3), **kw)
    assert not v.ok and "inside the pool limits" in v.issues[0]
    big = "sig_a"
    for i in range(9):
        big = f"w_avg({big}, 5)" if i % 2 else f"order({big})"
    assert check(big, outside_pool=(8, 3), **kw).ok
    assert not check(big, **{**kw, "max_operators": 8}).ok


# ---------- fix table ----------
def _sim(**kw: Any) -> SimulationORM:
    base = {"train_sharpe": 1.2, "train_fitness": 0.7, "train_turnover": 0.2, "settings": {}}
    return SimulationORM(**{**base, **kw})


def test_fix_table_maps_each_failure_to_one_change() -> None:
    g = FactoryConfig().gates
    fast = fixes.for_train(_sim(train_turnover=0.5), "R1", g, set(), (0.1, 0.3))
    assert fast is not None and (fast.change, fast.decay) == ("setting", 2)
    top = fixes.for_train(_sim(train_turnover=0.5, settings={"decay": 42}), "R1", g, set())
    assert top is not None and top.change == "window"
    slow = fixes.for_train(_sim(train_turnover=0.005), "R1", g, set())
    assert slow is not None and slow.change == "window" and "shorter" in slow.instruction
    unfit = fixes.for_train(_sim(train_sharpe=1.7, train_turnover=0.3), "R1", g, set())
    assert unfit is not None and unfit.change == "setting" and unfit.failure == "fitness too low"
    weak = fixes.for_train(_sim(), "R1", g, set())
    assert weak is not None and weak.change == "window"
    nxt = fixes.for_train(_sim(), "R1", g, {"window"})
    assert nxt is not None and nxt.change == "operator"
    assert fixes.for_train(_sim(), "R1", g, {"window", "operator", "field"}) is None
    assert fixes.for_platform(["PROD_CORRELATION"]).change == "rewrite"
    assert fixes.for_platform(["CONCENTRATED_WEIGHT"]).change == "operator"
    assert "keep the claim" in fixes.for_platform(["SOMETHING_NEW"]).instruction


def test_setting_fix_is_applied_by_the_program_without_an_llm_decision() -> None:
    fast = {"sharpe": 1.2, "fitness": 0.6, "turnover": 0.5, "test_sharpe": 0.5}
    answers = {
        "hypothesis": [HYP],  # expects mid turnover (10-30%)
        "expression": [lambda r: _expr(f"order({_field(r)})")],
        "diagnosis": [{"label": "idea_supported", "reason": "the predicted sign shows up clearly"}],
        "decision": [{"action": "drop", "reason": "x" * 10, "lesson": "y" * 10}],
    }
    factory, ledger, api, seen = _factory(answers, lambda e, st: fast)
    hid = factory.open_new()
    assert hid is not None
    from tests.factory.test_loop import _drain

    _drain(factory, ledger, api, rounds=1)
    with ledger.session() as s:
        child = s.scalar(select(ExpressionORM).where(ExpressionORM.parent_id.is_not(None)))
        assert child is not None and child.mutation == "setting"
        assert "fix table" in child.reason
        sim = s.scalar(select(SimulationORM).where(SimulationORM.expression_id == child.id))
        assert sim is not None and sim.settings["decay"] == 2
    steps = [r["step"] for r in seen]
    assert steps.count(STEP["diagnosis"]) == 1 and steps.count(STEP["decision"]) == 0


def test_llm_child_must_make_the_table_change() -> None:
    answers = {
        "hypothesis": [HYP],
        "expression": [lambda r: _expr(f"order({_field(r)})")],
        "diagnosis": [{"label": "idea_supported", "reason": "the predicted sign shows up clearly"}],
        "decision": [
            lambda r: {
                "action": "mutate",
                "reason": "try another operator",
                "lesson": "operators matter here",
                "mutations": [
                    {
                        "parent": r["input"]["parents"][0]["expression_id"],
                        "change": "operator",
                        "expression": f"flipit({r['input']['hypothesis']['field']})",
                        "reason": "a different transform of the same input",
                    }
                ],
            }
        ],
    }
    factory, ledger, api, seen = _factory(answers, lambda e, st: MID)
    hid = factory.open_new()
    assert hid is not None
    from tests.factory.test_loop import _drain

    _drain(factory, ledger, api, rounds=1)
    decision = next(r for r in seen if r["step"] == STEP["decision"])
    assert decision["input"]["required_fix"]["change"] == "window"
    with ledger.session() as s:
        child = s.scalar(select(ExpressionORM).where(ExpressionORM.parent_id.is_not(None)))
        assert child is not None and not child.valid
        assert "fix table asks for a window change" in child.rejection


# ---------- local checks ----------
def test_crowded_list_round_trip_and_excluded_from_helpers(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert guards.load_crowded(cfg) == set()
    assert guards.add_crowded(cfg, ["f1", "f2", "f1"], "test") == ["f1", "f2"]
    assert guards.add_crowded(cfg, ["f2"], "again") == []
    assert guards.load_crowded(cfg) == {"f1", "f2"}


class PnLAPI:
    def __init__(self, series: dict[str, dict[str, float]]) -> None:
        self.series = series
        self.calls: list[str] = []

    def daily_pnl(self, alpha_id: str) -> dict[str, float]:
        self.calls.append(alpha_id)
        return self.series[alpha_id]

    def submitted_alphas(self) -> list[str]:
        return ["S1", "S2"]


def _cum(changes: list[float]) -> dict[str, float]:
    start = date(2024, 1, 1)
    out, total = {}, 0.0
    for i, c in enumerate(changes):
        total += c
        out[(start + timedelta(days=i)).isoformat()] = total
    return out


def test_local_self_correlation_finds_the_closest_submitted_alpha(tmp_path: Path) -> None:
    import random

    rng = random.Random(3)
    base = [rng.gauss(0, 1) for _ in range(300)]
    other = [rng.gauss(0, 1) for _ in range(300)]
    near = [b + 0.2 * rng.gauss(0, 1) for b in base]
    api = PnLAPI({"C": _cum(base), "S1": _cum(other), "S2": _cum(near)})
    sc = guards.SelfCorrelation(api, _cfg(tmp_path))
    value, who = sc.max_with_submitted("C")
    assert who == "S2" and value is not None and value > 0.9
    sc2 = guards.SelfCorrelation(api, _cfg(tmp_path))
    sc2.max_with_submitted("C")
    assert api.calls.count("S2") == 1  # submitted PnL comes from the cache the 2nd time
    assert guards.correlation({"a": 1.0}, {"a": 1.0}) is None  # too few days


def test_local_self_correlation_failure_blocks_the_platform_check(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    ledger = Ledger("sqlite://")
    sid = _stored(ledger, "pool", 7, review=Review.NONE)
    with ledger.session() as s:
        sim = s.get(SimulationORM, sid)
        assert sim is not None
        sim.train_pass = True

    class Close:
        def max_with_submitted(self, alpha_id: str) -> tuple[float, str]:
            return 0.8, "S9"

    api = FakeAPI()
    assert gates.evaluate(api, ledger, cfg, Close()) == []  # type: ignore[arg-type]
    with ledger.session() as s:
        sim = s.get(SimulationORM, sid)
        assert sim is not None and sim.checks_pass is False
        assert sim.gate_note.startswith("local self-correlation 0.80")
        h = s.get(HypothesisORM, sim.hypothesis_id)
        assert h is not None and h.state["platform_failure"]["checks"] == ["SELF_CORRELATION"]


def test_prod_correlation_failure_crowds_fields_and_writes_a_lesson(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    ledger = Ledger("sqlite://")
    sid = _stored(ledger, "pool", 8, review=Review.NONE)
    with ledger.session() as s:
        sim = s.get(SimulationORM, sid)
        assert sim is not None
        sim.train_pass = True
    api = FakeAPI()
    api.checks["A8"] = [
        {"name": "LOW_SHARPE", "result": "PASS"},
        {"name": "SELF_CORRELATION", "result": "PASS", "value": 0.3},
        {"name": "PROD_CORRELATION", "result": "FAIL", "value": 0.75},
    ]
    gates.evaluate(api, ledger, cfg)  # type: ignore[arg-type]
    assert guards.load_crowded(cfg) == {"sig_a"}
    assert any("crowded list" in x for x in notes.recent_lessons(cfg.memory_dir))


# ---------- platform retry ----------
def test_platform_failure_is_retried_once_with_the_fix() -> None:
    answers = {
        "hypothesis": [HYP],
        "expression": [lambda r: _expr(f"order({_field(r)})", f"w_avg({_field(r)}, 21)")],
        "diagnosis": [{"label": "idea_failed", "reason": "no signal in either direction at all"}],
        "decision": [{"action": "drop", "reason": "x" * 10, "lesson": "y" * 10}],
    }
    factory, ledger, _, seen = _factory(answers, lambda e, st: LOW)
    with ledger.session() as s:
        old = HypothesisORM(
            region="R1",
            dataset_id="d1",
            field_id="sig_a",
            field_type="MATRIX",
            claim=HYP["claim"],
            direction=1,
            status=HStatus.CLOSED,
            verdict="supported",
            state={
                "slot": "pool",
                "helpers": {},
                "platform_failure": {
                    "expression": "order(sig_a)",
                    "checks": ["CONCENTRATED_WEIGHT"],
                },
            },
        )
        s.add(old)
        s.flush()
        old_id = old.id
    assert factory.retry_platform_failures() == 1
    assert factory.retry_platform_failures() == 0  # once only
    req = next(r for r in seen if r["step"] == STEP["expression"])
    assert req["input"]["required_fix"]["failed_expression"] == "order(sig_a)"
    assert "weight" in req["input"]["required_fix"]["instruction"]
    with ledger.session() as s:
        child = s.scalar(select(HypothesisORM).where(HypothesisORM.parent_id == old_id))
        assert child is not None and child.claim == HYP["claim"] and child.sims_budget == 6


# ---------- memory files ----------
def test_memory_files(tmp_path: Path) -> None:
    md = str(tmp_path / "m")
    assert "Rules" in notes.rules(md)
    assert notes.add_lesson(md, "quota", "2026-11-25", "stopped on quota") == 1
    assert notes.add_lesson(md, "quota", "2026-11-25", "again") is None  # same key
    assert notes.add_lesson(md, "wasted", "2026-11-25", "two empty batches") == 2
    assert notes.recent_lessons(md) == ["L1: stopped on quota", "L2: two empty batches"]
    notes.write_status(md, ["slot pool: 1 stored"])
    assert "slot pool" in (Path(md) / "STATUS.md").read_text()
    assert notes.rate_jumped(0.10, 0.20) and not notes.rate_jumped(0.18, 0.20)
    assert not notes.rate_jumped(0.1, 0.0)


def test_rules_and_lessons_reach_the_llm() -> None:
    answers = {"hypothesis": [HYP], "expression": [lambda r: _expr(f"order({_field(r)})")]}
    factory, _, _, seen = _factory(answers, lambda e, st: LOW)
    notes.add_lesson(factory.config.memory_dir, "quota", "d", "a lesson to remember")
    factory.refresh_notes()
    factory.open_new()
    assert all("Rules" in r["input"]["rules"] for r in seen)
    assert seen[0]["input"]["lessons"] == ["L1: a lesson to remember"]
    assert seen[0]["input"]["slot"]["name"] == "pool"


# ---------- submit switch ----------
def test_auto_submit_respects_mode_pause_and_daily_allowance(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    ledger = Ledger("sqlite://")
    for n in range(6):
        _stored(ledger, "pool", n)
    api = FakeAPI()
    assert (
        gates.auto_submit(api, ledger, cfg) == []
    )  # default: you approve each one  # type: ignore[arg-type]
    cfg.policy.submit_mode = "auto"
    pause = Path(cfg.memory_dir) / gates.PAUSE_FILE
    pause.parent.mkdir(parents=True)
    pause.touch()
    assert gates.submit_mode(cfg) == "approve"
    pause.unlink()
    done = gates.auto_submit(api, ledger, cfg)  # type: ignore[arg-type]
    assert done == ["A0", "A1", "A2", "A3"] and api.submitted == done
    assert gates.auto_submit(api, ledger, cfg) == []  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        type(cfg.policy)(submit_mode="sometimes")


class TwoDatasets(FakeAPI):
    def datasets(self, region: str, delay: int, universe: str) -> list[dict[str, Any]]:
        return [
            {"id": "d1", "name": "Data one", "category": {"id": "cat"}, "fieldCount": 5},
            {"id": "d2", "name": "Data two", "category": {"id": "cat2"}, "fieldCount": 5},
        ]

    def datafields(self, *args: Any) -> list[dict[str, Any]]:
        if args[-1] == "d2":
            return [{"id": "oth_x", "type": "MATRIX", "coverage": 0.9, "description": "other"}]
        return super().datafields(*args)


def test_multi_dataset_slot_gets_and_must_use_a_second_dataset() -> None:
    from alpha_orchestrator.factory import loop

    answers = {
        "hypothesis": [HYP],
        "expression": [
            lambda r: _expr(
                # 9 operators, hypothesis field only: rejected (no second-dataset field)
                "order(w_avg(order(w_avg(order(w_avg(order(w_avg(order("
                + _field(r)
                + "), 5)), 5)), 5)), 5))"
            )
        ],
    }
    factory, ledger, _, seen = _factory(answers, lambda e, st: LOW, api_cls=TwoDatasets)
    factory.config.slots.rules = [factory.config.slots.get("multi_dataset")]
    hid = factory.open_new()
    assert hid is not None
    with ledger.session() as s:
        h = s.get(HypothesisORM, hid)
        assert h is not None
        second = h.state["second_dataset"]
        assert h.state["slot"] == "multi_dataset" and second
        assert h.dataset_id not in set(second.values())
        assert set(h.helper_fields) & set(second)  # added by the program: the LLM picked none
        e = s.scalar(select(ExpressionORM).where(ExpressionORM.hypothesis_id == hid))
        assert e is not None and "needs a field from the second dataset" in e.rejection
    assert "second_dataset_candidates" in seen[0]["input"]
    assert isinstance(loop.Factory, type)
