"""设计文档 v1 end to end on a simulated platform whose truth is known.

The scripted LLM is honest: it proposes "EPS revisions", rates eps fields 9 and the rest 2,
and improves a parent by comparing within subindustry (worth +0.35 Sharpe in this world).
"""

import json
import random
import re
from typing import Any

import pytest
from sqlalchemy import select

from alpha_orchestrator.factory import engine as eng
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CellORM,
    EventORM,
    IdeaStatus,
    ResearchIdeaORM,
    Stage,
)

from .world import World, analyst_fields


def honest_llm(model: str, system: str, user: str) -> tuple[str, int, int]:
    p = json.loads(user)
    task = p["task"]
    if task.startswith("You are a quant researcher reading"):
        eps = [f["id"] for f in p["fields"] if "_eps_" in f["id"]]
        other = [f["id"] for f in p["fields"] if "_eps_" not in f["id"]]
        groups = [
            {
                "name": "EPS revisions",
                "meaning": "analysts revise slowly",
                "direction": "higher is better",
                "why": "underreaction",
                "fields": [*eps[:2], "not_a_field"],
            },
            {
                "name": "Other",
                "meaning": "noise",
                "direction": "higher is better",
                "why": "none",
                "fields": other[:1],
            },
        ]
        return json.dumps({"groups": groups}), 900, 200
    if task.startswith("Write 3 formulas that test this idea"):
        ids = [f["id"] for f in p["fields"]]
        a = ids[0]
        formulas = [
            {"formula": f"rank(ts_delta({a}, 21))", "why": "change"},
            {"formula": f"winsorize({a}, 0.01, 0.99)", "why": "invalid arity"},
            {"formula": f"rank({a})", "why": "level"},
            {"formula": f"ts_zscore({a}, 63)", "why": "own history"},
        ]
        return json.dumps({"formulas": formulas}), 400, 150
    if task.startswith("Diagnose why"):
        e = p["best"]["expression"]
        tried = {h["expression"] for h in p["history_all_tries"]}
        out = [
            {"expression": f"group_rank({e}, subindustry)", "change": "within subindustry"},
            {"expression": f"rank({e})", "change": "rank"},
            {"expression": f"ts_mean({e}, 5)", "change": "smooth"},
        ]
        out = [v for v in out if v["expression"] not in tried]
        return (
            json.dumps(
                {
                    "problem": "low fitness (returns too small for the turnover)",
                    "evidence": "fitness 0.6",
                    "decision": "change",
                    "why": "x",
                    "variants": out,
                }
            ),
            700,
            200,
        )
    if task.startswith("Propose ONE"):
        n = len(p["ideas_already_tried_here"])
        card = {
            "title": f"EPS revision drift {n}",
            "mechanism": "analysts revise slowly; prices underreact",
            "long": "upward revisions",
            "short": "downward revisions",
            "horizon_days": 21,
            "query": "eps earnings per share estimate revision",
            "evidence": {"type": "card", "id": "Q2"},
        }
        return json.dumps(card), 500, 100
    if task.startswith("Rate how well"):
        scores = [
            {
                "id": f["id"],
                "score": 9 if "_eps_" in f["id"] else 2,
                "reason": "eps" if "_eps_" in f["id"] else "other",
                # fy1 and fy2 versions measure the same quantity
                "measures": " ".join(f["id"].split("_")[1:3]),
            }
            for f in p["fields"]
        ]
        return json.dumps({"scores": scores}), 800, 300
    if "DIFFERENT expressions" in task:
        forms = [
            {"form": "rank(ts_delta({X}, 21))", "way": "change"},
            {"form": "rank(ts_delta({X}, 63))", "way": "same shape, other window"},
            {"form": "rank({X})", "way": "level"},
            {"form": "rank(ts_zscore({X}, 63))", "way": "own history"},
            {"form": "group_rank({X}, industry)", "way": "peers"},
            {"form": "madeup({X})", "way": "invalid"},
            {"form": "ts_rank({X}, 21)", "way": "recent rank"},
        ]
        return json.dumps({"forms": forms}), 400, 150
    if "DIFFERENT formulas" in task:
        ids = sorted((f["id"] for f in p["fields"]), key=lambda i: "_mean_" not in i)
        a, b = ids[0], ids[1]
        formulas = [
            {"formula": f"rank(ts_delta({a}, 21))", "why": "revision"},
            {"formula": f"rank(ts_delta({a}, 63))", "why": "same shape, other window"},
            {"formula": f"rank(ts_delta({a}, 21)) - rank({b})", "why": "spread"},
            {"formula": f"rank({b})", "why": "level"},
            {"formula": f"madeup({a})", "why": "invalid operator"},
            {"formula": f"group_rank(ts_delta({b}, 21), industry)", "why": "peers"},
            {"formula": f"ts_zscore({a}, 63)", "why": "own history"},
        ]
        return json.dumps({"formulas": formulas}), 400, 150
    if task.startswith("For each parent"):
        out = []
        for par in p["parents"]:
            e = par["expression"]
            out.append(
                {
                    "parent": par["id"],
                    "expression": f"group_rank({e}, subindustry)",
                    "step": "S3",
                    "change": "within subindustry",
                }
            )
            out.append(
                {"parent": par["id"], "expression": f"rank({e})", "step": "S0", "change": "rank"}
            )
        return json.dumps({"variants": out}), 600, 200
    return json.dumps({"description": "Idea: EPS revisions. Rationale: underreaction."}), 50, 50


def run(world: World, minutes: float = 10, **budget: Any):  # type: ignore[no-untyped-def]
    ledger = Ledger("sqlite://")
    api = world.platform()
    cfg = FactoryConfig()
    cfg.budget.min_batch_interval_s = 0
    for k, v in budget.items():
        setattr(cfg.budget, k, v)
    llm = FactoryLLM(cfg, ledger, complete=honest_llm)
    report = eng.run_day(
        cfg, api, ledger, llm, minutes=minutes, rng=random.Random(3), sleep=lambda _s: None
    )
    return ledger, api, report


def test_signal_is_found_improved_and_handed_over_never_submitted() -> None:
    ledger, api, report = run(World({"est": analyst_fields("eps")}))
    with ledger.session() as s:
        idea = s.scalars(select(ResearchIdeaORM)).first()
        assert idea is not None, report
        assert idea.extra.get("diagnoses"), idea.extra  # 第 5+6 步 ran on the whole history
        assert idea.extra["group"]["fields"] and "not_a_field" not in idea.extra["group"]["fields"]
        first = [c for c in s.scalars(select(CandidateORM)) if c.stage == Stage.FIELD]
        assert first and any("eps" in c.expression for c in first)
        improved = [c for c in s.scalars(select(CandidateORM)) if c.stage == Stage.IMPROVE]
        assert any("subindustry" in c.expression for c in improved)
    assert api.submitted == []  # handed over, never submitted


@pytest.mark.skip(reason="v3 设计，已被 10/14 晚现行设计取代")
def test_llm_only_stops_after_one_idea_and_five_formulas() -> None:
    ledger = Ledger("sqlite://")
    api = World({"est": analyst_fields("eps")}).platform()
    cfg = FactoryConfig()
    llm = FactoryLLM(cfg, ledger, complete=honest_llm)

    report = eng.run_llm_only(cfg, api, ledger, llm, rng=random.Random(3))

    assert report is not None
    assert report.title.startswith("EPS revision drift")
    assert len(report.formulas) == eng.FORMULAS
    assert report.simulations == 0
    with ledger.session() as s:
        candidates = list(s.scalars(select(CandidateORM)))
    assert len(candidates) == eng.FORMULAS
    assert all(candidate.status == "QUEUED" for candidate in candidates)
    assert not api.sims


@pytest.mark.skip(reason="v3 设计，已被 10/14 晚现行设计取代")
def test_noise_dataset_ideas_die_and_the_dataset_closes() -> None:
    ledger, _api, _report = run(World({"flat": analyst_fields(None)}), minutes=10)
    with ledger.session() as s:
        ideas = list(s.scalars(select(ResearchIdeaORM)))
        cell = s.get(CellORM, next(iter(c.cell_id for c in s.scalars(select(CellORM)))))
        kinds = [e.kind for e in s.scalars(select(EventORM))]
    assert ideas  # the same idea again is a duplicate: two in a row close the dataset
    assert all(i.status == IdeaStatus.DEAD for i in ideas)
    assert "idea_failed" in kinds
    assert "dataset_closed" in kinds
    assert cell is not None and cell.plan.get("outcome") == "noise"


@pytest.mark.skip(reason="v3 设计，已被 10/14 晚现行设计取代")
def test_formulas_state_the_idea_with_its_own_fields() -> None:
    ledger, _api, _ = run(World({"flat": analyst_fields(None)}), minutes=10)
    with ledger.session() as s:
        idea = s.scalars(select(ResearchIdeaORM)).first()
        assert idea is not None and idea.extra["evidence"]["id"] == "Q2"
        exprs = [f["expression"] for f in idea.extra["formulas"]]
        assert len(exprs) == eng.FORMULAS  # the window-only duplicate and madeup() dropped
        assert not any("madeup" in e for e in exprs)
        assert any(e.count("ts_backfill(") == 2 for e in exprs)  # a two-field formula
        assert all("{X}" not in e for e in exprs)  # no template slot
        tested = idea.extra["test_fields"]
        quantities = {" ".join(f.split("_")[1:3]) for f in tested}
        assert len(quantities) == len(tested)  # never fy1 and fy2 of the same thing


def test_main_formulas_deduplicate_outer_rank_and_zscore_shells() -> None:
    proposed = [
        {"form": "rank(ts_delta(signal, 21))", "why": "rank shell"},
        {"form": "zscore(ts_delta(signal, 21))", "why": "zscore shell"},
        {"form": "ts_delta(signal, 63)", "why": "same shape and other window"},
        {"form": "subtract(signal, benchmark)", "why": "different relation"},
    ]

    formulas = eng._main_formulas(
        proposed,
        {"signal": "MATRIX", "benchmark": "MATRIX"},
        {"signal": "signal", "benchmark": "benchmark"},
        {"rank", "zscore", "ts_delta", "subtract"},
    )

    assert [formula for formula, _why, _field in formulas] == [
        "rank(ts_delta(signal, 21))",
        "subtract(signal, benchmark)",
    ]


def test_no_example_formula_in_any_prompt() -> None:
    seen: list[str] = []

    def spy(model: str, system: str, user: str) -> tuple[str, int, int]:
        seen.append(user)
        return honest_llm(model, system, user)

    ledger = Ledger("sqlite://")
    api = World({"est": analyst_fields("eps")}).platform()
    cfg = FactoryConfig()
    cfg.budget.min_batch_interval_s = 0
    llm = FactoryLLM(cfg, ledger, complete=spy)
    eng.run_day(cfg, api, ledger, llm, minutes=3, rng=random.Random(1), sleep=lambda _s: None)
    assert seen
    for prompt in seen:
        if json.loads(prompt)["task"].startswith(("Propose ONE", "Write 5", "Write 1", "Rate how")):
            data = json.loads(prompt)
            # no formula anywhere in what we write (operators are listed by name only)
            ours = json.dumps({k: v for k, v in data.items() if k != "operators"})
            assert not re.search(r"[a-z_]+\(", ours), ours[:300]


def test_a_noise_idea_stops_at_the_first_gate() -> None:
    ledger, _api, _ = run(World({"flat": analyst_fields(None)}), minutes=10)
    with ledger.session() as s:
        ideas = list(s.scalars(select(ResearchIdeaORM)))
    assert ideas and all(i.status == IdeaStatus.DEAD for i in ideas)
    for i in ideas:  # 第 4 问 gate, or after improvement nothing passed the regular line
        why = i.extra.get("why_dead", "")
        assert "told from noise" in why or "regular line" in why, why


def test_first_gate_is_081_for_five_formulas() -> None:
    assert round(eng.first_gate(5, 0.316), 2) == 0.81


@pytest.mark.skip(reason="v3 设计，已被 10/14 晚现行设计取代")
def test_record_page_lists_every_step(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from alpha_orchestrator.factory.record import render_record

    ledger, _api, _ = run(World({"est": analyst_fields("eps")}))
    text = render_record(ledger, tmp_path / "record.html").read_text(encoding="utf-8")
    for marker in ("第1-2步", "第3步", "第4步", "第7-8步", "第8步", "EPS revision drift"):
        assert marker in text


def test_high_turnover_is_fixed_not_discarded() -> None:
    from alpha_orchestrator.factory.models import CandidateORM as C

    e = eng.Engine.__new__(eng.Engine)
    e.config = FactoryConfig()
    assert not e._suspicious(C(sharpe=0.7, turnover=0.8))  # goes to step 7: raise decay
    assert e._suspicious(C(sharpe=4.6, turnover=0.1))  # the user's line: a fake


def test_luck_line_grows_slowly() -> None:
    assert round(eng.luck_line(10, 0.40), 2) == 1.12
    assert round(eng.luck_line(40, 0.40), 2) == 1.29
    assert round(eng.luck_line(300, 0.40), 2) == 1.50


def test_an_unreachable_llm_stops_the_day_and_closes_nothing() -> None:
    def broke(model: str, system: str, user: str) -> tuple[str, int, int]:
        raise TypeError("'NoneType' object is not subscriptable")  # what an empty balance gave

    ledger = Ledger("sqlite://")
    api = World({"est": analyst_fields("eps")}).platform()
    cfg = FactoryConfig()
    cfg.budget.min_batch_interval_s = 0
    llm = FactoryLLM(cfg, ledger, complete=broke)
    report = eng.run_day(
        cfg, api, ledger, llm, minutes=3, rng=random.Random(1), sleep=lambda _s: None
    )
    assert "LLM" in report.stopped_because
    with ledger.session() as s:
        assert all(c.state != "EXHAUSTED" for c in s.scalars(select(CellORM)))


@pytest.mark.skip(reason="v3 设计，已被 10/14 晚现行设计取代")
def test_a_rejected_citation_asks_for_a_card_and_does_not_count_as_no_idea() -> None:
    asked: list[dict[str, Any]] = []

    def cites_badly(model: str, system: str, user: str) -> tuple[str, int, int]:
        p = json.loads(user)
        if p["task"].startswith("Propose ONE"):
            asked.append(p)
            card = json.loads(honest_llm(model, system, user)[0])
            card["evidence"] = {"type": "paper", "id": "nope", "quote": "x" * 30}
            return json.dumps(card), 10, 10
        return honest_llm(model, system, user)

    ledger = Ledger("sqlite://")
    api = World({"est": analyst_fields("eps")}).platform()
    cfg = FactoryConfig()
    cfg.budget.min_batch_interval_s = 0
    llm = FactoryLLM(cfg, ledger, complete=cites_badly)
    assert eng.run_llm_only(cfg, api, ledger, llm, rng=random.Random(1)) is None
    assert "method card" in asked[1].get("your_last_answer_was_rejected_because", "")
    with ledger.session() as s:
        cell = next(c for c in s.scalars(select(CellORM)) if c.plan)
        assert cell.plan.get("idea_fail_streak", 0) == 0


def test_refill_skips_repeated_groups_and_near_miss_keeps_improving() -> None:
    ledger, _, _ = run(World({"est": analyst_fields("eps")}))
    with ledger.session() as s:
        cell = s.scalars(select(CellORM)).first()
        names = [g["name"] for g in (cell.plan or {}).get("groups", [])]
    assert len(names) == len(set(n.lower() for n in names)), names  # no repeated group
    assert eng.NEAR_MISS == 1.2 and eng.GROUP_REFILLS == 10


def _kinds(ledger: Ledger) -> list[str]:
    with ledger.session() as s:
        return [e.kind for e in s.scalars(select(EventORM))]


def test_no_signal_after_scouting_means_next_dataset_no_refill() -> None:
    ledger, _, _ = run(World({"est": analyst_fields(None)}, noise=0.2))
    kinds = _kinds(ledger)
    assert "field_groups" in kinds and "field_groups_refill" not in kinds


def test_signal_found_means_more_groups_are_asked_for() -> None:
    ledger, _, _ = run(World({"est": analyst_fields("eps")}), minutes=30)
    assert "field_groups_refill" in _kinds(ledger)


class _SizeWorld(World):
    """The signal is concentrated in small caps: comparing within liquidity deciles adds
    `bucket_gain` Sharpe (like the 10/16 run's two 1.58s failing the sub-universe check)."""

    bucket_gain: float = 0.0

    def metrics(self, expr: str, settings: dict[str, Any]) -> dict[str, Any]:
        m = super().metrics(expr, settings)
        if "bucket(rank(close*volume)" in expr and m["sharpe"] > 0.5:
            sharpe = m["sharpe"] + self.bucket_gain
            fitness = sharpe * 0.75
            fails = "" if sharpe >= 1.58 and fitness >= 1 else "LOW_SHARPE"
            from .fakes import is_metrics

            return is_metrics(sharpe, fitness, turnover=0.12, margin=0.0012, fails=fails)
        return m


def _enhance_rows(ledger: Ledger) -> dict[str, list[CandidateORM]]:
    out: dict[str, list[CandidateORM]] = {}
    with ledger.session() as s:
        for c in s.scalars(select(CandidateORM).where(CandidateORM.stage == Stage.ENHANCE)):
            out.setdefault(c.extra["idea"], []).append(c)
            s.expunge(c)
    return out


def test_a_near_miss_is_enhanced_by_deciles_and_passes() -> None:
    world = _SizeWorld(
        {"est": analyst_fields("eps", strength=1.0, signal_stats=("mean", "median", "high", "low"))}
    )
    world.bucket_gain = 0.6
    ledger, _, _ = run(world, minutes=30)
    rows = _enhance_rows(ledger)
    assert rows, "a near miss (1.0 + 0.35 within subindustry) must be enhanced"
    for kids in rows.values():
        assert all("bucket(" in k.expression or "trade_when(" in k.expression for k in kids)
        assert len(kids) <= len(eng.ENHANCE_GROUPS) + len(eng.ENHANCE_TIMING)
    with ledger.session() as s:
        done = [i for i in s.scalars(select(ResearchIdeaORM)) if i.status == IdeaStatus.DONE]
        hit_ids = {h for i in done for h in i.extra.get("regular_hits", [])}
        hit_stages = {s.get(CandidateORM, h).stage for h in hit_ids}
    assert Stage.ENHANCE in hit_stages


def test_enhance_tries_trade_when_on_the_best_wrapper_then_closes() -> None:
    world = _SizeWorld(
        {"est": analyst_fields("eps", strength=1.0, signal_stats=("mean", "median", "high", "low"))}
    )
    ledger, _, _ = run(world, minutes=30)
    rows = _enhance_rows(ledger)
    assert rows
    for idea_id, kids in rows.items():
        timing = [k for k in kids if k.expression.startswith("trade_when(")]
        assert len([k for k in kids if k not in timing]) == len(eng.ENHANCE_GROUPS)
        assert len(timing) == len(eng.ENHANCE_TIMING), idea_id
        with ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea.status == IdeaStatus.DEAD and idea.extra["phase"] == "enhanced"


def test_a_noise_idea_is_never_enhanced() -> None:
    ledger, _, _ = run(World({"est": analyst_fields(None)}, noise=0.2))
    assert not _enhance_rows(ledger)
