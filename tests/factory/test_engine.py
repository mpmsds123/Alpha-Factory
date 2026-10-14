"""设计文档 v1 end to end on a simulated platform whose truth is known.

The scripted LLM is honest: it proposes "EPS revisions", rates eps fields 9 and the rest 2,
and improves a parent by comparing within subindustry (worth +0.35 Sharpe in this world).
"""

import json
import random
import re
from typing import Any

from sqlalchemy import select

from alpha_orchestrator.factory import engine as eng
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CellORM,
    EventORM,
    IdeaFieldORM,
    IdeaStatus,
    ResearchIdeaORM,
    ReviewORM,
    Stage,
)

from .world import World, analyst_fields


def honest_llm(model: str, system: str, user: str) -> tuple[str, int, int]:
    p = json.loads(user)
    task = p["task"]
    if task.startswith("Propose ONE"):
        n = len(p["ideas_already_tried_here"])
        card = {
            "title": f"EPS revision drift {n}",
            "mechanism": "analysts revise slowly; prices underreact",
            "long": "upward revisions",
            "short": "downward revisions",
            "horizon_days": 21,
            "query": "eps earnings per share estimate revision",
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
                    "change": "within subindustry",
                }
            )
            out.append({"parent": par["id"], "expression": f"rank({e})", "change": "rank"})
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


def test_signal_is_found_improved_and_handed_over_never_submitted(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(eng, "CONTROL_SHARE", 0.0)
    ledger, api, report = run(World({"est": analyst_fields("eps")}))
    with ledger.session() as s:
        idea = s.scalars(select(ResearchIdeaORM)).first()
        assert idea is not None, report
        assert idea.status in (IdeaStatus.PROMISING, IdeaStatus.DONE), idea.extra
        ranked = list(s.scalars(select(IdeaFieldORM).where(IdeaFieldORM.rank > 0)))
        assert ranked and all("_eps_" in r.field_id for r in ranked)  # step 4 kept only eps
        first = [c for c in s.scalars(select(CandidateORM)) if c.stage == Stage.FIELD]
        assert first and all("eps" in c.expression for c in first)
        improved = [c for c in s.scalars(select(CandidateORM)) if c.stage == Stage.IMPROVE]
        assert any("subindustry" in c.expression for c in improved)
        reviews = list(s.scalars(select(ReviewORM)))
    assert reviews, report
    assert api.submitted == []  # handed over, never submitted


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


def test_noise_dataset_ideas_die_and_the_dataset_closes() -> None:
    ledger, _api, _report = run(World({"flat": analyst_fields(None)}), minutes=10)
    with ledger.session() as s:
        ideas = list(s.scalars(select(ResearchIdeaORM)))
        cell = s.get(CellORM, next(iter(c.cell_id for c in s.scalars(select(CellORM)))))
        kinds = [e.kind for e in s.scalars(select(EventORM))]
    assert len(ideas) >= eng.DEAD_IN_ROW
    assert all(i.status == IdeaStatus.DEAD for i in ideas)
    assert "dataset_closed" in kinds
    assert cell is not None and cell.plan.get("outcome") == "noise"


def test_main_arm_formulas_state_the_idea_with_its_own_fields(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(eng, "CONTROL_SHARE", 0.0)
    ledger, _api, _ = run(World({"flat": analyst_fields(None)}), minutes=10)
    with ledger.session() as s:
        idea = s.scalars(select(ResearchIdeaORM)).first()
        assert idea is not None and idea.extra["arm"] == "main"
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


def test_control_arm_expands_one_template_over_random_fields(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(eng, "CONTROL_SHARE", 1.0)
    ledger, _api, _ = run(World({"flat": analyst_fields(None)}), minutes=10)
    with ledger.session() as s:
        idea = s.scalars(select(ResearchIdeaORM)).first()
        assert idea is not None and idea.extra["arm"] == "control"
        exprs = [f["expression"] for f in idea.extra["formulas"]]
        shapes = {e.split("ts_backfill")[0] for e in exprs}
        assert len(exprs) == eng.CONTROL_FIELDS and len(shapes) == 1  # one template, k fields


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


def test_an_idea_dies_only_after_improvement_fails() -> None:
    ledger, _api, _ = run(World({"flat": analyst_fields(None)}), minutes=10)
    with ledger.session() as s:
        ideas = list(s.scalars(select(ResearchIdeaORM)))
        improved = [c for c in s.scalars(select(CandidateORM)) if c.stage == Stage.IMPROVE]
    assert improved  # ideas reach step 7 before any verdict
    dead = [i for i in ideas if i.status == IdeaStatus.DEAD]
    assert dead and all("after improvement" in i.extra.get("why_dead", "") for i in dead)


def test_record_page_lists_every_step(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from alpha_orchestrator.factory.record import render_record

    ledger, _api, _ = run(World({"est": analyst_fields("eps")}))
    text = render_record(ledger, tmp_path / "record.html").read_text(encoding="utf-8")
    for marker in ("第1-2步", "第3步", "第4步", "第7-8步", "第8步", "EPS revision drift"):
        assert marker in text


def test_a_form_is_valid_even_when_the_dataset_has_only_vector_fields() -> None:
    ops = {"rank", "ts_delta", "ts_backfill", "vec_avg"}
    assert eng._form_ok("rank(ts_delta({X}, 21))", ops)
    assert not eng._form_ok("rank({X}) - rank({X})", ops)  # two placeholders
    assert not eng._form_ok("madeup({X})", ops)


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
