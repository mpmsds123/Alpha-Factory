"""设计文档 v1 end to end on a simulated platform whose truth is known.

The scripted LLM is honest: it proposes "EPS revisions", rates eps fields 9 and the rest 2,
and improves a parent by comparing within subindustry (worth +0.35 Sharpe in this world).
"""

import json
import random
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
            "form": "rank(ts_delta({X}, 21))",
            "query": "eps earnings per share estimate revision",
        }
        return json.dumps(card), 500, 100
    if task.startswith("Rate how well"):
        scores = [
            {
                "id": f["id"],
                "score": 9 if "_eps_" in f["id"] else 2,
                "reason": "eps" if "_eps_" in f["id"] else "other",
            }
            for f in p["fields"]
        ]
        return json.dumps({"scores": scores}), 800, 300
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


def test_signal_is_found_improved_and_handed_over_never_submitted() -> None:
    ledger, api, report = run(World({"est": analyst_fields("eps")}))
    with ledger.session() as s:
        idea = s.scalars(select(ResearchIdeaORM)).first()
        assert idea is not None and idea.status in (IdeaStatus.PROMISING, IdeaStatus.DONE)
        ranked = list(s.scalars(select(IdeaFieldORM).where(IdeaFieldORM.rank > 0)))
        assert ranked and all("_eps_" in r.field_id for r in ranked)  # step 4 kept only eps
        first = [c for c in s.scalars(select(CandidateORM)) if c.stage == Stage.FIELD]
        assert first and all("eps" in c.expression for c in first)
        improved = [c for c in s.scalars(select(CandidateORM)) if c.stage == Stage.IMPROVE]
        assert any("subindustry" in c.expression for c in improved)
        reviews = list(s.scalars(select(ReviewORM)))
    assert reviews, report
    assert api.submitted == []  # handed over, never submitted


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


def test_each_idea_first_tests_at_most_ten_fields() -> None:
    ledger, _api, _ = run(World({"flat": analyst_fields(None)}), minutes=10)
    with ledger.session() as s:
        for idea in s.scalars(select(ResearchIdeaORM)):
            n = sum(
                1 for c in s.scalars(select(CandidateORM)) if (c.extra or {}).get("idea") == idea.id
            )
            assert n <= eng.FIRST_TESTS


def test_luck_line_grows_slowly() -> None:
    assert round(eng.luck_line(10, 0.40), 2) == 1.12
    assert round(eng.luck_line(40, 0.40), 2) == 1.29
    assert round(eng.luck_line(300, 0.40), 2) == 1.50
