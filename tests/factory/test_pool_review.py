from __future__ import annotations

import json
import random
from typing import Any

from sqlalchemy import select

from alpha_orchestrator.factory import engine as eng
from alpha_orchestrator.factory import pool_review as pr
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import ResearchIdeaORM

from .test_engine import eps_fields, honest_llm
from .world import World


def test_band_rates_are_smoothed_and_zero_hits_is_not_zero() -> None:
    rates = pr.band_rates([(0.5, False)] * 4 + [(1.1, True), (1.1, False), (1.5, True)])
    assert [r["n"] for r in rates] == [4, 2, 1]
    assert 0 < rates[0]["p"] < 0.15  # no hit in 4 -> small but not zero
    assert rates[1]["p"] == 0.5
    assert rates[2]["p"] == 0.75


def test_parse_keeps_only_allowed_actions_one_per_idea() -> None:
    allowed = {"aaaaaaaa-1": ["continue", "stop"], "bbbbbbbb-2": ["continue", "stop", "enhance"]}
    raw = {
        "decisions": [
            {"idea": "aaaaaaaa", "action": "enhance", "why": "not allowed here"},
            {"idea": "aaaaaaaa", "action": "stop", "why": "ok"},
            {"idea": "aaaaaaaa", "action": "continue", "why": "second"},
            {"idea": "bbbbbbbb", "action": "ENHANCE", "why": "near miss"},
            {"idea": "zzzzzzzz", "action": "stop", "why": "unknown"},
            "junk",
        ]
    }
    ok, problems = pr.parse(raw, allowed)
    assert [(d.idea, d.action) for d in ok] == [("aaaaaaaa-1", "stop"), ("bbbbbbbb-2", "enhance")]
    assert len(problems) == 4


def _run(propose: str) -> tuple[Ledger, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def complete(model: str, system: str, user: str) -> tuple[str, int, int]:
        p = json.loads(user)
        if p["task"].startswith("Review this pool"):
            seen.append(p)
            ds = [
                {"idea": r["id"], "action": propose, "why": f"best_now {r['best_now']}"}
                for r in p["ideas"]
                if r["allowed_actions"]
            ]
            return json.dumps({"decisions": ds}), 300, 100
        return honest_llm(model, system, user)

    ledger = Ledger("sqlite://")
    cfg = FactoryConfig()
    cfg.pool_review = True
    cfg.budget.min_batch_interval_s = 0
    llm = FactoryLLM(cfg, ledger, complete=complete)
    api = World({"est": eps_fields()}).platform()
    eng.run_day(cfg, api, ledger, llm, minutes=10, rng=random.Random(3), sleep=lambda _s: None)
    return ledger, seen


def test_a_stop_proposal_ends_the_idea_with_the_reason_recorded() -> None:
    ledger, seen = _run("stop")
    assert seen, "the reviewer was never asked"
    assert all("allowed_actions" in r for r in seen[0]["ideas"])
    with ledger.session() as s:
        dead = [
            i
            for i in s.scalars(select(ResearchIdeaORM))
            if "pool review" in str(i.extra.get("why_dead", ""))
        ]
    assert dead


def test_an_enhance_proposal_is_refused_for_ideas_below_the_near_miss_line() -> None:
    ledger, seen = _run("enhance")
    assert seen
    with ledger.session() as s:
        for i in s.scalars(select(ResearchIdeaORM)):
            if i.best_sharpe < eng.NEAR_MISS:
                assert "pool review" not in str(i.extra.get("why_dead", ""))
