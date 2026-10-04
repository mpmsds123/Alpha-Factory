"""A whole day against an in-memory platform and a scripted LLM researcher.

`good` holds a real effect: the value-revision idea has ore (and one sub-idea has the
sign reversed); `dead` has none. Expected: both are read with the 6-way probes, ideas are
tested round by round, `good` is dug and yields reviewable alphas, `dead` is stopped by the
no-ore rule, and nothing is submitted without approval.
"""

import json
import random
from typing import Any

from sqlalchemy import select

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.daily import run_day
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CellORM,
    FieldORM,
    HypothesisORM,
    LLMCallORM,
    ReviewORM,
)
from alpha_orchestrator.factory.select import approve

from .fakes import FakePlatform, is_metrics

DIG_EXPRESSION = "group_zscore(ts_av_diff(goodf_fy1, 126), subindustry)"
FIELDS = {
    "good": [
        {"id": "goodf_fy1", "type": "MATRIX", "coverage": 0.9, "description": "EPS revision"},
        {"id": "goodf_fy2", "type": "MATRIX", "coverage": 0.8, "description": "EPS revision"},
        {"id": "goodg", "type": "MATRIX", "coverage": 0.7, "description": "guidance"},
    ],
    "dead": [
        {"id": "deadf", "type": "MATRIX", "coverage": 0.9},
        {"id": "deadg", "type": "MATRIX", "coverage": 0.9},
    ],
}


def platform_metrics(expr: str, settings: dict[str, Any]) -> dict[str, Any]:
    if settings.get("neutralization") == "NONE":  # 6-way probes: a daily, unscaled field
        return is_metrics(0.0, 0.0, long_count=2700, short_count=0)
    if "dead" in expr:
        return is_metrics(0.1, 0.05, fails="LOW_SHARPE,LOW_FITNESS")
    sharpe = 1.3
    if "ts_delta" in expr:
        sharpe = -1.3
    if "group_rank" in expr:
        sharpe += 0.5 if sharpe > 0 else -0.5
    if expr.startswith("multiply(-1"):
        sharpe = abs(sharpe)
    if expr.startswith("sign("):
        sharpe *= 0.7
    fitness = sharpe * 0.75
    fails = "" if sharpe >= 1.58 and fitness >= 1 else "LOW_SHARPE"
    return is_metrics(sharpe, fitness, margin=0.0012, fails=fails)


def scripted_llm(model: str, system: str, user: str) -> tuple[str, int, int]:
    prompt = json.loads(user)
    task = prompt["task"]
    if task.startswith("Propose"):
        if prompt["dataset"]["id"] == "dead":
            ideas = [
                {
                    "title": "Dead level",
                    "mechanism": "m",
                    "template": "rank({A})",
                    "slots": {"A": ["deadf", "deadg"]},
                    "windows": [63],
                }
            ]
        else:
            ideas = [
                {
                    "title": "Revision drift",
                    "mechanism": "analysts under-react to revisions",
                    "direction": "upward revisions -> long",
                    "sources": ["Research Paper 9"],
                    "template": "group_rank(ts_av_diff({A}, {w}), industry)",
                    "slots": {"A": ["goodf_fy1", "goodf_fy2"]},
                    "windows": [63, 126],
                    "decay": 4,
                    "neutralization": "INDUSTRY",
                },
                {
                    "title": "Revision change",
                    "mechanism": "fresh revisions",
                    "template": "rank(ts_delta({A}, {w}))",
                    "slots": {"A": ["goodf_fy1"]},
                    "windows": [21],
                },
            ]
        return json.dumps({"ideas": ideas}), 10, 10
    if task.startswith("Read this round"):
        verdicts = []
        for idea in prompt["ideas"]:
            positive = (idea["positive_share"] or 0) > 0.5
            if idea["ore_count"] and positive:
                verdicts.append(
                    {
                        "id": idea["id"],
                        "verdict": "dig",
                        "reason": "ore",
                        "expressions": [{"expression": DIG_EXPRESSION, "why": "peer compare"}],
                    }
                )
            elif idea["ore_count"]:
                verdicts.append({"id": idea["id"], "verdict": "flip", "reason": "reversed"})
            else:
                verdicts.append({"id": idea["id"], "verdict": "drop", "reason": "flat"})
        return json.dumps({"ideas": verdicts, "dataset": {"decision": "continue"}}), 10, 10
    return json.dumps({"description": "Idea: ... Rationale: ..."}), 5, 5


def make() -> tuple[Ledger, FakePlatform, FactoryConfig, FactoryLLM]:
    ledger = Ledger("sqlite://")
    api = FakePlatform(platform_metrics, FIELDS)
    cfg = FactoryConfig()
    cfg.budget.min_batch_interval_s = 0
    return ledger, api, cfg, FactoryLLM(cfg, ledger, complete=scripted_llm)


def test_one_day_end_to_end() -> None:
    ledger, api, cfg, llm = make()
    report = run_day(cfg, api, ledger, llm, minutes=5, rng=random.Random(7), sleep=lambda _s: None)
    with ledger.session() as s:
        cells = {c.dataset_id: c for c in s.scalars(select(CellORM))}
        cands = list(s.scalars(select(CandidateORM)))
        reviews = list(s.scalars(select(ReviewORM)))
        rep = s.scalar(select(FieldORM).where(FieldORM.field_id == "goodf_fy1"))
        member = s.scalar(select(FieldORM).where(FieldORM.field_id == "goodf_fy2"))
        hyps = {h.title: h for h in s.scalars(select(HypothesisORM)) if h.rounds}
        purposes = {c.purpose for c in s.scalars(select(LLMCallORM))}

    # 1. the data was read before any idea: probes on representatives, profile on every member
    assert rep is not None and member is not None
    assert rep.profile["cadence"] == "daily" and member.profile == rep.profile
    assert rep.prep == "winsorize(goodf_fy1, std=4)"  # daily, full coverage, unscaled
    assert all(c.settings["neutralization"] == "NONE" for c in cands if c.stage == "PROFILE")

    # 2. ideas were expanded with the prepared fields and the idea's own settings
    idea_rows = [c for c in cands if c.stage == "IDEA"]
    assert idea_rows and all("winsorize(" in c.expression for c in idea_rows)
    assert any(c.settings["decay"] == 4 for c in idea_rows)

    # 3. the LLM review routed each idea: dig near ore, flip the reversed one, drop the dead
    stages = {c.stage for c in cands}
    assert {"PROFILE", "IDEA", "DIG", "FLIP", "ROBUST"} <= stages
    assert any(c.expression == DIG_EXPRESSION for c in cands)
    assert hyps["Revision change"].verdicts[0]["verdict"] == "flip"
    assert hyps["Dead level"].verdicts[0]["verdict"] == "drop"
    assert {"ideas", "review", "description"} <= purposes

    # 4. the stop rule closed the dataset with nothing in it, with the reason kept
    assert cells["dead"].state == "EXHAUSTED"
    assert cells["dead"].hits == 0 and cells["dead"].rounds_without_ore >= 2
    assert cells["good"].hits > 0

    # 5. ready alphas were stress-tested and reached you; nothing was submitted
    ready = [c for c in cands if c.extra.get("ready")]
    assert ready and any((c.extra.get("robust") or {}).get("pass") for c in ready)
    assert 1 <= len(reviews) <= 4 and report.reviews
    assert api.submitted == []
    assert report.simulations + report.failed <= 2 * cfg.budget.dataset_sim_cap

    assert approve(api, ledger, reviews[0].id)  # type: ignore[arg-type]
    assert api.submitted == [reviews[0].alpha_id]


def test_stopped_datasets_are_replaced_by_the_next_in_rank() -> None:
    ledger, api, cfg, llm = make()
    cfg.budget.active_datasets = 1
    run_day(cfg, api, ledger, llm, minutes=5, rng=random.Random(1), sleep=lambda _s: None)
    with ledger.session() as s:
        states = {c.dataset_id: c.state for c in s.scalars(select(CellORM))}
    # Both get worked within one day: when the first is stopped the second is opened.
    assert "NEW" not in states.values()


def test_ideas_can_wait_for_approval() -> None:
    ledger, api, cfg, llm = make()
    cfg.policy.hypotheses_need_approval = True
    run_day(cfg, api, ledger, llm, minutes=5, rng=random.Random(2), sleep=lambda _s: None)
    with ledger.session() as s:
        stages = {c.stage for c in s.scalars(select(CandidateORM))}
        statuses = {h.status for h in s.scalars(select(HypothesisORM))}
    assert stages == {"PROFILE"} and statuses == {"PROPOSED"}
