import json
import random
from typing import Any

from sqlalchemy import select

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.daily import run_day
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import CandidateORM, CellORM, ReviewORM
from alpha_orchestrator.factory.select import approve

from .fakes import FakePlatform, is_metrics

FIELDS = {
    "good": [{"id": "goodf_fy1", "type": "MATRIX", "coverage": 0.9},
             {"id": "goodf_fy2", "type": "MATRIX", "coverage": 0.8}],
    "dead": [{"id": "deadf", "type": "MATRIX", "coverage": 0.9},
             {"id": "deadg", "type": "MATRIX", "coverage": 0.9}],
}


def platform_metrics(expr: str, settings: dict[str, Any]) -> dict[str, Any]:
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
    task = json.loads(user)["task"]
    if "hypotheses" in task:
        return json.dumps({"hypotheses": [{"title": "Revision drift", "statement": "s",
                                           "direction": "high -> long", "fields": ["goodf_fy1"],
                                           "sources": ["paper"]}]}), 10, 10
    if task.startswith("Write"):
        return json.dumps({"expressions": [
            {"expression": "group_zscore(ts_av_diff(ts_backfill(goodf_fy1, 63), 126), subindustry)",
             "rationale": "r"}]}), 10, 10
    if "implement" in task:
        return json.dumps({"consistent": True, "reason": "ok"}), 5, 5
    return json.dumps({"description": "Idea: ... Rationale: ..."}), 5, 5


def test_one_day_end_to_end() -> None:
    ledger = Ledger("sqlite://")
    api = FakePlatform(platform_metrics, FIELDS)
    cfg = FactoryConfig()
    cfg.budget.min_batch_interval_s = 0
    cfg.policy.hypotheses_need_approval = False
    llm = FactoryLLM(cfg, ledger, complete=scripted_llm)

    report = run_day(cfg, api, ledger, llm, minutes=5, rng=random.Random(7),  # type: ignore[arg-type]
                     sleep=lambda _s: None)
    with ledger.session() as s:
        cells = {c.dataset_id: c for c in s.scalars(select(CellORM))}
        cands = list(s.scalars(select(CandidateORM)))
        reviews = list(s.scalars(select(ReviewORM)))
    assert cells["dead"].state == "KILLED"
    assert cells["good"].state == "SIGNAL" and cells["good"].hits > 0
    stages = {c.stage for c in cands}
    assert {"PROBE", "FLIP", "SECOND", "ROBUST"} <= stages
    ready = [c for c in cands if c.extra.get("ready")]
    assert ready and any((c.extra.get("robust") or {}).get("pass") for c in ready)
    assert 1 <= len(reviews) <= 4 and report.reviews
    # bounded: two tiny datasets must not eat hundreds of simulations
    assert report.simulations + report.failed <= 2 * (50 + 100) + 60
    assert api.submitted == []  # nothing is submitted without approval

    # Day 2: the dataset with signal is exploited with LLM expressions from its hypothesis.
    run_day(cfg, api, ledger, llm, minutes=5, rng=random.Random(8),  # type: ignore[arg-type]
            sleep=lambda _s: None)
    with ledger.session() as s:
        llm_rows = [c for c in s.scalars(select(CandidateORM)) if c.stage == "LLM"]
        total_reviews = len(list(s.scalars(select(ReviewORM))))
    assert llm_rows and llm_rows[0].hypothesis_id
    assert total_reviews <= 4  # four per day, counted by review creation date

    assert approve(api, ledger, reviews[0].id)  # type: ignore[arg-type]
    assert api.submitted == [reviews[0].alpha_id]
