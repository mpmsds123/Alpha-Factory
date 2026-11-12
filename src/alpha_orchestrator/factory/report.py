"""The cost guard and the progress CSVs, all recomputed from the database.

CSVs (in reports_dir, git-ignored because they hold real expressions):
  stages.csv          per quota day: how many items entered and survived each stage
  rejections.csv      per quota day: why expressions were rejected before simulation
  by_prompt_model.csv per step, prompt version and model: calls, dollars, passes per dollar
  hypotheses.csv      one row per hypothesis (Memory B)
  priors.csv          one row per prior with its current confidence (Memory A)
  review_queue.csv    alphas stored for your review
"""

import csv
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import func, select

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import (
    DecisionORM,
    ExpressionORM,
    HypothesisORM,
    PriorORM,
    Review,
    SimStatus,
    SimulationORM,
)

STORED = (Review.PENDING, Review.SUBMITTED)
STAGE_COLUMNS = (
    "day",
    "hypotheses_opened",
    "hypotheses_with_expressions",
    "expressions_proposed",
    "expressions_valid",
    "simulations_queued",
    "simulations_done",
    "simulations_failed",
    "round1_cleared",
    "diag_idea_supported",
    "diag_idea_failed",
    "diag_implementation_failed",
    "decision_mutate",
    "decision_revise",
    "decision_drop",
    "program_overrides",
    "train_pass",
    "oos_pass",
    "platform_pass",
    "stored_for_review",
    "hypotheses_closed",
    "llm_usd",
    "usd_per_stored",
    "uncited_decisions",
    "invalid_llm_answers",
)
REJECTION_KINDS = (
    ("does not parse", "parse"),
    ("unknown operator", "unknown operator"),
    ("positional input", "arity"),
    ("grouping input", "group input"),
    ("not allowed", "field not allowed"),
    ("needs a reducer", "vector reducer"),
    ("operators >", "too many operators"),
    ("data fields >", "too many fields"),
    ("window", "window"),
    ("hypothesis field", "missing hypothesis field"),
    ("against the hypothesis direction", "wrong direction"),
    ("same structure", "not original"),
    ("already tried", "repeat"),
    ("identical simulation", "repeat"),
    ("declared a", "change not as declared"),
    ("identical to its parent", "no change"),
    ("parent is not", "bad parent"),
    ("budget spent", "budget"),
)


def guard(ledger: Ledger, config: FactoryConfig, now: datetime | None = None) -> str:
    """'' when a new hypothesis may open; otherwise the reason it may not."""
    now = now or datetime.now(UTC)
    c = config.cost
    with ledger.session() as s:
        today = float(
            s.scalar(
                select(func.coalesce(func.sum(DecisionORM.cost_usd), 0.0)).where(
                    DecisionORM.created_at >= config.budget.window_start(now)
                )
            )
            or 0.0
        )
        if today >= c.daily_usd_cap:
            return f"daily LLM cap: ${today:.2f} of ${c.daily_usd_cap:.2f} spent"
        since = now - timedelta(days=c.guard_window_days)
        spent = float(
            s.scalar(
                select(func.coalesce(func.sum(DecisionORM.cost_usd), 0.0)).where(
                    DecisionORM.created_at >= since
                )
            )
            or 0.0
        )
        passes = int(
            s.scalar(
                select(func.count())
                .select_from(SimulationORM)
                .where(SimulationORM.review.in_(STORED), SimulationORM.created_at >= since)
            )
            or 0
        )
    if spent >= c.guard_min_spend_usd and spent > c.usd_per_pass_max * passes:
        return (
            f"${spent:.2f} spent for {passes} stored alpha(s) in {c.guard_window_days} days: "
            f"above ${c.usd_per_pass_max:.2f} per alpha"
        )
    return ""


def _day(t: datetime | None, config: FactoryConfig) -> str:
    """The quota day a moment belongs to, named by the calendar date (in the quota time
    zone) that most of its window covers: a window opening 21:00 on the 11th is the 12th."""
    if t is None:
        return ""
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    b = config.budget
    start = b.window_start(t) + timedelta(hours=(24 - b.quota_reset_hour) % 24)
    return start.astimezone(ZoneInfo(b.quota_timezone)).date().isoformat()


def rejection_kind(text: str) -> str:
    low = text.lower()
    for needle, kind in REJECTION_KINDS:
        if needle in low:
            return kind
    return "other"


def _write(path: Path, columns: tuple[str, ...] | list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def stage_rows(
    ledger: Ledger, config: FactoryConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    days: dict[str, Counter[str]] = defaultdict(Counter)
    rejections: dict[str, Counter[str]] = defaultdict(Counter)
    gate = config.search.first_gate
    with ledger.session() as s:
        for h in s.scalars(select(HypothesisORM).where(HypothesisORM.source == "factory")):
            d = days[_day(h.created_at, config)]
            d["hypotheses_opened"] += 1
            if h.round > 0:
                d["hypotheses_with_expressions"] += 1
            if h.closed_at is not None:
                days[_day(h.closed_at, config)]["hypotheses_closed"] += 1
                if h.best_train_sharpe >= gate:
                    days[_day(h.closed_at, config)]["round1_cleared"] += 1
        for e in s.scalars(select(ExpressionORM)):
            key = _day(e.created_at, config)
            days[key]["expressions_proposed"] += 1
            if e.valid:
                days[key]["expressions_valid"] += 1
            else:
                rejections[key][rejection_kind(e.rejection)] += 1
        for x in s.scalars(select(SimulationORM)):
            days[_day(x.created_at, config)]["simulations_queued"] += 1
            done_day = days[_day(x.completed_at, config)] if x.completed_at else None
            if done_day is None:
                continue
            if x.status == SimStatus.DONE:
                done_day["simulations_done"] += 1
            elif x.status == SimStatus.FAILED:
                done_day["simulations_failed"] += 1
            done_day["train_pass"] += int(x.train_pass)
            done_day["oos_pass"] += int(bool(x.oos_pass))
            done_day["platform_pass"] += int(bool(x.checks_pass))
            done_day["stored_for_review"] += int(x.review in STORED)
        for dec in s.scalars(select(DecisionORM)):
            d = days[_day(dec.created_at, config)]
            d["llm_usd_micro"] += round(dec.cost_usd * 1_000_000)
            if dec.actor == "program":
                d["program_overrides"] += 1
                continue
            if "uncited" in (dec.flags or []):
                d["uncited_decisions"] += 1
            if not dec.ok:
                d["invalid_llm_answers"] += 1
            out = dec.output or {}
            if dec.step == "diagnosis" and out.get("label"):
                d[f"diag_{out['label']}"] += 1
            if dec.step == "decision" and out.get("action"):
                d[f"decision_{out['action']}"] += 1
    rows = []
    for day in sorted(k for k in days if k):
        c = days[day]
        usd = c.pop("llm_usd_micro", 0) / 1_000_000
        stored = c["stored_for_review"]
        rows.append(
            {
                "day": day,
                **{
                    k: c.get(k, 0)
                    for k in STAGE_COLUMNS[1:]
                    if k not in ("llm_usd", "usd_per_stored")
                },
                "llm_usd": round(usd, 4),
                "usd_per_stored": round(usd / stored, 4) if stored else "",
            }
        )
    kinds = sorted({k for c in rejections.values() for k in c})
    rej = [
        {"day": day, **{k: rejections[day].get(k, 0) for k in kinds}} for day in sorted(rejections)
    ]
    return rows, [{"_kinds": kinds}, *rej]


def prompt_model_rows(ledger: Ledger) -> list[dict[str, Any]]:
    """Passing alphas per dollar, by step, prompt version and model. A hypothesis counts
    as a pass for every (step, version, model) that took part in it."""
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    with ledger.session() as s:
        passed = {
            x.hypothesis_id
            for x in s.scalars(select(SimulationORM).where(SimulationORM.review.in_(STORED)))
        }
        for d in s.scalars(select(DecisionORM).where(DecisionORM.actor == "llm")):
            g = groups.setdefault(
                (d.step, d.prompt_version, d.model),
                {
                    "calls": 0,
                    "ok": 0,
                    "uncited": 0,
                    "retried": 0,
                    "usd": 0.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "hypotheses": set(),
                },
            )
            g["calls"] += 1
            g["ok"] += int(d.ok)
            g["uncited"] += int("uncited" in (d.flags or []))
            g["retried"] += int(d.attempts > 1)
            g["usd"] += d.cost_usd
            g["tokens_in"] += d.prompt_tokens
            g["tokens_out"] += d.completion_tokens
            if d.hypothesis_id is not None:
                g["hypotheses"].add(d.hypothesis_id)
    rows = []
    for (step, version, model), g in sorted(groups.items()):
        n_pass = len(g["hypotheses"] & passed)
        rows.append(
            {
                "step": step,
                "prompt_version": version,
                "model": model,
                "calls": g["calls"],
                "valid": g["ok"],
                "uncited": g["uncited"],
                "retried": g["retried"],
                "hypotheses": len(g["hypotheses"]),
                "passed_hypotheses": n_pass,
                "usd": round(g["usd"], 4),
                "passes_per_usd": round(n_pass / g["usd"], 3) if g["usd"] else "",
                "avg_tokens_in": g["tokens_in"] // g["calls"],
                "avg_tokens_out": g["tokens_out"] // g["calls"],
            }
        )
    return rows


def write_all(ledger: Ledger, config: FactoryConfig, status: str = "") -> Path:
    """Rewrite every CSV; return the folder."""
    out = Path(config.reports_dir)
    stages, rejections = stage_rows(ledger, config)
    if stages and status:
        stages[-1]["status"] = status
    _write(out / "stages.csv", [*STAGE_COLUMNS, "status"], stages)
    kinds = rejections[0]["_kinds"]
    _write(out / "rejections.csv", ["day", *kinds], rejections[1:])
    _write(
        out / "by_prompt_model.csv",
        [
            "step",
            "prompt_version",
            "model",
            "calls",
            "valid",
            "uncited",
            "retried",
            "hypotheses",
            "passed_hypotheses",
            "usd",
            "passes_per_usd",
            "avg_tokens_in",
            "avg_tokens_out",
        ],
        prompt_model_rows(ledger),
    )
    with ledger.session() as s:
        cost = dict(
            s.execute(
                select(DecisionORM.hypothesis_id, func.sum(DecisionORM.cost_usd)).group_by(
                    DecisionORM.hypothesis_id
                )
            ).all()
        )
        hyp = [
            {
                "id": f"H{h.id}",
                "parent": f"H{h.parent_id}" if h.parent_id else "",
                "source": h.source,
                "region": h.region,
                "dataset": h.dataset_id,
                "category": h.category,
                "field": h.field_id,
                "type": h.hypothesis_type,
                "direction": h.direction,
                "holding_days": h.holding_days,
                "status": h.status,
                "verdict": h.verdict,
                "best_train_sharpe": round(h.best_train_sharpe, 3),
                "simulations": h.sims_used,
                "rounds": h.round,
                "usd": round(float(cost.get(h.id) or 0.0), 4),
                "claim": h.claim,
                "reason": h.reason,
                "lesson": h.lesson,
            }
            for h in s.scalars(select(HypothesisORM).order_by(HypothesisORM.id))
        ]
        priors = [
            {
                "id": f"P{p.id}",
                "source_type": p.source_type,
                "source": p.source_title,
                "claim": p.claim,
                "a": p.a,
                "b": p.b,
                "confidence": round(p.confidence, 3),
                "updates": max(0, len(p.history or []) - 1),
            }
            for p in s.scalars(select(PriorORM).order_by(PriorORM.id))
        ]
        queue = []
        for x in s.scalars(
            select(SimulationORM)
            .where(SimulationORM.review != Review.NONE)
            .order_by(SimulationORM.id)
        ):
            h = s.get(HypothesisORM, x.hypothesis_id)
            queue.append(
                {
                    "simulation": x.id,
                    "alpha_id": x.alpha_id,
                    "review": x.review,
                    "hypothesis": f"H{x.hypothesis_id}",
                    "claim": h.claim if h else "",
                    "expression": x.expression,
                    "decay": x.settings.get("decay"),
                    "neutralization": x.settings.get("neutralization"),
                    "train_sharpe": x.train_sharpe,
                    "test_sharpe": x.test_sharpe,
                    "is_sharpe": x.is_sharpe,
                    "is_fitness": x.is_fitness,
                    "is_turnover": x.is_turnover,
                }
            )
    _write(out / "hypotheses.csv", list(hyp[0]) if hyp else ["id"], hyp)
    _write(out / "priors.csv", list(priors[0]) if priors else ["id"], priors)
    _write(out / "review_queue.csv", list(queue[0]) if queue else ["simulation"], queue)
    return out
