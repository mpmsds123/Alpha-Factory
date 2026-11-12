"""Design Q10: every alpha that passes is stored for you to pick from, and exported to a CSV.

store_passing() is what the factory runs (10/14). select_today() is the older daily-four
selection, kept for the review commands. Nothing is submitted here: `review approve` is
the only path that calls submit.
"""

import csv
import math
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.domain.models import utc_now
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CellORM,
    EventORM,
    HypothesisORM,
    ReviewORM,
    ReviewStatus,
)


def pnl_corr(a: dict[str, float], b: dict[str, float]) -> float:
    """Correlation of daily PnL changes. The platform's series is cumulative, and two
    upward-sloping cumulative curves correlate highly whatever they hold."""
    keys = sorted(set(a) & set(b))
    if len(keys) < 61:
        return 0.0
    x = [a[k] - a[p] for p, k in pairwise(keys)]
    y = [b[k] - b[p] for p, k in pairwise(keys)]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((i - mx) * (j - my) for i, j in zip(x, y, strict=True))
    sxx = sum((i - mx) ** 2 for i in x)
    syy = sum((j - my) ** 2 for j in y)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else 0.0


def check_value(checks: list[dict[str, Any]], name: str) -> float | None:
    for c in checks:
        if c.get("name") == name and isinstance(c.get("value"), (int, float)):
            return float(c["value"])
    return None


def checks_passed(checks: list[dict[str, Any]], corr_max: float) -> bool:
    """Empty, pending, unknown or incomplete evidence never authorizes promotion."""
    if not checks or any(c.get("result") != "PASS" for c in checks):
        return False
    for name in ("SELF_CORRELATION", "PROD_CORRELATION"):
        value = check_value(checks, name)
        if value is None or not math.isfinite(value) or value > corr_max:
            return False
    return True


def robust_passed(extra: dict[str, Any], keep_ratio: float) -> bool:
    robust = extra.get("robust") or {}
    ratios = robust.get("ratios") or {}
    return bool(robust.get("pass") and ratios) and all(
        isinstance(value, (int, float)) and math.isfinite(value) and value >= keep_ratio
        for value in ratios.values()
    )


def select_today(
    api: BrainAPI, ledger: Ledger, config: FactoryConfig, llm: FactoryLLM | None
) -> list[str]:
    start = datetime.combine(datetime.now(UTC).date(), datetime.min.time(), tzinfo=UTC)
    with ledger.session() as s:
        used_today = int(
            s.scalar(
                select(func.count()).select_from(ReviewORM).where(ReviewORM.created_at >= start)
            )
            or 0
        )
        slots = config.policy.max_submissions_per_day - used_today
        reviewed = set(s.scalars(select(ReviewORM.candidate_id)))
        keep = [
            r.alpha_id
            for r in s.scalars(
                select(ReviewORM).where(
                    ReviewORM.status.in_(
                        [ReviewStatus.PENDING, ReviewStatus.APPROVED, ReviewStatus.SUBMITTED]
                    )
                )
            )
        ]
        value_of = {c.cell_id: c.rank_score or 1.0 for c in s.scalars(select(CellORM))}
        pool: list[tuple[str, str, float, bool, int, float]] = []
        for c in s.scalars(select(CandidateORM).where(CandidateORM.status == "DONE")):
            if c.id in reviewed or not c.extra.get("ready") or not c.alpha_id:
                continue
            if not robust_passed(c.extra, config.thresholds.robust_keep_ratio):
                continue
            value = value_of.get(c.cell_id, 1.0)
            pool.append(
                (
                    c.id,
                    c.alpha_id,
                    float(value),
                    bool(c.extra.get("atom", False)),
                    int(c.extra.get("operators", 8)),
                    c.fitness or 0.0,
                )
            )
    if slots <= 0 or not pool:
        return []

    def prelim(item: tuple[Any, ...]) -> float:
        _, _, value, atom, ops, fitness = item
        return float(value * (1.15 if atom else 1.0) * (1 + 0.1 * (8 - ops) / 8) * fitness)

    pool.sort(key=prelim, reverse=True)
    kept_pnl = {aid: api.daily_pnl(aid) for aid in keep}
    chosen: list[str] = []
    for cid, alpha_id, value, atom, ops, fitness in pool:
        if len(chosen) >= slots:
            break
        checks = api.check(alpha_id)
        prod = check_value(checks, "PROD_CORRELATION")
        self_corr = check_value(checks, "SELF_CORRELATION")
        with ledger.session() as s:
            row = s.get(CandidateORM, cid)
            assert row is not None
            row.extra = {**row.extra, "submission_check": checks}
        if not checks_passed(checks, config.thresholds.pnl_corr_max):
            continue
        pnl = api.daily_pnl(alpha_id)
        worst = max((abs(pnl_corr(pnl, other)) for other in kept_pnl.values()), default=0.0)
        if worst > config.thresholds.pnl_corr_max:
            continue
        description = _describe(ledger, llm, cid)
        score = {
            "value": round(value, 3),
            "atom": atom,
            "operators": ops,
            "fitness": fitness,
            "prod_corr": prod,
            "self_corr": self_corr,
            "max_pnl_corr_vs_pool": round(worst, 3),
            "final": round(
                prelim((cid, alpha_id, value, atom, ops, fitness)) * (1 - (prod or 0) / 2), 4
            ),
        }
        with ledger.session() as s:
            s.add(
                ReviewORM(candidate_id=cid, alpha_id=alpha_id, score=score, description=description)
            )
            ledger.log(s, "review_created", alpha=alpha_id, score=score)
        kept_pnl[alpha_id] = pnl
        chosen.append(alpha_id)
    return chosen


def _describe(ledger: Ledger, llm: FactoryLLM | None, candidate_id: str) -> str:
    with ledger.session() as s:
        c = s.get(CandidateORM, candidate_id)
        assert c is not None
        cell = s.get(CellORM, c.cell_id)
        h = s.get(HypothesisORM, c.hypothesis_id) if c.hypothesis_id else None
        info = {
            "expression": c.expression,
            "settings": c.settings,
            "dataset": cell.dataset_name if cell else c.cell_id,
            "dataset_description": (cell.description[:800] if cell else ""),
            "hypothesis": h.statement if h else None,
            "lineage_change": c.change,
        }
    if llm is None:
        return ""
    try:
        return llm.describe(info)
    except RuntimeError:
        return ""


def approve(
    api: BrainAPI, ledger: Ledger, review_id: str, config: FactoryConfig | None = None
) -> bool:
    """The only code path that submits. Called by `alpha-factory review approve`."""
    with ledger.session() as s:
        r = s.get(ReviewORM, review_id)
        if r is None or r.status != ReviewStatus.PENDING:
            return False
        alpha_id, description = r.alpha_id, r.description
    checks = api.check(alpha_id)
    if not checks_passed(checks, (config or FactoryConfig()).thresholds.pnl_corr_max):
        with ledger.session() as s:
            ledger.log(s, "submission_check_blocked", alpha=alpha_id, checks=checks)
        return False
    if description:
        api.set_properties(alpha_id, description=description, tags=["alpha-factory"])
    ok = api.submit(alpha_id)
    with ledger.session() as s:
        r = s.get(ReviewORM, review_id)
        assert r is not None
        r.status = ReviewStatus.SUBMITTED if ok else ReviewStatus.FAILED
        r.decided_at = utc_now()
        if ok:
            c = s.get(CandidateORM, r.candidate_id)
            cell = s.get(CellORM, c.cell_id) if c else None
            if cell:
                cell.submitted += 1
        ledger.log(s, "submitted" if ok else "submit_failed", alpha=alpha_id)
    return ok


def reject(ledger: Ledger, review_id: str, note: str = "") -> bool:
    with ledger.session() as s:
        r = s.get(ReviewORM, review_id)
        if r is None or r.status != ReviewStatus.PENDING:
            return False
        r.status = ReviewStatus.REJECTED
        r.note = note
        r.decided_at = utc_now()
        return True


EXPORT_NAME = "工厂_待挑选.csv"
EXPORT_COLUMNS = (
    "review_id",
    "alpha_id",
    "dataset",
    "idea",
    "mechanism",
    "evidence_type",
    "evidence_id",
    "evidence_title",
    "evidence_quote",
    "registered_long",
    "registered_short",
    "mirrored",
    "expression",
    "decay",
    "neutralization",
    "sharpe",
    "fitness",
    "turnover",
    "margin_bps",
    "platform_fails",
    "self_corr",
    "prod_corr",
    "other_neutralization_ratio",
    "rounds",
    "steps",
    "idea_sims",
    "idea_cost_cny",
    "status",
    "created",
)


def quota_only(checks: list[dict[str, Any]]) -> bool:
    """Recognize the observed exhausted regular-submission allowance, fail closed otherwise."""
    failed = [check for check in checks if check.get("result") == "FAIL"]
    if len(failed) != 1 or failed[0].get("name") != "REGULAR_SUBMISSION":
        return False
    check = failed[0]
    limit, value = check.get("limit"), check.get("value")
    return (
        isinstance(limit, (int, float))
        and not isinstance(limit, bool)
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(limit)
        and limit > 0
        and value == limit
    )


def _eligible_check(s: Any, candidate: CandidateORM, config: FactoryConfig, window: str) -> bool:
    extra = candidate.extra or {}
    if extra.get("quota_deferred_window") == window:
        return False
    if not extra.get("platform_failed"):
        return True
    # Reclassify old rows only when the saved response and failure event agree.
    if extra.get("platform_failed") != ["REGULAR_SUBMISSION"] or not quota_only(
        extra.get("submission_check") or []
    ):
        return False
    event = s.scalar(
        select(EventORM)
        .where(
            EventORM.kind == "platform_check_failed",
            EventORM.payload["alpha"].as_string() == candidate.alpha_id,
        )
        .order_by(EventORM.id.desc())
        .limit(1)
    )
    if event is None:
        return False
    timestamp = event.at if event.at.tzinfo else event.at.replace(tzinfo=UTC)
    previous_window = config.budget.window_start(timestamp).isoformat()
    candidate.extra = {
        **extra,
        "platform_failed": [],
        "quota_deferred_window": previous_window,
        "platform_check_state": "deferred_quota",
    }
    s.add(EventORM(kind="platform_check_quota_reclassified", payload={"alpha": candidate.alpha_id}))
    return previous_window != window


def store_passing(
    api: BrainAPI, ledger: Ledger, config: FactoryConfig, *, now: datetime | None = None
) -> list[str]:
    """Design Q7, Q10: every regular-line alpha that kept its Sharpe under the other
    neutralization gets the platform's submission check (IS ladder, sub-universe,
    correlation) and, if nothing FAILs, a row in the pick-list. No daily limit, no ranking:
    you pick. Then the whole pick-list is written to the CSV."""
    window = config.budget.window_start(now).isoformat()
    with ledger.session() as s:
        reviewed = set(s.scalars(select(ReviewORM.candidate_id)))
        pool = [
            (c.id, c.alpha_id)
            for c in s.scalars(select(CandidateORM).where(CandidateORM.status == "DONE"))
            if c.id not in reviewed
            and c.alpha_id
            and (c.extra or {}).get("ready")
            and robust_passed(c.extra or {}, config.thresholds.robust_keep_ratio)
            and _eligible_check(s, c, config, window)
        ]
    stored: list[str] = []
    for cid, alpha_id in pool:
        checks = api.check(alpha_id)
        fails = [str(c.get("name")) for c in checks if c.get("result") == "FAIL"]
        with ledger.session() as s:
            row = s.get(CandidateORM, cid)
            assert row is not None
            row.extra = {**row.extra, "submission_check": checks}
            if quota_only(checks):
                row.extra = {
                    **row.extra,
                    "platform_check_state": "deferred_quota",
                    "quota_deferred_window": window,
                }
                ledger.log(s, "platform_check_quota_deferred", alpha=alpha_id, window=window)
                continue
            if fails:
                row.extra = {
                    **row.extra,
                    "platform_failed": fails,
                    "platform_check_state": "failed",
                }
                ledger.log(s, "platform_check_failed", alpha=alpha_id, fails=fails)
                continue
            if not checks_passed(checks, config.thresholds.pnl_corr_max):
                row.extra = {**row.extra, "platform_check_state": "pending"}
                ledger.log(s, "platform_check_deferred", alpha=alpha_id, checks=checks)
                continue
            score = {
                "prod_corr": check_value(checks, "PROD_CORRELATION"),
                "self_corr": check_value(checks, "SELF_CORRELATION"),
            }
            row.extra = {**row.extra, "platform_check_state": "passed"}
            s.add(ReviewORM(candidate_id=cid, alpha_id=alpha_id, score=score))
            ledger.log(s, "stored_for_pick", alpha=alpha_id, score=score)
        stored.append(alpha_id)
    export_csv(ledger, config)
    return stored


def _lineage(s: Any, c: CandidateORM) -> tuple[int, list[str], bool]:
    rounds, steps, mirrored = 0, [], False
    node: CandidateORM | None = c
    while node is not None:
        if node.change == "mirrored":
            mirrored = True
        elif node.stage == "IMPROVE":
            rounds += 1
            steps.append(node.change.split(":")[0])
        node = s.get(CandidateORM, node.parent_id) if node.parent_id else None
    return rounds, list(reversed(steps)), mirrored


def export_csv(ledger: Ledger, config: FactoryConfig) -> Path:
    """Design Q10: the pick-list, one row per stored alpha, in the repo's reports folder."""
    from alpha_orchestrator.factory.models import LLMCallORM, ResearchIdeaORM

    out = Path(config.reports_dir) / EXPORT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with ledger.session() as s:
        for r in s.scalars(select(ReviewORM).order_by(ReviewORM.created_at)):
            c = s.get(CandidateORM, r.candidate_id)
            if c is None:
                continue
            idea = s.get(ResearchIdeaORM, (c.extra or {}).get("idea") or "")
            ev = ((idea.extra or {}).get("evidence") or {}) if idea else {}
            rounds, steps, mirrored = _lineage(s, c)
            robust = (c.extra or {}).get("robust") or {}
            idea_rows = (idea.extra or {}).get("rows", []) if idea else []
            cost = (
                s.scalar(
                    select(func.coalesce(func.sum(LLMCallORM.cost_cny), 0.0)).where(
                        LLMCallORM.idea_id == idea.id
                    )
                )
                if idea
                else 0.0
            )
            rows.append(
                {
                    "review_id": r.id,
                    "alpha_id": r.alpha_id,
                    "dataset": c.cell_id,
                    "idea": idea.title if idea else "",
                    "mechanism": idea.mechanism if idea else "",
                    "evidence_type": ev.get("type", ""),
                    "evidence_id": ev.get("id", ""),
                    "evidence_title": ev.get("title", ""),
                    "evidence_quote": ev.get("quote", ""),
                    "registered_long": idea.long_side if idea else "",
                    "registered_short": idea.short_side if idea else "",
                    "mirrored": "yes" if mirrored else "no",
                    "expression": c.expression,
                    "decay": c.settings.get("decay"),
                    "neutralization": c.settings.get("neutralization"),
                    "sharpe": c.sharpe,
                    "fitness": c.fitness,
                    "turnover": c.turnover,
                    "margin_bps": round((c.margin or 0) * 10_000, 2),
                    "platform_fails": c.fails,
                    "self_corr": (r.score or {}).get("self_corr"),
                    "prod_corr": (r.score or {}).get("prod_corr"),
                    "other_neutralization_ratio": (robust.get("ratios") or {}).get(
                        "neutralization"
                    ),
                    "rounds": rounds,
                    "steps": " > ".join(steps),
                    "idea_sims": len(idea_rows),
                    "idea_cost_cny": round(float(cost or 0.0), 4),
                    "status": r.status,
                    "created": r.created_at.isoformat() if r.created_at else "",
                }
            )
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return out
