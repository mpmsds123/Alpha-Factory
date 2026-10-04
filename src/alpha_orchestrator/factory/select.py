"""Choose today's (at most) four alphas and put them in front of you.

Nothing is submitted here. A review row is created with the platform's own check results,
correlations, robustness verdict and an LLM-written description; `review approve` is the
only path that calls submit.
"""

import math
from datetime import UTC, datetime
from itertools import pairwise
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
            if not (c.extra.get("robust") or {}).get("pass"):
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
        fails = [c.get("name") for c in checks if c.get("result") == "FAIL"]
        prod = check_value(checks, "PROD_CORRELATION")
        self_corr = check_value(checks, "SELF_CORRELATION")
        with ledger.session() as s:
            row = s.get(CandidateORM, cid)
            assert row is not None
            row.extra = {**row.extra, "submission_check": checks}
        if fails:
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


def approve(api: BrainAPI, ledger: Ledger, review_id: str) -> bool:
    """The only code path that submits. Called by `alpha-factory review approve`."""
    with ledger.session() as s:
        r = s.get(ReviewORM, review_id)
        if r is None or r.status != ReviewStatus.PENDING:
            return False
        alpha_id, description = r.alpha_id, r.description
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
