"""Which dataset to research next. Decided once, before any simulation is spent.

Simulations test ideas; they never rank datasets against each other. The order comes
from what is known in advance:

    score = papers x payoff x (1 - crowding) x portfolio_gap x cost x difficulty

papers         the platform's Research Papers name datasets in which researchers already
               found alpha (零基础 AI 课 第4课); every mention raises the prior, capped x2.
payoff         pyramid multiplier, plus a bonus for the pyramid you have not completed.
crowding       alpha count on the dataset: crowded data rarely clears prod correlation.
portfolio_gap  official "value add": a category you have barely submitted adds more.
cost           higher regional cost floors kill more alphas after fees.
difficulty     a region with a higher sharpe floor (CHN) needs a stronger signal.

A dataset leaves the queue when its stop rule fires (see research.py); the reason is kept.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha_orchestrator.factory.catalog import latest_pyramid_counts
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.models import CellORM, CellState, IdeaORM

PAPER_SOURCES = frozenset({"Research_Papers_for_Consultants", "Research_Papers_for_Users"})
PAPER_TITLE = re.compile(r"paper|论文|research", re.I)


@dataclass(frozen=True)
class Ranked:
    cell_id: str
    score: float
    reason: dict[str, Any]


def paper_bonus(refs: int) -> float:
    return 1.0 + 0.25 * min(refs, 4)


def gap_bonus(submitted: int) -> float:
    """A pyramid completes at 3 submissions; the last missing one is worth the most."""
    return {0: 1.15, 1: 1.2, 2: 1.3}.get(submitted, 1.0)


def crowding_discount(alpha_count: int) -> float:
    """10 alphas -> 0.10, 1,000 -> 0.30, capped at 0.6."""
    return min(0.6, math.log10(1 + max(alpha_count, 0)) / 10)


def portfolio_gap(category_share: float) -> float:
    """Under 10% of your submissions in this category: bonus. Dominant category: discount."""
    if category_share < 0.10:
        return 1.2
    return max(0.6, 1.0 - 0.4 * (category_share - 0.10))


def cost_factor(margin_floor_bps: float) -> float:
    return max(0.5, 1.0 - (margin_floor_bps - 5.0) / 75.0)


def count_paper_refs(s: Session, dataset_ids: list[str]) -> dict[str, int]:
    """How many research-paper posts mention each dataset id (as a whole word)."""
    patterns = {d: re.compile(rf"(?<![\w]){re.escape(d)}(?![\w])", re.I) for d in dataset_ids}
    out: Counter[str] = Counter()
    for idea in s.scalars(select(IdeaORM)):
        if idea.source not in PAPER_SOURCES and not PAPER_TITLE.search(idea.title):
            continue
        text = f"{idea.title}\n{idea.body}"
        for d, pattern in patterns.items():
            if pattern.search(text):
                out[d] += 1
    return dict(out)


def rank(s: Session, config: FactoryConfig) -> list[Ranked]:
    """Scores every dataset still open and stores the score and its reasons on the row."""
    cells = list(
        s.scalars(
            select(CellORM).where(
                CellORM.active.is_(True),
                CellORM.state.not_in([CellState.EXHAUSTED, CellState.SATURATED]),
            )
        )
    )
    pyramids = latest_pyramid_counts(s)
    by_category: Counter[str] = Counter()
    for (_, _, cat), n in pyramids.items():
        by_category[cat] += n
    total = sum(by_category.values())
    refs = count_paper_refs(s, sorted({c.dataset_id for c in cells}))

    out = []
    for c in cells:
        if c.submitted >= config.thresholds.dataset_cap:
            continue
        cat = c.category.lower()
        share = by_category[cat] / total if total else 0.0
        c.paper_refs = refs.get(c.dataset_id, 0)
        reason = {
            "papers": round(paper_bonus(c.paper_refs), 3),
            "paper_mentions": c.paper_refs,
            "multiplier": c.multiplier,
            "pyramid_gap": gap_bonus(pyramids.get((c.region, c.delay, cat), 0)),
            "themes": 1.2 if c.themes else 1.0,
            "crowding": round(1 - crowding_discount(c.alpha_count), 3),
            "portfolio_gap": round(portfolio_gap(share), 3),
            "cost": round(cost_factor(config.margin_floor(c.region)), 3),
            "difficulty": round(config.thresholds.submit_sharpe / config.sharpe_floor(c.region), 3),
        }
        score = 1.0
        for key in (
            "papers",
            "multiplier",
            "pyramid_gap",
            "themes",
            "crowding",
            "portfolio_gap",
            "cost",
            "difficulty",
        ):
            score *= float(reason[key])
        c.rank_score, c.rank_reason = score, reason
        out.append(Ranked(c.cell_id, score, reason))
    out.sort(key=lambda r: -r.score)
    return out
