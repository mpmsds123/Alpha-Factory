"""Which dataset to research next: a multi-armed bandit over datasets.

Each dataset has an estimated chance of "the survey finds more signal than noise",
written as a Beta distribution (successes, failures):

    successes  1 per 2 official research papers naming the dataset (capped at 2)
               + 1 for each region where this same dataset was found to have signal
               + 0.5 for each dataset of the same category in the same region with signal
    failures   the same, for datasets whose survey said "noise"

The next dataset is the one with the best draw of (chance x payoff), Thompson sampling:
well-supported datasets are picked most, untried ones still get picked sometimes, and a
dataset that failed elsewhere falls back on its own. Payoff is what a found alpha is worth:
pyramid multiplier and gap, crowding, regional cost and sharpe floor.

A dataset the catalog already shows as below the coverage floor is never opened, and one
found too sparse at the health check counts as a failure like a "noise" survey does.

`rank()` gives the same numbers without the random draw, for the dashboard.
"""

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha_orchestrator.factory.catalog import latest_pyramid_counts
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.models import CellORM, CellState, IdeaORM

PAPER_SOURCES = frozenset({"Research_Papers_for_Consultants", "Research_Papers_for_Users"})
PAPER_TITLE = re.compile(r"^research paper", re.I)  # official paper posts only


@dataclass(frozen=True)
class Ranked:
    cell_id: str
    score: float
    reason: dict[str, Any]


def gap_bonus(submitted: int) -> float:
    """A pyramid completes at 3 submissions; the last missing one is worth the most."""
    return {0: 1.15, 1: 1.2, 2: 1.3}.get(submitted, 1.0)


def crowding_discount(alpha_count: int) -> float:
    """10 alphas -> 0.10, 1,000 -> 0.30, capped at 0.6."""
    return min(0.6, math.log10(1 + max(alpha_count, 0)) / 10)


def cost_factor(margin_floor_bps: float) -> float:
    return max(0.5, 1.0 - (margin_floor_bps - 5.0) / 75.0)


def is_paper(idea: IdeaORM) -> bool:
    return idea.source in PAPER_SOURCES or bool(PAPER_TITLE.search(idea.title))


def count_paper_refs(s: Session, dataset_ids: list[str]) -> dict[str, int]:
    """How many official research-paper posts name each dataset id (as a whole word)."""
    patterns = {d: re.compile(rf"(?<![\w]){re.escape(d)}(?![\w])", re.I) for d in dataset_ids}
    out: Counter[str] = Counter()
    for idea in s.scalars(select(IdeaORM)):
        if not is_paper(idea):
            continue
        text = f"{idea.title}\n{idea.body}"
        for d, pattern in patterns.items():
            if pattern.search(text):
                out[d] += 1
    return dict(out)


def outcome_of(cell: CellORM) -> str | None:
    """'signal' once a survey found signal; 'noise' when it found none, or when the data was
    too sparse to survey at all. Both are lessons for datasets like this one."""
    plan = cell.plan or {}
    if plan.get("verdict") in ("signal", "noise"):
        return str(plan["verdict"])
    if plan.get("outcome") in ("no_eligible", "no_fields"):
        return "noise"
    return None


def evidence(cells: list[CellORM], refs: dict[str, int]) -> dict[str, tuple[float, float]]:
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    by_cat_region: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for c in cells:
        o = outcome_of(c)
        if o:
            by_dataset[c.dataset_id][o] += 1
            by_cat_region[(c.category.lower(), c.region)][o] += 1
    out = {}
    for c in cells:
        own = by_dataset[c.dataset_id]
        near = by_cat_region[(c.category.lower(), c.region)]
        mine = outcome_of(c)
        succ = min(refs.get(c.dataset_id, 0), 4) / 2 + own["signal"] + 0.5 * near["signal"]
        fail = own["noise"] + 0.5 * near["noise"]
        if mine:  # do not count a dataset's own result toward itself
            succ -= 1.5 if mine == "signal" else 0
            fail -= 1.5 if mine == "noise" else 0
        out[c.cell_id] = (max(succ, 0.0), max(fail, 0.0))
    return out


def payoff(
    c: CellORM, config: FactoryConfig, pyramids: Counter[tuple[str, int, str]]
) -> dict[str, float]:
    return {
        "multiplier": c.multiplier,
        "pyramid_gap": gap_bonus(pyramids.get((c.region, c.delay, c.category.lower()), 0)),
        "themes": 1.2 if c.themes else 1.0,
        "crowding": round(1 - crowding_discount(c.alpha_count), 3),
        "cost": round(cost_factor(config.margin_floor(c.region)), 3),
        "difficulty": round(config.thresholds.submit_sharpe / config.sharpe_floor(c.region), 3),
    }


def _open(s: Session) -> list[CellORM]:
    return list(s.scalars(select(CellORM).where(CellORM.active.is_(True))))


def rank(s: Session, config: FactoryConfig, rng: random.Random | None = None) -> list[Ranked]:
    """Scores every dataset still NEW; with `rng`, scores are Thompson draws."""
    cells = _open(s)
    pyramids = latest_pyramid_counts(s)
    refs = count_paper_refs(s, sorted({c.dataset_id for c in cells}))
    ev = evidence(cells, refs)
    out = []
    for c in cells:
        c.paper_refs = refs.get(c.dataset_id, 0)
        succ, fail = ev[c.cell_id]
        value = 1.0
        pay = payoff(c, config, pyramids)
        for v in pay.values():
            value *= float(v)
        chance = rng.betavariate(1 + succ, 1 + fail) if rng else (1 + succ) / (2 + succ + fail)
        reason = {
            **pay,
            "paper_mentions": c.paper_refs,
            "successes": round(succ, 2),
            "failures": round(fail, 2),
            "chance": round((1 + succ) / (2 + succ + fail), 3),
        }
        c.rank_score, c.rank_reason = round(value * reason["chance"], 4), reason
        if c.state != CellState.NEW or c.submitted >= config.thresholds.dataset_cap:
            continue
        if 0 < c.coverage < config.thresholds.min_coverage:
            continue  # the catalog already says too sparse: costs nothing to skip
        out.append(Ranked(c.cell_id, value * chance, reason))
    out.sort(key=lambda r: -r.score)
    return out
