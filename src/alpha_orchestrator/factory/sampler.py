"""Which field gets the next hypothesis. The program decides, never the LLM.

Dataset: stay on the current one until it is used up (hypothesis cap, submission cap or a
streak of round-1 misses); then a Thompson draw over all datasets. Each dataset's draw
comes from Beta(1 + cleared, 1 + missed) of its (category, region): how often past
hypotheses on that kind of data cleared the round-1 gate.

Field: coverage at least the floor, no identifier/date fields, one field per family of
similar fields, a family not hypothesised on yet, and preferably one few alphas use
(official docs, "Understanding simulation limits").
"""

import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.models import (
    CatalogORM,
    HStatus,
    HypothesisORM,
    Review,
    SimulationORM,
)

# Official docs: technical fields (timestamps, dates, symbol identifiers) rarely carry alpha.
TECHNICAL = re.compile(
    r"\b(identifier|id code|isin|cusip|sedol|ticker|symbol|timestamp|date of|dates?"
    r"|domain names?|url|urls|website|web site|web address|email)\b",
    re.IGNORECASE,
)
FIELD_TYPES = ("MATRIX", "VECTOR")  # a grouping field is not a signal on its own
UNTAPPED_POOL = 10  # draw among the 10 least-used eligible fields
HELPERS_SHOWN = 20  # other fields offered as helpers (one per family)


@dataclass
class Pick:
    dataset: CatalogORM
    field: CatalogORM
    family: list[CatalogORM]  # same-family fields (for "field" mutations)
    helpers: list[CatalogORM]  # other fields of the dataset, one per family


def family_key(field_id: str) -> str:
    """Fields that differ only in numbers (horizons, versions) are one family."""
    return re.sub(r"\d+", "#", field_id.lower())


def _key(row: CatalogORM | HypothesisORM) -> tuple[str, str]:
    return row.region, row.dataset_id


def dataset_status(s: Session, config: FactoryConfig, region: str, dataset_id: str) -> str:
    """'' while the dataset may take more hypotheses, else why it is used up."""
    sc = config.search
    rows = list(
        s.scalars(
            select(HypothesisORM)
            .where(
                HypothesisORM.region == region,
                HypothesisORM.dataset_id == dataset_id,
                HypothesisORM.source == "factory",
                HypothesisORM.parent_id.is_(None),
            )
            .order_by(HypothesisORM.id)
        )
    )
    if len(rows) >= sc.dataset_hypotheses_max:
        return "hypothesis cap"
    reviewed = s.scalar(
        select(SimulationORM.id)
        .join(HypothesisORM, HypothesisORM.id == SimulationORM.hypothesis_id)
        .where(
            HypothesisORM.region == region,
            HypothesisORM.dataset_id == dataset_id,
            SimulationORM.review.in_([Review.PENDING, Review.SUBMITTED]),
        )
        .offset(sc.dataset_submissions_max - 1)
        .limit(1)
    )
    if reviewed is not None:
        return "submission cap"
    streak = 0
    for h in reversed(rows):
        if h.status != HStatus.CLOSED:
            continue
        if h.best_train_sharpe >= sc.first_gate:
            break
        streak += 1
    if streak >= sc.dataset_dry_streak:
        return f"{streak} round-1 misses in a row"
    return ""


def _current(s: Session, config: FactoryConfig) -> CatalogORM | None:
    last = s.scalar(
        select(HypothesisORM)
        .where(HypothesisORM.source == "factory")
        .order_by(HypothesisORM.id.desc())
        .limit(1)
    )
    if last is None or dataset_status(s, config, last.region, last.dataset_id):
        return None
    return s.scalar(
        select(CatalogORM).where(
            CatalogORM.kind == "dataset",
            CatalogORM.region == last.region,
            CatalogORM.dataset_id == last.dataset_id,
        )
    )


def arm_counts(s: Session, config: FactoryConfig) -> dict[tuple[str, str], tuple[int, int]]:
    """(cleared, missed) round 1, per (category, region), over closed hypotheses."""
    cleared: Counter[tuple[str, str]] = Counter()
    missed: Counter[tuple[str, str]] = Counter()
    for h in s.scalars(select(HypothesisORM).where(HypothesisORM.status == HStatus.CLOSED)):
        arm = ((h.category or "").lower(), h.region)
        if h.best_train_sharpe >= config.search.first_gate:
            cleared[arm] += 1
        else:
            missed[arm] += 1
    return {a: (cleared[a], missed[a]) for a in set(cleared) | set(missed)}


def choose_dataset(
    s: Session,
    config: FactoryConfig,
    rng: random.Random,
    skip: set[tuple[str, str]] | None = None,
) -> CatalogORM | None:
    """The current dataset, or a new one by Thompson draw."""
    skip = skip or set()
    current = _current(s, config)
    if current is not None and _key(current) not in skip:
        return current
    arms = arm_counts(s, config)
    best: tuple[float, CatalogORM] | None = None
    for d in s.scalars(select(CatalogORM).where(CatalogORM.kind == "dataset")):
        if d.delay != config.policy.delay or _key(d) in skip:
            continue
        if config.policy.regions and d.region not in config.policy.regions:
            continue
        if dataset_status(s, config, d.region, d.dataset_id):
            continue
        hit, miss = arms.get((d.category, d.region), (0, 0))
        draw = rng.betavariate(1 + hit, 1 + miss)
        if best is None or draw > best[0]:
            best = (draw, d)
    return best[1] if best else None


def eligible(f: CatalogORM, config: FactoryConfig) -> bool:
    """Coverage floor, signal-bearing type, not a technical field."""
    return (
        f.field_type in FIELD_TYPES
        and f.coverage >= config.search.min_field_coverage
        and not TECHNICAL.search(f"{f.field_id.replace('_', ' ')} {f.description}")
    )


def dataset_fields(s: Session, d: CatalogORM) -> list[CatalogORM]:
    return list(
        s.scalars(
            select(CatalogORM)
            .where(
                CatalogORM.kind == "field",
                CatalogORM.region == d.region,
                CatalogORM.delay == d.delay,
                CatalogORM.universe == d.universe,
                CatalogORM.dataset_id == d.dataset_id,
            )
            .order_by(CatalogORM.field_id)
        )
    )


def partner_datasets(s: Session, d: CatalogORM, rng: random.Random) -> list[CatalogORM]:
    """Other datasets of the same region, delay and universe, in random order (the second
    dataset of a multi-dataset hypothesis)."""
    rows = list(
        s.scalars(
            select(CatalogORM).where(
                CatalogORM.kind == "dataset",
                CatalogORM.region == d.region,
                CatalogORM.delay == d.delay,
                CatalogORM.universe == d.universe,
                CatalogORM.dataset_id != d.dataset_id,
            )
        )
    )
    rng.shuffle(rows)
    return rows


def partner_fields(
    s: Session, d: CatalogORM, config: FactoryConfig, exclude: set[str], k: int = 3
) -> list[CatalogORM]:
    """Up to k eligible fields of a dataset, one per family, least used first."""
    best: dict[str, CatalogORM] = {}
    for f in dataset_fields(s, d):
        if f.field_type not in FIELD_TYPES or f.field_id in exclude or not eligible(f, config):
            continue
        fam = family_key(f.field_id)
        if fam not in best or (f.alpha_count, -f.coverage) < (
            best[fam].alpha_count,
            -best[fam].coverage,
        ):
            best[fam] = f
    return sorted(best.values(), key=lambda f: (f.alpha_count, -f.coverage, f.field_id))[:k]


def choose_field(
    s: Session,
    d: CatalogORM,
    config: FactoryConfig,
    rng: random.Random,
    exclude: set[str] | None = None,
) -> Pick | None:
    """A field of an unused family, drawn among the least-used eligible ones. Fields in
    `exclude` (the crowded list) are never drawn or offered as helpers."""
    exclude = exclude or set()
    fields = [f for f in dataset_fields(s, d) if f.field_id not in exclude]
    used = {
        family_key(h.field_id)
        for h in s.scalars(
            select(HypothesisORM).where(
                HypothesisORM.region == d.region, HypothesisORM.dataset_id == d.dataset_id
            )
        )
    }
    by_family: dict[str, list[CatalogORM]] = {}
    for f in fields:
        by_family.setdefault(family_key(f.field_id), []).append(f)
    candidates = []
    for fam, members in by_family.items():
        if fam in used:
            continue
        ok = [f for f in members if eligible(f, config)]
        if ok:
            candidates.append(min(ok, key=lambda f: (f.alpha_count, -f.coverage, f.field_id)))
    if not candidates:
        return None
    candidates.sort(key=lambda f: (f.alpha_count, -f.coverage, f.field_id))
    field = rng.choice(candidates[:UNTAPPED_POOL])
    fam = family_key(field.field_id)
    helpers = [
        members[0]
        for key, members in sorted(by_family.items())
        if key != fam and members[0].field_type in (*FIELD_TYPES, "GROUP")
    ][:HELPERS_SHOWN]
    return Pick(dataset=d, field=field, family=by_family[fam], helpers=helpers)


def field_view(f: CatalogORM) -> dict[str, Any]:
    """What the LLM sees of a field."""
    return {
        "id": f.field_id,
        "type": f.field_type,
        "description": f.description[:300],
        "coverage": round(f.coverage, 2),
    }
