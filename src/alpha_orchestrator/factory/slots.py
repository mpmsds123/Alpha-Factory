"""Daily slots: which kind of alpha the next hypothesis works for.

AlphaForge splits each day's submissions into fixed slots, one per kind of alpha the
platform pays for, so a day does not end with four near-copies of one kind. Here a
hypothesis is given a slot when it opens; the slot sets its expression limits. A slot is
filled once an alpha of that kind is stored for review (or submitted) in the current
quota window. When every slot is filled and the day's target is met, no new hypothesis
opens.
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha_orchestrator.factory.config import FactoryConfig, Slot
from alpha_orchestrator.factory.models import HStatus, HypothesisORM, Review, SimulationORM


def slot_of(h: HypothesisORM) -> str:
    """The slot a hypothesis works for ('pool' for hypotheses opened before slots)."""
    return str((h.state or {}).get("slot") or "pool")


def stored_today(s: Session, config: FactoryConfig, now: datetime | None = None) -> dict[str, int]:
    """Alphas stored for review or submitted in this quota window, by slot."""
    start = config.budget.window_start(now)
    out = {r.name: 0 for r in config.slots.rules}
    rows = s.execute(
        select(SimulationORM, HypothesisORM)
        .join(HypothesisORM, HypothesisORM.id == SimulationORM.hypothesis_id)
        .where(
            SimulationORM.review.in_([Review.PENDING, Review.SUBMITTED]),
            SimulationORM.completed_at >= start,
        )
    )
    for _, h in rows:
        name = slot_of(h)
        out[name] = out.get(name, 0) + 1
    return out


def open_by_slot(s: Session) -> dict[str, int]:
    out: dict[str, int] = {}
    for h in s.scalars(select(HypothesisORM).where(HypothesisORM.status == HStatus.OPEN)):
        out[slot_of(h)] = out.get(slot_of(h), 0) + 1
    return out


def next_slot(
    s: Session,
    config: FactoryConfig,
    now: datetime | None = None,
    skip: frozenset[str] = frozenset(),
) -> Slot | None:
    """The slot the next hypothesis should work for; None when the day's target is met.

    Empty slots first; once each has an alpha, any slot until the daily target (the
    spare place). Among the candidates, the one with the fewest open hypotheses."""
    stored = stored_today(s, config, now)
    if sum(stored.values()) >= config.slots.per_day:
        return None
    rules = config.slots.rules
    usable = [r for r in rules if r.name not in skip]
    empty = [r for r in usable if stored.get(r.name, 0) == 0]
    candidates = empty or usable
    if not candidates:
        return None
    busy = open_by_slot(s)
    return min(candidates, key=lambda r: (busy.get(r.name, 0), rules.index(r)))


def limits(slot: Slot, config: FactoryConfig) -> tuple[int, int]:
    """(max operators, max fields) for expressions of this slot."""
    sc = config.search
    ops = slot.max_operators or sc.max_operators
    return min(ops, config.checks.hard_max_operators), slot.max_fields or sc.max_fields
