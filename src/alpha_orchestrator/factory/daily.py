"""One day of research, start to finish. Safe to re-run: everything resumes from the ledger.

1. refresh the catalog (once a day); datasets the earlier logic stopped go back to the queue
2. keep `active_datasets` datasets open (one, by default); the next one is chosen by a
   Thompson draw over what is known about each dataset (datasets.py)
3. simulate in chunks; after each chunk record results and move the dataset one step
   (health -> survey -> research rounds), see research.py
4. pick at most four ready alphas for you to review; nothing is submitted here
"""

import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.factory import catalog, datasets
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import CellORM, CellState, EventORM
from alpha_orchestrator.factory.research import Factory, Metadata, reset_legacy
from alpha_orchestrator.factory.scheduler import QUOTA_RESERVE, Simulator
from alpha_orchestrator.factory.select import select_today

CHUNK = 100  # simulations between research steps (~30 minutes at 10 in flight)


@dataclass
class DayReport:
    started: list[str] = field(default_factory=list)
    simulations: int = 0
    failed: int = 0
    ready: int = 0
    reviews: list[str] = field(default_factory=list)
    stopped_because: str = ""


def catalog_is_stale(ledger: Ledger, hours: float = 20.0) -> bool:
    with ledger.session() as s:
        last = s.scalar(select(func.max(EventORM.at)).where(EventORM.kind == "catalog"))
    if last is None:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    return datetime.now(UTC) - last > timedelta(hours=hours)


OPEN = [CellState.STUDYING, CellState.SURVEY, CellState.RESEARCH]


def fill_active(
    factory: Factory, ledger: Ledger, config: FactoryConfig, rng: random.Random
) -> list[str]:
    """Opens the next datasets, by Thompson draw, until `active_datasets` are worked."""
    started = []
    with ledger.session() as s:
        ranked = datasets.rank(s, config, rng)
        open_now = int(
            s.scalar(select(func.count()).select_from(CellORM).where(CellORM.state.in_(OPEN))) or 0
        )
        states = {c.cell_id: c.state for c in s.scalars(select(CellORM))}
    for r in ranked:
        if open_now >= config.budget.active_datasets:
            break
        if states.get(r.cell_id) != CellState.NEW:
            continue
        factory.start(r.cell_id)
        started.append(r.cell_id)
        with ledger.session() as s:
            if s.get(CellORM, r.cell_id).state != CellState.EXHAUSTED:  # type: ignore[union-attr]
                open_now += 1
            ledger.log(s, "dataset_opened", r.cell_id, score=r.score, reason=r.reason)
    return started


def run_day(
    config: FactoryConfig,
    api: BrainAPI,
    ledger: Ledger,
    llm: FactoryLLM | None,
    *,
    minutes: float = 720.0,
    rng: random.Random | None = None,
    meta: Metadata | None = None,
    sleep: Any = time.sleep,
    on_progress: Any = None,
) -> DayReport:
    report = DayReport()
    deadline = time.monotonic() + minutes * 60
    meta = meta or Metadata.fetch(api)
    if catalog_is_stale(ledger):
        catalog.refresh(api, ledger, config)
    reset_legacy(ledger)
    rng = rng or random.Random()

    with ledger.session() as s:
        used = ledger.sims_used_today(s)
    budget = config.budget.daily_sim_cap - used
    quota = api.client.quota.remaining
    if quota is not None:
        budget = min(budget, quota - QUOTA_RESERVE)
    if budget <= 0:
        report.stopped_because = "daily simulation budget already used"
        return report

    factory = Factory(api, ledger, config, meta, llm, rng=rng)
    simulator = Simulator(api, ledger, config, sleep=sleep)
    spent = 0
    idle = 0
    while spent < budget and time.monotonic() < deadline:
        report.started += fill_active(factory, ledger, config, rng)
        moved = factory.advance()
        run = simulator.run(min(CHUNK, budget - spent), deadline=deadline)
        spent += run.completed + run.failed
        report.simulations += run.completed
        report.failed += run.failed
        report.ready += factory.evaluate()["ready"]
        if on_progress is not None:
            on_progress()
        report.stopped_because = run.stopped_because
        if run.stopped_because in ("quota", "deadline", "platform kept refusing batches"):
            break
        if run.batches == 0 and not any(moved.values()):
            idle += 1
            if idle >= 2:  # nothing to simulate and no dataset could move
                report.stopped_because = report.stopped_because or "nothing left to do"
                break
        else:
            idle = 0
    factory.evaluate()
    report.reviews = select_today(api, ledger, config, llm)
    with ledger.session() as s:
        ledger.log(
            s,
            "day_done",
            simulations=report.simulations,
            failed=report.failed,
            ready=report.ready,
            started=report.started,
            reviews=report.reviews,
            reason=report.stopped_because,
            quota=api.client.quota.remaining,
        )
    return report
