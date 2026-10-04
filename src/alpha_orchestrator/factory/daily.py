"""One day of research, start to finish. Safe to re-run: everything resumes from the ledger."""

import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.factory import catalog
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CandidateStatus,
    CellORM,
    CellState,
    EventORM,
    HypothesisORM,
    HypothesisStatus,
)
from alpha_orchestrator.factory.pipeline import DEFAULT_CELL_BUDGET, Factory, Metadata
from alpha_orchestrator.factory.scheduler import QUOTA_RESERVE, Simulator
from alpha_orchestrator.factory.scoring import plan_day
from alpha_orchestrator.factory.select import select_today

CHUNK = 200  # simulations between evaluation passes


@dataclass
class DayReport:
    planned: list[dict[str, Any]] = field(default_factory=list)
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


def run_day(
    config: FactoryConfig,
    api: BrainAPI,
    ledger: Ledger,
    llm: FactoryLLM | None,
    *,
    minutes: float = 600.0,
    rng: random.Random | None = None,
    meta: Metadata | None = None,
    sleep: Any = time.sleep,
) -> DayReport:
    report = DayReport()
    deadline = time.monotonic() + minutes * 60
    meta = meta or Metadata.fetch(api)
    if catalog_is_stale(ledger):
        catalog.refresh(api, ledger, config)

    with ledger.session() as s:
        used = ledger.sims_used_today(s)
        stats = catalog.cell_stats(s)
    budget = config.budget.daily_sim_cap - used
    quota = api.client.quota.remaining
    if quota is not None:
        budget = min(budget, quota - QUOTA_RESERVE)
    if budget <= 0:
        report.stopped_because = "daily simulation budget already used"
        return report

    factory = Factory(api, ledger, config, meta, llm, rng=rng)
    plan = plan_day(stats, config, budget, rng=rng)
    factory.cell_budget = {
        a.cell_id: a.budget + (DEFAULT_CELL_BUDGET if a.mode == "explore" else 0) for a in plan
    }
    for a in plan:
        report.planned.append(
            {"cell": a.cell_id, "mode": a.mode, "budget": a.budget, "ev": round(a.score, 4)}
        )
        with ledger.session() as s:
            cell = s.get(CellORM, a.cell_id)
            state = cell.state if cell else None
            queued = int(
                s.scalar(
                    select(func.count())
                    .select_from(CandidateORM)
                    .where(
                        CandidateORM.cell_id == a.cell_id,
                        CandidateORM.status == CandidateStatus.QUEUED,
                    )
                )
                or 0
            )
            has_hyp = (
                s.scalar(
                    select(HypothesisORM.id)
                    .where(
                        HypothesisORM.cell_id == a.cell_id,
                        HypothesisORM.status.in_(
                            [HypothesisStatus.PROPOSED, HypothesisStatus.APPROVED]
                        ),
                    )
                    .limit(1)
                )
                is not None
            )
        priority = 100 * max(a.score, 1e-4)
        if a.mode == "explore":
            if state == CellState.NEW:
                factory.profile(a.cell_id)
            factory.queue_probes(a.cell_id, max(a.budget - queued, 0), priority)
        elif llm is not None:
            if not has_hyp:
                factory.propose_hypotheses(a.cell_id)
            factory.queue_llm_expressions(a.cell_id, priority)
        with ledger.session() as s:
            ledger.log(s, "planned", a.cell_id, mode=a.mode, budget=a.budget, ev=a.score)

    simulator = Simulator(api, ledger, config, sleep=sleep)
    spent = 0
    while spent < budget and time.monotonic() < deadline:
        run = simulator.run(min(CHUNK, budget - spent), deadline=deadline)
        spent += run.completed + run.failed
        report.simulations += run.completed
        report.failed += run.failed
        counts = factory.evaluate()
        report.ready += counts["ready"]
        report.stopped_because = run.stopped_because
        if run.batches == 0 and counts["queued"] == 0:
            break
    factory.evaluate()
    report.reviews = select_today(api, ledger, config, llm)
    with ledger.session() as s:
        ledger.log(
            s,
            "day_done",
            simulations=report.simulations,
            failed=report.failed,
            ready=report.ready,
            reviews=report.reviews,
            reason=report.stopped_because,
            quota=api.client.quota.remaining,
        )
    return report
