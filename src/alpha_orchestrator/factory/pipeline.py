"""The research loop: profile -> probe -> (kill | signal) -> hypotheses -> expressions ->
orders 2-3, flips, repairs -> robustness -> ready for selection.

Producers here only write QUEUED candidates; the Simulator consumes them. `evaluate()`
reads finished simulations and decides what each result earns next.
"""

import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.domain.models import utc_now
from alpha_orchestrator.factory import generators as gen
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ideas import IdeaIndex
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM, FieldInfo
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CandidateStatus,
    CellORM,
    CellState,
    FieldORM,
    HypothesisORM,
    HypothesisStatus,
    Stage,
)
from alpha_orchestrator.factory.validate import check

STAGE_WEIGHT = {
    Stage.ROBUST: 3.0,
    Stage.REPAIR: 2.0,
    Stage.THIRD: 2.0,
    Stage.FLIP: 2.2,
    Stage.SECOND: 1.8,
    Stage.LLM: 1.6,
    Stage.FIRST: 1.2,
    Stage.PROBE: 1.0,
}
MAX_DEPTH = 5
KEEP_PER_FIELD = 3  # expansions per (dataset, field set, stage)
DEFAULT_CELL_BUDGET = 100  # expansions/day for a dataset the planner did not pick today
EXPANDABLE = {Stage.PROBE, Stage.FIRST, Stage.LLM, Stage.FLIP, Stage.REPAIR}
GROUP_WRAP = re.compile(r"\bgroup_(?:rank|zscore|neutralize)\(")


@dataclass
class Metadata:
    """Account-level facts refreshed once per run."""

    operators: set[str] = field(default_factory=set)
    definitions: dict[str, str] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fetch(cls, api: BrainAPI) -> "Metadata":
        ops = [o for o in api.operators() if "REGULAR" in (o.get("scope") or ["REGULAR"])]
        return cls(
            operators={str(o["name"]).lower() for o in ops},
            definitions={str(o["name"]): str(o.get("definition") or o["name"]) for o in ops},
            options=api.simulation_options(),
        )

    def neutralizations(self, region: str) -> list[str]:
        return list(self.options.get(region, {}).get("neutralizations", []))

    def universes(self, region: str) -> list[str]:
        return list(self.options.get(region, {}).get("universes", []))


class Factory:
    def __init__(
        self,
        api: BrainAPI,
        ledger: Ledger,
        config: FactoryConfig,
        meta: Metadata,
        llm: FactoryLLM | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.api = api
        self.ledger = ledger
        self.config = config
        self.meta = meta
        self.llm = llm
        self.rng = rng or random.Random()
        self._ideas: IdeaIndex | None = None
        self.cell_budget: dict[str, int] = {}  # set by the daily planner

    # ================= producers =================
    def profile(self, cell_id: str) -> int:
        with self.ledger.session() as s:
            cell = _cell(s, cell_id)
            region, delay, universe, dataset = (
                cell.region,
                cell.delay,
                cell.universe,
                cell.dataset_id,
            )
        rows = [
            r
            for r in self.api.datafields(region, delay, universe, dataset)
            if r.get("type") in ("MATRIX", "VECTOR")
        ]
        families = gen.group_families(rows)
        with self.ledger.session() as s:
            cell = _cell(s, cell_id)
            for family, members in families.items():
                for i, r in enumerate(members):
                    s.add(
                        FieldORM(
                            cell_id=cell_id,
                            field_id=str(r["id"]),
                            type=str(r["type"]),
                            description=str(r.get("description") or "")[:1000],
                            coverage=float(r.get("coverage") or 0),
                            alpha_count=int(r.get("alphaCount") or 0),
                            family=family,
                            representative=(i == 0),
                        )
                    )
            cell.state = CellState.PROFILED if rows else CellState.KILLED
            self.ledger.log(s, "profiled", cell_id, fields=len(rows), families=len(families))
        return len(rows)

    def queue_probes(self, cell_id: str, budget: int, priority: float) -> int:
        with self.ledger.session() as s:
            cell = _cell(s, cell_id)
            reps = list(
                s.scalars(
                    select(FieldORM).where(
                        FieldORM.cell_id == cell_id, FieldORM.representative.is_(True)
                    )
                )
            )
            fields = self._field_types(s, cell_id)
            settings = self._settings(cell)
            proposals = [
                p
                for f in reps
                for p in gen.probe_proposals(f.field_id, f.type, cell.category, settings)
            ]
            self.rng.shuffle(proposals)  # 官方: shuffle, then sample to judge the dataset fast
            added = self._enqueue(s, cell, proposals, fields, priority, limit=budget)
            cell.state = CellState.PROBING
            self.ledger.log(s, "probes_queued", cell_id, count=added)
        return added

    def propose_hypotheses(self, cell_id: str, n: int = 4) -> int:
        if self.llm is None:
            return 0
        with self.ledger.session() as s:
            cell = _cell(s, cell_id)
            fields = self._field_infos(s, cell_id, limit=40)
            query = " ".join(
                [cell.dataset_name, cell.dataset_id, cell.category, cell.description]
                + [f.description for f in fields[:15]]
            )
            idea_rows = self._idea_index(s).search(query, 6)
            ideas = [{"title": i.title, "excerpt": i.body[:1200]} for i in idea_rows]
            cell_info = {
                "id": cell.dataset_id,
                "name": cell.dataset_name,
                "category": cell.category,
                "region": cell.region,
                "description": cell.description[:1500],
            }
        proposals = self.llm.propose_hypotheses(cell_info, fields, ideas, n=n)
        status = (
            HypothesisStatus.PROPOSED
            if self.config.policy.hypotheses_need_approval
            else HypothesisStatus.APPROVED
        )
        with self.ledger.session() as s:
            for h in proposals:
                s.add(
                    HypothesisORM(
                        cell_id=cell_id,
                        title=str(h["title"])[:255],
                        statement=str(h["statement"]),
                        direction=str(h.get("direction", "")),
                        fields=h["fields"],
                        sources=h["sources"],
                        status=status,
                    )
                )
            self.ledger.log(s, "hypotheses_proposed", cell_id, count=len(proposals))
        return len(proposals)

    def queue_llm_expressions(self, cell_id: str, priority: float, n: int = 6) -> int:
        if self.llm is None:
            return 0
        added = 0
        with self.ledger.session() as s:
            hyps = list(
                s.scalars(
                    select(HypothesisORM).where(
                        HypothesisORM.cell_id == cell_id,
                        HypothesisORM.status == HypothesisStatus.APPROVED,
                    )
                )
            )
            jobs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
            for h in hyps:
                history = [
                    {
                        "expression": c.expression,
                        "sharpe": c.sharpe,
                        "fitness": c.fitness,
                        "turnover": c.turnover,
                        "fails": c.fails,
                        "error": c.error,
                    }
                    for c in s.scalars(
                        select(CandidateORM).where(CandidateORM.hypothesis_id == h.id)
                    )
                ][-12:]
                jobs.append(
                    (
                        {
                            "id": h.id,
                            "title": h.title,
                            "statement": h.statement,
                            "direction": h.direction,
                            "fields": h.fields,
                        },
                        history,
                    )
                )
            infos = self._field_infos(s, cell_id, limit=60)
        ops = sorted(f"{k}: {v}" for k, v in self.meta.definitions.items())
        for hyp, history in jobs:
            wanted = set(hyp["fields"])
            fields = [f for f in infos if f.id in wanted] or infos[:20]
            exprs = self.llm.write_expressions(hyp, fields, ops, history, n=n)
            with self.ledger.session() as s:
                cell = _cell(s, cell_id)
                ftypes = self._field_types(s, cell_id)
                settings = self._settings(cell)
                proposals = []
                for e in exprs:
                    ok, why = self.llm.aligned(hyp, e["expression"])
                    if ok:
                        proposals.append(
                            gen.Proposal(
                                e["expression"],
                                settings,
                                Stage.LLM,
                                role="llm",
                                change=e.get("rationale", ""),
                            )
                        )
                    else:
                        self.ledger.log(
                            s, "misaligned", cell_id, expression=e["expression"], reason=why
                        )
                added += self._enqueue(
                    s, cell, proposals, ftypes, priority, hypothesis_id=hyp["id"]
                )
                record = s.get(HypothesisORM, hyp["id"])
                assert record is not None
                record.rounds += 1
                if (
                    record.rounds >= self.config.policy.hypothesis_rounds_before_retire
                    and record.best_sharpe < 1.0
                ):
                    record.status = HypothesisStatus.RETIRED
        return added

    # ================= evaluation =================
    def evaluate(self) -> dict[str, int]:
        """Turn every new result into its next step. Idempotent per candidate."""
        counts = {"evaluated": 0, "queued": 0, "ready": 0}
        t = self.config.thresholds
        with self.ledger.session() as s:
            fresh = list(
                s.scalars(
                    select(CandidateORM).where(
                        CandidateORM.status.in_([CandidateStatus.DONE, CandidateStatus.FAILED]),
                        CandidateORM.evaluated.is_(False),
                    )
                )
            )
            for c in fresh:
                c.evaluated = True
                counts["evaluated"] += 1
                cell = _cell(s, c.cell_id)
                cell.sims_used += 1
                cell.updated_at = utc_now()
                if c.status == CandidateStatus.FAILED or c.sharpe is None:
                    if c.stage == Stage.PROBE:
                        cell.probes += 1
                    continue
                sharpe = c.sharpe
                cell.best_sharpe = max(cell.best_sharpe, abs(sharpe))
                if c.stage == Stage.PROBE:
                    cell.probes += 1
                    cell.hits += abs(sharpe) >= t.probe_hit_sharpe
                if c.hypothesis_id:
                    h = s.get(HypothesisORM, c.hypothesis_id)
                    if h:
                        h.best_sharpe = max(h.best_sharpe, sharpe)
                if c.stage == Stage.ROBUST:
                    continue
                counts["queued"] += self._next_steps(s, c, cell)
                if self._passes_is(c, cell.region):
                    c.extra = {**c.extra, "ready": True}
                    counts["ready"] += 1
            counts["queued"] += self._schedule_robustness(s)
            self._settle_cells(s)
            self._settle_robustness(s)
        return counts

    def _next_steps(self, s: Session, c: CandidateORM, cell: CellORM) -> int:
        t = self.config.thresholds
        depth = int(c.extra.get("depth", 0))
        if depth >= MAX_DEPTH:
            return 0
        sharpe, fitness = c.sharpe or 0.0, c.fitness or 0.0
        ftypes = self._field_types(s, cell.cell_id)
        base_priority = c.priority
        out: list[gen.Proposal] = []
        if (
            sharpe <= t.flip_sharpe
            and fitness <= t.flip_fitness
            and cell.region != "CHN"
            and c.stage != Stage.FLIP
        ):
            p = gen.flip(c.expression, c.settings, sharpe)
            if p:
                out.append(p)
        if sharpe >= t.expand_sharpe and fitness >= t.expand_fitness and self._may_expand(s, c):
            settings = {
                **c.settings,
                "decay": max(int(c.settings.get("decay", 0)), gen.decay_for_turnover(c.turnover)),
            }
            has_group = bool(GROUP_WRAP.search(c.expression))
            if not has_group and c.stage in EXPANDABLE:
                out += gen.second_order(c.expression, settings)
            elif has_group and "trade_when(" not in c.expression:
                out += gen.third_order(c.expression, settings)
            c.extra = {**c.extra, "expanded": True}
        if sharpe >= 1.0 and c.fails and c.stage != Stage.REPAIR:
            out += gen.repair_proposals(
                c.expression,
                c.settings,
                c.fails.split(","),
                c.turnover,
                self.meta.neutralizations(cell.region),
                may_finetune=sharpe >= t.finetune_sharpe and fitness >= t.finetune_fitness,
            )
        if c.stage == Stage.PROBE and sharpe >= t.probe_hit_sharpe:
            out += self._siblings(s, c, cell)
        return self._enqueue(
            s, cell, out, ftypes, base_priority, parent=c, hypothesis_id=c.hypothesis_id
        )

    def _may_expand(self, s: Session, c: CandidateORM) -> bool:
        """官方剪枝: keep a few representatives per field set and stage, not every variant."""
        fields = c.extra.get("fields") or []
        expanded = 0
        for other in s.scalars(
            select(CandidateORM).where(
                CandidateORM.cell_id == c.cell_id,
                CandidateORM.stage == c.stage,
                CandidateORM.id != c.id,
            )
        ):
            if other.extra.get("expanded") and (other.extra.get("fields") or []) == fields:
                expanded += 1
        return expanded < KEEP_PER_FIELD

    def _siblings(self, s: Session, c: CandidateORM, cell: CellORM) -> list[gen.Proposal]:
        """A probe hit on a family representative: try the same template on its family."""
        rep = next(
            (
                f
                for f in s.scalars(
                    select(FieldORM).where(
                        FieldORM.cell_id == cell.cell_id, FieldORM.representative.is_(True)
                    )
                )
                if f.field_id in c.expression
            ),
            None,
        )
        if rep is None:
            return []
        members = s.scalars(
            select(FieldORM)
            .where(
                FieldORM.cell_id == cell.cell_id,
                FieldORM.family == rep.family,
                FieldORM.representative.is_(False),
            )
            .limit(5)
        )
        return [
            gen.Proposal(
                c.expression.replace(rep.field_id, m.field_id),
                c.settings,
                Stage.FIRST,
                role=c.role,
                change=f"family sibling of {rep.field_id}",
            )
            for m in members
        ]

    def _passes_is(self, c: CandidateORM, region: str) -> bool:
        t = self.config.thresholds
        if c.fails or c.sharpe is None or c.fitness is None or c.turnover is None:
            return False
        margin_bps = (c.margin or 0.0) * 10_000
        return (
            c.sharpe >= self.config.sharpe_floor(region)
            and c.fitness >= t.submit_fitness
            and t.turnover_min <= c.turnover <= t.turnover_max
            and margin_bps >= self.config.margin_floor(region)
        )

    def _schedule_robustness(self, s: Session) -> int:
        """Stress-test only alphas that could be chosen: per dataset the best few by fitness
        with distinct field sets, and at most twice the daily submission slots per day."""
        start = datetime.combine(datetime.now(UTC).date(), datetime.min.time(), tzinfo=UTC)
        tested_today = {
            c.parent_id
            for c in s.scalars(
                select(CandidateORM).where(
                    CandidateORM.stage == Stage.ROBUST, CandidateORM.created_at >= start
                )
            )
        }
        room = 2 * self.config.policy.max_submissions_per_day - len(tested_today)
        if room <= 0:
            return 0
        waiting = [
            c
            for c in s.scalars(select(CandidateORM).where(CandidateORM.status == "DONE"))
            if c.extra.get("ready") and not c.extra.get("robust_queued")
        ]
        waiting.sort(key=lambda c: -(c.fitness or 0))
        taken: dict[str, set[tuple[str, ...]]] = {}
        queued = 0
        for c in waiting:
            if room <= 0:
                break
            signature = tuple(c.extra.get("fields") or [])
            seen = taken.setdefault(c.cell_id, set())
            if signature in seen or len(seen) >= 2:
                continue
            seen.add(signature)
            cell = _cell(s, c.cell_id)
            queued += self._queue_robustness(s, c, cell)
            c.extra = {**c.extra, "robust_queued": True}
            room -= 1
        return queued

    def _queue_robustness(self, s: Session, c: CandidateORM, cell: CellORM) -> int:
        proposals = gen.robustness_proposals(
            c.expression,
            c.settings,
            self.meta.neutralizations(cell.region),
            self.meta.universes(cell.region),
        )
        for p in proposals:
            p.extra["robust_of"] = c.id
        return self._enqueue(
            s,
            cell,
            proposals,
            self._field_types(s, cell.cell_id),
            c.priority,
            parent=c,
            hypothesis_id=c.hypothesis_id,
        )

    def _settle_robustness(self, s: Session) -> None:
        t = self.config.thresholds
        ready = [
            c
            for c in s.scalars(select(CandidateORM).where(CandidateORM.status == "DONE"))
            if c.extra.get("ready") and "robust" not in c.extra
        ]
        for parent in ready:
            kids = list(
                s.scalars(
                    select(CandidateORM).where(
                        CandidateORM.parent_id == parent.id, CandidateORM.stage == Stage.ROBUST
                    )
                )
            )
            if not kids or any(
                k.status in (CandidateStatus.QUEUED, CandidateStatus.RUNNING) for k in kids
            ):
                continue
            base = parent.sharpe or 0.0
            results: dict[str, float] = {}
            for k in kids:
                ratio = (k.sharpe or 0.0) / base if base else 0.0
                results[k.role] = min(results.get(k.role, 9.9), ratio)
            verdict = all(r >= t.robust_keep_ratio for r in results.values())
            parent.extra = {**parent.extra, "robust": {"pass": verdict, "ratios": results}}
            self.ledger.log(
                s, "robustness", parent.cell_id, candidate=parent.id, passed=verdict, ratios=results
            )

    def _settle_cells(self, s: Session) -> None:
        budget = self.config.budget.probe_budget
        for cell in s.scalars(select(CellORM).where(CellORM.state == CellState.PROBING)):
            pending = s.scalar(
                select(CandidateORM.id)
                .where(
                    CandidateORM.cell_id == cell.cell_id,
                    CandidateORM.stage == Stage.PROBE,
                    CandidateORM.status.in_([CandidateStatus.QUEUED, CandidateStatus.RUNNING]),
                )
                .limit(1)
            )
            if cell.hits > 0 and cell.probes >= 10:
                cell.state = CellState.SIGNAL
            elif cell.probes >= budget or (pending is None and cell.probes > 0):
                cell.state = CellState.SIGNAL if cell.hits else CellState.KILLED
            if cell.state == CellState.KILLED:
                for q in s.scalars(
                    select(CandidateORM).where(
                        CandidateORM.cell_id == cell.cell_id,
                        CandidateORM.status == CandidateStatus.QUEUED,
                    )
                ):
                    q.status = CandidateStatus.INVALID
                    q.error = "dataset stopped: no signal within the probe budget"
            if cell.state != CellState.PROBING:
                self.ledger.log(
                    s,
                    "cell_" + cell.state.lower(),
                    cell.cell_id,
                    probes=cell.probes,
                    hits=cell.hits,
                )

        for cell in s.scalars(
            select(CellORM).where(
                CellORM.submitted >= self.config.thresholds.dataset_cap,
                CellORM.state != CellState.SATURATED,
            )
        ):
            cell.state = CellState.SATURATED

    # ================= helpers =================
    def _enqueue(
        self,
        s: Session,
        cell: CellORM,
        proposals: Iterable[gen.Proposal],
        fields: dict[str, str],
        priority: float,
        *,
        limit: int | None = None,
        parent: CandidateORM | None = None,
        hypothesis_id: str | None = None,
    ) -> int:
        added = 0
        depth = int(parent.extra.get("depth", 0)) + 1 if parent else 0
        allowance = self.cell_budget.get(cell.cell_id, DEFAULT_CELL_BUDGET) - self._created_today(
            s, cell.cell_id
        )
        for p in proposals:
            if limit is not None and added >= limit:
                break
            if p.stage not in (Stage.ROBUST, Stage.FLIP) and allowance <= 0:
                self.ledger.log(s, "cell_budget_reached", cell.cell_id, stage=p.stage)
                break
            verdict = check(p.expression, operators=self.meta.operators, fields=fields)
            if not verdict.ok:
                self.ledger.log(
                    s, "invalid", cell.cell_id, expression=p.expression, issues=verdict.issues[:3]
                )
                continue
            row = self.ledger.add_candidate(
                s,
                cell_id=cell.cell_id,
                hypothesis_id=hypothesis_id,
                parent_id=parent.id if parent else None,
                stage=p.stage,
                role=p.role,
                change=p.change,
                expression=p.expression,
                settings=p.settings,
                priority=priority * STAGE_WEIGHT.get(p.stage, 1.0),
                extra={
                    **p.extra,
                    "depth": depth,
                    "atom": verdict.atom,
                    "operators": verdict.operator_count,
                    "fields": sorted(verdict.fields_used),
                },
            )
            if row is not None:
                added += 1
                if p.stage not in (Stage.ROBUST, Stage.FLIP):
                    allowance -= 1
        return added

    @staticmethod
    def _created_today(s: Session, cell_id: str) -> int:
        start = datetime.combine(datetime.now(UTC).date(), datetime.min.time(), tzinfo=UTC)
        return int(
            s.scalar(
                select(func.count())
                .select_from(CandidateORM)
                .where(
                    CandidateORM.cell_id == cell_id,
                    CandidateORM.created_at >= start,
                    CandidateORM.stage.not_in([Stage.ROBUST, Stage.FLIP]),
                )
            )
            or 0
        )

    def _settings(self, cell: CellORM) -> dict[str, Any]:
        return gen.base_settings(
            cell.region,
            cell.universe,
            cell.delay,
            cell.category.lower(),
            self.meta.neutralizations(cell.region),
        )

    @staticmethod
    def _field_types(s: Session, cell_id: str) -> dict[str, str]:
        return {
            f.field_id: f.type
            for f in s.scalars(select(FieldORM).where(FieldORM.cell_id == cell_id))
        }

    @staticmethod
    def _field_infos(s: Session, cell_id: str, limit: int) -> list[FieldInfo]:
        rows = s.scalars(
            select(FieldORM)
            .where(FieldORM.cell_id == cell_id)
            .order_by(FieldORM.representative.desc(), FieldORM.coverage.desc())
            .limit(limit)
        )
        return [FieldInfo(f.field_id, f.type, f.description) for f in rows]

    def _idea_index(self, s: Session) -> IdeaIndex:
        if self._ideas is None:
            self._ideas = IdeaIndex.load(s)
        return self._ideas


def _cell(s: Session, cell_id: str) -> CellORM:
    cell = s.get(CellORM, cell_id)
    if cell is None:
        raise KeyError(cell_id)
    return cell


def approve_hypotheses(ledger: Ledger, ids: Sequence[str], approve: bool = True) -> int:
    with ledger.session() as s:
        n = 0
        for hid in ids:
            h = s.get(HypothesisORM, hid)
            if h and h.status == HypothesisStatus.PROPOSED:
                h.status = HypothesisStatus.APPROVED if approve else HypothesisStatus.REJECTED
                h.decided_at = utc_now()
                n += 1
        return n
