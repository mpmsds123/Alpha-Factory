"""The research loop for one dataset, from first look to stop.

    start     fetch fields, group near-duplicates into families, queue the 6-way probes
              on the representatives                                      (STUDYING)
    profile   read the probes' long/short counts -> per-field profile and `prep`
    ideas     LLM: profile + papers naming the dataset + forum/homework templates
              -> ideas (mechanism, direction, template, fields, windows, decay)
    round     each idea's template expanded over its fields and windows   (RESEARCH)
    review    LLM reads every idea's results -> dig / flip / fix / drop per idea, and
              continue / new ideas / exhausted for the dataset
    stop      exhausted when the LLM says so, after `rounds_without_ore_to_stop` rounds
              without any |sharpe| >= ore line, or when the dataset budget is spent

Simulations never rank datasets (datasets.py does that before any is spent). Producers
here only write QUEUED candidates; the Simulator consumes them.
"""

import random
import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.domain.models import utc_now
from alpha_orchestrator.factory import generators as gen
from alpha_orchestrator.factory import profile as prof
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.datasets import PAPER_SOURCES, PAPER_TITLE
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
    IdeaORM,
    Stage,
)
from alpha_orchestrator.factory.validate import check

STAGE_WEIGHT = {
    Stage.PROFILE: 3.0,  # finish reading the data first: the LLM is waiting on it
    Stage.ROBUST: 3.0,
    Stage.FLIP: 2.2,
    Stage.REPAIR: 2.0,
    Stage.DIG: 2.0,
    Stage.SECOND: 1.8,
    Stage.THIRD: 1.8,
    Stage.IDEA: 1.2,
}
PENDING = (CandidateStatus.QUEUED, CandidateStatus.RUNNING)
FIX_TO_FAIL = {
    "turnover": "HIGH_TURNOVER",
    "concentration": "CONCENTRATED_WEIGHT",
    "sub_universe": "LOW_SUB_UNIVERSE_SHARPE",
}
LLM_FIELDS = 80
SLOT_LABEL = re.compile(r"\b([A-Za-z]\w*)=([\w.]+)")


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
        llm: FactoryLLM,
        rng: random.Random | None = None,
    ) -> None:
        self.api = api
        self.ledger = ledger
        self.config = config
        self.meta = meta
        self.llm = llm
        self.rng = rng or random.Random()
        self._ideas: IdeaIndex | None = None

    # ================= step 1: start and read the data =================
    def start(self, cell_id: str) -> int:
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
            if not rows:
                self._stop(s, cell, "no MATRIX or VECTOR fields")
                return 0
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
            s.flush()
            # Probe the biggest families first: they carry most of the dataset.
            reps = sorted(
                families.items(),
                key=lambda kv: (-len(kv[1]), -float(kv[1][0].get("coverage") or 0)),
            )[: self.config.budget.profile_fields]
            settings = {**self._settings(cell), "neutralization": "NONE", "decay": 0}
            proposals = []
            for _, members in reps:
                fid, ftype = str(members[0]["id"]), str(members[0]["type"])
                for name, expr in prof.probe_expressions(fid, ftype):
                    proposals.append(
                        gen.Proposal(
                            expr,
                            settings,
                            Stage.PROFILE,
                            role=name,
                            change=f"6-way probe {name}",
                            extra={"field": fid, "probe": name},
                        )
                    )
            added = self._enqueue(s, cell, proposals)
            cell.state = CellState.STUDYING
            self.ledger.log(s, "dataset_started", cell_id, fields=len(rows), probes=added)
        return added

    def finish_profile(self, s: Session, cell: CellORM) -> None:
        counts: dict[str, dict[str, int | None]] = {}
        for c in s.scalars(
            select(CandidateORM).where(
                CandidateORM.cell_id == cell.cell_id, CandidateORM.stage == Stage.PROFILE
            )
        ):
            total = None
            if c.long_count is not None or c.short_count is not None:
                total = (c.long_count or 0) + (c.short_count or 0)
            counts.setdefault(str(c.extra.get("field")), {})[str(c.extra.get("probe"))] = total
        fields = list(s.scalars(select(FieldORM).where(FieldORM.cell_id == cell.cell_id)))
        by_family = {
            f.family: prof.interpret(counts[f.field_id], cell.universe)
            for f in fields
            if f.representative and f.field_id in counts
        }
        for f in fields:
            f.profile = by_family.get(f.family, {})
            f.prep = prof.prep_for(f.field_id, f.type, cell.category, f.profile)
        cell.state = CellState.RESEARCH
        self.ledger.log(
            s,
            "dataset_profiled",
            cell.cell_id,
            profiles={k: v for k, v in list(by_family.items())[:20]},
        )

    # ================= step 2: ideas =================
    def propose_ideas(self, cell_id: str) -> int:
        with self.ledger.session() as s:
            cell = _cell(s, cell_id)
            fields = self._field_infos(s, cell_id)
            references = self._references(s, cell, fields)
            tried = [
                {
                    "title": h.title,
                    "template": h.template,
                    "status": h.status,
                    "last_verdict": (h.verdicts or [{}])[-1],
                }
                for h in s.scalars(select(HypothesisORM).where(HypothesisORM.cell_id == cell_id))
            ]
            dataset = self._dataset_brief(cell)
        ops = sorted(f"{k}: {v}" for k, v in self.meta.definitions.items())
        ideas = self.llm.ideas(
            dataset, fields, references, ops, tried, n=self.config.budget.ideas_per_round
        )
        status = (
            HypothesisStatus.PROPOSED
            if self.config.policy.hypotheses_need_approval
            else HypothesisStatus.APPROVED
        )
        neutralizations = set(self.meta.neutralizations(dataset["region"]))
        with self.ledger.session() as s:
            for idea in ideas:
                settings: dict[str, Any] = {"decay": idea["decay"]}
                if idea["neutralization"] in neutralizations:
                    settings["neutralization"] = idea["neutralization"]
                s.add(
                    HypothesisORM(
                        cell_id=cell_id,
                        title=idea["title"],
                        statement=idea["mechanism"],
                        direction=idea["direction"],
                        fields=sorted({f for v in idea["slots"].values() for f in v}),
                        sources=idea["sources"],
                        template=idea["template"],
                        slots=idea["slots"],
                        windows=idea["windows"],
                        settings=settings,
                        status=status,
                    )
                )
            self.ledger.log(s, "ideas_proposed", cell_id, count=len(ideas))
        return len(ideas)

    def queue_idea(self, s: Session, h: HypothesisORM) -> int:
        cell = _cell(s, h.cell_id)
        prep = {
            f.field_id: f.prep
            for f in s.scalars(
                select(FieldORM).where(
                    FieldORM.cell_id == h.cell_id, FieldORM.field_id.in_(h.fields)
                )
            )
        }
        proposals = gen.expand_template(
            h.template,
            h.slots,
            h.windows,
            prep,
            {**self._settings(cell), **h.settings},
            self.config.budget.expansions_per_idea,
            self.rng,
        )
        return self._start_round(s, cell, h, proposals)

    # ================= step 3: review a finished round =================
    def review(self, cell_id: str) -> dict[str, Any]:
        t = self.config.thresholds
        with self.ledger.session() as s:
            cell = _cell(s, cell_id)
            hyps = [
                h
                for h in s.scalars(
                    select(HypothesisORM).where(
                        HypothesisORM.cell_id == cell_id,
                        HypothesisORM.status == HypothesisStatus.APPROVED,
                    )
                )
                if h.rounds > h.reviewed
            ]
            summaries = [self._summary(s, h) for h in hyps]
            fields = self._field_infos(s, cell_id)
            dataset = {
                **self._dataset_brief(cell),
                "rounds_done": cell.rounds,
                "rounds_without_ore": cell.rounds_without_ore,
                "ore_line": t.ore_sharpe,
                "simulations_left": self.config.budget.dataset_sim_cap - cell.sims_used,
            }
        if not summaries:
            return {}
        ops = sorted(f"{k}: {v}" for k, v in self.meta.definitions.items())
        result = self.llm.review(dataset, summaries, ops, fields, self.config.budget.dig_per_idea)
        verdicts = {v["id"]: v for v in result["ideas"]}
        ore = any(x["ore_count"] for x in summaries)

        with self.ledger.session() as s:
            cell = _cell(s, cell_id)
            cell.rounds += 1
            cell.rounds_without_ore = 0 if ore else cell.rounds_without_ore + 1
            queued = 0
            for summary in summaries:
                h = s.get(HypothesisORM, summary["id"])
                assert h is not None
                v = verdicts.get(h.id) or {"verdict": "drop", "reason": "no verdict given"}
                h.reviewed = h.rounds
                h.verdicts = [
                    *h.verdicts,
                    {"round": h.rounds, "verdict": v["verdict"], "reason": v["reason"]},
                ]
                added = self._apply(s, cell, h, v, summary)
                if added == 0 and h.status == HypothesisStatus.APPROVED:
                    h.status = HypothesisStatus.RETIRED
                queued += added
            self.ledger.log(
                s,
                "round_reviewed",
                cell_id,
                round=cell.rounds,
                ore=ore,
                verdicts={k: v["verdict"] for k, v in verdicts.items()},
                decision=result["dataset"],
                queued=queued,
            )
            decision = result["dataset"]["decision"]
            if cell.rounds_without_ore >= self.config.policy.rounds_without_ore_to_stop:
                self._stop(
                    s, cell, f"no |sharpe| >= {t.ore_sharpe} in {cell.rounds_without_ore} rounds"
                )
            elif decision == "exhausted":
                self._stop(s, cell, "LLM: " + result["dataset"]["reason"])
        if self._is_researching(cell_id) and (decision == "new_ideas" or queued == 0):
            self.propose_ideas(cell_id)
        return result

    def _apply(
        self,
        s: Session,
        cell: CellORM,
        h: HypothesisORM,
        v: dict[str, Any],
        summary: dict[str, Any],
    ) -> int:
        verdict = v["verdict"]
        settings = {**self._settings(cell), **h.settings}
        best = summary["best"]
        out: list[gen.Proposal] = []
        if verdict == "drop":
            h.status = HypothesisStatus.RETIRED
            return 0
        if verdict == "flip":
            for b in [b for b in best if b["sharpe"] < 0][:5]:
                p = gen.flip(b["expression"], b["settings"], b["sharpe"])
                if p:
                    out.append(p)
        elif verdict == "fix":
            fail = FIX_TO_FAIL.get(v.get("fix", ""), "")
            for b in [b for b in best if b["sharpe"] > 0][:3]:
                out += gen.repair_proposals(
                    b["expression"],
                    b["settings"],
                    [fail, *b["fails"].split(",")],
                    b["turnover"],
                    self.meta.neutralizations(cell.region),
                    may_finetune=False,
                )
        elif verdict == "dig":
            for e in v.get("expressions") or []:
                out.append(
                    gen.Proposal(
                        e["expression"], settings, Stage.DIG, role="dig", change=e.get("why", "")
                    )
                )
            top = next((b for b in best if b["sharpe"] > 0), None)
            if top and not gen.GROUP_WRAP.search(top["expression"]):
                out += [
                    p
                    for p in gen.second_order(top["expression"], top["settings"])
                    if p.role == "group_rank"
                ]
        return self._start_round(s, cell, h, out)

    def _summary(self, s: Session, h: HypothesisORM) -> dict[str, Any]:
        t = self.config.thresholds
        rows = [
            c
            for c in s.scalars(select(CandidateORM).where(CandidateORM.hypothesis_id == h.id))
            if c.extra.get("round") == h.rounds and c.stage != Stage.ROBUST
        ]
        done = [c for c in rows if c.sharpe is not None]
        sharpes = sorted(c.sharpe for c in done if c.sharpe is not None)
        by_field: dict[str, float] = {}
        for c in done:
            for _, fid in SLOT_LABEL.findall(c.change or ""):
                if abs(c.sharpe or 0) > abs(by_field.get(fid, 0.0)):
                    by_field[fid] = round(c.sharpe or 0.0, 2)
        best = sorted(done, key=lambda c: -abs(c.sharpe or 0))[:6]
        errors = [c.error for c in rows if c.status == CandidateStatus.FAILED and c.error]

        def q(p: float) -> float | None:
            if not sharpes:
                return None
            return round(sharpes[min(int(p * len(sharpes)), len(sharpes) - 1)], 2)

        return {
            "id": h.id,
            "title": h.title,
            "mechanism": h.statement,
            "direction": h.direction,
            "template": h.template,
            "round": h.rounds,
            "previous_verdicts": h.verdicts[-3:],
            "simulated": len(rows),
            "errors": len(rows) - len(done),
            "error_samples": errors[:3],
            "ore_count": sum(abs(x) >= t.ore_sharpe for x in sharpes),
            "positive_share": round(sum(x > 0 for x in sharpes) / len(sharpes), 2)
            if sharpes
            else None,
            "sharpe_p10": q(0.1),
            "sharpe_median": round(statistics.median(sharpes), 2) if sharpes else None,
            "sharpe_p90": q(0.9),
            "best_by_field": dict(sorted(by_field.items(), key=lambda kv: -abs(kv[1]))[:10]),
            "best": [
                {
                    "expression": c.expression,
                    "settings": c.settings,
                    "sharpe": round(c.sharpe or 0.0, 2),
                    "fitness": round(c.fitness or 0.0, 2),
                    "turnover": round(c.turnover or 0.0, 3),
                    "fails": c.fails,
                    "change": c.change,
                }
                for c in best
            ],
        }

    # ================= what to do next =================
    def advance(self) -> dict[str, int]:
        """Moves every open dataset one step if its current step is finished."""
        moved = {"profiled": 0, "queued": 0, "reviewed": 0, "ideas": 0}
        with self.ledger.session() as s:
            open_cells = [
                c.cell_id
                for c in s.scalars(
                    select(CellORM).where(
                        CellORM.state.in_([CellState.STUDYING, CellState.RESEARCH])
                    )
                )
            ]
        for cell_id in open_cells:
            with self.ledger.session() as s:
                cell = _cell(s, cell_id)
                if cell.sims_used >= self.config.budget.dataset_sim_cap:
                    self._stop(s, cell, "dataset simulation budget spent")
                    continue
                if self._pending(s, cell_id):
                    continue
                if cell.state == CellState.STUDYING:
                    self.finish_profile(s, cell)
                    moved["profiled"] += 1
                hyps = list(
                    s.scalars(
                        select(HypothesisORM).where(
                            HypothesisORM.cell_id == cell_id,
                            HypothesisORM.status.in_(
                                [HypothesisStatus.APPROVED, HypothesisStatus.PROPOSED]
                            ),
                        )
                    )
                )
                fresh = [h for h in hyps if h.status == HypothesisStatus.APPROVED and h.rounds == 0]
                for h in fresh:
                    added = self.queue_idea(s, h)
                    if added == 0:
                        h.status = HypothesisStatus.RETIRED
                        h.verdicts = [
                            {
                                "round": 0,
                                "verdict": "drop",
                                "reason": "nothing new: every expression invalid or already tested",
                            }
                        ]
                    moved["queued"] += added
                if fresh and not any(h.rounds for h in fresh):
                    # A whole batch of ideas that cannot be expressed counts as a dry round.
                    cell.rounds_without_ore += 1
                    if cell.rounds_without_ore >= self.config.policy.rounds_without_ore_to_stop:
                        self._stop(s, cell, "new ideas produced nothing untested")
                        continue
                fresh = [h for h in fresh if h.rounds]
                needs_review = any(
                    h.status == HypothesisStatus.APPROVED and h.rounds > h.reviewed for h in hyps
                )
                needs_ideas = not [h for h in hyps if h.status != HypothesisStatus.RETIRED]
            if fresh:
                continue
            if needs_review:
                self.review(cell_id)
                moved["reviewed"] += 1
            elif needs_ideas:
                proposed = self.propose_ideas(cell_id)
                moved["ideas"] += proposed
                if proposed == 0:
                    with self.ledger.session() as s:
                        self._stop(s, _cell(s, cell_id), "LLM produced no usable idea")
        return moved

    # ================= results =================
    def evaluate(self) -> dict[str, int]:
        """Records every new result. Idempotent per candidate."""
        counts = {"evaluated": 0, "ready": 0}
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
                if c.sharpe is None or c.stage == Stage.PROFILE:
                    continue
                sharpe = c.sharpe
                cell.best_sharpe = max(cell.best_sharpe, abs(sharpe))
                if c.stage != Stage.ROBUST:
                    cell.hits += abs(sharpe) >= t.ore_sharpe
                if c.hypothesis_id:
                    h = s.get(HypothesisORM, c.hypothesis_id)
                    if h:
                        h.best_sharpe = max(h.best_sharpe, sharpe)
                if c.stage != Stage.ROBUST and self._passes_is(c, cell.region):
                    c.extra = {**c.extra, "ready": True}
                    counts["ready"] += 1
            counts["robust_queued"] = self._schedule_robustness(s)
            self._settle_robustness(s)
            for cell in s.scalars(
                select(CellORM).where(
                    CellORM.submitted >= t.dataset_cap, CellORM.state != CellState.SATURATED
                )
            ):
                cell.state = CellState.SATURATED
                cell.stop_reason = f"{cell.submitted} submissions from this dataset"
        return counts

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
            proposals = gen.robustness_proposals(
                c.expression,
                c.settings,
                self.meta.neutralizations(cell.region),
                self.meta.universes(cell.region),
            )
            for p in proposals:
                p.extra["robust_of"] = c.id
            queued += self._enqueue(s, cell, proposals, parent=c, hypothesis_id=c.hypothesis_id)
            c.extra = {**c.extra, "robust_queued": True}
            room -= 1
        return queued

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
            if not kids or any(k.status in PENDING for k in kids):
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

    # ================= helpers =================
    def _start_round(
        self, s: Session, cell: CellORM, h: HypothesisORM, proposals: Sequence[gen.Proposal]
    ) -> int:
        for p in proposals:
            p.extra["round"] = h.rounds + 1
        added = self._enqueue(s, cell, proposals, hypothesis_id=h.id)
        if added:
            h.rounds += 1
        return added

    def _stop(self, s: Session, cell: CellORM, reason: str) -> None:
        cell.state = CellState.EXHAUSTED
        cell.stop_reason = reason
        for q in s.scalars(
            select(CandidateORM).where(
                CandidateORM.cell_id == cell.cell_id,
                CandidateORM.status == CandidateStatus.QUEUED,
                CandidateORM.stage != Stage.ROBUST,
            )
        ):
            q.status = CandidateStatus.INVALID
            q.error = f"dataset stopped: {reason}"
        self.ledger.log(s, "dataset_stopped", cell.cell_id, reason=reason, sims=cell.sims_used)

    def _is_researching(self, cell_id: str) -> bool:
        with self.ledger.session() as s:
            return _cell(s, cell_id).state == CellState.RESEARCH

    @staticmethod
    def _pending(s: Session, cell_id: str) -> bool:
        """Anything still to simulate for this dataset, robustness tests aside."""
        return (
            s.scalar(
                select(CandidateORM.id)
                .where(
                    CandidateORM.cell_id == cell_id,
                    CandidateORM.status.in_(PENDING),
                    CandidateORM.stage != Stage.ROBUST,
                )
                .limit(1)
            )
            is not None
        )

    def _enqueue(
        self,
        s: Session,
        cell: CellORM,
        proposals: Iterable[gen.Proposal],
        *,
        parent: CandidateORM | None = None,
        hypothesis_id: str | None = None,
    ) -> int:
        fields = {
            f.field_id: f.type
            for f in s.scalars(select(FieldORM).where(FieldORM.cell_id == cell.cell_id))
        }
        room = self.config.budget.dataset_sim_cap - cell.sims_used - self._queued(s, cell.cell_id)
        priority = 100 * max(cell.rank_score, 1e-3)
        added = 0
        for p in proposals:
            if p.stage != Stage.ROBUST and room <= 0:
                self.ledger.log(s, "dataset_budget_reached", cell.cell_id, stage=p.stage)
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
                    "atom": verdict.atom,
                    "operators": verdict.operator_count,
                    "fields": sorted(verdict.fields_used),
                },
            )
            if row is not None:
                added += 1
                if p.stage != Stage.ROBUST:
                    room -= 1
        return added

    @staticmethod
    def _queued(s: Session, cell_id: str) -> int:
        return int(
            s.scalar(
                select(func.count())
                .select_from(CandidateORM)
                .where(CandidateORM.cell_id == cell_id, CandidateORM.status.in_(PENDING))
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
    def _dataset_brief(cell: CellORM) -> dict[str, Any]:
        return {
            "id": cell.dataset_id,
            "name": cell.dataset_name,
            "category": cell.category,
            "region": cell.region,
            "universe": cell.universe,
            "delay": cell.delay,
            "description": cell.description[:1500],
        }

    @staticmethod
    def _field_infos(s: Session, cell_id: str) -> list[FieldInfo]:
        rows = s.scalars(
            select(FieldORM)
            .where(FieldORM.cell_id == cell_id)
            .order_by(FieldORM.representative.desc(), FieldORM.coverage.desc())
            .limit(LLM_FIELDS)
        )
        return [
            FieldInfo(f.field_id, f.type, f.description, prof.describe(f.profile), f.prep)
            for f in rows
        ]

    def _references(
        self, s: Session, cell: CellORM, fields: Sequence[FieldInfo]
    ) -> list[dict[str, str]]:
        """Papers naming this dataset first, then the closest forum posts and homework."""
        pattern = re.compile(rf"(?<![\w]){re.escape(cell.dataset_id)}(?![\w])", re.I)
        papers = [
            i
            for i in s.scalars(select(IdeaORM))
            if (i.source in PAPER_SOURCES or PAPER_TITLE.search(i.title))
            and pattern.search(f"{i.title}\n{i.body}")
        ][:3]
        if self._ideas is None:
            self._ideas = IdeaIndex.load(s)
        query = " ".join(
            [cell.dataset_name, cell.dataset_id, cell.category, cell.description]
            + [f.description for f in fields[:15]]
        )
        seen = {i.id for i in papers}
        near = [i for i in self._ideas.search(query, 9) if i.id not in seen][:6]
        return [
            {"kind": "paper naming this dataset", "title": i.title, "excerpt": i.body[:2000]}
            for i in papers
        ] + [{"kind": i.source, "title": i.title, "excerpt": i.body[:1200]} for i in near]


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
