"""设计文档 v1, steps 3-9, and one day of running them (第 5 节).

Every rule below names its step in the design document. Numbers marked 提议 there are the
module constants here; nothing else decides.
"""

import copy
import math
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.factory import catalog, evidence, study
from alpha_orchestrator.factory import generators as gen
from alpha_orchestrator.factory.cards import CHANGE_CARDS, IDEA_CARDS
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ideas import tokens
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CandidateStatus,
    CellORM,
    CellState,
    FieldORM,
    IdeaFieldORM,
    IdeaStatus,
    ResearchIdeaORM,
    ReviewORM,
    Stage,
)
from alpha_orchestrator.factory.research import Metadata
from alpha_orchestrator.factory.scheduler import QUOTA_RESERVE, Simulator
from alpha_orchestrator.factory.select import store_passing
from alpha_orchestrator.factory.validate import check

# 第 3 步
FIELD_LINES = 200  # field families shown to the LLM
MAX_FORM_OPERATORS = 4  # 第 4 问: 4 + one per round x 4 rounds = 8, the Power Pool limit
IDEA_FAILS_TO_STOP = 2  # 第 6 问: 2 failed ideas in a row (cost cap) -> the dataset is done
# 第 4 步
COARSE = 300
SCORE_BATCH = 100
KEEP_SCORE = 6
# 第 5 步 (2026-10-05 修改)
FORMULAS = 5  # per idea; a batch of 10 holds two ideas (scheduler mixes ideas in a batch)
FORMULA_FIELDS = 8  # distinct quantities the LLM may combine in its formulas
# 第 3 问 (10/14): no control arm; the template scanner is the control
# 第 4 问 (10/14): an idea whose best first formula is under the luck line of its tries
# (0.81 for 5 tries at noise 0.316) cannot be told from noise and stops there
GATE_CONFIDENCE = 0.95
# 第 1 问: an idea "hits" when one of its formulas passes the regular line (_passes_floor)
# 第 5 问 (10/14)
PARENTS = 2
VARIANTS = 3
DRY_ROUNDS = 2  # a parent stops after 2 rounds without a 0.35 gain
MAX_ROUNDS = 4  # 4 + 4 operators = 8, the Power Pool limit
SAME_FAIL_LIMIT = 3  # alpha-mining: 3 variants failing the same check -> change the formula
# 第 5 节
ACTIVE_IDEAS = 2  # 提议
CHUNK = 20  # simulations between decisions
NO_FLIP_REGIONS = frozenset({"CHN"})  # 第 6 步: short side is constrained in China
ROBUST_ROLES = frozenset({"neutralization"})  # 第 7 问: the platform checks sub-universe
PENDING = (CandidateStatus.QUEUED, CandidateStatus.RUNNING)
WORD = re.compile(r"[a-z0-9]+")


def luck_line(n: int, sigma: float) -> float:
    """The |Sharpe| the best of n pure-noise formulas exceeds with only 5% chance."""
    n = max(n, 1)
    p = 1 - GATE_CONFIDENCE ** (1 / n)
    return sigma * NormalDist().inv_cdf(1 - p / 2)


def first_gate(n: int, sigma: float) -> float:
    """第 4 问: 0.81 for 5 first formulas at noise 0.316."""
    return luck_line(n, sigma)


class Engine:
    def __init__(
        self,
        api: BrainAPI,
        ledger: Ledger,
        config: FactoryConfig,
        meta: Metadata,
        llm: FactoryLLM,
        rng: random.Random | None = None,
    ) -> None:
        self.api, self.ledger, self.config, self.meta, self.llm = api, ledger, config, meta, llm
        self.rng = rng or random.Random()

    # ======================= driving =======================
    def advance(self) -> dict[str, int]:
        """Moves every idea one step when its simulations are back; keeps ACTIVE_IDEAS
        ideas going on the current dataset; opens the next dataset when needed."""
        moved = {"ideas_moved": 0, "ideas_new": 0}
        cell_id = self._current_dataset() or self._open_dataset()
        if cell_id is None:
            return moved
        for idea_id in self._active_ideas(cell_id):
            moved["ideas_moved"] += self._step_idea(idea_id)
        attempts = 0
        while len(self._active_ideas(cell_id)) < ACTIVE_IDEAS and attempts < ACTIVE_IDEAS:
            attempts += 1
            if self._dataset_finished(cell_id):
                break
            if self._new_idea(cell_id):
                moved["ideas_new"] += 1
        if not self._active_ideas(cell_id) and self._dataset_finished(cell_id):
            self._close_dataset(cell_id)
        return moved

    def generate_one(self) -> str | None:
        """Create exactly one idea through formulas, without running a simulation."""
        cell_id = self._current_dataset() or self._open_dataset()
        if cell_id is None or self._dataset_finished(cell_id):
            return None
        return self._new_idea(cell_id)

    def _current_dataset(self) -> str | None:
        with self.ledger.session() as s:
            for c in s.scalars(select(CellORM).where(CellORM.state == CellState.RESEARCH)):
                if (c.plan or {}).get("design") == study.DESIGN:
                    return c.cell_id
        return None

    def _open_dataset(self) -> str | None:
        """第 1-2 步, repeated until a dataset has usable fields."""
        for _ in range(30):
            with self.ledger.session() as s:
                cell_id = study.choose_dataset(s, self.config)
            if cell_id is None:
                return None
            if study.load_fields(self.api, self.ledger, cell_id, self.config):
                with self.ledger.session() as s:
                    cell = s.get(CellORM, cell_id)
                    assert cell is not None
                    # 第 6 问: the 1000-simulation budget counts from here
                    cell.plan = {**(cell.plan or {}), "sims_at_open": cell.sims_used}
                    self.ledger.log(s, "dataset_opened", cell_id)
                return cell_id
        return None

    def _active_ideas(self, cell_id: str) -> list[str]:
        with self.ledger.session() as s:
            return list(
                s.scalars(
                    select(ResearchIdeaORM.id)
                    .where(
                        ResearchIdeaORM.cell_id == cell_id,
                        ResearchIdeaORM.status.in_([IdeaStatus.TESTING, IdeaStatus.PROMISING]),
                    )
                    .order_by(ResearchIdeaORM.created_at)
                )
            )

    def _dataset_finished(self, cell_id: str) -> bool:
        """第 6 问: the dataset's 1000 simulations are spent, or the LLM failed to give a
        valid new idea IDEA_FAILS_TO_STOP times in a row."""
        with self.ledger.session() as s:
            cell = s.get(CellORM, cell_id)
            if cell is None:
                return True
            plan = cell.plan or {}
            spent = cell.sims_used - int(plan.get("sims_at_open", 0))
            if spent >= self.config.budget.dataset_sim_cap:
                return True
            return int(plan.get("idea_fail_streak", 0)) >= IDEA_FAILS_TO_STOP

    def _close_dataset(self, cell_id: str) -> None:
        with self.ledger.session() as s:
            cell = s.get(CellORM, cell_id)
            if cell is None:
                return
            ideas = list(
                s.scalars(
                    select(ResearchIdeaORM).where(
                        ResearchIdeaORM.cell_id == cell_id,
                        ResearchIdeaORM.status != IdeaStatus.RETIRED,
                    )
                )
            )
            worked = sum(1 for i in ideas if i.status == IdeaStatus.DONE)
            outcome = "signal" if worked else "noise"
            plan = cell.plan or {}
            spent = cell.sims_used - int(plan.get("sims_at_open", 0))
            cell.state = CellState.EXHAUSTED
            cell.plan = {**plan, "verdict": outcome, "outcome": outcome, "sims_spent": spent}
            why = (
                "budget spent"
                if spent >= self.config.budget.dataset_sim_cap
                else "no valid new idea twice in a row"
            )
            cell.stop_reason = (
                f"{why}; {len(ideas)} ideas, {worked} reached the regular line, {spent} sims"
            )
            self.ledger.log(s, "dataset_closed", cell_id, outcome=outcome, ideas=len(ideas))

    # ======================= 第 3 步 想点子 =======================
    def _new_idea(self, cell_id: str) -> str | None:
        """第 3 步 + 第 2、6 问: ask for one idea with evidence; the program checks format,
        evidence and duplicates (and, for a paper, whether the quote supports the claim).
        A rejected answer is sent back with its reason once; two rejections in a row count
        as one failure for the dataset (第 6 问)."""
        from alpha_orchestrator.factory.models import LLMCallORM

        with self.ledger.session() as s:
            cell = s.get(CellORM, cell_id)
            assert cell is not None
            brief = {
                "id": cell.dataset_id,
                "name": cell.dataset_name,
                "category": cell.category,
                "region": cell.region,
                "universe": cell.universe,
                "description": cell.description[:1500],
            }
            fields = study.usable_fields(s, cell_id)
            lines = _family_lines(fields)
            meaning = " ".join(
                [cell.dataset_name, cell.description[:1500]]
                + [f["description"] for f in lines[:60]]
            )
            documents = evidence.retrieve(meaning, evidence.official_docs(s))
            earlier = [
                {
                    "title": i.title,
                    "query": i.query,
                    "evidence_key": str((i.extra or {}).get("evidence_key", "")),
                }
                for i in s.scalars(
                    select(ResearchIdeaORM).where(
                        ResearchIdeaORM.cell_id == cell_id,
                        ResearchIdeaORM.status != IdeaStatus.RETIRED,
                    )
                )
            ]
            dead_ends, worked = _knowledge(s)
        docs_by_id = {d["id"]: d for d in documents}
        self.llm.idea_id = None
        self.llm.unbilled = []
        card: dict[str, Any] | None = None
        rejected = ""
        reasons: list[str] = []
        for _ in range(2):
            try:
                answer = self.llm.idea(
                    brief,
                    lines,
                    IDEA_CARDS,
                    documents,
                    [e["title"] for e in earlier],
                    dead_ends,
                    worked,
                    rejected,
                )
            except RuntimeError as exc:
                reasons.append(str(exc)[:200])
                continue
            ok, rejected = evidence.check_idea(answer, docs_by_id)
            if ok:
                same = evidence.duplicate_of(answer, earlier)
                if same:
                    ok, rejected = False, f"it repeats the idea already tried here: {same}"
            ev = answer.get("evidence") or {}
            if ok and str(ev.get("type", "")).lower() == "paper":
                try:
                    supported, why = self.llm.supports(
                        str(answer.get("mechanism", "")), str(ev.get("quote", ""))
                    )
                except RuntimeError:
                    supported, why = False, "the support check failed"
                if not supported:
                    ok, rejected = False, f"the quote does not support the mechanism: {why}"
            if ok:
                card = answer
                break
            reasons.append(rejected)
        with self.ledger.session() as s:
            cell = s.get(CellORM, cell_id)
            assert cell is not None
            plan = dict(cell.plan or {})
            if card is None:
                plan["idea_fail_streak"] = int(plan.get("idea_fail_streak", 0)) + 1
                cell.plan = plan
                self.ledger.log(s, "idea_failed", cell_id, reasons=reasons)
                return None
            plan["idea_fail_streak"] = 0
            cell.plan = plan
            ev = card["evidence"]
            idea = ResearchIdeaORM(
                cell_id=cell_id,
                title=str(card.get("title", ""))[:300],
                mechanism=str(card.get("mechanism", "")),
                long_side=str(card.get("long", "")),
                short_side=str(card.get("short", "")),
                horizon_days=_int(card.get("horizon_days"), 21),
                query=str(card.get("query", "")),
                extra={
                    "evidence": {
                        "type": str(ev.get("type", "")).lower(),
                        "id": str(ev.get("id", "")).strip().upper()
                        if str(ev.get("type", "")).lower() == "card"
                        else str(ev.get("id", "")).strip(),
                        "quote": str(ev.get("quote", ""))[: evidence.QUOTE_MAX],
                        "title": docs_by_id.get(str(ev.get("id", "")).strip(), {}).get(
                            "title", IDEA_CARDS.get(str(ev.get("id", "")).strip().upper(), "")
                        ),
                    },
                    "evidence_key": evidence.evidence_key(card),
                    "rejected_before": reasons,
                    "n": len(earlier),
                },
            )
            s.add(idea)
            s.flush()
            idea_id = idea.id
            for call_id in self.llm.unbilled:
                call = s.get(LLMCallORM, call_id)
                if call is not None:
                    call.idea_id = idea_id  # 第 9 问: the calls that produced it are its cost
            self.llm.unbilled = []
            self.ledger.log(s, "idea_created", cell_id, idea=idea_id, title=idea.title)
        self.llm.idea_id = idea_id
        self._narrow(idea_id)
        self._queue_tests(idea_id)
        return idea_id

    # ======================= 第 4 步 缩字段 =======================
    def _narrow(self, idea_id: str) -> None:
        self.llm.idea_id = idea_id
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            fields = study.usable_fields(s, idea.cell_id)
            query = f"{idea.query} {idea.title} {idea.mechanism}"
            coarse = _bm25(query, fields)[:COARSE]
            card = _card(idea)
            rows = [{"id": f.field_id, "description": f.description[:200]} for f in coarse]
        scores: dict[str, tuple[int, str, str]] = {}
        for i in range(0, len(rows), SCORE_BATCH):
            try:
                scores.update(self.llm.score_fields(card, rows[i : i + SCORE_BATCH]))
            except RuntimeError:
                continue
        order = {r["id"]: k for k, r in enumerate(rows)}
        kept = sorted(
            (fid for fid, v in scores.items() if v[0] >= KEEP_SCORE),
            key=lambda fid: (-scores[fid][0], order[fid]),
        )
        rank = {fid: k + 1 for k, fid in enumerate(kept)}
        with self.ledger.session() as s:
            for r in rows:
                sc, why, measures = scores.get(r["id"], (0, "", ""))
                s.add(
                    IdeaFieldORM(
                        idea_id=idea_id,
                        field_id=r["id"],
                        score=sc,
                        reason=why,
                        measures=measures,
                        rank=rank.get(r["id"], 0),
                    )
                )
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            idea.extra = {**idea.extra, "coarse": len(rows), "kept": len(kept)}
            self.ledger.log(
                s, "fields_ranked", idea.cell_id, idea=idea_id, coarse=len(rows), kept=len(kept)
            )
            if not kept:
                idea.status = IdeaStatus.DEAD
                idea.extra = {**idea.extra, "why_dead": f"no field scored {KEEP_SCORE}+"}

    # ======================= 第 5 步 写公式、送回测 =======================
    def _distinct_fields(self, s: Session, idea_id: str, limit: int) -> list[IdeaFieldORM]:
        """The most relevant fields, one per measured quantity: versions of the same
        quantity (e.g. nine "total assets" fields) are one piece of evidence, not nine."""
        chosen: list[IdeaFieldORM] = []
        seen: set[str] = set()
        for r in s.scalars(
            select(IdeaFieldORM)
            .where(IdeaFieldORM.idea_id == idea_id, IdeaFieldORM.rank > 0)
            .order_by(IdeaFieldORM.rank)
        ):
            key = (r.measures or r.field_id).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            chosen.append(r)
            if len(chosen) >= limit:
                break
        return chosen

    def _queue_tests(self, idea_id: str) -> int:
        """第 5 步 + 第 4 问: the LLM writes FORMULAS formulas stating the idea with its own
        fields (1-3 per formula, at most MAX_FORM_OPERATORS operators)."""
        self.llm.idea_id = idea_id
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            if idea.status == IdeaStatus.DEAD:
                return 0
            usable = {f.field_id: f for f in study.usable_fields(s, idea.cell_id)}
            rows = self._distinct_fields(s, idea_id, FORMULA_FIELDS)
            pool = [usable[r.field_id] for r in rows if r.field_id in usable]
            info = [{"id": f.field_id, "description": f.description[:200]} for f in pool]
            card = _card(idea)
            types = {f.field_id: f.type for f in pool}
            preps = {f.field_id: f.prep for f in pool}
        ops = sorted(f"{k}: {v}"[:160] for k, v in self.meta.definitions.items())
        exprs: list[tuple[str, str, str | None]] = []  # (expression, why, field)
        for _ in range(2):
            try:
                proposed = self.llm.formulas(card, info, ops, FORMULAS)
                exprs = _main_formulas(proposed, types, preps, self.meta.operators)[:FORMULAS]
            except RuntimeError:
                continue
            if exprs:
                break
        added = 0
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            cell = s.get(CellORM, idea.cell_id)
            assert cell is not None
            if not exprs:
                idea.status = IdeaStatus.DEAD
                idea.extra = {**idea.extra, "why_dead": "no valid formula after 2 tries"}
                self.ledger.log(s, "idea_dead", idea.cell_id, idea=idea_id, reason="no formulas")
                return 0
            idea.form = exprs[0][0]
            idea.extra = {
                **idea.extra,
                "formulas": [{"expression": e, "why": w, "field": f} for e, w, f in exprs],
                "test_fields": [i["id"] for i in info],
            }
            settings = self._settings(cell)
            for expr, why, fid in exprs:
                if self._add(s, cell, expr, settings, Stage.FIELD, idea_id, field=fid, change=why):
                    added += 1
            self.ledger.log(
                s,
                "tests_queued",
                idea.cell_id,
                idea=idea_id,
                queued=added,
                formulas=[e for e, _, _ in exprs],
            )
        return added

    # ======================= 第 6-7 步 =======================
    def _step_idea(self, idea_id: str) -> int:
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            if self._pending(s, idea_id):
                return 0
            status, phase = idea.status, (idea.extra or {}).get("phase")
        if status == IdeaStatus.TESTING:
            self._judge(idea_id)
            return 1
        if status == IdeaStatus.PROMISING and phase != "improve":
            self._start_improving(idea_id)
            return 1
        if status == IdeaStatus.PROMISING:
            self._settle_round(idea_id)
            return 1
        return 0

    def _judge(self, idea_id: str) -> None:
        """第 4 问: the best first formula must clear the luck line of the tries (0.81 for
        5 at noise 0.316); then the best two go to step 7, a negative one mirrored first
        (第 3 问: direct mirroring, never in CHN)."""
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            cell = s.get(CellORM, idea.cell_id)
            assert cell is not None
            done = [c for c in self._results(s, idea_id, Stage.FIELD) if not self._suspicious(c)]
            usable = [c for c in done if (c.sharpe or 0) > 0 or cell.region not in NO_FLIP_REGIONS]
            usable.sort(key=lambda c: -abs(c.sharpe or 0))
            best = usable[:PARENTS]
            idea.best_sharpe = max((abs(c.sharpe or 0) for c in usable), default=0.0)
            gate = first_gate(len(done), self.config.thresholds.noise_sigma)
            if not best or idea.best_sharpe < gate:
                idea.status = IdeaStatus.DEAD
                idea.extra = {
                    **idea.extra,
                    "gate": round(gate, 3),
                    "why_dead": f"first {len(done)} formulas best |Sharpe| "
                    f"{idea.best_sharpe:.2f} < {gate:.2f}: cannot be told from noise",
                }
                self.ledger.log(
                    s, "idea_dead", idea.cell_id, idea=idea_id, tested=len(done), gate=gate
                )
                return
            idea.status = IdeaStatus.PROMISING
            for c in best:
                if (c.sharpe or 0) < 0:
                    self._add(
                        s,
                        cell,
                        f"multiply(-1, {c.expression})",
                        c.settings,
                        Stage.FIELD,
                        idea_id,
                        field=c.extra.get("field"),
                        parent=c,
                        change="mirrored",
                    )
            idea.extra = {
                **idea.extra,
                "gate": round(gate, 3),
                "parents_from": [c.id for c in best],
            }
            self.ledger.log(
                s,
                "idea_to_improve",
                idea.cell_id,
                idea=idea_id,
                tested=len(done),
                best=idea.best_sharpe,
            )

    def _start_improving(self, idea_id: str) -> None:
        """第 7 步: the best of step 6 become parents (their mirror when negative)."""
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            parents: list[dict[str, Any]] = []
            for cid in idea.extra.get("parents_from", []):
                c = s.get(CandidateORM, cid)
                if c is None:
                    continue
                if (c.sharpe or 0) < 0:
                    c = s.scalar(
                        select(CandidateORM).where(
                            CandidateORM.parent_id == c.id, CandidateORM.change == "mirrored"
                        )
                    )
                if c is None or c.sharpe is None or self._suspicious(c):
                    continue
                parents.append({"id": c.id, "best": c.sharpe, "dry": 0, "rounds": 0, "pending": []})
            idea.extra = {**idea.extra, "phase": "improve", "parents": parents}
        self._queue_round(idea_id)

    def _queue_round(self, idea_id: str) -> None:
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            cell = s.get(CellORM, idea.cell_id)
            assert cell is not None
            parents = [
                p
                for p in idea.extra.get("parents", [])
                if p["dry"] < DRY_ROUNDS and p["rounds"] < MAX_ROUNDS
            ]
            if not parents:
                self._finish_idea(s, idea)
                return
            card = _card(idea)
            rows = {p["id"]: s.get(CandidateORM, p["id"]) for p in parents}
            payload = [
                {
                    "id": cid,
                    "expression": c.expression,
                    "decay": c.settings.get("decay"),
                    "neutralization": c.settings.get("neutralization"),
                    "sharpe": c.sharpe,
                    "fitness": c.fitness,
                    "turnover": c.turnover,
                    "failing_checks": c.fails,
                }
                for cid, c in rows.items()
                if c is not None
            ]
        self.llm.idea_id = idea_id
        try:
            variants = self.llm.improve(card, payload, CHANGE_CARDS)
        except RuntimeError:
            variants = []
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            cell = s.get(CellORM, idea.cell_id)
            assert cell is not None
            plist = copy.deepcopy(idea.extra.get("parents", []))  # a new object, so it is saved
            by_id = {p["id"]: p for p in plist}
            counts: dict[str, int] = {}
            for v in variants:
                p = by_id.get(v["parent"])
                parent = s.get(CandidateORM, v["parent"])
                if p is None or parent is None or counts.get(p["id"], 0) >= VARIANTS:
                    continue
                settings = dict(parent.settings)
                if "decay" in v:
                    settings["decay"] = max(0, min(512, int(v["decay"])))
                c = self._add(
                    s,
                    cell,
                    v["expression"],
                    settings,
                    Stage.IMPROVE,
                    idea_id,
                    field=parent.extra.get("field"),
                    parent=parent,
                    change=v["change"],
                )
                if c is not None:
                    p["pending"] = [*p.get("pending", []), c.id]
                    counts[p["id"]] = counts.get(p["id"], 0) + 1
            for p in plist:
                if not p.get("pending") and p["dry"] < DRY_ROUNDS and p["rounds"] < MAX_ROUNDS:
                    p["dry"] += 1  # nothing valid proposed counts as a round without progress
                    p["rounds"] += 1
            idea.extra = {**idea.extra, "parents": plist}
            if not any(p.get("pending") for p in plist):
                self._finish_idea(s, idea)

    def _settle_round(self, idea_id: str) -> None:
        """第 5 问: a variant replaces its parent only when it is 0.35 better."""
        gain = self.config.thresholds.refine_min_gain
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            plist = copy.deepcopy(idea.extra.get("parents", []))  # a new object, so it is saved
            for p in plist:
                if not p.get("pending"):
                    continue
                found = [s.get(CandidateORM, k) for k in p["pending"]]
                kids = [
                    k
                    for k in found
                    if k is not None and k.sharpe is not None and not self._suspicious(k)
                ]
                best = max(kids, key=lambda k: k.sharpe or 0, default=None)
                if best is not None and (best.sharpe or 0) >= p["best"] + gain:
                    p.update(id=best.id, best=best.sharpe, dry=0)
                    idea.best_sharpe = max(idea.best_sharpe, best.sharpe or 0)
                else:
                    p["dry"] += 1
                p["rounds"] += 1
                p["pending"] = []
                # the same first failing check on every variant, SAME_FAIL_LIMIT times: the
                # structure is blocked, stop fixing this one (alpha-mining 停止条件)
                first_fails = {(k.fails or "").split(",")[0] for k in kids if k.fails}
                if len(first_fails) == 1:
                    fail = first_fails.pop()
                    p["same_fail"] = p.get("same_fail", 0) + 1 if p.get("last_fail") == fail else 1
                    p["last_fail"] = fail
                    if p["same_fail"] >= SAME_FAIL_LIMIT:
                        p["dry"] = DRY_ROUNDS
            idea.extra = {**idea.extra, "parents": plist}
            self.ledger.log(
                s,
                "round_settled",
                idea.cell_id,
                idea=idea_id,
                parents=[(p["best"], p["dry"], p["rounds"]) for p in plist],
            )
        self._queue_round(idea_id)

    # ======================= 第 8 步 排除运气 =======================
    def evaluate(self) -> int:
        """Records new results; anything passing the platform floor is stress-tested."""
        ready = 0
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
                cell = s.get(CellORM, c.cell_id)
                if cell is None:
                    continue
                cell.sims_used += 1
                if c.stage == Stage.ROBUST or c.sharpe is None or self._suspicious(c):
                    continue
                if (c.extra or {}).get("design") == "v1" and self._passes_floor(c, cell):
                    c.extra = {**c.extra, "ready": True}
                    ready += 1
                    self._queue_robust(s, cell, c)
            self._settle_robust(s)
        return ready

    def _queue_robust(self, s: Session, cell: CellORM, c: CandidateORM) -> None:
        for p in gen.robustness_proposals(
            c.expression,
            c.settings,
            self.meta.neutralizations(cell.region),
            self.meta.universes(cell.region),
        ):
            if p.role in ROBUST_ROLES:
                self._add(
                    s,
                    cell,
                    p.expression,
                    p.settings,
                    Stage.ROBUST,
                    c.extra.get("idea"),
                    parent=c,
                    role=p.role,
                )
        c.extra = {**c.extra, "robust_queued": True}

    def _settle_robust(self, s: Session) -> None:
        keep = self.config.thresholds.robust_keep_ratio
        for parent in s.scalars(
            select(CandidateORM).where(CandidateORM.status == CandidateStatus.DONE)
        ):
            extra = parent.extra or {}
            if not extra.get("robust_queued") or "robust" in extra:
                continue
            kids = list(
                s.scalars(
                    select(CandidateORM).where(
                        CandidateORM.parent_id == parent.id, CandidateORM.stage == Stage.ROBUST
                    )
                )
            )
            if any(k.status in PENDING for k in kids):
                continue
            # 第 7 问: no luck line (the regular line is far above it); the other
            # neutralization must keep robust_keep_ratio of the Sharpe
            base = parent.sharpe or 0.0
            ratios = {k.role: round((k.sharpe or 0.0) / base, 3) if base else 0.0 for k in kids}
            tries = self._tries(s, extra.get("idea"))
            ok = all(r >= keep for r in ratios.values())
            parent.extra = {**extra, "robust": {"pass": ok, "ratios": ratios, "tries": tries}}
            self.ledger.log(
                s,
                "robustness",
                parent.cell_id,
                candidate=parent.id,
                passed=ok,
                ratios=ratios,
                tries=tries,
            )

    # ======================= helpers =======================
    def _add(
        self,
        s: Session,
        cell: CellORM,
        expression: str,
        settings: dict[str, Any],
        stage: str,
        idea_id: str | None,
        *,
        field: str | None = None,
        parent: CandidateORM | None = None,
        change: str = "",
        role: str = "",
    ) -> CandidateORM | None:
        types = {
            f.field_id: f.type
            for f in s.scalars(select(FieldORM).where(FieldORM.cell_id == cell.cell_id))
        }
        verdict = check(expression, operators=self.meta.operators, fields=types)
        if not verdict.ok:
            self.ledger.log(
                s,
                "invalid",
                cell.cell_id,
                idea=idea_id,
                expression=expression,
                issues=verdict.issues[:3],
            )
            return None
        row, created = self.ledger.get_or_add(
            s,
            cell_id=cell.cell_id,
            parent_id=parent.id if parent else None,
            stage=stage,
            role=role or stage.lower(),
            change=change,
            expression=expression,
            settings=settings,
            priority={Stage.ROBUST: 3.0, Stage.IMPROVE: 2.0}.get(stage, 1.0),
            extra={
                "design": "v1",
                "idea": idea_id,
                "field": field,
                "fields": sorted(verdict.fields_used),
            },
        )
        if not created and row.status == CandidateStatus.INVALID:
            row.status, row.error = CandidateStatus.QUEUED, None
        if idea_id:  # an earlier identical simulation counts for this idea too
            idea = s.get(ResearchIdeaORM, idea_id)
            if idea is not None and row.id not in idea.extra.get("rows", []):
                idea.extra = {**idea.extra, "rows": [*idea.extra.get("rows", []), row.id]}
        return row

    def _settings(self, cell: CellORM) -> dict[str, Any]:
        return gen.base_settings(
            cell.region,
            cell.universe,
            cell.delay,
            cell.category.lower(),
            self.meta.neutralizations(cell.region),
        )

    @staticmethod
    def _idea_rows(s: Session, idea_id: str) -> list[CandidateORM]:
        """Every simulation this idea asked for, including identical ones it reused."""
        idea = s.get(ResearchIdeaORM, idea_id)
        ids = list((idea.extra or {}).get("rows", [])) if idea is not None else []
        if not ids:
            return []
        return list(s.scalars(select(CandidateORM).where(CandidateORM.id.in_(ids))))

    def _pending(self, s: Session, idea_id: str) -> int:
        return sum(
            1
            for c in self._idea_rows(s, idea_id)
            if c.status in PENDING and c.stage != Stage.ROBUST
        )

    def _results(self, s: Session, idea_id: str, stage: str) -> list[CandidateORM]:
        return [
            c
            for c in self._idea_rows(s, idea_id)
            if c.stage == stage and c.status == CandidateStatus.DONE and c.sharpe is not None
        ]

    def _tries(self, s: Session, idea_id: str | None) -> int:
        if not idea_id:
            return 1
        return len(
            {
                c.expression
                for c in self._idea_rows(s, idea_id)
                if c.stage != Stage.ROBUST and c.status == CandidateStatus.DONE
            }
        )

    def _suspicious(self, c: CandidateORM) -> bool:
        """A fake: |Sharpe| above the line the user set. High turnover is not a fake, it is
        a failing check that step 7 fixes (raise decay, DECISION_LAYER)."""
        return abs(c.sharpe or 0) > self.config.thresholds.suspicious_sharpe

    def _finish_idea(self, s: Session, idea: ResearchIdeaORM) -> None:
        """第 1 问: an idea hits when one of its formulas passes the regular line."""
        cell = s.get(CellORM, idea.cell_id)
        rows = [
            c
            for c in self._idea_rows(s, idea.id)
            if c.stage in (Stage.FIELD, Stage.IMPROVE)
            and c.status == CandidateStatus.DONE
            and not self._suspicious(c)
        ]
        idea.best_sharpe = max((c.sharpe or 0.0 for c in rows), default=0.0)
        hits = [c.id for c in rows if cell is not None and self._passes_floor(c, cell)]
        if hits:
            idea.status = IdeaStatus.DONE
            idea.extra = {**idea.extra, "regular_hits": hits}
            self.ledger.log(s, "idea_done", idea.cell_id, idea=idea.id, best=idea.best_sharpe)
        else:
            idea.status = IdeaStatus.DEAD
            idea.extra = {
                **idea.extra,
                "why_dead": f"after improvement the best Sharpe is {idea.best_sharpe:.2f}; "
                "no formula passed the regular line",
            }
            self.ledger.log(s, "idea_dead", idea.cell_id, idea=idea.id, best=idea.best_sharpe)

    def _passes_floor(self, c: CandidateORM, cell: CellORM) -> bool:
        t = self.config.thresholds
        if c.fails or c.fitness is None or c.turnover is None:
            return False
        return (
            (c.sharpe or 0) >= self.config.sharpe_floor(cell.region)
            and c.fitness >= t.submit_fitness
            and t.turnover_min <= c.turnover <= t.turnover_max
            and (c.margin or 0) * 10_000 >= self.config.margin_floor(cell.region)
        )


# ======================= small pieces =======================
def _int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _card(idea: ResearchIdeaORM) -> dict[str, Any]:
    return {
        "title": idea.title,
        "mechanism": idea.mechanism,
        "long": idea.long_side,
        "short": idea.short_side,
        "horizon_days": idea.horizon_days,
    }


def _main_formulas(
    proposed: list[dict[str, str]],
    types: dict[str, str],
    preps: dict[str, str],
    operators: set[str],
) -> list[tuple[str, str, str | None]]:
    """Formulas written with field ids: valid, at most MAX_FORM_OPERATORS as written, no
    window-only or outer rank/zscore duplicates; fields are then replaced by their prep."""
    plain = {
        fid: "MATRIX" for fid in types
    }  # checked as written; vectors get their reducer in prep
    out: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()
    for f in proposed:
        text = f["form"].strip()
        v = check(text, operators=operators, fields=plain)
        if not v.ok or v.operator_count > MAX_FORM_OPERATORS or not (v.fields_used & set(types)):
            continue
        shape = _formula_shape(text)
        if shape in seen:
            continue
        seen.add(shape)
        expr = text
        for fid in sorted(v.fields_used & set(types), key=len, reverse=True):
            expr = re.sub(rf"(?<![\w]){re.escape(fid)}(?![\w])", preps[fid], expr)
        first = sorted(v.fields_used & set(types))[0]
        out.append((expr, f.get("way", ""), first))
    return out


def _formula_shape(expression: str) -> str:
    """Canonical comparison shape, ignoring windows and cosmetic outer scaling."""
    shape = re.sub(r"\s+", "", expression)
    while True:
        match = re.fullmatch(r"(?:rank|zscore)\((.*)\)", shape, flags=re.IGNORECASE)
        if match is None:
            break
        inner = match.group(1)
        depth = 0
        encloses_whole_expression = True
        for char in inner:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                encloses_whole_expression = False
                break
        if not encloses_whole_expression:
            break
        shape = inner
    return re.sub(r"\d+(\.\d+)?", "#", shape)


def _knowledge(s: Session) -> tuple[list[str], list[str]]:
    """第 9 步 fed back to 第 3 步: recent dead ends and what reached Sharpe 1."""
    dead: list[str] = []
    worked: list[str] = []
    for i in s.scalars(
        select(ResearchIdeaORM)
        .where(ResearchIdeaORM.status != IdeaStatus.RETIRED)
        .order_by(ResearchIdeaORM.created_at.desc())
        .limit(200)
    ):
        if i.status == IdeaStatus.DEAD and len(dead) < 20:
            dead.append(f"{i.title}: {i.extra.get('why_dead', '')}"[:200])
        elif i.status == IdeaStatus.DONE and len(worked) < 20:
            worked.append(f"{i.title} (best Sharpe {i.best_sharpe:.2f}): {i.form}"[:300])
    return dead, worked


def _family_lines(fields: list[FieldORM]) -> list[dict[str, str]]:
    seen: dict[str, FieldORM] = {}
    for f in fields:
        seen.setdefault(gen.family_key(f.field_id), f)
    return [
        {"id": f.field_id, "description": f.description[:150]}
        for f in list(seen.values())[:FIELD_LINES]
    ]


def _bm25(query: str, fields: list[FieldORM], k1: float = 1.4, b: float = 0.75) -> list[FieldORM]:
    """第 4 步 coarse filter: keyword relevance of name + description to the idea."""
    docs = [tokens(f"{f.field_id.replace('_', ' ')} {f.description}") for f in fields]
    if not docs:
        return []
    avg = sum(len(d) for d in docs) / len(docs) or 1.0
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    n = len(docs)
    q = set(tokens(query))
    scored = []
    for i, d in enumerate(docs):
        tf: dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        scored.append((score, i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [fields[i] for _, i in scored]


# ======================= 第 5 节 one day =======================
@dataclass
class DayReport:
    simulations: int = 0
    failed: int = 0
    ready: int = 0
    reviews: list[str] = field(default_factory=list)
    stopped_because: str = ""


@dataclass
class LLMOnlyReport:
    idea_id: str
    title: str
    mechanism: str
    formulas: list[str]
    simulations: int = 0


def run_llm_only(
    config: FactoryConfig,
    api: BrainAPI,
    ledger: Ledger,
    llm: FactoryLLM,
    *,
    rng: random.Random | None = None,
    meta: Metadata | None = None,
) -> LLMOnlyReport | None:
    """Generate one idea and its formulas, deliberately never invoking Simulator."""
    from alpha_orchestrator.factory.daily import catalog_is_stale

    meta = meta or Metadata.fetch(api)
    if catalog_is_stale(ledger):
        catalog.refresh(api, ledger, config)
    study.reset_old_verdicts(ledger)
    idea_id = Engine(api, ledger, config, meta, llm, rng).generate_one()
    if idea_id is None:
        return None
    with ledger.session() as s:
        idea = s.get(ResearchIdeaORM, idea_id)
        assert idea is not None
        formulas = [str(row["expression"]) for row in (idea.extra or {}).get("formulas", [])]
        ledger.log(s, "llm_only_done", idea.cell_id, idea=idea_id, formulas=len(formulas))
        return LLMOnlyReport(
            idea_id=idea_id,
            title=idea.title,
            mechanism=idea.mechanism,
            formulas=formulas,
        )


def run_day(
    config: FactoryConfig,
    api: BrainAPI,
    ledger: Ledger,
    llm: FactoryLLM,
    *,
    minutes: float = 720.0,
    rng: random.Random | None = None,
    meta: Metadata | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[], None] | None = None,
) -> DayReport:
    from alpha_orchestrator.factory.daily import catalog_is_stale

    report = DayReport()
    deadline = time.monotonic() + minutes * 60
    meta = meta or Metadata.fetch(api)
    if catalog_is_stale(ledger):
        catalog.refresh(api, ledger, config)
    study.reset_old_verdicts(ledger)
    with ledger.session() as s:
        budget = config.budget.v1_daily_cap - ledger.sims_used_today(s)
    quota = api.client.quota.remaining
    if quota is not None:
        budget = min(budget, quota - QUOTA_RESERVE)
    engine = Engine(api, ledger, config, meta, llm, rng)
    simulator = Simulator(api, ledger, config, sleep=sleep)
    spent, idle = 0, 0
    while spent < budget and time.monotonic() < deadline:
        moved = engine.advance()
        run = simulator.run(min(CHUNK, budget - spent), deadline=deadline)
        spent += run.completed + run.failed
        report.simulations += run.completed
        report.failed += run.failed
        report.ready += engine.evaluate()
        if on_progress is not None:
            on_progress()
        report.stopped_because = run.stopped_because
        if run.stopped_because in ("quota", "deadline", "platform kept refusing batches"):
            break
        if run.batches == 0 and not any(moved.values()):
            idle += 1
            if idle >= 3:
                report.stopped_because = report.stopped_because or "nothing left to do"
                break
        else:
            idle = 0
    else:
        report.stopped_because = report.stopped_because or "daily budget spent"
    engine.evaluate()
    report.reviews = store_passing(api, ledger, config)  # 第 10 问: all of them, you pick
    with ledger.session() as s:
        ledger.log(
            s,
            "day_done",
            design="v1",
            simulations=report.simulations,
            failed=report.failed,
            ready=report.ready,
            reviews=report.reviews,
            reason=report.stopped_because,
        )
    return report


def _reviews_today(ledger: Ledger) -> int:
    from datetime import UTC, datetime

    start = datetime.combine(datetime.now(UTC).date(), datetime.min.time(), tzinfo=UTC)
    with ledger.session() as s:
        return int(
            s.scalar(
                select(func.count()).select_from(ReviewORM).where(ReviewORM.created_at >= start)
            )
            or 0
        )


def _robust_passed_waiting(ledger: Ledger) -> int:
    with ledger.session() as s:
        reviewed = set(s.scalars(select(ReviewORM.candidate_id)))
        return sum(
            1
            for c in s.scalars(
                select(CandidateORM).where(CandidateORM.status == CandidateStatus.DONE)
            )
            if c.id not in reviewed and ((c.extra or {}).get("robust") or {}).get("pass")
        )
