"""Design document v1, steps 3-9, and one day of running them (section 5).

Every rule below names its step in the design document. Numbers marked as proposals there
are the module constants here; nothing else decides.
"""

import copy
import math
import random
import re
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import NormalDist
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.factory import catalog, fixes, refinement_trial, study
from alpha_orchestrator.factory import generators as gen
from alpha_orchestrator.factory import pool_review as pr
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.direction import direction_label, expected_sign
from alpha_orchestrator.factory.ideas import tokens
from alpha_orchestrator.factory.ledger import Ledger, fingerprint, ledger_locked
from alpha_orchestrator.factory.llm import FactoryLLM, LLMUnavailable
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CandidateStatus,
    CellORM,
    CellState,
    FieldORM,
    IdeaStatus,
    ResearchIdeaORM,
    ReviewORM,
    Stage,
)
from alpha_orchestrator.factory.quarantine import exception_scope
from alpha_orchestrator.factory.research import Metadata
from alpha_orchestrator.factory.scheduler import Simulator
from alpha_orchestrator.factory.select import store_passing
from alpha_orchestrator.factory.validate import MAX_OPERATORS, check

# Step 3
FIELD_LINES = 200  # field families shown to the LLM
MAX_FORM_OPERATORS = 4  # Design Q4: 4 + one per round x 4 rounds = 8, the Power Pool limit
IDEA_FAILS_TO_STOP = 2  # Design Q6: 2 failed ideas in a row (cost cap) -> the dataset is done
# Step 4: coarse keyword filter, then LLM scoring in batches; keep fields scoring >= 6/10
COARSE = 300  # fields kept by the keyword filter before LLM scoring (bounds LLM cost)
SCORE_BATCH = 100  # fields scored per LLM call (keeps each prompt within limits)
KEEP_SCORE = 6  # minimum LLM relevance score (0-10) for a field to stay in the pool
# Step 5 (revised 2026-10-05)
FORMULAS = 5  # per idea; a batch of 10 holds two ideas (scheduler mixes ideas in a batch)
FORMULA_FIELDS = 8  # distinct quantities the LLM may combine in its formulas
# Design Q3 (10/14): no control arm; the template scanner is the control
# Design Q4 (10/14): an idea whose best first formula is under the luck line of its tries
# (0.81 for 5 tries at noise 0.316) cannot be told from noise and stops there
GATE_CONFIDENCE = 0.95
# Design Q1: an idea "hits" when one of its formulas passes the regular line (_passes_floor)
# Design Q5 (10/14)
PARENTS = 1  # Current design steps 5+6: every change starts from the current best formula
VARIANTS = 3  # one-change variants per round; 3 x 3 rounds fits the 12-per-idea budget
DRY_ROUNDS = 1  # Current design step 7: a round that does not add refine_min_gain stops it
MAX_ROUNDS = 3  # Current design step 7: at most 3 + 3x3 = 12 simulations per idea
# Current design (10/14 evening)
GROUPS = 8  # Step 2: 100 simulations / 12 per idea
FIRST_FORMULAS = 3  # Step 3
FIRST_OPERATORS = 5  # Step 3: 5 + one change per round x 3 rounds = 8, the Power Pool limit
FORMULA_MAX_FIELDS = 3  # Power Pool
# Step 7. Was 0.8 (the 5% luck line of 3 noise formulas at sigma 0.316). Measured sigma is
# 0.36-0.58 per dataset (pooled 0.49), so 0.8 is passed by noise 9%-77% of the time. At 1.0
# the 23 ideas whose best first formula sat in 0.8-1.0 are no longer improved: they cost 49
# backtests (6% of 817) and none later reached 1.2; all 17 ideas that reached 1.2, and all 8
# that reached 1.58, had a first formula of at least 1.0. Ledger of 2026-10-31.
FIRST_GATE = 1.0
HORIZONS = (5, 21, 63, 126, 252)  # the windows the system prompt allows
GROUP_FIELD_LINES = 400  # field families shown when grouping fields (prompt-size cap)
DIRECTION_SHORTLIST = 8  # candidate datasets offered to the research-direction trial
# Revised after the first production run (10/15)
GROUP_REFILLS = 10  # Safety cap; the real stop is the dataset budget (100 simulations) running out
# Decided 10/15: scout 8 groups first; none above 0.8 -> no signal, move on; any above ->
# add new groups until the budget is spent. The chance of wrongly dropping a dataset with
# signal is about 0.8^8 = 17% (assuming a 20% chance per group of passing 0.8).
# One step from the submission line: keep refining up to 3 rounds even if gains are < 0.35
NEAR_MISS = 1.2
# Enhance (10/16): formulas still below the line after refinement but with Sharpe >= NEAR_MISS get
# structures validated by the scanner's 2nd/3rd-order runs. Stage 1, 4 tries: compare within 10
# liquidity or size buckets. In round 2 both 1.58 formulas failed only the sub-universe check
# (signal concentrated in small caps); in the scanner, these buckets passed most often.
# Single quotes: the exact form the scanner has run thousands of times.
_DECILES = "densify(bucket(rank({x}), range='0.1, 1, 0.1'))"
ENHANCE_GROUPS = tuple(
    f"{op}({{f}}, {_DECILES.format(x=x)})"
    for x in ("close*volume", "cap")
    for op in ("group_zscore", "group_neutralize")
)
# Stage 2, 2 tries: wrap the best stage-1 formula in a trade condition (-1 = never force-close).
# In the scanner's 3rd-order runs these two conditions passed most often
# (volume surge, high volatility).
ENHANCE_TIMING = (
    "trade_when(ts_mean(volume, 10) > ts_mean(volume, 60), {f}, -1)",
    "trade_when(group_rank(ts_std_dev(returns, 60), sector) > 0.7, {f}, -1)",
)
PROBLEMS = (
    "no signal",
    "direction reversed",
    "turnover too high",
    "works only in some years",
    "concentrated in few stocks",
    "low fitness (returns too small for the turnover)",
    "other (say what)",
)
SAME_FAIL_LIMIT = 3  # alpha-mining: 3 variants failing the same check -> change the formula
# Section 5: one day of running
ACTIVE_IDEAS = 2  # proposed value in the design document
# v7: 6 ideas in flight. In v6 two ideas used 17% of the daily quota (peak 321 of 1,000):
# each idea waits for its own backtests, so more ideas in parallel keep the platform busy.
V7_ACTIVE_IDEAS = 6
V7_QUEUE_CAP = 40  # v7 backlog cap: 6 ideas in flight need more queued work than v6's 20
V7_DRY_ROUNDS = 2  # v7: two rounds in a row without a better variant end the idea
PAST_IDEAS = 40  # v7: finished ideas shown to idea generation
# Item 3. No new idea is opened while this many simulations are already waiting or running.
# A batch holds 10 simulations: one batch on the platform plus one ready behind it keeps the
# platform busy and builds no backlog. The ledger of 2026-10-31 shows up to 48 waiting at once.
QUEUE_CAP = 20
CHUNK = 20  # simulations between decisions
NO_FLIP_REGIONS = frozenset({"CHN"})  # Step 6: short side is constrained in China
ROBUST_ROLES = frozenset({"neutralization"})  # Design Q7: the platform checks sub-universe
PENDING = (CandidateStatus.QUEUED, CandidateStatus.RUNNING)
# Platform limits and unit conversions.
PLATFORM_MAX_DECAY = 512  # largest decay the platform accepts
MIN_DOUBLED_DECAY = 5  # a "raise decay" fix starts from at least this many days
BPS_PER_UNIT = 10_000  # margins are stored as fractions, configured in basis points
# Safety cap on dataset draws per call: each draw that yields no usable fields is retried,
# and 30 bounds a pathological catalog without stopping a normal day.
MAX_DATASET_DRAWS = 30
POOL_REVIEW_ROWS = 40  # most recent ideas shown to the pool review (keeps the prompt bounded)
MAX_REJECTIONS_SHOWN = 5  # validation errors echoed back to the LLM per retry
# Idea memory fed to idea generation: scan the latest 200 ideas, show at most 20 dead ends
# and 20 successes, enough for the model to avoid repeats without crowding the prompt.
MEMORY_SCAN_IDEAS = 200
MEMORY_EXAMPLES = 20
DEFAULT_HORIZON_DAYS = 21  # about one trading month, used when the LLM gives no horizon
# Queue priority: finish robustness checks first, then enhancements, then refinements, so
# work closest to a submittable alpha is simulated first.
STAGE_PRIORITY = {Stage.ROBUST: 3.0, Stage.ENHANCE: 2.5, Stage.IMPROVE: 2.0}
DEFAULT_STAGE_PRIORITY = 1.0
DEFAULT_RUN_MINUTES = 720.0  # 12 hours: one unattended overnight run
WORD = re.compile(r"[a-z0-9]+")


def luck_line(n: int, sigma: float) -> float:
    """The |Sharpe| the best of n pure-noise formulas exceeds with only 5% chance."""
    n = max(n, 1)
    p = 1 - GATE_CONFIDENCE ** (1 / n)
    return sigma * NormalDist().inv_cdf(1 - p / 2)


def first_gate(n: int, sigma: float) -> float:
    """Design Q4: 0.81 for 5 first formulas at noise 0.316."""
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
        for idea_id in self._enhancing_ideas():
            moved["ideas_moved"] += self._step_idea(idea_id)
        cell_id = self._current_dataset() or self._open_dataset()
        if cell_id is None:
            return moved
        if self.config.pool_review:  # before stepping: ideas whose results just came back
            moved["pool_decisions"] = self._pool_review(cell_id)
        for idea_id in self._active_ideas(cell_id):
            moved["ideas_moved"] += self._step_idea(idea_id)
        attempts = 0
        v7 = self.config.design == "v7"
        active, cap = (V7_ACTIVE_IDEAS, V7_QUEUE_CAP) if v7 else (ACTIVE_IDEAS, QUEUE_CAP)
        while len(self._active_ideas(cell_id)) < active and attempts < active:
            attempts += 1
            if self._dataset_finished(cell_id):
                break
            if self._pending_count() >= cap:
                moved["queue_full"] = moved.get("queue_full", 0) + 1
                break
            if self._new_idea(cell_id):
                moved["ideas_new"] += 1
        if not self._active_ideas(cell_id) and self._dataset_finished(cell_id):
            self._close_dataset(cell_id)
            moved["datasets_closed"] = 1
        return moved

    def generate_one(self) -> str | None:
        """Create exactly one idea through formulas, without running a simulation."""
        cell_id = self._current_dataset() or self._open_dataset()
        if cell_id is None or self._dataset_finished(cell_id):
            return None
        return self._new_idea(cell_id)

    def _current_dataset(self) -> str | None:
        with self.ledger.session() as s:
            _, excluded = exception_scope(s, self.config)
            for c in s.scalars(select(CellORM).where(CellORM.state == CellState.RESEARCH)):
                if c.cell_id in excluded:
                    continue
                if (c.plan or {}).get("design") == study.DESIGN:
                    return c.cell_id
            # A pilot cap is an operational stop, not a permanent research verdict.
            # Resume only explicitly selected cells and retain their prior evidence.
            for cell_id in self.config.only_datasets:
                if cell_id in excluded:
                    continue
                cell = s.get(CellORM, cell_id)
                if (
                    cell is None
                    or not cell.active
                    or cell.state != CellState.EXHAUSTED
                    or not cell.stop_reason.startswith("budget spent;")
                ):
                    continue
                plan = cell.plan or {}
                spent = cell.sims_used - int(plan.get("sims_at_open", 0))
                if (
                    plan.get("design") != study.DESIGN
                    or spent >= self.config.budget.dataset_sim_cap
                ):
                    continue
                has_work = int(plan.get("next_group", 0)) < len(plan.get("groups") or []) or (
                    int(plan.get("refills", 0)) < GROUP_REFILLS and self._has_signal(s, cell_id)
                )
                if not has_work:
                    continue
                previous = {"reason": cell.stop_reason, "outcome": plan.get("outcome")}
                cell.plan = {k: v for k, v in plan.items() if k not in {"outcome", "verdict"}}
                cell.state, cell.stop_reason = CellState.RESEARCH, ""
                self.ledger.log(
                    s,
                    "dataset_budget_resumed",
                    cell_id,
                    previous=previous,
                    new_cap=self.config.budget.dataset_sim_cap,
                )
                return cell_id
        return None

    def _open_dataset(self) -> str | None:
        """Steps 1-2, repeated until a dataset has usable fields."""
        for _ in range(MAX_DATASET_DRAWS):
            if self.config.design == "v7" and self.config.direction_trial:
                cell_id = self._choose_direction()
            else:
                with self.ledger.session() as s:
                    cell_id = study.choose_dataset(s, self.config, self.rng)
            if cell_id is None:
                return None
            if study.load_fields(self.api, self.ledger, cell_id, self.config):
                with self.ledger.session() as s:
                    cell = s.get(CellORM, cell_id)
                    assert cell is not None
                    # Design Q6: the 1000-simulation budget counts from here
                    cell.plan = {**(cell.plan or {}), "sims_at_open": cell.sims_used}
                    self.ledger.log(s, "dataset_opened", cell_id)
                return cell_id
        return None

    def _choose_direction(self) -> str | None:
        with self.ledger.session() as s:
            if self.config.only_datasets:
                return study.choose_dataset(s, self.config, self.rng)
            _, excluded = exception_scope(s, self.config)
            pool = study.v7_dataset_pool(s, self.config, excluded)
            for cell in pool:
                if (cell.plan or {}).get("direction_trial"):
                    # A metadata request can fail after assignment. Reuse that assignment.
                    return cell.cell_id
            if len(pool) < 2:
                return pool[0].cell_id if pool else None
            choices = [
                {
                    **self._brief(cell),
                    "cell_id": cell.cell_id,
                    "delay": cell.delay,
                    "catalogue_coverage": cell.coverage,
                    "catalogue_field_count": cell.field_count,
                }
                for cell in self.rng.sample(pool, min(DIRECTION_SHORTLIST, len(pool)))
            ]

        self.llm.idea_id = None
        proposal = self.llm.research_direction(choices)
        # Randomize after collecting the proposal so both arms have the same entry criteria.
        guided = self.rng.random() < 0.5
        cell_id = proposal["cell_id"] if guided else str(self.rng.choice(choices)["cell_id"])
        trial = {
            "version": 1,
            "arm": "guided" if guided else "random",
            "guided_probability": 0.5,
            "pool_size": len(pool),
            "shortlist": [choice["cell_id"] for choice in choices],
            "proposal": proposal,
            "model": self.config.llm.strong_model,
            "dataset_cap": self.config.budget.dataset_sim_cap,
        }
        with self.ledger.session() as s:
            chosen = s.get(CellORM, cell_id)
            assert chosen is not None
            chosen.plan = {
                **(chosen.plan or {}),
                "direction_trial": trial,
                "sims_at_open": chosen.sims_used,
            }
            self.ledger.log(s, "direction_assigned", cell_id, **trial, catalogue=choices)
        return cell_id

    def _band_sample(self, s: Session) -> list[tuple[float, bool]]:
        """(best first-formula |Sharpe|, whether the idea later reached the Sharpe line) for
        every finished idea of the current design, to read hit probabilities off."""
        out: list[tuple[float, bool]] = []
        for idea in s.scalars(
            select(ResearchIdeaORM).where(
                ResearchIdeaORM.status.in_([IdeaStatus.DEAD, IdeaStatus.DONE])
            )
        ):
            cell = s.get(CellORM, idea.cell_id)
            rows = [c for c in self._idea_rows(s, idea.id) if c.sharpe is not None]
            first = [abs(c.sharpe or 0) for c in rows if c.stage == Stage.FIELD]
            if cell is None or not first:
                continue
            line = self.config.sharpe_floor(cell.region)
            out.append((max(first), any((c.sharpe or 0) >= line for c in rows)))
        return out

    def _pool_review(self, cell_id: str) -> int:
        """Item 7. DeepSeek reads the dataset's whole pool of ideas next to numbers the program
        computed, and proposes stop / enhance per active idea; only allowed proposals run."""
        with self.ledger.session() as s:
            cell = s.get(CellORM, cell_id)
            assert cell is not None
            state = dict((cell.plan or {}).get("pool_review") or {})
            if int(state.get("calls", 0)) >= self.config.pool_review_max_calls:
                return 0
            rates = pr.band_rates(self._band_sample(s))
            table: list[dict[str, Any]] = []
            allowed: dict[str, list[str]] = {}
            for idea in s.scalars(
                select(ResearchIdeaORM)
                .where(
                    ResearchIdeaORM.cell_id == cell_id, ResearchIdeaORM.status != IdeaStatus.RETIRED
                )
                .order_by(ResearchIdeaORM.created_at)
            ):
                rows = [c for c in self._idea_rows(s, idea.id) if c.sharpe is not None]
                first = [abs(c.sharpe or 0) for c in rows if c.stage == Stage.FIELD]
                best = max(rows, key=lambda c: c.sharpe or 0, default=None)
                parents = (idea.extra or {}).get("parents") or [{}]
                is_active = (
                    idea.status == IdeaStatus.PROMISING
                    and (idea.extra or {}).get("phase") == "improve"
                    and not self._pending(s, idea.id)
                )
                actions = ["continue", "stop"] if is_active else []
                if is_active and idea.best_sharpe >= NEAR_MISS and "enhance" not in idea.extra:
                    actions.append("enhance")
                if actions:
                    allowed[idea.id] = actions
                b = pr.band(max(first, default=0.0))
                table.append(
                    {
                        "id": idea.id[:8],
                        "title": idea.title[:60],
                        "status": idea.status,
                        "first_formula_best": round(max(first, default=0.0), 2),
                        "best_now": round(idea.best_sharpe, 2),
                        "rounds_used": parents[0].get("rounds", 0),
                        "rounds_without_gain": parents[0].get("dry", 0),
                        "first_failed_check": ((best.fails or "").split(",")[0] if best else ""),
                        "simulations": len(rows),
                        "hit_probability": round(rates[b]["p"], 3),
                        "allowed_actions": actions,
                    }
                )
            if not allowed:
                return 0
            sig = repr(sorted((k, repr(v)) for k, v in allowed.items()))
            sig += repr([(t["id"], t["status"], t["rounds_used"], t["best_now"]) for t in table])
            if sig == state.get("sig"):
                return 0
            spent = cell.sims_used - int((cell.plan or {}).get("sims_at_open", 0))
            left = self.config.budget.dataset_sim_cap - spent
            brief = self._brief(cell)
        self.llm.idea_id = None
        try:
            raw = self.llm.pool_review(brief, table[-POOL_REVIEW_ROWS:], left)
        except LLMUnavailable:
            raise
        except RuntimeError:
            return 0
        decisions, problems = pr.parse(raw, allowed)
        done = 0
        with self.ledger.session() as s:
            cell = s.get(CellORM, cell_id)
            assert cell is not None
            for d in decisions:
                target = s.get(ResearchIdeaORM, d.idea)
                if (
                    target is None
                    or target.status != IdeaStatus.PROMISING
                    or self._pending(s, d.idea)
                ):
                    continue
                if d.action == "stop":
                    target.status = IdeaStatus.DEAD
                    target.extra = {**target.extra, "why_dead": f"pool review: {d.why}"}
                    done += 1
                elif d.action == "enhance":
                    rows = [
                        c
                        for c in self._idea_rows(s, target.id)
                        if c.stage in (Stage.FIELD, Stage.IMPROVE, Stage.ENHANCE)
                        and c.status == CandidateStatus.DONE
                        and not self._suspicious(c)
                    ]
                    done += bool(self._start_enhance(s, target, cell, rows))
            plan = dict(cell.plan or {})
            plan["pool_review"] = {"calls": int(state.get("calls", 0)) + 1, "sig": sig}
            cell.plan = plan
            self.ledger.log(
                s,
                "pool_review",
                cell_id,
                proposed=[(d.idea[:8], d.action, d.why) for d in decisions],
                executed=done,
                rejected=problems[:MAX_REJECTIONS_SHOWN],
            )
        return done

    def _pending_count(self) -> int:
        """Simulations created but not finished (waiting or running), across the ledger."""
        with self.ledger.session() as s:
            return int(
                s.scalar(
                    select(func.count())
                    .select_from(CandidateORM)
                    .where(CandidateORM.status.in_(PENDING))
                )
                or 0
            )

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
        """Design Q6: the dataset's 1000 simulations are spent, or the LLM failed to give a
        valid new idea IDEA_FAILS_TO_STOP times in a row."""
        with self.ledger.session() as s:
            cell = s.get(CellORM, cell_id)
            if cell is None:
                return True
            plan = cell.plan or {}
            spent = cell.sims_used - int(plan.get("sims_at_open", 0))
            if spent >= self.config.budget.dataset_sim_cap:
                return True
            if (
                plan.get("groups") is not None
                and int(plan.get("next_group", 0)) >= len(plan["groups"])
                and (
                    int(plan.get("refills", 0)) >= GROUP_REFILLS or not self._has_signal(s, cell_id)
                )
            ):
                return True  # no signal after scouting, or group refills hit the cap
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
            plan = cell.plan or {}
            spent = cell.sims_used - int(plan.get("sims_at_open", 0))
            budget_stop = spent >= self.config.budget.dataset_sim_cap
            outcome = (
                "unavailable"
                if plan.get("no_usable_groups")
                else "budget" if budget_stop else "signal" if worked else "noise"
            )
            cell.state = CellState.EXHAUSTED
            cell.plan = {**plan, "verdict": outcome, "outcome": outcome, "sims_spent": spent}
            if plan.get("direction_trial") and "direction_result" not in plan:
                hits = {
                    cid
                    for idea in ideas
                    for cid in idea.extra.get("regular_hits", [])
                    if (candidate := s.get(CandidateORM, cid)) is not None
                    and (candidate.extra or {}).get("idea") == idea.id
                }
                cell.plan = {
                    **cell.plan,
                    "direction_result": {"settled": spent, "regular_candidates": len(hits)},
                }
            why = (
                "no usable field groups"
                if plan.get("no_usable_groups")
                else (
                    "budget spent"
                    if spent >= self.config.budget.dataset_sim_cap
                    else "no valid new idea twice in a row"
                )
            )
            cell.stop_reason = (
                f"{why}; {len(ideas)} ideas, {worked} reached the regular line, {spent} sims"
            )
            self.ledger.log(s, "dataset_closed", cell_id, outcome=outcome, ideas=len(ideas))

    # ======================= Current design steps 2-4 =======================
    def _brief(self, cell: CellORM) -> dict[str, Any]:
        brief = {
            "id": cell.dataset_id,
            "name": cell.dataset_name,
            "category": cell.category,
            "region": cell.region,
            "universe": cell.universe,
            "description": cell.description[:1500],
        }
        trial = (cell.plan or {}).get("direction_trial") or {}
        if trial.get("arm") == "guided":
            brief["research_direction"] = trial["proposal"]
        return brief

    def _groups(self, cell_id: str, refill: bool = False) -> list[dict[str, Any]] | None:
        """Step 2: one LLM call per dataset reads every field description and returns up
        to GROUPS field groups; field ids that do not exist are dropped. refill=True asks
        for GROUPS more, different from those already tried (budget left over)."""
        with self.ledger.session() as s:
            cell = s.get(CellORM, cell_id)
            assert cell is not None
            plan0 = cell.plan or {}
            if plan0.get("groups") is not None and not refill:
                return list(plan0["groups"])
            tried = [g["name"] for g in plan0.get("groups") or []]
            brief = self._brief(cell)
            fields = study.usable_fields(s, cell_id)
            lines = [
                {
                    "id": f.field_id,
                    "description": f.description[:120],
                    "coverage": round(f.coverage or 0, 2),
                }
                for f in _one_per_family(fields)[:GROUP_FIELD_LINES]
            ]
            known = {f.field_id for f in fields}
            past: list[dict[str, Any]] = []
            if self.config.design == "v7":
                past = _past_ideas(s, PAST_IDEAS)
                signal = (
                    study.scanner_signal_fields(
                        self.config.scanner_runs, cell.region, cell.dataset_id
                    )
                    if self.config.scanner_runs
                    else None
                )
                for line in lines:
                    if signal and line["id"] in signal:
                        line["scanner_found_signal"] = True
        self.llm.idea_id = None
        groups: list[dict[str, Any]] = []
        for _ in range(2):
            try:
                raw = self.llm.field_groups(brief, lines, GROUPS, tried, past)
            except LLMUnavailable:
                raise
            except RuntimeError:
                continue
            for g in raw:
                ids = [str(x) for x in (g.get("fields") or []) if str(x) in known]
                if ids:
                    card = _idea_card(g, ids[:FORMULA_MAX_FIELDS])
                    if card is not None:
                        groups.append(card)
            if groups:
                break
        with self.ledger.session() as s:
            cell = s.get(CellORM, cell_id)
            assert cell is not None
            plan = dict(cell.plan or {})
            if refill:
                plan["refills"] = int(plan.get("refills", 0)) + 1
                old = list(plan.get("groups") or [])
                seen = {g["name"].lower() for g in old} | {tuple(sorted(g["fields"])) for g in old}
                groups = [
                    g
                    for g in groups
                    if g["name"].lower() not in seen and tuple(sorted(g["fields"])) not in seen
                ]
                merged = old + groups
                if not groups:
                    # No new field relation was proposed. Do not repeat the same
                    # expensive request while ready simulations sit in the queue.
                    plan["refills"] = GROUP_REFILLS
                cell.plan = {**plan, "groups": merged}
                self.ledger.log(
                    s, "field_groups_refill", cell_id, groups=[g["name"] for g in groups]
                )
                return merged
            if not groups:
                self.ledger.log(s, "field_groups_unusable", cell_id)
                return None
            cell.plan = {**plan, "groups": groups, "next_group": 0}
            self.ledger.log(s, "field_groups", cell_id, groups=[g["name"] for g in groups])
        return groups

    def _new_idea(self, cell_id: str) -> str | None:
        """Step 3: the next field group becomes an idea; its 3 first formulas are queued."""
        groups = self._groups(cell_id)
        if groups is None:
            with self.ledger.session() as s:
                cell = s.get(CellORM, cell_id)
                assert cell is not None
                cell.plan = {**(cell.plan or {}), "groups": [], "no_usable_groups": True}
            return None
        with self.ledger.session() as s:
            cell = s.get(CellORM, cell_id)
            assert cell is not None
            plan = dict(cell.plan or {})
            k = int(plan.get("next_group", 0))
            refill = (
                k >= len(groups)
                and int(plan.get("refills", 0)) < GROUP_REFILLS
                and self._has_signal(s, cell_id)
            )
        if refill:
            groups = self._groups(cell_id, refill=True) or groups
        with self.ledger.session() as s:
            cell = s.get(CellORM, cell_id)
            assert cell is not None
            plan = dict(cell.plan or {})
            k = int(plan.get("next_group", 0))
            if k >= len(groups):
                return None
            g = groups[k]
            plan["next_group"] = k + 1
            cell.plan = plan
            idea = ResearchIdeaORM(
                cell_id=cell_id,
                title=g["name"],
                mechanism=g["meaning"] + " " + g["why"],
                long_side=g["direction"],
                short_side="",
                # groups saved in a dataset's plan before this field existed ran on 21 days
                horizon_days=int(g.get("horizon_days") or DEFAULT_HORIZON_DAYS),
                query=" ".join(g["fields"]),
                extra={
                    "group": g,
                    "n": k,
                    "scanner_fields": [
                        f for f in g["fields"] if f in set(plan.get("scanner_fields") or [])
                    ],
                },
            )
            s.add(idea)
            s.flush()
            idea_id = idea.id
            self.ledger.log(s, "idea_created", cell_id, idea=idea_id, title=idea.title)
        self.llm.idea_id = idea_id
        self._queue_tests(idea_id)
        return idea_id

    def _queue_tests(self, idea_id: str) -> int:
        """Steps 3-4: FIRST_FORMULAS formulas; invalid ones go back to the LLM once with the
        exact errors; still invalid -> dropped, never sent (one bad formula cancels a batch)."""
        self.llm.idea_id = idea_id
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            cell = s.get(CellORM, idea.cell_id)
            assert cell is not None
            brief = self._brief(cell)
            group = dict(idea.extra.get("group") or {})
            usable = {f.field_id: f for f in study.usable_fields(s, idea.cell_id)}
            pool = [usable[f] for f in group.get("fields", []) if f in usable]
            info = [{"id": f.field_id, "description": f.description[:200]} for f in pool]
            types = {f.field_id: f.type for f in pool}
            preps = {f.field_id: f.prep for f in pool}
        ops = self._operator_lines()
        exprs: list[tuple[str, str, str | None]] = []
        rejected = ""
        for _ in range(2):
            try:
                proposed = self.llm.first_formulas(
                    brief, group, info, ops, FIRST_FORMULAS, rejected
                )
            except LLMUnavailable:
                raise
            except RuntimeError:
                continue
            exprs, problems = _first_formulas(
                proposed, types, preps, self.meta.operators, self.meta.signatures
            )
            exprs = exprs[:FIRST_FORMULAS]
            if len(exprs) >= FIRST_FORMULAS or (exprs and not problems):
                break
            rejected = "; ".join(problems[:MAX_REJECTIONS_SHOWN])
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
            signs = _group_signs(group)
            idea.extra = {
                **idea.extra,
                "formulas": [
                    {
                        "expression": e,
                        "why": w,
                        "field": f,
                        "direction": direction_label(e, signs),
                    }
                    for e, w, f in exprs
                ],
            }
            settings = self._settings(cell)
            for expr, why, fid in exprs:
                candidate = self._add(
                    s, cell, expr, settings, Stage.FIELD, idea_id, field=fid, change=why
                )
                if candidate is not None:
                    added += 1
                    if added == 1 and (cell.plan or {}).get("direction_trial"):
                        idea.extra = {
                            **idea.extra,
                            "direction_probe": {
                                "candidate": candidate.id,
                                "alignment": direction_label(expr, signs),
                                "reused": (candidate.extra or {}).get("idea") != idea_id,
                            },
                        }
            self.ledger.log(
                s,
                "tests_queued",
                idea.cell_id,
                idea=idea_id,
                queued=added,
                formulas=[e for e, _, _ in exprs],
            )
        return added

    @staticmethod
    def _has_signal(s: Session, cell_id: str) -> bool:
        """At least one field group of this dataset passed the first gate (FIRST_GATE)."""
        return (
            s.scalar(
                select(func.count())
                .select_from(ResearchIdeaORM)
                .where(
                    ResearchIdeaORM.cell_id == cell_id,
                    ResearchIdeaORM.status != IdeaStatus.RETIRED,
                    ResearchIdeaORM.best_sharpe >= FIRST_GATE,
                )
            )
            or 0
        ) > 0

    def _operator_lines(self) -> list[str]:
        return sorted(f"{k}: {v}"[:160] for k, v in self.meta.definitions.items())

    # ======================= Steps 6-7 =======================
    def _step_idea(self, idea_id: str) -> int:
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            if self._pending(s, idea_id):
                return 0
            needs_formulas = not (idea.extra or {}).get("rows")
            status, phase = idea.status, (idea.extra or {}).get("phase")
        if phase == "enhance":
            self._settle_enhance(idea_id)
            return 1
        if status == IdeaStatus.TESTING:
            if needs_formulas:
                return self._queue_tests(idea_id)
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
        """Design Q4: the best first formula must clear the luck line of the tries (0.81 for
        5 at noise 0.316); then the best two go to step 7, a negative one mirrored first
        (Design Q3: direct mirroring, never in CHN)."""
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            cell = s.get(CellORM, idea.cell_id)
            assert cell is not None
            probe = (idea.extra or {}).get("direction_probe")
            if probe and "signed_sharpe" not in probe:
                first = s.get(CandidateORM, probe["candidate"])
                signed = None
                if (
                    first is not None
                    and first.status == CandidateStatus.DONE
                    and first.sharpe is not None
                    and math.isfinite(first.sharpe)
                    and not self._suspicious(first)
                    and probe["alignment"] == "match"
                    and not probe["reused"]
                ):
                    signed = first.sharpe
                idea.extra = {**idea.extra, "direction_probe": {**probe, "signed_sharpe": signed}}
            done = [c for c in self._results(s, idea_id, Stage.FIELD) if not self._suspicious(c)]
            usable = [c for c in done if (c.sharpe or 0) > 0 or cell.region not in NO_FLIP_REGIONS]
            usable.sort(key=lambda c: -abs(c.sharpe or 0))
            best = usable[:PARENTS]
            idea.best_sharpe = max((abs(c.sharpe or 0) for c in usable), default=0.0)
            gate = FIRST_GATE
            if any(self._passes_floor(c, cell) for c in done):
                self._finish_idea(s, idea)
                return
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
        """Step 7: the best of step 6 become parents (their mirror when negative)."""
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
        """Steps 5+6: the LLM sees every formula this idea tried (worse ones included), the
        best one's yearly Sharpe and the field descriptions; it gives up or proposes
        VARIANTS one-change variants of the best. Repeats and invalid ones go back once."""
        if self.config.design == "v7":
            self._queue_fix_round(idea_id)
            return
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
            parent = s.get(CandidateORM, parents[0]["id"])
            assert parent is not None
            card = {
                "name": idea.title,
                "meaning": idea.mechanism,
                "direction": idea.long_side,
            }
            history = [_row(c) for c in self._idea_rows(s, idea_id) if c.stage != Stage.ROBUST]
            tried = {re.sub(r"\s+", "", c.expression) for c in self._idea_rows(s, idea_id)}
            best = _row(parent)
            best["neutralization"] = (parent.settings or {}).get("neutralization")
            best["neutralization_options"] = self.meta.neutralizations(cell.region)
            alpha_id = parent.alpha_id
            requirements = {
                "sharpe_min": self.config.sharpe_floor(cell.region),
                "fitness_min": self.config.thresholds.submit_fitness,
                "turnover_min": self.config.thresholds.turnover_min,
                "turnover_max": self.config.thresholds.turnover_max,
                "margin_min": self.config.margin_floor(cell.region) / BPS_PER_UNIT,
            }
            usable = {f.field_id: f for f in study.usable_fields(s, idea.cell_id)}
            group = idea.extra.get("group") or {}
            info = [
                {"id": f, "description": usable[f].description[:200]}
                for f in group.get("fields", [])
                if f in usable
            ]
            types = {
                f.field_id: f.type
                for f in s.scalars(select(FieldORM).where(FieldORM.cell_id == cell.cell_id))
            }
        if alpha_id:
            with suppress(Exception):
                best["yearly_sharpe"] = {
                    str(y.get("year")): y.get("sharpe")
                    for y in self.api.yearly_stats(alpha_id)
                    if isinstance(y, dict)
                }
        self.llm.idea_id = idea_id
        variants: list[dict[str, Any]] = []
        verdict: dict[str, Any] = {}
        rejected = ""
        for _ in range(2):
            try:
                verdict = self.llm.diagnose(
                    card,
                    history,
                    best,
                    info,
                    self._operator_lines(),
                    PROBLEMS,
                    VARIANTS,
                    rejected,
                    requirements=requirements,
                )
            except LLMUnavailable:
                raise
            except RuntimeError:
                continue
            if str(verdict.get("decision", "")).lower() == "give_up":
                break
            variants, problems = [], []
            tried_shapes = {_formula_shape(e) for e in tried}
            for v in verdict.get("variants") or []:
                if not isinstance(v, dict) or not isinstance(v.get("expression"), str):
                    continue
                expr = v["expression"].strip()
                setting_change = "decay" in v or "neutralization" in v
                if re.sub(r"\s+", "", expr) in tried and not setting_change:
                    problems.append(f"{expr}: already tried")
                    continue
                if _formula_shape(re.sub(r"\s+", "", expr)) in tried_shapes and not setting_change:
                    problems.append(
                        f"{expr}: only a window or number changed; change the "
                        "structure or a setting instead"
                    )
                    continue
                chk = check(
                    expr,
                    operators=self.meta.operators,
                    fields=types,
                    signatures=self.meta.signatures,
                )
                if not chk.ok or len(chk.fields_used) > FORMULA_MAX_FIELDS:
                    problems.append(f"{expr}: {'; '.join(chk.issues[:2]) or 'too many fields'}")
                    continue
                variants.append(v)
            if len(variants) >= VARIANTS or (variants and not problems):
                break
            rejected = "; ".join(problems[:MAX_REJECTIONS_SHOWN]) or "no valid variant"
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            cell = s.get(CellORM, idea.cell_id)
            assert cell is not None
            plist = copy.deepcopy(idea.extra.get("parents", []))
            p = plist[0]
            diag = {
                "round": p["rounds"] + 1,
                "problem": str(verdict.get("problem", ""))[:120],
                "evidence": str(verdict.get("evidence", ""))[:300],
                "decision": str(verdict.get("decision", ""))[:20],
                "why": str(verdict.get("why", ""))[:300],
            }
            idea.extra = {**idea.extra, "diagnoses": [*idea.extra.get("diagnoses", []), diag]}
            self.ledger.log(s, "diagnosis", idea.cell_id, idea=idea_id, **diag)
            if diag["decision"].lower() == "give_up" or not variants:
                idea.extra = {**idea.extra, "parents": plist, "gave_up": diag}
                self._finish_idea(s, idea)
                return
            parent = s.get(CandidateORM, p["id"])
            assert parent is not None
            neutral = {n.upper() for n in self.meta.neutralizations(cell.region)}
            for v in variants[:VARIANTS]:
                settings = dict(parent.settings)
                if isinstance(v.get("decay"), int):
                    settings["decay"] = max(0, min(PLATFORM_MAX_DECAY, int(v["decay"])))
                if str(v.get("neutralization", "")).upper() in neutral:
                    settings["neutralization"] = str(v["neutralization"]).upper()
                c = self._add(
                    s,
                    cell,
                    v["expression"],
                    settings,
                    Stage.IMPROVE,
                    idea_id,
                    field=parent.extra.get("field"),
                    parent=parent,
                    change=f"{diag['problem']}: {str(v.get('change', ''))[:160]}",
                )
                if c is not None:
                    p["pending"] = [*p.get("pending", []), c.id]
            idea.extra = {**idea.extra, "parents": plist}
            if not p.get("pending"):
                self._finish_idea(s, idea)

    def _settle_round(self, idea_id: str) -> None:
        """Design Q5: a variant replaces its parent only when it is 0.35 better."""
        if self.config.design == "v7":
            self._settle_fix_round(idea_id)
            return
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
                elif p["best"] >= NEAR_MISS:
                    # near the line: keep the better one (if any) and keep trying, up to
                    # MAX_ROUNDS; a smaller gain is worth it this close
                    if best is not None and (best.sharpe or 0) > p["best"]:
                        p.update(id=best.id, best=best.sharpe)
                        idea.best_sharpe = max(idea.best_sharpe, best.sharpe or 0)
                else:
                    p["dry"] += 1
                p["rounds"] += 1
                p["pending"] = []
                # the same first failing check on every variant, SAME_FAIL_LIMIT times: the
                # structure is blocked, stop fixing this one (alpha-mining stop condition)
                first_fails = {(k.fails or "").split(",")[0] for k in kids if k.fails}
                if len(first_fails) == 1:
                    fail = first_fails.pop()
                    p["same_fail"] = p.get("same_fail", 0) + 1 if p.get("last_fail") == fail else 1
                    p["last_fail"] = fail
                    if p["same_fail"] >= SAME_FAIL_LIMIT:
                        p["dry"] = DRY_ROUNDS
            idea.extra = {**idea.extra, "parents": plist}
            cell = s.get(CellORM, idea.cell_id)
            if cell is not None and any(
                c.status == CandidateStatus.DONE and self._passes_floor(c, cell)
                for c in self._idea_rows(s, idea_id)
            ):
                self._finish_idea(s, idea)
                return
            self.ledger.log(
                s,
                "round_settled",
                idea.cell_id,
                idea=idea_id,
                parents=[(p["best"], p["dry"], p["rounds"]) for p in plist],
            )
        self._queue_round(idea_id)

    # ======================= v7 steps 5-7: menu + Thompson sampling =======================
    def _fix_stats(self, s: Session) -> fixes.Stats:
        """Every finished variant whose change is a menu fix, v6 enhancement rows included
        (same wrappers): (problem, fix) -> [successes, failures]. Success = fewer failed
        platform checks than the parent."""
        rows: list[tuple[str, str, bool]] = []
        for c in s.scalars(
            select(CandidateORM).where(
                CandidateORM.stage.in_([Stage.IMPROVE, Stage.ENHANCE]),
                CandidateORM.status == CandidateStatus.DONE,
                CandidateORM.parent_id.is_not(None),
            )
        ):
            if c.sharpe is None or self._suspicious(c):
                continue
            parent = s.get(CandidateORM, c.parent_id)
            if parent is None or parent.sharpe is None:
                continue
            extra = c.extra or {}
            fix = extra.get("fix") or fixes.recognize(
                c.expression, c.settings or {}, parent.expression, parent.settings or {}
            )
            if fix is None:
                continue
            problem = extra.get("problem") or fixes.diagnose(parent.fails)
            rows.append((problem, fix, fixes.success(c.fails, parent.fails)))
        return fixes.tally(rows)

    def _queue_fix_round(self, idea_id: str) -> None:
        """v7: the program diagnoses the best formula from its failed checks; the LLM picks
        fixes from that problem's menu; Thompson sampling chooses VARIANTS of them; the
        program applies each fix to the formula or its settings."""
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            cell = s.get(CellORM, idea.cell_id)
            assert cell is not None
            plist = copy.deepcopy(idea.extra.get("parents", []))
            live = [p for p in plist if p["dry"] < V7_DRY_ROUNDS and p["rounds"] < MAX_ROUNDS]
            if not live:
                self._finish_idea(s, idea)
                return
            p = live[0]
            parent = s.get(CandidateORM, p["id"])
            assert parent is not None
            if self._passes_floor(parent, cell):
                self._finish_idea(s, idea)
                return
            parent_id, parent_fails = parent.id, parent.fails
            problem = fixes.diagnose(parent_fails)
            tried = list(p.get("tried", []))
            stats = self._fix_stats(s)
            menu = []
            for name in fixes.MENU[problem]:
                if name in tried or fixes.already_applied(name, parent.expression):
                    continue
                won, lost = stats.get((problem, name), [0, 0])
                menu.append(
                    {"fix": name, "what": fixes.FIXES[name].what, "worked": f"{won}/{won + lost}"}
                )
            card: dict[str, Any] = {
                "name": idea.title,
                "meaning": idea.mechanism,
                "direction": idea.long_side,
            }
            if (cell.plan or {}).get("direction_trial"):
                card.update(
                    first_formula_observation=idea.extra.get("direction_probe"),
                    falsifier=(idea.extra.get("group") or {}).get("falsifier"),
                )
            best = _row(parent)
            used_neutral = {
                str((c.settings or {}).get("neutralization", "")).upper()
                for c in self._idea_rows(s, idea_id)
            }
            options = [
                n.upper()
                for n in self.meta.neutralizations(cell.region)
                if n.upper() not in used_neutral
            ]
            paired = self.config.refinement_trial and not idea.extra.get("refinement_trial")
        if paired and self._queue_refinement_trial(idea_id, parent, problem, menu, stats, card):
            return
        if not menu:
            with self.ledger.session() as s:
                idea = s.get(ResearchIdeaORM, idea_id)
                assert idea is not None
                self._finish_idea(s, idea)
            return
        self.llm.idea_id = idea_id
        try:
            verdict = self.llm.pick_fixes(card, best, problem, menu, VARIANTS)
        except LLMUnavailable:
            raise
        except RuntimeError:
            verdict = {"decision": "change", "picks": []}
        picks = [
            str(x.get("fix"))
            for x in verdict.get("picks") or []
            if isinstance(x, dict) and x.get("fix")
        ]
        whys = {
            str(x.get("fix")): str(x.get("why", ""))[:200]
            for x in verdict.get("picks") or []
            if isinstance(x, dict)
        }
        gave_up = verdict.get("decision") == "give_up"
        chosen, source = (
            ([], "llm")
            if gave_up
            else fixes.choose(problem, picks, stats, VARIANTS, self.rng, tried)
        )
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            cell = s.get(CellORM, idea.cell_id)
            assert cell is not None
            plist = copy.deepcopy(idea.extra.get("parents", []))
            p = next(q for q in plist if q["id"] == parent_id)
            diag = {
                "round": p["rounds"] + 1,
                "problem": problem,
                "failing_checks": sorted(fixes.fail_set(parent_fails)),
                "llm_picks": picks,
                "chosen": chosen,
                "source": source,
                "decision": "give_up" if gave_up else "change",
                "why": str(verdict.get("why") or "")[:500],
                "proposed_reasons": whys,
            }
            idea.extra = {**idea.extra, "diagnoses": [*idea.extra.get("diagnoses", []), diag]}
            self.ledger.log(s, "diagnosis", idea.cell_id, idea=idea_id, **diag)
            parent_row = s.get(CandidateORM, parent_id)
            assert parent_row is not None
            pending: list[str] = []
            for name in chosen:
                fix = fixes.FIXES[name]
                settings = dict(parent_row.settings)
                expression = parent_row.expression
                if fix.shape is not None:
                    expression = fix.shape.format(f=parent_row.expression)
                elif fix.setting == "decay":
                    settings["decay"] = min(
                        PLATFORM_MAX_DECAY,
                        max(MIN_DOUBLED_DECAY, 2 * int(settings.get("decay") or 0)),
                    )
                elif fix.setting == "neutralization":
                    if not options:
                        continue
                    settings["neutralization"] = options.pop(0)
                c = self._add(
                    s,
                    cell,
                    expression,
                    settings,
                    Stage.IMPROVE,
                    idea_id,
                    field=(parent_row.extra or {}).get("field"),
                    parent=parent_row,
                    change=f"{problem}: {name}. {whys.get(name, '')}"[:300],
                    role="fix",
                )
                if c is not None:
                    c.extra = {**(c.extra or {}), "fix": name, "problem": problem, "source": source}
                    pending.append(c.id)
            p["tried"] = [*p.get("tried", []), *chosen]
            p["pending"] = pending
            idea.extra = {**idea.extra, "parents": plist}
            if not pending:
                self._finish_idea(s, idea)

    def _queue_refinement_trial(
        self,
        idea_id: str,
        parent: CandidateORM,
        problem: str,
        menu: list[dict[str, str]],
        stats: fixes.Stats,
        card: dict[str, Any],
    ) -> bool:
        targets = fixes.target_checks(parent.fails, problem)
        if not targets:
            return False
        expressions: dict[str, str] = {}
        with self.ledger.session() as s:
            if s.scalar(
                select(ResearchIdeaORM.id)
                .where(
                    func.json_extract(ResearchIdeaORM.extra, "$.refinement_trial.parent")
                    == parent.id
                )
                .limit(1)
            ):
                return False
            fields = {
                f.field_id: f.type
                for f in s.scalars(select(FieldORM).where(FieldORM.cell_id == parent.cell_id))
            }
            for item in menu:
                name = item["fix"]
                shape = fixes.FIXES[name].shape
                if shape is None:  # Compare formula changes at identical simulation settings.
                    continue
                expression = shape.format(f=parent.expression)
                if not check(
                    expression,
                    operators=self.meta.operators,
                    fields=fields,
                    signatures=self.meta.signatures,
                    max_operators=None,
                ).ok:
                    continue
                exists = s.scalar(
                    select(CandidateORM.id).where(
                        CandidateORM.fingerprint == fingerprint(expression, parent.settings)
                    )
                )
                if exists is None:
                    expressions[name] = expression
        if len(expressions) < 2:
            return False

        rule = refinement_trial.rule_choice(list(expressions), problem, stats)
        evidence = {
            "idea": card,
            "parent": _row(parent),
            "problem": problem,
            "target_checks": targets,
            "success_rule": "all target checks explicitly PASS and no new failed checks",
            "menu": [item for item in menu if item["fix"] in expressions],
        }
        self.llm.idea_id = idea_id
        proposal = self.llm.plan_refinement(evidence)
        choices = {"rule": rule, "llm": proposal["fix"]}
        order = [name for name in dict.fromkeys(choices.values()) if name is not None]
        self.rng.shuffle(order)
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            cell = s.get(CellORM, parent.cell_id)
            assert idea is not None and cell is not None
            parents = copy.deepcopy(idea.extra["parents"])
            work = next(p for p in parents if p["id"] == parent.id)
            candidates: dict[str, str] = {}
            for name in order:
                c = self._add(
                    s,
                    cell,
                    expressions[name],
                    dict(parent.settings),
                    Stage.IMPROVE,
                    idea_id,
                    field=(parent.extra or {}).get("field"),
                    parent=parent,
                    change=f"{problem}: {name}",
                    role="fix",
                )
                if c is None:
                    raise RuntimeError("Validated paired refinement could not be queued")
                c.extra = {
                    **c.extra,
                    "fix": name,
                    "problem": problem,
                    "source": "refinement_trial",
                }
                candidates[name] = c.id
            work["pending"] = list(candidates.values())
            work["tried"] = [*work.get("tried", []), *order]
            trial = {
                "version": refinement_trial.VERSION,
                "parent": parent.id,
                "parent_checks": sorted(fixes.fail_set(parent.fails)),
                "target_checks": targets,
                "suspicious_sharpe": self.config.thresholds.suspicious_sharpe,
                "choices": choices,
                "candidates": candidates,
                "evidence": evidence,
                "proposal": proposal,
                "model": self.config.llm.strong_model,
            }
            idea.extra = {**idea.extra, "parents": parents, "refinement_trial": trial}
            self.ledger.log(
                s,
                "refinement_trial_queued",
                cell.cell_id,
                idea=idea_id,
                parent=parent.id,
                candidates=candidates,
            )
        return True

    def _settle_fix_round(self, idea_id: str) -> None:
        """v7: the best variant (fewest failed checks, then Sharpe) replaces its parent when it
        scores higher; otherwise the round is dry. V7_DRY_ROUNDS dry rounds in a row or
        MAX_ROUNDS rounds end the idea."""
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            cell = s.get(CellORM, idea.cell_id)
            assert cell is not None
            plist = copy.deepcopy(idea.extra.get("parents", []))
            for p in plist:
                if not p.get("pending"):
                    continue
                parent = s.get(CandidateORM, p["id"])
                kids = [
                    k
                    for k in (s.get(CandidateORM, i) for i in p["pending"])
                    if k is not None and k.sharpe is not None and not self._suspicious(k)
                ]
                best = max(kids, key=lambda k: fixes.score(k.fails, k.sharpe), default=None)
                if (
                    best is not None
                    and parent is not None
                    and fixes.score(best.fails, best.sharpe)
                    > fixes.score(parent.fails, parent.sharpe)
                ):
                    p.update(id=best.id, best=best.sharpe, dry=0, tried=[])
                    idea.best_sharpe = max(idea.best_sharpe, best.sharpe or 0)
                else:
                    p["dry"] += 1
                p["rounds"] += 1
                p["pending"] = []
            idea.extra = {**idea.extra, "parents": plist}
            if any(
                c.status == CandidateStatus.DONE and self._passes_floor(c, cell)
                for c in self._idea_rows(s, idea_id)
            ):
                self._finish_idea(s, idea)
                return
            self.ledger.log(
                s,
                "round_settled",
                idea.cell_id,
                idea=idea_id,
                parents=[(p["best"], p["dry"], p["rounds"]) for p in plist],
            )
        self._queue_fix_round(idea_id)

    # ======================= Step 8: rule out luck =======================
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
            refinement_trial.settle(s)
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
            # Design Q7: no luck line (the regular line is far above it); the other
            # neutralization must keep robust_keep_ratio of the Sharpe
            base = parent.sharpe or 0.0
            ratios = {k.role: round((k.sharpe or 0.0) / base, 3) if base else 0.0 for k in kids}
            tries = self._tries(s, extra.get("idea"))
            ok = bool(ratios) and all(r >= keep for r in ratios.values())
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
        verdict = check(
            expression,
            operators=self.meta.operators,
            fields=types,
            signatures=self.meta.signatures,
            max_operators=None if stage == Stage.ENHANCE or role == "fix" else MAX_OPERATORS,
        )
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
            priority=STAGE_PRIORITY.get(stage, DEFAULT_STAGE_PRIORITY),
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
        """Design Q1: an idea hits when one of its formulas passes the regular line."""
        cell = s.get(CellORM, idea.cell_id)
        rows = [
            c
            for c in self._idea_rows(s, idea.id)
            if c.stage in (Stage.FIELD, Stage.IMPROVE, Stage.ENHANCE)
            and c.status == CandidateStatus.DONE
            and not self._suspicious(c)
        ]
        hits = [c.id for c in rows if cell is not None and self._passes_floor(c, cell)]
        if (
            not hits
            and cell is not None
            and self.config.design != "v7"  # v7: the menu holds the enhancement wrappers
            and self._start_enhance(s, idea, cell, rows)
        ):
            return
        idea.best_sharpe = max((c.sharpe or 0.0 for c in rows), default=0.0)
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

    # ======================= Enhance =======================
    def _start_enhance(
        self, s: Session, idea: ResearchIdeaORM, cell: CellORM, rows: list[CandidateORM]
    ) -> bool:
        """A near miss (best positive Sharpe >= NEAR_MISS, not past the line) gets the
        ENHANCE_GROUPS wrappers once, before the idea is closed. False when not eligible."""
        if "enhance" in (idea.extra or {}):
            return False
        base = max(
            (c for c in rows if c.stage != Stage.ENHANCE and (c.sharpe or 0) >= NEAR_MISS),
            key=lambda c: c.sharpe or 0,
            default=None,
        )
        if base is None:
            return False
        made = self._wrap(s, cell, idea.id, base, ENHANCE_GROUPS, "分档比较")
        if not made:
            return False
        idea.status = IdeaStatus.PROMISING
        idea.extra = {
            **idea.extra,
            "phase": "enhance",
            "enhance": {"step": 1, "base": base.id, "pending": made},
        }
        self.ledger.log(s, "enhance_start", idea.cell_id, idea=idea.id, base=base.sharpe)
        return True

    def _settle_enhance(self, idea_id: str) -> None:
        """Step 1 back: the best of base + wrappers (fewest failed checks first, then Sharpe)
        gets the ENHANCE_TIMING wrappers. Step 2 back: the idea closes."""
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            cell = s.get(CellORM, idea.cell_id)
            assert cell is not None
            e = dict(idea.extra.get("enhance") or {})
            kids = [
                k
                for k in (s.get(CandidateORM, i) for i in e.get("pending", []))
                if k is not None and k.sharpe is not None and not self._suspicious(k)
            ]
            done = any(self._passes_floor(k, cell) for k in kids)
            if e.get("step") == 1 and not done:
                base = s.get(CandidateORM, e.get("base"))
                pool = [k for k in [base, *kids] if k is not None and (k.sharpe or 0) > 0]
                best = max(pool, key=_enhance_key, default=None)
                made = (
                    self._wrap(s, cell, idea_id, best, ENHANCE_TIMING, "条件交易")
                    if best is not None
                    else []
                )
                if made:
                    assert best is not None
                    idea.extra = {
                        **idea.extra,
                        "enhance": {"step": 2, "base": best.id, "pending": made},
                    }
                    return
            idea.extra = {**idea.extra, "phase": "enhanced", "enhance": {**e, "pending": []}}
            self._finish_idea(s, idea)

    def _wrap(
        self,
        s: Session,
        cell: CellORM,
        idea_id: str,
        base: CandidateORM,
        shapes: tuple[str, ...],
        why: str,
    ) -> list[str]:
        made = []
        for shape in shapes:
            c = self._add(
                s,
                cell,
                shape.format(f=base.expression),
                dict(base.settings),
                Stage.ENHANCE,
                idea_id,
                field=(base.extra or {}).get("field"),
                parent=base,
                change=f"加强·{why}",
            )
            if c is not None:
                made.append(c.id)
        return made

    def _enhancing_ideas(self) -> list[str]:
        with self.ledger.session() as s:
            _, excluded = exception_scope(s, self.config)
            return [
                i.id
                for i in s.scalars(
                    select(ResearchIdeaORM).where(ResearchIdeaORM.status == IdeaStatus.PROMISING)
                )
                if i.cell_id not in excluded and (i.extra or {}).get("phase") == "enhance"
            ]

    def revive_near_misses(self) -> int:
        """Ideas closed before the enhance stage existed whose best Sharpe reached NEAR_MISS
        get it now (the datasets in only_datasets, current design)."""
        n = 0
        with self.ledger.session() as s:
            _, excluded = exception_scope(s, self.config)
            for cell_id in self.config.only_datasets:
                if cell_id in excluded:
                    continue
                cell = s.get(CellORM, cell_id)
                if cell is None or (cell.plan or {}).get("design") != study.DESIGN:
                    continue
                for idea in s.scalars(
                    select(ResearchIdeaORM).where(
                        ResearchIdeaORM.cell_id == cell_id,
                        ResearchIdeaORM.status == IdeaStatus.DEAD,
                        ResearchIdeaORM.best_sharpe >= NEAR_MISS,
                    )
                ):
                    rows = [
                        c
                        for c in self._idea_rows(s, idea.id)
                        if c.stage in (Stage.FIELD, Stage.IMPROVE)
                        and c.status == CandidateStatus.DONE
                        and not self._suspicious(c)
                    ]
                    if self._start_enhance(s, idea, cell, rows):
                        n += 1
        return n

    def _passes_floor(self, c: CandidateORM, cell: CellORM) -> bool:
        t = self.config.thresholds
        if c.fails or c.fitness is None or c.turnover is None:
            return False
        return (
            (c.sharpe or 0) >= self.config.sharpe_floor(cell.region)
            and c.fitness >= t.submit_fitness
            and t.turnover_min <= c.turnover <= t.turnover_max
            and (c.margin or 0) * BPS_PER_UNIT >= self.config.margin_floor(cell.region)
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


def _group_signs(group: dict[str, Any]) -> dict[str, int]:
    """field id -> +1 (higher value should predict higher returns) / -1, as the idea states."""
    out: dict[str, int] = {}
    for fid, word in (group.get("field_signs") or {}).items():
        sign = expected_sign(str(word))
        if sign is not None:
            out[str(fid)] = sign
    return out


def _idea_card(g: dict[str, Any], ids: list[str]) -> dict[str, Any] | None:
    """A field group becomes an idea only if it can be proven wrong: it names its window
    (one of HORIZONS, not a default), what result would show it wrong, and for each field
    whether higher or lower should predict higher returns. Otherwise it is dropped."""
    try:
        horizon = int(g.get("horizon_days"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    falsifier = str(g.get("falsifier") or "").strip()
    raw = g.get("field_signs")
    signs = {
        fid: ("high" if expected_sign(str(raw.get(fid, ""))) == 1 else "low")
        for fid in ids
        if isinstance(raw, dict) and expected_sign(str(raw.get(fid, ""))) is not None
    }
    if horizon not in HORIZONS or len(falsifier) < 10 or len(signs) != len(ids):
        return None
    highs = sum(v == "high" for v in signs.values())
    return {
        "name": str(g.get("name", ""))[:120],
        "meaning": str(g.get("meaning", ""))[:300],
        "direction": "higher is better" if highs * 2 >= len(ids) else "lower is better",
        "why": str(g.get("why", ""))[:300],
        "fields": ids,
        "horizon_days": horizon,
        "falsifier": falsifier[:300],
        "field_signs": signs,
    }


def _first_formulas(
    proposed: list[dict[str, str]],
    types: dict[str, str],
    preps: dict[str, str],
    operators: set[str],
    signatures: dict[str, tuple[int, int]],
) -> tuple[list[tuple[str, str, str | None]], list[str]]:
    """Steps 3-4: valid formulas (fields replaced by their prep) and the reasons the others
    were rejected, to send back to the LLM."""
    plain = {fid: "GROUP" if t == "GROUP" else "MATRIX" for fid, t in types.items()}
    out: list[tuple[str, str, str | None]] = []
    problems: list[str] = []
    seen: set[str] = set()
    for f in proposed:
        text = f["form"].strip()
        v = check(text, operators=operators, fields=plain, signatures=signatures)
        if not v.ok:
            problems.append(f"{text}: {'; '.join(v.issues[:2])}")
            continue
        used = v.fields_used & set(types)
        if not used or len(v.fields_used) > FORMULA_MAX_FIELDS:
            problems.append(f"{text}: must use 1-{FORMULA_MAX_FIELDS} of the given fields")
            continue
        if v.operator_count > FIRST_OPERATORS:
            problems.append(f"{text}: {v.operator_count} operators > {FIRST_OPERATORS}")
            continue
        shape = _formula_shape(text)
        if shape in seen:
            problems.append(f"{text}: same as another formula except windows")
            continue
        seen.add(shape)
        expr = text
        for fid in sorted(used, key=len, reverse=True):
            expr = re.sub(rf"(?<![\w]){re.escape(fid)}(?![\w])", preps[fid] or fid, expr)
        out.append((expr, f.get("way", ""), sorted(used)[0]))
    return out, problems


def _enhance_key(c: CandidateORM) -> tuple[int, float]:
    """Fewer failed platform checks first (a fixed check is worth more), then Sharpe."""
    fails = [f for f in (c.fails or "").split(",") if f]
    return (-len(fails), c.sharpe or 0.0)


def _one_per_family(fields: list[FieldORM]) -> list[FieldORM]:
    seen: dict[str, FieldORM] = {}
    for f in sorted(fields, key=lambda f: -(f.coverage or 0)):
        seen.setdefault(gen.family_key(f.field_id), f)
    return list(seen.values())


def _row(c: CandidateORM) -> dict[str, Any]:
    return {
        "expression": c.expression,
        "change": c.change,
        "decay": (c.settings or {}).get("decay"),
        "status": str(c.status),
        "sharpe": c.sharpe,
        "fitness": c.fitness,
        "turnover": c.turnover,
        "margin": c.margin,
        "returns": c.returns,
        "drawdown": c.drawdown,
        "long_count": c.long_count,
        "short_count": c.short_count,
        "failing_checks": c.fails,
        "error": (c.error or "")[:150] or None,
    }


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
    """Step 9 fed back to step 3: recent dead ends and what reached Sharpe 1."""
    dead: list[str] = []
    worked: list[str] = []
    for i in s.scalars(
        select(ResearchIdeaORM)
        .where(ResearchIdeaORM.status != IdeaStatus.RETIRED)
        .order_by(ResearchIdeaORM.created_at.desc())
        .limit(MEMORY_SCAN_IDEAS)
    ):
        if i.status == IdeaStatus.DEAD and len(dead) < MEMORY_EXAMPLES:
            dead.append(f"{i.title}: {i.extra.get('why_dead', '')}"[:200])
        elif i.status == IdeaStatus.DONE and len(worked) < MEMORY_EXAMPLES:
            worked.append(f"{i.title} (best Sharpe {i.best_sharpe:.2f}): {i.form}"[:300])
    return dead, worked


def _past_ideas(s: Session, limit: int) -> list[dict[str, Any]]:
    """v7 step 2 memory (R&D-Agent(Q) knowledge forest): the latest finished ideas from
    every dataset and design, with what happened to them. No formula text."""
    out: list[dict[str, Any]] = []
    for i in s.scalars(
        select(ResearchIdeaORM)
        .where(ResearchIdeaORM.status.in_([IdeaStatus.DONE, IdeaStatus.DEAD, IdeaStatus.RETIRED]))
        .order_by(ResearchIdeaORM.created_at.desc())
        .limit(limit * 3)
    ):
        if len(out) >= limit:
            break
        cell = s.get(CellORM, i.cell_id)
        if i.status != IdeaStatus.DONE and not (i.extra or {}).get("why_dead"):
            continue  # unfinished (e.g. retired mid-test): no outcome to learn from
        probe = (i.extra or {}).get("direction_probe") or {}
        out.append(
            {
                "idea": i.title[:80],
                "meaning": i.mechanism[:160],
                "dataset_category": cell.category if cell else "",
                "region": cell.region if cell else "",
                "best_sharpe": round(i.best_sharpe, 2),
                "signed_first_formula_sharpe": probe.get("signed_sharpe"),
                "outcome": (
                    "passed the submission line"
                    if i.status == IdeaStatus.DONE
                    else str((i.extra or {}).get("why_dead", "stopped"))[:120]
                ),
            }
        )
    return out


def _family_lines(fields: list[FieldORM]) -> list[dict[str, str]]:
    seen: dict[str, FieldORM] = {}
    for f in fields:
        seen.setdefault(gen.family_key(f.field_id), f)
    return [
        {"id": f.field_id, "description": f.description[:150]}
        for f in list(seen.values())[:FIELD_LINES]
    ]


def _bm25(query: str, fields: list[FieldORM], k1: float = 1.4, b: float = 0.75) -> list[FieldORM]:
    """Step 4 coarse filter: keyword relevance of name + description to the idea."""
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


# ======================= Section 5: one day =======================
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


@ledger_locked
def run_day(
    config: FactoryConfig,
    api: BrainAPI,
    ledger: Ledger,
    llm: FactoryLLM,
    *,
    minutes: float = DEFAULT_RUN_MINUTES,
    rng: random.Random | None = None,
    meta: Metadata | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[], None] | None = None,
) -> DayReport:
    """Run the factory for one allocation day and return what happened."""
    from alpha_orchestrator.factory.daily import catalog_is_stale

    report = DayReport()
    deadline = time.monotonic() + minutes * 60
    if config.quarantine_exception:
        cutoff = datetime.fromisoformat(config.quarantine_exception["expires_at"])
        deadline = min(
            deadline, time.monotonic() + max(0, (cutoff - datetime.now(UTC)).total_seconds())
        )
    simulator = Simulator(api, ledger, config, sleep=sleep)
    simulator.assert_recoverable()
    # Drain durable requests before an unavailable LLM can prevent their recovery.
    recovered = simulator.run(0, deadline=deadline)
    report.simulations += recovered.completed
    report.failed += recovered.failed
    if recovered.stopped_because == "deadline":
        report.stopped_because = "deadline"
        return report
    meta = meta or Metadata.fetch(api)
    if catalog_is_stale(ledger):
        catalog.refresh(api, ledger, config)
    study.reset_old_verdicts(ledger)
    with ledger.session() as s:
        budget = config.budget.v1_daily_cap - ledger.sims_reserved_today(
            s, config.budget.window_start()
        )
    quota = api.client.quota.remaining
    if quota is not None:
        budget = min(budget, quota - config.budget.quota_reserve)
    engine = Engine(api, ledger, config, meta, llm, rng)
    revived = engine.revive_near_misses()
    if revived:
        with ledger.session() as s:
            ledger.log(s, "enhance_revived", ideas=revived)
    spent, idle = 0, 0
    while spent < budget and time.monotonic() < deadline:
        try:
            with ledger.session() as s:
                _, excluded = exception_scope(s, config)
                queued = bool(ledger.queue_groups(s, excluded))
            moved = {"queued_work": 1} if queued else engine.advance()
        except LLMUnavailable as exc:
            report.stopped_because = f"LLM 调用失败（余额或网络）：{exc}"[:300]
            # Already queued work remains useful even when proposal generation fails.
            remaining = simulator.run(min(CHUNK, budget - spent), deadline=deadline)
            report.simulations += remaining.completed
            report.failed += remaining.failed
            break
        run = simulator.run(min(CHUNK, budget - spent), deadline=deadline)
        spent += run.completed + run.failed
        report.simulations += run.completed
        report.failed += run.failed
        report.ready += engine.evaluate()
        if on_progress is not None:
            on_progress()
        report.stopped_because = run.stopped_because
        if simulator._remaining_daily() <= 0:
            report.stopped_because = "daily budget spent"
            break
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
    report.reviews = store_passing(api, ledger, config)  # Design Q10: all of them, you pick
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
