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
from contextlib import suppress
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.factory import catalog, study
from alpha_orchestrator.factory import generators as gen
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
    IdeaStatus,
    ResearchIdeaORM,
    ReviewORM,
    Stage,
)
from alpha_orchestrator.factory.research import Metadata
from alpha_orchestrator.factory.scheduler import QUOTA_RESERVE, Simulator
from alpha_orchestrator.factory.select import store_passing
from alpha_orchestrator.factory.validate import MAX_OPERATORS, check

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
PARENTS = 1  # 现行设计 第 5+6 步: every change starts from the current best formula
VARIANTS = 3
DRY_ROUNDS = 1  # 现行设计 第 7 步: a round that does not add refine_min_gain (0.1) stops it
MAX_ROUNDS = 3  # 现行设计 第 7 步: at most 3 + 3x3 = 12 simulations per idea
# 现行设计 (10/14 晚)
GROUPS = 8  # 第 2 步: 100 simulations / 12 per idea
FIRST_FORMULAS = 3  # 第 3 步
FIRST_OPERATORS = 5  # 第 3 步: 5 + one change per round x 3 rounds = 8, the Power Pool limit
FORMULA_MAX_FIELDS = 3  # Power Pool
FIRST_GATE = 0.8  # 第 7 步: the luck line of 3 noise formulas is 0.76
GROUP_FIELD_LINES = 400
# 10/15 第一次正式跑后的修改
GROUP_REFILLS = 10  # 安全上限；真正的停止条件是数据集预算（100 次）用完
# 10/15 定：先侦察 8 组；一个都没过 0.8 → 这个数据集没信号，换下一个；有过的 → 补新组
# 直到预算用完。误扔一个有信号数据集的概率约 0.8^8 = 17%（每组过 0.8 的机会按 20% 估）
NEAR_MISS = 1.2  # Sharpe 到这里离正式线只差一步：改进没涨够 0.35 也继续改，直到 3 轮
# 加强（10/16）：改进完仍没过线、但 Sharpe ≥ NEAR_MISS 的公式，按扫描器二阶/三阶验证过的结构再试。
# 第 1 步 4 次：按成交额或市值分 10 档，在档内比较。第二轮两条 1.58 都只卡在子股票池检查
# （信号集中在小盘股）；扫描器期权数据集二阶里，成交额分档 15/31、市值分档 10/31 过三条线，
# 其他分组 ≤ 4/16。
_DECILES = "densify(bucket(rank({x}), range='0.1, 1, 0.1'))"  # 单引号：扫描器跑过上千次的写法
ENHANCE_GROUPS = tuple(
    f"{op}({{f}}, {_DECILES.format(x=x)})"
    for x in ("close*volume", "cap")
    for op in ("group_zscore", "group_neutralize")
)
# 第 2 步 2 次：在第 1 步最好的那条外面加 trade_when（只在条件成立时换仓，-1 = 不强制平仓）。
# 扫描器三阶里这两个条件各 13/25 过三条线，并列最高，一个看放量、一个看高波动。
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
# 第 5 节
ACTIVE_IDEAS = 2  # 提议
CHUNK = 20  # simulations between decisions
NO_FLIP_REGIONS = frozenset({"CHN"})  # 第 6 步: short side is constrained in China
ROBUST_ROLES = frozenset({"neutralization"})  # 第 7 问: the platform checks sub-universe
PENDING = (CandidateStatus.QUEUED, CandidateStatus.RUNNING)
WORD = re.compile(r"[a-z0-9]+")


class LLMUnavailable(RuntimeError):
    """The LLM could not be reached (balance, network, malformed replies). Not the
    dataset's fault: the day stops and no dataset is marked done because of it."""


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
        for idea_id in self._enhancing_ideas():
            moved["ideas_moved"] += self._step_idea(idea_id)
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
            if (
                plan.get("groups") is not None
                and int(plan.get("next_group", 0)) >= len(plan["groups"])
                and (
                    int(plan.get("refills", 0)) >= GROUP_REFILLS or not self._has_signal(s, cell_id)
                )
            ):
                return True  # 侦察完没信号，或补组到上限
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

    # ======================= 现行设计 第 2-4 步 =======================
    def _brief(self, cell: CellORM) -> dict[str, Any]:
        return {
            "id": cell.dataset_id,
            "name": cell.dataset_name,
            "category": cell.category,
            "region": cell.region,
            "universe": cell.universe,
            "description": cell.description[:1500],
        }

    def _groups(self, cell_id: str, refill: bool = False) -> list[dict[str, Any]] | None:
        """第 2 步: one LLM call per dataset reads every field description and returns up
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
        self.llm.idea_id = None
        groups: list[dict[str, Any]] = []
        for _ in range(2):
            try:
                raw = self.llm.field_groups(brief, lines, GROUPS, tried)
            except RuntimeError:
                continue
            for g in raw:
                ids = [str(x) for x in (g.get("fields") or []) if str(x) in known]
                if ids:
                    groups.append(
                        {
                            "name": str(g.get("name", ""))[:120],
                            "meaning": str(g.get("meaning", ""))[:300],
                            "direction": str(g.get("direction", ""))[:40],
                            "why": str(g.get("why", ""))[:300],
                            "fields": ids[:FORMULA_MAX_FIELDS],
                        }
                    )
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
                cell.plan = {**plan, "groups": merged}
                self.ledger.log(
                    s, "field_groups_refill", cell_id, groups=[g["name"] for g in groups]
                )
                return merged
            if not groups:
                self.ledger.log(s, "llm_unavailable", cell_id, reasons=["no field groups"])
                return None
            cell.plan = {**plan, "groups": groups, "next_group": 0}
            self.ledger.log(s, "field_groups", cell_id, groups=[g["name"] for g in groups])
        return groups

    def _new_idea(self, cell_id: str) -> str | None:
        """第 3 步: the next field group becomes an idea; its 3 first formulas are queued."""
        groups = self._groups(cell_id)
        if groups is None:
            raise LLMUnavailable("the LLM gave no field groups")
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
                horizon_days=21,
                query=" ".join(g["fields"]),
                extra={"group": g, "n": k},
            )
            s.add(idea)
            s.flush()
            idea_id = idea.id
            self.ledger.log(s, "idea_created", cell_id, idea=idea_id, title=idea.title)
        self.llm.idea_id = idea_id
        self._queue_tests(idea_id)
        return idea_id

    def _queue_tests(self, idea_id: str) -> int:
        """第 3-4 步: FIRST_FORMULAS formulas; invalid ones go back to the LLM once with the
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
            except RuntimeError:
                continue
            exprs, problems = _first_formulas(
                proposed, types, preps, self.meta.operators, self.meta.signatures
            )
            exprs = exprs[:FIRST_FORMULAS]
            if len(exprs) >= FIRST_FORMULAS or (exprs and not problems):
                break
            rejected = "; ".join(problems[:5])
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

    @staticmethod
    def _has_signal(s: Session, cell_id: str) -> bool:
        """At least one field group of this dataset passed the first gate (0.8)."""
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

    # ======================= 第 6-7 步 =======================
    def _step_idea(self, idea_id: str) -> int:
        with self.ledger.session() as s:
            idea = s.get(ResearchIdeaORM, idea_id)
            assert idea is not None
            if self._pending(s, idea_id):
                return 0
            status, phase = idea.status, (idea.extra or {}).get("phase")
        if phase == "enhance":
            self._settle_enhance(idea_id)
            return 1
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
        """第 5+6 步: the LLM sees every formula this idea tried (worse ones included), the
        best one's yearly Sharpe and the field descriptions; it gives up or proposes
        VARIANTS one-change variants of the best. Repeats and invalid ones go back once."""
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
                )
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
            rejected = "; ".join(problems[:5]) or "no valid variant"
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
                    settings["decay"] = max(0, min(512, int(v["decay"])))
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
                # structure is blocked, stop fixing this one (alpha-mining 停止条件)
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
        verdict = check(
            expression,
            operators=self.meta.operators,
            fields=types,
            signatures=self.meta.signatures,
            max_operators=None if stage == Stage.ENHANCE else MAX_OPERATORS,
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
            priority={Stage.ROBUST: 3.0, Stage.ENHANCE: 2.5, Stage.IMPROVE: 2.0}.get(stage, 1.0),
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
            if c.stage in (Stage.FIELD, Stage.IMPROVE, Stage.ENHANCE)
            and c.status == CandidateStatus.DONE
            and not self._suspicious(c)
        ]
        hits = [c.id for c in rows if cell is not None and self._passes_floor(c, cell)]
        if not hits and cell is not None and self._start_enhance(s, idea, cell, rows):
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

    # ======================= 加强 =======================
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
            return [
                i.id
                for i in s.scalars(
                    select(ResearchIdeaORM).where(ResearchIdeaORM.status == IdeaStatus.PROMISING)
                )
                if (i.extra or {}).get("phase") == "enhance"
            ]

    def revive_near_misses(self) -> int:
        """Ideas closed before 加强 existed whose best Sharpe reached NEAR_MISS get it now
        (the datasets in only_datasets, current design)."""
        n = 0
        with self.ledger.session() as s:
            for cell_id in self.config.only_datasets:
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


def _first_formulas(
    proposed: list[dict[str, str]],
    types: dict[str, str],
    preps: dict[str, str],
    operators: set[str],
    signatures: dict[str, tuple[int, int]],
) -> tuple[list[tuple[str, str, str | None]], list[str]]:
    """第 3-4 步: valid formulas (fields replaced by their prep) and the reasons the others
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
    revived = engine.revive_near_misses()
    if revived:
        with ledger.session() as s:
            ledger.log(s, "enhance_revived", ideas=revived)
    simulator = Simulator(api, ledger, config, sleep=sleep)
    spent, idle = 0, 0
    while spent < budget and time.monotonic() < deadline:
        try:
            moved = engine.advance()
        except LLMUnavailable as exc:
            report.stopped_because = f"LLM 调用失败（余额或网络）：{exc}"[:300]
            break
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
