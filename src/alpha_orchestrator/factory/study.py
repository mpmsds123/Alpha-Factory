"""The research loop of 设计文档 v1 (Obsidian「Alpha工厂 工作台」), one step per function.

Every decision here cites the step of the design document it implements. A decision the
document does not contain does not belong in this file.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.factory import datasets
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import CellORM, CellState, FieldORM

FILL_DAYS = 63  # 第 2 步: quarterly data carried one quarter (官方小贴士 10)
NUMERIC_TYPES = ("MATRIX", "VECTOR")


# ---------- before the first run ----------
def reset_old_verdicts(ledger: Ledger) -> int:
    """Datasets closed by the logic before design v1 go back to NEW and their verdicts stop
    counting as evidence (that logic was wrong). Their simulations stay in the ledger and
    are reused when the same formula comes up again."""
    n = 0
    with ledger.session() as s:
        for c in s.scalars(select(CellORM).where(CellORM.state != CellState.NEW)):
            plan = c.plan or {}
            if plan.get("design") == "v1":
                continue
            c.plan = {"legacy": plan}
            c.state, c.stop_reason = CellState.NEW, ""
            n += 1
        if n:
            ledger.log(s, "old_verdicts_reset", count=n)
    return n


# ---------- 第 1 步 选数据集 ----------
def choose_dataset(s: Session, config: FactoryConfig) -> str | None:
    """The best-scoring dataset not yet worked whose catalog coverage is known and at least
    `min_coverage`, or None when none is left. Unknown coverage (0 in the catalog) is not
    taken on trust."""
    floor = config.thresholds.min_coverage
    for r in datasets.rank(s, config):
        cell = s.get(CellORM, r.cell_id)
        if cell is not None and cell.coverage >= floor:
            return r.cell_id
    return None


# ---------- 第 2 步 读字段 ----------
def prep(field_id: str, field_type: str) -> str:
    """How a field enters every formula: vectors averaged (官方课), then carried for one
    quarter so quarterly data is not empty between releases (官方小贴士 10)."""
    base = f"vec_avg({field_id})" if field_type == "VECTOR" else field_id
    return f"ts_backfill({base}, {FILL_DAYS})"


def load_fields(api: BrainAPI, ledger: Ledger, cell_id: str, config: FactoryConfig) -> int:
    """Fetches the dataset's fields and keeps the numeric ones covering at least
    `min_coverage` of the universe. Returns how many are usable."""
    with ledger.session() as s:
        cell = _cell(s, cell_id)
        region, delay, universe, dataset = cell.region, cell.delay, cell.universe, cell.dataset_id
    rows = api.datafields(region, delay, universe, dataset)
    floor = config.thresholds.min_coverage
    usable = 0
    with ledger.session() as s:
        existing = {
            f.field_id: f for f in s.scalars(select(FieldORM).where(FieldORM.cell_id == cell_id))
        }
        for r in rows:
            field_type = str(r.get("type") or "")
            coverage = float(r.get("coverage") or 0.0)
            ok = field_type in NUMERIC_TYPES and coverage >= floor
            f = existing.get(str(r["id"])) or FieldORM(cell_id=cell_id, field_id=str(r["id"]))
            f.type = field_type
            f.description = str(r.get("description") or "")[:1000]
            f.coverage = coverage
            f.alpha_count = int(r.get("alphaCount") or 0)
            f.eligible = ok
            f.prep = prep(f.field_id, field_type) if ok else ""
            s.add(f)
            usable += ok
        cell = _cell(s, cell_id)
        cell.state = CellState.RESEARCH if usable else CellState.EXHAUSTED
        cell.plan = {**(cell.plan or {}), "design": "v1", "usable_fields": usable}
        if not usable:
            cell.stop_reason = f"no numeric field covers {floor:.0%} of the universe"
        ledger.log(s, "fields_loaded", cell_id, total=len(rows), usable=usable)
    return usable


def usable_fields(s: Session, cell_id: str) -> list[FieldORM]:
    return list(
        s.scalars(select(FieldORM).where(FieldORM.cell_id == cell_id, FieldORM.eligible.is_(True)))
    )


def _cell(s: Session, cell_id: str) -> CellORM:
    cell = s.get(CellORM, cell_id)
    if cell is None:
        raise KeyError(cell_id)
    return cell


def summary(ledger: Ledger, cell_id: str) -> dict[str, Any]:
    with ledger.session() as s:
        cell = _cell(s, cell_id)
        fields = usable_fields(s, cell_id)
        return {
            "dataset": cell.dataset_id,
            "region": cell.region,
            "universe": cell.universe,
            "usable_fields": len(fields),
            "examples": [f.prep for f in fields[:3]],
        }
