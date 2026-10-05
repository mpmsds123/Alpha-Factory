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

DESIGN = "v2"  # verdicts reached under an earlier design do not count (reset_old_verdicts)
FILL_DAYS = 63  # 第 2 步: quarterly data carried one quarter (官方小贴士 10)
NUMERIC_TYPES = ("MATRIX", "VECTOR")


# ---------- before the first run ----------
def reset_old_verdicts(ledger: Ledger) -> int:
    """Datasets judged under an earlier design go back to NEW and their verdicts stop
    counting as evidence (that logic was wrong); their ideas are retired, so they are
    neither repeated as "already tried" nor fed back as dead ends. Simulations stay in the
    ledger and are reused when the same formula comes up again."""
    from alpha_orchestrator.factory.models import IdeaStatus, ResearchIdeaORM

    n = 0
    with ledger.session() as s:
        reset: set[str] = set()
        for c in s.scalars(select(CellORM).where(CellORM.state != CellState.NEW)):
            plan = c.plan or {}
            if plan.get("design") == DESIGN:
                continue
            c.plan = {"legacy": plan}
            c.state, c.stop_reason = CellState.NEW, ""
            reset.add(c.cell_id)
            n += 1
        for i in s.scalars(
            select(ResearchIdeaORM).where(ResearchIdeaORM.status != IdeaStatus.RETIRED)
        ):
            cell = s.get(CellORM, i.cell_id)
            if cell is None or (cell.plan or {}).get("design") != DESIGN or i.cell_id in reset:
                i.status = IdeaStatus.RETIRED
        if n:
            ledger.log(s, "old_verdicts_reset", count=n)
    return n


# ---------- 第 1 步 选数据集 ----------
def choose_dataset(s: Session, config: FactoryConfig) -> str | None:
    """The best-scoring dataset not yet worked whose catalog coverage is known and at least
    `min_coverage`, or None when none is left. Unknown coverage (0 in the catalog) is not
    taken on trust."""
    floor = config.thresholds.min_coverage
    evidence = region_evidence(s)
    best: tuple[float, str] | None = None
    for r in datasets.rank(s, config):
        cell = s.get(CellORM, r.cell_id)
        if cell is None or cell.coverage < floor:
            continue
        score = r.score * evidence.get(cell.region, 1.0)
        if best is None or score > best[0]:
            best = (score, r.cell_id)
    return best[1] if best else None


PRIOR_TESTS = 20  # weight of the overall rate for a region: about two batches of tests
# Like for like: only first formulas of an idea, never improved ones. Improvement (DIG,
# REFINE, ...) ran in some regions and not others, so it would credit the region with
# what the method did.
FIRST_PASS = ("IDEA", "SURVEY", "FIELD")


def region_evidence(s: Session) -> dict[str, float]:
    """How often a backtest in each region reached Sharpe > 1 and Fitness > 0.5 in this
    ledger, relative to all regions (1.0 = average). Shrunk toward the overall rate so a
    region with few tests is neither favoured nor ruled out on noise."""
    from alpha_orchestrator.factory.models import CandidateORM

    rows = s.execute(
        select(CellORM.region, CandidateORM.sharpe, CandidateORM.fitness, CandidateORM.turnover)
        .join(CellORM, CellORM.cell_id == CandidateORM.cell_id)
        .where(
            CandidateORM.status == "DONE",
            CandidateORM.sharpe.is_not(None),
            CandidateORM.stage.in_(FIRST_PASS),
        )
    ).all()
    hits: dict[str, int] = {}
    tests: dict[str, int] = {}
    for region, sh_, fit, turn in rows:
        sh = float(sh_ or 0.0)
        if abs(sh) > 3 or (turn or 0) > 0.7:
            continue
        tests[region] = tests.get(region, 0) + 1
        hits[region] = hits.get(region, 0) + int(
            abs(sh) > 1 and abs(fit or 0) > 0.5 and sh * (fit or 0) > 0
        )
    total = sum(tests.values())
    if not total:
        return {}
    overall = (sum(hits.values()) + 1) / (total + 2)
    return {
        r: ((hits[r] + PRIOR_TESTS * overall) / (tests[r] + PRIOR_TESTS)) / overall for r in tests
    }


# ---------- 第 2 步 读字段 ----------
# Released a few times a year: carried forward so they are not empty between releases
# (官方小贴士 10). Everything else (news, sentiment, price-volume, ...) is used as is:
# "新闻是最讲时效性的东西，你把它填充过来干嘛呢……不是什么都该填的" (官方零基础课).
LOW_FREQUENCY = frozenset(
    {"fundamental", "analyst", "earnings", "insiders", "institutions", "shortinterest"}
)


def prep(field_id: str, field_type: str, category: str = "fundamental") -> str:
    """How a field enters every formula: vectors averaged (官方课); low-frequency data
    carried for one quarter, everything else left as is."""
    base = f"vec_avg({field_id})" if field_type == "VECTOR" else field_id
    if category.lower() in LOW_FREQUENCY:
        return f"ts_backfill({base}, {FILL_DAYS})"
    return base


def load_fields(api: BrainAPI, ledger: Ledger, cell_id: str, config: FactoryConfig) -> int:
    """Fetches the dataset's fields and keeps the numeric ones covering at least
    `min_coverage` of the universe. Returns how many are usable."""
    with ledger.session() as s:
        cell = _cell(s, cell_id)
        region, delay, universe, dataset = cell.region, cell.delay, cell.universe, cell.dataset_id
        category = cell.category
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
            f.prep = prep(f.field_id, field_type, category) if ok else ""
            s.add(f)
            usable += ok
        cell = _cell(s, cell_id)
        cell.state = CellState.RESEARCH if usable else CellState.EXHAUSTED
        cell.plan = {**(cell.plan or {}), "design": DESIGN, "usable_fields": usable}
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
