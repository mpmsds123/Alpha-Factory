"""The research loop of 设计文档 v1 (Obsidian「Alpha工厂 工作台」), one step per function.

Every decision here cites the step of the design document it implements. A decision the
document does not contain does not belong in this file.
"""

import csv
import re
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.factory import datasets
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import CellORM, CellState, FieldORM

DESIGN = "v5"  # 10/15 晚: 侦察+有信号才加码; earlier verdicts do not count (reset_old_verdicts)
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
    """第 3 问: only datasets the template scanner found worth digging (first-pass hit rate
    >= 2%), highest hit rate first, each once under this design. Without the scanner's
    folder, the old ranking (best score, coverage known and >= min_coverage)."""
    if config.only_datasets:
        for cell_id in config.only_datasets:
            cell = s.get(CellORM, cell_id)
            if cell is None:
                continue
            if cell.state != CellState.NEW and (cell.plan or {}).get("design") == DESIGN:
                continue
            return cell.cell_id
        return None
    if config.scanner_runs:
        for w in scanner_worth(config.scanner_runs):
            cell = s.get(CellORM, f"{w['region']}|1|{w['universe']}|{w['dataset']}")
            if cell is None:
                continue
            if cell.state != CellState.NEW and (cell.plan or {}).get("design") == DESIGN:
                continue  # already dug under this design
            return cell.cell_id
        return None
    floor = config.thresholds.min_coverage
    evidence = region_evidence(s)
    judged = scanner_judged(config.scanner_progress)
    best: tuple[float, str] | None = None
    best_judged: tuple[float, str] | None = None
    for r in datasets.rank(s, config):
        cell = s.get(CellORM, r.cell_id)
        if cell is None or cell.coverage < floor:
            continue
        score = r.score * evidence.get(cell.region, 1.0)
        if best is None or score > best[0]:
            best = (score, r.cell_id)
        if (cell.region, cell.universe, cell.dataset_id) in judged and (
            best_judged is None or score > best_judged[0]
        ):
            best_judged = (score, r.cell_id)
    pick = best_judged or best
    return pick[1] if pick else None


WORTH_RATE = 0.02  # 第 3 问: the scanner's "worth digging" line (its SPRT keeps >= 2%)
FIRST_PASS_FILE = re.compile(r"^([A-Z]+)_(.+?)(?:_v\d+)?_1阶\.csv$")
FIELDS_FILE = re.compile(r"^fields_([A-Z]+)_([A-Z0-9]+)_(\d)_(.+)\.json$")


def scanner_worth(folder: str) -> list[dict[str, Any]]:
    """Datasets in the scanner's first-pass results with |Sharpe| >= 1 and turnover 1-70%
    in at least 2% of simulations, highest rate first. Each row also carries the scanner's
    regular-line count (Sharpe >= 1.58, Fitness >= 1) for the comparison (第 8 问)."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        return []
    universe: dict[tuple[str, str], str] = {}
    for f in root.iterdir():
        m = FIELDS_FILE.match(f.name)
        if m and m.group(3) == "1":
            universe[(m.group(1), m.group(4))] = m.group(2)
    rows: list[dict[str, Any]] = []
    for f in sorted(root.iterdir()):
        m = FIRST_PASS_FILE.match(f.name)
        if not m or (m.group(1), m.group(2)) not in universe:
            continue
        n = pp = regular = 0
        with f.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    sharpe, turnover = float(r["sharpe"]), float(r["turnover"])
                    fitness = float(r.get("fitness") or 0)
                except (KeyError, TypeError, ValueError):
                    continue
                n += 1
                if 0.01 <= turnover <= 0.70 and abs(sharpe) >= 1:
                    pp += 1
                    if abs(sharpe) >= 1.58 and abs(fitness) >= 1:
                        regular += 1
        if n and pp / n >= WORTH_RATE:
            region, dataset = m.group(1), m.group(2)
            rows.append(
                {
                    "region": region,
                    "universe": universe[(region, dataset)],
                    "dataset": dataset,
                    "sims": n,
                    "rate": pp / n,
                    "regular": regular,
                }
            )
    rows.sort(key=lambda r: -float(r["rate"]))
    return rows


SCANNER_JUDGED = ("止损", "值得跑")  # 提案 1: scanner verdicts that count for the comparison


def scanner_judged(path: str) -> set[tuple[str, str, str]]:
    """(region, universe, dataset) the template scanner has judged, read from its
    数据集进度.csv, so both methods are compared on the same datasets. Empty if no file."""
    if not path:
        return set()
    p = Path(path).expanduser()
    if not p.exists():
        return set()
    with p.open(encoding="utf-8", newline="") as f:
        return {
            (r["region"], r["universe"], r["dataset"])
            for r in csv.DictReader(f)
            if r.get("result") in SCANNER_JUDGED
        }


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
