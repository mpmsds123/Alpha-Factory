"""The factory progress CSV: one row per allocation day, so the factory's state is visible
from the folder the same way the scanner's runs/*.csv are. Numbers come from the ledger;
the status column is what the latest invocation reported (kept from the previous file for
past days).
"""

import csv
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import select

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CellORM,
    EventORM,
    LLMCallORM,
    ResearchIdeaORM,
)

FILE_NAME = "工厂_进度.csv"
COLUMNS = [
    "日期",
    "状态",
    "回测完成",
    "每日上限",
    "过第一道门槛",
    "可提交",
    "当前数据集",
    "当前点子",
    "DeepSeek花费(元)",
    "更新时间",
]


def _aware(t: datetime) -> datetime:
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def _day(t: datetime, cfg: FactoryConfig) -> str:
    """The allocation day a timestamp belongs to, labelled by the date it ends on."""
    local = _aware(t).astimezone(ZoneInfo(cfg.budget.quota_timezone))
    return (local + timedelta(hours=24 - cfg.budget.quota_reset_hour)).date().isoformat()


def first_gate(c: CandidateORM, cfg: FactoryConfig) -> bool:
    th = cfg.thresholds
    if c.sharpe is None or c.fitness is None or c.turnover is None:
        return False
    line = th.submit_sharpe_chn if (c.cell_id or "").startswith("CHN") else th.submit_sharpe
    return (
        c.sharpe >= line
        and c.fitness >= th.submit_fitness
        and 0.01 <= c.turnover <= th.suspicious_turnover
        and c.sharpe <= th.suspicious_sharpe
    )


def no_platform_fail(c: CandidateORM) -> bool:
    checks = c.checks
    if isinstance(checks, str):
        try:
            checks = json.loads(checks)
        except ValueError:
            return False
    if not isinstance(checks, list) or not checks:
        return False
    return not any(isinstance(x, dict) and x.get("result") == "FAIL" for x in checks)


def _old_status(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["日期"]: r.get("状态", "") for r in csv.DictReader(f) if r.get("日期")}


def write_progress(ledger: Ledger, cfg: FactoryConfig, status: str) -> Path:
    path = Path(cfg.reports_dir) / FILE_NAME
    statuses = _old_status(path)
    done: dict[str, int] = defaultdict(int)
    gate: dict[str, int] = defaultdict(int)
    ready: dict[str, int] = defaultdict(int)
    cost: dict[str, float] = defaultdict(float)
    with ledger.session() as s:
        for c in s.scalars(select(CandidateORM).where(CandidateORM.completed_at.is_not(None))):
            if c.sharpe is None or c.completed_at is None:
                continue
            d = _day(c.completed_at, cfg)
            done[d] += 1
            if first_gate(c, cfg):
                gate[d] += 1
                if no_platform_fail(c):
                    ready[d] += 1
        for call in s.scalars(select(LLMCallORM)):
            cost[_day(call.created_at, cfg)] += float(call.cost_cny or 0.0)
        opened = s.scalars(
            select(EventORM).where(EventORM.kind == "dataset_opened").order_by(EventORM.id.desc())
        ).first()
        dataset = ""
        if opened is not None and opened.cell_id:
            cell = s.get(CellORM, opened.cell_id)
            dataset = opened.cell_id if cell is None else f"{cell.region} {cell.dataset_id}"
        idea = s.scalars(
            select(ResearchIdeaORM)
            .where(ResearchIdeaORM.status == "TESTING")
            .order_by(ResearchIdeaORM.created_at.desc())
        ).first()
    today = _day(datetime.now(UTC), cfg)
    statuses[today] = status
    days = sorted(set(done) | set(cost) | set(statuses), reverse=True)
    now = datetime.now(ZoneInfo(cfg.budget.quota_timezone)).strftime("%m-%d %H:%M")
    rows = []
    for d in days:
        live = d == today
        rows.append(
            {
                "日期": d,
                "状态": statuses.get(d, ""),
                "回测完成": done.get(d, 0),
                "每日上限": cfg.budget.v1_daily_cap,
                "过第一道门槛": gate.get(d, 0),
                "可提交": ready.get(d, 0),
                "当前数据集": dataset if live else "",
                "当前点子": (idea.title if idea is not None else "") if live else "",
                "DeepSeek花费(元)": round(cost.get(d, 0.0), 2),
                "更新时间": now if live else "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)
    return path
