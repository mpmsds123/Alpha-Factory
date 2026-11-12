"""v7 weekly report (every Friday): budget use, the idea funnel, the bottleneck, how each menu
fix has worked, and cost. Read-only on the ledger; one CSV, overwritten each time."""

from __future__ import annotations

import csv
import tempfile
from collections import defaultdict
from contextlib import suppress
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import select

from alpha_orchestrator.factory import fixes
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.engine import FIRST_GATE, NEAR_MISS
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import (
    BatchORM,
    CandidateORM,
    CellORM,
    LLMCallORM,
    ResearchIdeaORM,
    ReviewORM,
    Stage,
)

FILE_NAME = "工厂_周报.csv"
FUNNEL = (
    "点子有有效回测",
    "过运气线（首批最好 |Sharpe| ≥ 1.0）",
    "改进后 Sharpe ≥ 1.2",
    "过四项线（Sharpe、Fitness、换手、margin）",
    "回测内平台检查全过",
    "通过提交检查 = 进待挑选表",
)


def _aware(t: datetime) -> datetime:
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def _four_lines(c: CandidateORM, cell: CellORM, cfg: FactoryConfig) -> bool:
    t = cfg.thresholds
    if c.fitness is None or c.turnover is None:
        return False
    return (
        (c.sharpe or 0) >= cfg.sharpe_floor(cell.region)
        and c.fitness >= t.submit_fitness
        and t.turnover_min <= c.turnover <= t.turnover_max
        and (c.margin or 0) * 10_000 >= cfg.margin_floor(cell.region)
    )


def weekly_rows(
    ledger: Ledger, cfg: FactoryConfig, days: int = 7, now: datetime | None = None
) -> list[tuple[str, str]]:
    """Build the rows of the weekly report for the last `days` days."""
    end = now or datetime.now(UTC)
    start = end - timedelta(days=days)
    rows: list[tuple[str, str]] = [
        ("期间", f"{start.isoformat()} ≤ 时间 < {end.isoformat()}"),
    ]
    with ledger.session() as s:
        batches = list(
            s.scalars(
                select(BatchORM).where(
                    BatchORM.created_at >= start.astimezone(UTC).replace(tzinfo=None),
                    BatchORM.created_at < end.astimezone(UTC).replace(tzinfo=None),
                    BatchORM.status.not_in(["REFUSED", "REJECTED"]),
                )
            )
        )
        reserved = sum(b.size for b in batches)
        cohort = (
            list(
                s.scalars(
                    select(CandidateORM).where(CandidateORM.batch_id.in_([b.id for b in batches]))
                )
            )
            if batches
            else []
        )
        cohort_finished = [
            c for c in cohort if c.completed_at is not None and _aware(c.completed_at) < end
        ]
        done = [
            c
            for c in s.scalars(select(CandidateORM).where(CandidateORM.completed_at.is_not(None)))
            if c.sharpe is not None
            and c.completed_at is not None
            and start <= _aware(c.completed_at) < end
        ]
        cap = cfg.budget.v1_daily_cap * days
        rows += [
            ("回测完成", str(len(done))),
            (
                "期间进入待挑选",
                str(
                    len(
                        list(
                            s.scalars(
                                select(ReviewORM.id).where(
                                    ReviewORM.created_at
                                    >= start.astimezone(UTC).replace(tzinfo=None),
                                    ReviewORM.created_at < end.astimezone(UTC).replace(tzinfo=None),
                                )
                            )
                        )
                    )
                ),
            ),
            ("本期发起队列：配额预留", str(reserved)),
            ("额度利用率", f"{reserved}/{cap} = {reserved / cap:.1%}" if cap else "无配额"),
            ("本期发起队列：已结算", str(len(cohort_finished))),
            ("本期发起队列：有指标", str(sum(c.sharpe is not None for c in cohort_finished))),
            ("本期发起队列：未结算或未匹配", str(max(0, reserved - len(cohort_finished)))),
        ]
        ideas = [
            i
            for i in s.scalars(
                select(ResearchIdeaORM).where(
                    ResearchIdeaORM.created_at >= start.astimezone(UTC).replace(tzinfo=None),
                    ResearchIdeaORM.created_at < end.astimezone(UTC).replace(tzinfo=None),
                )
            )
        ]
        reviewed = set(
            s.scalars(
                select(ReviewORM.candidate_id).where(
                    ReviewORM.created_at < end.astimezone(UTC).replace(tzinfo=None)
                )
            )
        )
        rows.append(("本期新建点子", str(len(ideas))))
        fake = cfg.thresholds.suspicious_sharpe
        counts = [0] * len(FUNNEL)
        for i in ideas:
            cell = s.get(CellORM, i.cell_id)
            ids = list((i.extra or {}).get("rows", []))
            mine = [
                c
                for c in (s.get(CandidateORM, k) for k in ids)
                if c is not None
                and c.sharpe is not None
                and c.completed_at is not None
                and _aware(c.completed_at) < end
                and c.stage != Stage.ROBUST
                and abs(c.sharpe) <= fake  # |Sharpe| > 3 is an artefact, not a find
            ]
            first = [abs(c.sharpe or 0) for c in mine if c.stage == Stage.FIELD]
            reached = [
                bool(mine),
                bool(first) and max(first) >= FIRST_GATE,
                any((c.sharpe or 0) >= NEAR_MISS for c in mine),
                cell is not None and any(_four_lines(c, cell, cfg) for c in mine),
                cell is not None and any(_four_lines(c, cell, cfg) and not c.fails for c in mine),
                any(k in reviewed for k in ids),
            ]
            for k, ok in enumerate(reached):
                if not ok:
                    break
                counts[k] += 1
        for name, n in zip(FUNNEL, counts, strict=True):
            rows.append((f"漏斗：{name}", str(n)))
        # The bottleneck is the step with the lowest pass rate, not the largest loss: the
        # first steps always lose the most ideas because they are cheap screens.
        rates = [
            (counts[k + 1] / counts[k], FUNNEL[k + 1], counts[k], counts[k + 1])
            for k in range(1, len(FUNNEL) - 1)
            if counts[k]
        ]
        if rates:
            rate, step, before, after = min(rates)
            rows.append(("瓶颈（通过率最低的一步）", f"{step}：{after}/{before} = {rate:.0%}"))
        ready = counts[-1]
        rows.append(
            (
                "每 1000 次回测的可提交点子",
                f"{ready / len(done) * 1000:.2f}" if done else "无回测",
            )
        )
        cost = sum(
            float(c.cost_cny or 0)
            for c in s.scalars(select(LLMCallORM))
            if start <= _aware(c.created_at) < end
        )
        rows.append(("DeepSeek 花费（元）", f"{cost:.2f}"))
        rows.append(
            ("每个可提交点子的花费（元）", f"{cost / ready:.2f}" if ready else "还没有可提交点子")
        )
        table: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        for c in done:
            extra = c.extra or {}
            if c.stage != Stage.IMPROVE or "fix" not in extra or c.parent_id is None:
                continue
            parent = s.get(CandidateORM, c.parent_id)
            if parent is None:
                continue
            ok = fixes.success(c.fails, parent.fails)
            table[(extra.get("problem", ""), extra["fix"])][0 if ok else 1] += 1
            table[("大模型挑的" if extra.get("source") == "llm" else "老虎机兜底", "合计")][
                0 if ok else 1
            ] += 1
        for (problem, fix), (won, lost) in sorted(table.items()):
            rows.append((f"改法：{problem} → {fix}", f"成功 {won}/{won + lost}"))
    return rows


def friday_cutoff(cfg: FactoryConfig, now: datetime, ending: date | None = None) -> datetime:
    """Return the end of the most recent completed Friday report window."""
    tz = ZoneInfo(cfg.budget.quota_timezone)
    local = now.astimezone(tz)
    friday = ending or (local.date() - timedelta(days=(local.weekday() - 4) % 7))
    if friday.weekday() != 4:
        raise ValueError("week ending must be a Friday")
    end = datetime.combine(friday, time(cfg.budget.quota_reset_hour), tz)
    if local < end:
        if ending is not None:
            raise ValueError("the requested reporting week has not closed")
        end -= timedelta(days=7)
    return end


def write_weekly(
    ledger: Ledger, cfg: FactoryConfig, days: int = 7, ending: datetime | None = None
) -> Path:
    """Write the weekly report CSV and return its path."""
    filename = FILE_NAME if ending is None else f"工厂_周报_{ending:%Y-%m-%d}.csv"
    path = Path(cfg.reports_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if ending is not None and path.exists():
        return path
    rows = weekly_rows(ledger, cfg, days, now=ending)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False
    ) as f:
        tmp = Path(f.name)
        try:
            w = csv.writer(f)
            w.writerow(["项目", "数值"])
            w.writerows(rows)
            f.flush()
            if ending is None:
                tmp.replace(path)
            else:
                with suppress(FileExistsError):
                    path.hardlink_to(tmp)
        finally:
            tmp.unlink(missing_ok=True)
    return path
