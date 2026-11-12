"""The factory funnel CSVs: how many ideas and simulations enter and survive each stage, and
why the others were lost. One row per allocation day, counted by the day each idea was
created (a cohort), so a day's stages can be read left to right as conversion rates.
Everything is recomputed from the ledger, so past days appear without any extra logging.
"""

import csv
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import CandidateORM, ResearchIdeaORM, ReviewORM
from alpha_orchestrator.factory.progress import _day

FILE_NAME = "工厂_漏斗.csv"
REASONS_FILE = "工厂_漏斗_原因.csv"
COLUMNS = [
    "日期",
    "新点子",
    "点子用扫描器字段",
    "写出公式",
    "过本地检查",
    "回测完成",
    "回测报错",
    "过第一道门槛",
    "第一道门槛通过率",
    "过常规线",
    "过稳健性",
    "过平台检查",
    "进待挑选",
    "更新时间",
]
REASON_COLUMNS = ["日期", "阶段", "原因", "次数"]
# Bands of the best first-formula |Sharpe| of an idea that died at the first gate: they show
# whether ideas missed the gate narrowly (fixable input) or were pure noise.
GATE_BANDS = (0.3, 0.5, 0.8, 1.0)
ERROR_CHARS = 60  # enough of a platform error to tell the kinds apart without the details
FIRST_STAGE = "FIELD"


def _band(x: float | None) -> str:
    if x is None:
        return "no usable result"
    low = 0.0
    for high in GATE_BANDS:
        if x < high:
            return f"best |Sharpe| {low}-{high}"
        low = high
    return f"best |Sharpe| >= {GATE_BANDS[-1]}"


def _passed_gate(idea: ResearchIdeaORM) -> bool:
    extra = idea.extra or {}
    return (
        str(idea.status) in ("DONE", "PROMISING")
        or "parents_from" in extra
        or str(extra.get("why_dead", "")).startswith("after improvement")
    )


def _error_kind(text: str | None) -> str:
    words = "".join(ch if not ch.isdigit() else "#" for ch in (text or "unknown error"))
    return words[:ERROR_CHARS]


def compute(ledger: Ledger, cfg: FactoryConfig) -> tuple[dict[str, Counter[str]], Counter[Any]]:
    """Per-day stage counts and (day, stage, reason) loss counts."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    reasons: Counter[Any] = Counter()
    suspicious = cfg.thresholds.suspicious_sharpe
    with ledger.session() as s:
        cands = {c.id: c for c in s.scalars(select(CandidateORM))}
        reviewed = set(s.scalars(select(ReviewORM.candidate_id)))
        idea_day: dict[str, str] = {}
        for idea in s.scalars(select(ResearchIdeaORM)):
            if str(idea.status) == "RETIRED":
                continue
            d = _day(idea.created_at, cfg)
            idea_day[idea.id] = d
            extra = idea.extra or {}
            n = counts[d]
            n["新点子"] += 1
            n["点子用扫描器字段"] += bool(extra.get("scanner_fields"))
            n["写出公式"] += len(extra.get("formulas") or [])
            rows = [cands[i] for i in extra.get("rows") or [] if i in cands]
            first = [c for c in rows if c.stage == FIRST_STAGE and c.change != "mirrored"]
            n["过本地检查"] += len(first)
            best: float | None = None
            for c in first:
                if str(c.status) == "DONE" and c.sharpe is not None:
                    n["回测完成"] += 1
                    if abs(c.sharpe) <= suspicious:
                        best = max(best or 0.0, abs(c.sharpe))
                elif str(c.status) == "FAILED":
                    n["回测报错"] += 1
                    reasons[(d, "回测", _error_kind(c.error))] += 1
            if not extra.get("formulas") and str(idea.status) == "DEAD":
                reasons[(d, "写公式", str(extra.get("why_dead", ""))[:ERROR_CHARS])] += 1
            if _passed_gate(idea):
                n["过第一道门槛"] += 1
                if str(idea.status) == "DEAD":
                    done = [c for c in rows if c.sharpe is not None and c.fails]
                    top = max(done, key=lambda c: c.sharpe or 0.0, default=None)
                    first_fail = (top.fails or "").split(",")[0] if top else "no result"
                    reasons[(d, "改进", f"best formula still fails {first_fail}")] += 1
            elif str(idea.status) == "DEAD" and extra.get("formulas"):
                reasons[(d, "第一道门槛", _band(best))] += 1
        for c in cands.values():
            extra = c.extra or {}
            if not extra.get("ready"):
                continue
            owner = str(extra.get("idea") or "")
            day: str | None = idea_day.get(owner)
            if day is None and c.completed_at is not None:
                day = _day(c.completed_at, cfg)
            if day is None:
                continue
            d = day
            n = counts[d]
            n["过常规线"] += 1
            robust = extra.get("robust")
            if robust is not None and robust.get("pass"):
                n["过稳健性"] += 1
            elif robust is not None:
                reasons[(d, "稳健性", "Sharpe dropped under the other neutralization")] += 1
            state = extra.get("platform_check_state")
            if state == "passed":
                n["过平台检查"] += 1
            for name in extra.get("platform_failed") or []:
                reasons[(d, "平台检查", str(name))] += 1
            if state in (None, "pending", "deferred_quota") and robust and robust.get("pass"):
                reasons[(d, "平台检查", f"not checked yet ({state or 'waiting'})")] += 1
            n["进待挑选"] += c.id in reviewed
    return counts, reasons


def _write(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def write_funnel(ledger: Ledger, cfg: FactoryConfig) -> Path:
    """Write both funnel CSVs into the reports folder; return the counts file."""
    counts, reasons = compute(ledger, cfg)
    today = _day(datetime.now(UTC), cfg)
    now = datetime.now(ZoneInfo(cfg.budget.quota_timezone)).strftime("%m-%d %H:%M")
    rows = []
    for d in sorted(counts, reverse=True):
        n = counts[d]
        tested = n["新点子"]
        rows.append(
            {
                **{k: n.get(k, 0) for k in COLUMNS[1:-1]},
                "日期": d,
                "第一道门槛通过率": f"{n['过第一道门槛'] / tested:.0%}" if tested else "",
                "更新时间": now if d == today else "",
            }
        )
    out = Path(cfg.reports_dir)
    _write(out / FILE_NAME, COLUMNS, rows)
    reason_rows = [
        {"日期": d, "阶段": stage, "原因": why, "次数": k}
        for (d, stage, why), k in sorted(reasons.items(), key=lambda x: (x[0][0], x[0][1], -x[1]))
    ]
    reason_rows.sort(key=lambda r: r["日期"], reverse=True)
    _write(out / REASONS_FILE, REASON_COLUMNS, reason_rows)
    return out / FILE_NAME
