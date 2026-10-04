"""Static local dashboard rendered from the ledger. Contains real expressions: stays local."""

import html
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from alpha_orchestrator.brain.session import SessionStore
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CellORM,
    EventORM,
    LLMCallORM,
    ReviewORM,
    ReviewStatus,
)
from alpha_orchestrator.factory.story import render_story

CSS = """
:root{--bg:#f4f6f8;--card:#fff;--ink:#17202a;--muted:#5e6b78;--line:#dfe5eb;--accent:#1f5fae;
--good:#1d7a4f;--bad:#b0412e;--warn:#99650b;--code:#eef2f6}
@media (prefers-color-scheme:dark){:root{--bg:#0f1419;--card:#161d24;--ink:#e4e9ee;--muted:#9aa6b2;
--line:#2a333d;--accent:#7fb0ec;--good:#6fcf9c;--bad:#ef8f7c;--warn:#e7b35c;--code:#1d2630}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 -apple-system,"PingFang SC","Noto Sans SC",system-ui,sans-serif;padding:24px}
main{max-width:1200px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:28px 0 10px}.muted{color:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px;margin-top:16px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.tile .v{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}.tile .l{color:var(--muted);font-size:12.5px}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px 10px;border-bottom:1px solid var(--line);
text-align:left;vertical-align:top}th{color:var(--muted);font-weight:500;background:var(--bg)}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
code{font:12px ui-monospace,Menlo,monospace;background:var(--code);padding:1px 5px;border-radius:4px;word-break:break-all}
.pill{display:inline-block;padding:0 8px;border-radius:99px;font-size:12px;font-weight:600}
.RESEARCH,.APPROVED,.ok{color:var(--good)}.EXHAUSTED,.RETIRED,.bad{color:var(--bad)}.STUDYING,.NEW,.PROPOSED,.warn{color:var(--warn)}
.story{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 16px 12px;margin:12px 0}
h3{font-size:15px;margin:10px 0 4px}h4{font-size:13.5px;margin:14px 0 4px;color:var(--accent)}
.alert{background:var(--card);border:1px solid var(--bad);border-radius:8px;padding:12px 14px;margin-top:14px}
"""


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _f(x: float | None, digits: int = 2) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def render(ledger: Ledger, config: FactoryConfig, out: Path) -> Path:
    start = datetime.combine(datetime.now(UTC).date(), datetime.min.time(), tzinfo=UTC)
    store = SessionStore()
    meta = store.meta()
    with ledger.session() as s:
        today = list(s.scalars(select(CandidateORM).where(CandidateORM.completed_at >= start)))
        reviews = list(
            s.scalars(
                select(ReviewORM)
                .where(ReviewORM.status == ReviewStatus.PENDING)
                .order_by(ReviewORM.created_at.desc())
            )
        )
        review_rows = []
        for r in reviews:
            c = s.get(CandidateORM, r.candidate_id)
            review_rows.append((r, c))
        cells = {c.cell_id: c for c in s.scalars(select(CellORM))}
        last_day = s.scalar(
            select(EventORM)
            .where(EventORM.kind == "day_done")
            .order_by(EventORM.id.desc())
            .limit(1)
        )
        tokens = s.execute(
            select(
                func.sum(LLMCallORM.prompt_tokens), func.sum(LLMCallORM.completion_tokens)
            ).where(LLMCallORM.created_at >= start)
        ).one()
        submitted = int(
            s.scalar(
                select(func.count())
                .select_from(ReviewORM)
                .where(ReviewORM.status == ReviewStatus.SUBMITTED)
            )
            or 0
        )
        events = list(s.scalars(select(EventORM).order_by(EventORM.id.desc()).limit(25)))
        story = render_story(s, config)

    stage_counts = Counter(c.stage for c in today)
    fails = Counter(f for c in today for f in (c.fails or "").split(",") if f)
    ready_today = sum(1 for c in today if c.extra.get("ready"))
    states = Counter(c.state for c in cells.values())
    quota = last_day.payload.get("quota") if last_day else None
    expires = meta.get("expires_at")
    session_line = (
        "需要刷脸：运行 <code>alpha-factory login</code>"
        if meta.get("pending_biometric")
        else (
            f"会话剩余约 {max(0, (float(expires) - time.time()) / 3600):.1f} 小时"
            if expires
            else "会话状态未知"
        )
    )

    tiles = [
        ("今日回测", len(today)),
        ("今日过门槛", ready_today),
        ("待你确认", len(reviews)),
        ("已提交（累计）", submitted),
        (
            "体检 / 普查 / 挖 / 已停",
            f"{states.get('STUDYING', 0)} / {states.get('SURVEY', 0)} / {states.get('RESEARCH', 0)} / "
            f"{states.get('EXHAUSTED', 0) + states.get('SATURATED', 0)}",
        ),
        ("平台剩余配额", quota if quota is not None else "–"),
        ("今日 LLM tokens", f"{(tokens[0] or 0) + (tokens[1] or 0):,}"),
    ]
    parts = [
        f"<title>Alpha 工厂面板</title><style>{CSS}</style><main>",
        "<h1>Alpha 工厂</h1>",
        f"<div class='muted'>更新于 {datetime.now().strftime('%Y-%m-%d %H:%M')} · {session_line}</div>",
    ]
    if meta.get("pending_biometric"):
        parts.append(
            "<div class='alert'>平台要求刷脸，流水线已暂停。运行 <code>alpha-factory login</code>"
            " 完成验证后会从断点继续。</div>"
        )
    parts.append(
        "<div class='tiles'>"
        + "".join(
            f"<div class='tile'><div class='v'>{_e(v)}</div><div class='l'>{_e(label)}</div></div>"
            for label, v in tiles
        )
        + "</div>"
    )

    parts.append(story)
    parts.append("<h2>待你确认</h2>")
    if review_rows:
        rows = "".join(
            f"<tr><td><code>{_e(r.id[:8])}</code></td><td>{_e(c.cell_id.split('|')[-1] if c else '')}<br>"
            f"<span class='muted'>{_e(c.cell_id.rsplit('|', 1)[0] if c else '')}</span></td>"
            f"<td><code>{_e(c.expression if c else '')}</code><div class='muted'>{_e(r.description[:400])}</div></td>"
            f"<td class='n'>{_f(c.sharpe if c else None)}</td><td class='n'>{_f(c.fitness if c else None)}</td>"
            f"<td class='n'>{_f((c.turnover or 0) * 100 if c else None, 1)}%</td>"
            f"<td class='n'>{_f((c.margin or 0) * 1e4 if c else None, 1)}</td>"
            f"<td class='n'>{_f(r.score.get('prod_corr'))}</td>"
            f"<td>{'ATOM' if r.score.get('atom') else ''}</td></tr>"
            for r, c in review_rows
        )
        parts.append(
            "<div class='wrap'><table><tr><th>ID</th><th>数据集</th><th>表达式 / 说明</th>"
            "<th>Sharpe</th><th>Fitness</th><th>换手</th><th>Margin bps</th>"
            f"<th>Prod corr</th><th></th></tr>{rows}</table></div>"
            "<p class='muted'>确认：<code>alpha-factory review approve ID</code> · "
            "放弃：<code>alpha-factory review reject ID</code></p>"
        )
    else:
        parts.append("<p class='muted'>暂无。</p>")

    parts.append(
        "<h2>今日漏斗与失败原因</h2><div class='tiles'>"
        + "".join(
            f"<div class='tile'><div class='v'>{n}</div><div class='l'>{_e(stage)}</div></div>"
            for stage, n in stage_counts.most_common()
        )
        + "</div>"
    )
    if fails:
        parts.append(
            "<div class='wrap' style='margin-top:10px'><table><tr><th>失败检查</th><th>次数</th></tr>"
            + "".join(
                f"<tr><td>{_e(k)}</td><td class='n'>{v}</td></tr>" for k, v in fails.most_common(10)
            )
            + "</table></div>"
        )

    parts.append("<h2>数据集队列（回测前按资料排序）</h2>")
    queue = sorted(cells.values(), key=lambda c: -c.rank_score)[:60]
    rows = "".join(
        f"<tr><td>{_e(c.region)}</td><td>{_e(c.dataset_id)}</td><td>{_e(c.category)}</td>"
        f"<td><span class='pill {c.state}'>{_e(c.state)}</span>"
        f"<div class='muted'>{_e(c.stop_reason)}</div></td>"
        f"<td class='n'>{c.rank_score:.3f}</td><td class='n'>{c.paper_refs}</td>"
        f"<td class='n'>{c.multiplier:.2f}</td><td class='n'>{c.alpha_count}</td>"
        f"<td class='muted'>{_e(', '.join(f'{k} {v}' for k, v in (c.rank_reason or {}).items()))}</td></tr>"
        for c in queue
    )
    parts.append(
        "<div class='wrap'><table><tr><th>地区</th><th>数据集</th><th>类别</th><th>状态</th>"
        "<th>得分</th><th>论文提及</th><th>乘数</th><th>已有 alpha</th><th>各项因子</th>"
        f"</tr>{rows}</table></div>"
    )

    parts.append(
        "<h2>最近事件</h2><div class='wrap'><table>"
        + "".join(
            f"<tr><td class='muted'>{_e(e.at.strftime('%m-%d %H:%M'))}</td><td>{_e(e.kind)}</td>"
            f"<td>{_e(e.cell_id or '')}</td><td class='muted'>{_e(str(e.payload)[:160])}</td></tr>"
            for e in events
        )
        + "</table></div></main>"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
