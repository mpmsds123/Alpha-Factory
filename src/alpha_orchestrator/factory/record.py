"""Intermediate-results page: what every step of design document v1 actually used and produced, straight from the
ledger. Nothing here is a summary written by the program; every row is stored data."""

# ruff: noqa: E501  (HTML table rows read better on one line)

import html
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select

from alpha_orchestrator.factory import refinement_trial
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CellORM,
    IdeaFieldORM,
    ResearchIdeaORM,
    ReviewORM,
    Stage,
)
from alpha_orchestrator.factory.study import DESIGN

STATUS = {"TESTING": "测试中", "PROMISING": "有苗头，改进中", "DEAD": "死胡同", "DONE": "做完"}
CSS = """
:root{--bg:#f6f4ef;--card:#fff;--ink:#1d2733;--soft:#6b7580;--ok:#2f7a4a;--bad:#ad3e2a;--hi:#c0632a}
@media (prefers-color-scheme:dark){:root{--bg:#141a21;--card:#1b232c;--ink:#e6ebf0;--soft:#97a3ae;
--ok:#8fd2a6;--bad:#f0998a;--hi:#f0a06b}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,"PingFang SC",sans-serif;
padding:20px 16px}
main{max-width:1100px;margin:0 auto;display:grid;gap:14px}
details{background:var(--card);border-radius:12px;padding:6px 14px}
summary{cursor:pointer;font-weight:700;padding:6px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:6px 0 12px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--bg);vertical-align:top}
th{color:var(--soft);font-weight:600}
code{font:12.5px ui-monospace,Menlo,monospace;word-break:break-all}
.ok{color:var(--ok);font-weight:700}.bad{color:var(--bad)}.soft{color:var(--soft)}
.step{display:inline-block;background:var(--hi);color:var(--card);border-radius:6px;padding:0 7px;
font-size:12px;font-weight:700;margin-right:6px}
"""


def _e(x: object) -> str:
    return html.escape("" if x is None else str(x))


def _num(x: float | None) -> str:
    return "–" if x is None else f"{x:.2f}"


def render_record(ledger: Ledger, path: Path) -> Path:
    """Render the research record page from raw ledger rows."""
    out = [f"<!doctype html><meta charset=utf-8><title>研究记录</title><style>{CSS}</style><main>"]
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    out.append(
        f"<h1>研究记录</h1><div class=soft>生成于 {now}。每一行都是数据库里的原始数据。</div>"
    )
    with ledger.session() as s:
        pairs = s.scalars(
            select(ResearchIdeaORM.extra).where(
                func.json_type(ResearchIdeaORM.extra, "$.refinement_trial") == "object"
            )
        )
        pair_counts = Counter(refinement_trial.outcome(row["refinement_trial"]) for row in pairs)
        if pair_counts:
            labels = {
                "llm_won": "LLM 胜",
                "rule_won": "规则胜",
                "same_choice": "同一改法",
                "same_outcome": "不同改法、结果相同",
                "pending": "待结算",
                "unusable": "缺可用证据",
                "abstained": "LLM 弃权",
            }
            summary = " · ".join(f"{label} {pair_counts[key]}" for key, label in labels.items())
            out.append(f"<h2>配对改进试验汇总</h2><div>{_e(summary)}</div>")
        cells = [
            c
            for c in s.scalars(select(CellORM))
            if (c.plan or {}).get("design") == DESIGN or (c.plan or {}).get("direction_trial")
        ]
        trial_counts: dict[tuple[int, str, int, str], list[int]] = defaultdict(
            lambda: [0, 0, 0, 0, 0]
        )
        for cell in cells:
            trial = cell.plan.get("direction_trial")
            if not trial:
                continue
            key = (trial["version"], trial["model"], trial["dataset_cap"], trial["arm"])
            counts = trial_counts[key]
            counts[0] += 1
            result = cell.plan.get("direction_result")
            if result is not None:
                counts[1] += 1
                counts[2] += int(result["regular_candidates"] > 0)
                counts[3] += result["settled"]
                counts[4] += result["regular_candidates"]
        if trial_counts:
            out.append(
                "<h2>研究方向试验</h2><table><tr><th>版本</th><th>模型</th><th>数据集预算</th>"
                "<th>选法</th><th>已分配</th><th>已收口</th><th>有过常规门槛的组合</th>"
                "<th>已收口组合的结算记录</th><th>过常规门槛的候选</th></tr>"
            )
            for key, counts in sorted(trial_counts.items()):
                arm = "LLM 定向" if key[-1] == "guided" else "随机对照"
                values = (*key[:-1], arm, *counts)
                out.append("<tr>" + "".join(f"<td>{_e(value)}</td>" for value in values) + "</tr>")
            out.append("</table>")
        cells.sort(key=lambda c: c.updated_at or datetime.min.replace(tzinfo=UTC), reverse=True)
        for cell in cells:
            ideas = list(
                s.scalars(
                    select(ResearchIdeaORM)
                    .where(ResearchIdeaORM.cell_id == cell.cell_id)
                    .order_by(ResearchIdeaORM.created_at)
                )
            )
            plan = cell.plan or {}
            out.append(
                f"<details open><summary><span class=step>第1-2步</span>{_e(cell.region)} · "
                f"{_e(cell.dataset_name or cell.dataset_id)}（{_e(cell.dataset_id)}）"
                f" — 可用字段 {_e(plan.get('usable_fields'))}，点子 {len(ideas)} 个，"
                f"回测 {cell.sims_used} 次{('，结束：' + _e(cell.stop_reason)) if cell.stop_reason else ''}"
                "</summary>"
            )
            trial = plan.get("direction_trial")
            if trial:
                proposal = trial["proposal"]
                selected_by = "LLM 定向" if trial["arm"] == "guided" else "随机对照"
                out.append(
                    f"<div>分配方式 {_e(selected_by)}，模型建议 {_e(proposal['cell_id'])}</div>"
                    f"<div>依据 {_e(proposal['metadata_basis'])}</div>"
                    f"<div>文献依据 {_e(proposal.get('theory_source') or '参考库中暂无匹配')}</div>"
                    f"<div>机制 {_e(proposal['mechanism'])}</div>"
                    f"<div>为什么先试 {_e(proposal['why_first'])}</div>"
                    f"<div>替代解释 {_e(proposal['competing_explanation'])}</div>"
                    f"<div>削弱假设的结果 {_e(proposal['falsifier'])}</div>"
                )
            else:
                out.append(f"<div class=soft>目录评分 <code>{_e(cell.rank_reason)}</code></div>")
            for idea in ideas:
                out.append(_idea_block(s, idea))
            out.append("</details>")
        reviews = list(s.scalars(select(ReviewORM).order_by(ReviewORM.created_at.desc())))
        total = s.scalar(select(func.count()).select_from(CandidateORM)) or 0
    out.append(f"<details open><summary>待你确认（{len(reviews)}）</summary><table>")
    out.append("<tr><th>平台编号</th><th>状态</th><th>成绩</th></tr>")
    for r in reviews:
        out.append(
            f"<tr><td>{_e(r.alpha_id)}</td><td>{_e(r.status)}</td><td><code>{_e(r.score)}</code></td></tr>"
        )
    out.append(f"</table></details><div class=soft>数据库里共 {total} 次回测记录。</div></main>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(out), encoding="utf-8")
    return path


def _idea_block(s, idea: ResearchIdeaORM) -> str:  # type: ignore[no-untyped-def]
    extra = idea.extra or {}
    probe = extra.get("direction_probe") or {}
    group = extra.get("group") or {}
    out = [
        f"<details><summary><span class=step>第3步</span>{_e(idea.title)} — "
        f"{_e(STATUS.get(idea.status, idea.status))}，最好 Sharpe {_num(idea.best_sharpe)}</summary>",
        "<table>",
        f"<tr><th>机制</th><td>{_e(idea.mechanism)}</td></tr>",
        f"<tr><th>做多 / 做空</th><td>{_e(idea.long_side)} / {_e(idea.short_side)}</td></tr>",
        f"<tr><th>看几天</th><td>{_e(idea.horizon_days)}</td></tr>",
        f"<tr><th>最简写法</th><td><code>{_e(idea.form)}</code></td></tr>",
        f"<tr><th>参考论文</th><td>{_e(', '.join(extra.get('papers', [])) or '无')}</td></tr>",
        f"<tr><th>削弱假设的结果</th><td>{_e(group.get('falsifier'))}</td></tr>",
        f"<tr><th>首条公式的样本内方向</th><td>{_e(probe.get('alignment'))} "
        f"{_num(probe.get('signed_sharpe'))}</td></tr>",
        f"<tr><th>为什么结束</th><td>{_e(extra.get('why_dead', ''))}</td></tr>",
        "</table>",
        f"<div><span class=step>第4步</span>关键词粗筛 {_e(extra.get('coarse'))} 个 → LLM 打 6 分以上 "
        f"{_e(extra.get('kept'))} 个</div>",
    ]
    rows = list(
        s.scalars(
            select(IdeaFieldORM)
            .where(IdeaFieldORM.idea_id == idea.id)
            .order_by(IdeaFieldORM.rank == 0, IdeaFieldORM.rank, IdeaFieldORM.score.desc())
        )
    )
    out.append(
        "<table><tr><th>排名</th><th>字段</th><th>LLM 分</th><th>理由</th><th>第5步 Sharpe</th><th>Fitness</th></tr>"
    )
    for r in rows[:80]:
        c = s.get(CandidateORM, r.candidate_id) if r.candidate_id else None
        sh = c.sharpe if c is not None else None
        cls = "ok" if sh is not None and abs(sh) > 1 else ""
        out.append(
            f"<tr><td>{r.rank or '–'}</td><td><code>{_e(r.field_id)}</code></td><td>{r.score}</td>"
            f"<td>{_e(r.reason)}</td><td class={cls}>{_num(sh)}</td>"
            f"<td>{_num(c.fitness if c is not None else None)}</td></tr>"
        )
    out.append("</table>")
    trial = extra.get("refinement_trial")
    if trial:
        proposal = trial["proposal"]
        out.append(
            "<h3>配对改进试验</h3>"
            f"<div>假设 {_e(proposal['hypothesis'])} · 替代解释 {_e(proposal['alternative'])}</div>"
            f"<div>预测 {_e(proposal['prediction'])} · 反证 {_e(proposal['falsifier'])}</div>"
            f"<div>行动依据 {_e(proposal['why'])}</div>"
            f"<div>目标检查 {_e(', '.join(trial['target_checks']))}</div>"
        )
        out.append("<table><tr><th>策略</th><th>改法</th><th>结算状态</th><th>Sharpe</th></tr>")
        for arm, fix in trial["choices"].items():
            result = (trial.get("result") or {}).get(arm, {})
            state = result.get("state", "待结算") if fix else "弃权"
            if state == "observed":
                state = "解除目标检查" if result["success"] else "未达预定判定条件"
            elif state == "unusable":
                state = "缺可用证据"
            out.append(
                f"<tr><td>{'LLM' if arm == 'llm' else '固定规则'}</td><td>{_e(fix)}</td>"
                f"<td>{_e(state)}</td><td>{_num(result.get('sharpe'))}</td></tr>"
            )
        out.append("</table>")
    for diagnosis in extra.get("diagnoses", []):
        if diagnosis.get("why"):
            out.append(
                f"<div>改进决策 {_e(diagnosis.get('decision'))} · {_e(diagnosis.get('problem'))}"
                f" · {_e(diagnosis['why'])}</div>"
            )
    cands = [
        c
        for c in s.scalars(
            select(CandidateORM).where(func.json_extract(CandidateORM.extra, "$.idea") == idea.id)
        )
        if c.stage in (Stage.IMPROVE, Stage.ROBUST)
    ]
    if cands:
        out.append(
            "<div><span class=step>第7-8步</span>改进和复测</div><table><tr><th>类型</th><th>改了什么</th>"
            "<th>公式</th><th>设置</th><th>Sharpe</th><th>Fitness</th><th>换手</th><th>没过的检查</th></tr>"
        )
        for c in sorted(cands, key=lambda c: c.created_at):
            kind = "改进" if c.stage == Stage.IMPROVE else f"复测·{c.role}"
            out.append(
                f"<tr><td>{kind}</td><td>{_e(c.change)}</td><td><code>{_e(c.expression)}</code></td>"
                f"<td>decay {_e(c.settings.get('decay'))} · {_e(c.settings.get('neutralization'))} · "
                f"{_e(c.settings.get('universe'))}</td><td>{_num(c.sharpe)}</td><td>{_num(c.fitness)}</td>"
                f"<td>{_num(c.turnover)}</td><td>{_e(c.fails)}</td></tr>"
            )
        out.append("</table>")
    passed = [
        c
        for c in s.scalars(
            select(CandidateORM).where(func.json_extract(CandidateORM.extra, "$.idea") == idea.id)
        )
        if (c.extra or {}).get("robust")
    ]
    for c in passed:
        rb = c.extra["robust"]
        mark = "<span class=ok>通过</span>" if rb.get("pass") else "<span class=bad>没通过</span>"
        out.append(
            f"<div><span class=step>第8步</span>{mark}：<code>{_e(c.expression)}</code> Sharpe "
            f"{_num(c.sharpe)}，运气线 {_e(rb.get('luck_line'))}（试了 {_e(rb.get('tries'))} 个），"
            f"复测保留比例 {_e(rb.get('ratios'))}</div>"
        )
    out.append("</details>")
    return "".join(out)
