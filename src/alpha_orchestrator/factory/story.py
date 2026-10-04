"""The research log: what the pipeline did for each dataset, step by step, and why, in plain
words. Every sentence is built from ledger rows, so it can be checked, not trusted."""

import html
import re
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.datasets import PAPER_SOURCES, PAPER_TITLE
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CellORM,
    CellState,
    FieldORM,
    HypothesisORM,
    IdeaORM,
    Stage,
)

STEP = {
    CellState.STUDYING: "第 2 步：读数据",
    CellState.RESEARCH: "第 3–4 步：试点子、看结果",
    CellState.EXHAUSTED: "已停",
    CellState.SATURATED: "已交满",
}
STAGE_WORDS = {
    Stage.PROFILE: "读数据的探测",
    Stage.IDEA: "点子展开的表达式",
    Stage.DIG: "在矿附近挖的变体",
    Stage.SECOND: "分组比较版本",
    Stage.THIRD: "加交易条件版本",
    Stage.FLIP: "取反版本",
    Stage.REPAIR: "修检查的版本",
    Stage.ROBUST: "提交前的稳健性检验",
}
VERDICT_WORDS = {
    "dig": "有矿，在附近深挖",
    "flip": "方向反了，取反再试",
    "fix": "有信号但卡在某项检查，修",
    "drop": "没信号，放弃这个点子",
}
CADENCE_WORDS = {
    "daily": "几乎每天变",
    "weekly": "大约每周变",
    "monthly": "大约每月变",
    "quarterly": "大约每季度变",
    "slow": "很少变",
}


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def now_line(s: Session) -> str:
    pending = Counter(
        c.stage
        for c in s.scalars(
            select(CandidateORM).where(CandidateORM.status.in_(["QUEUED", "RUNNING"]))
        )
    )
    if not pending:
        return "现在没有排队的回测。"
    parts = [f"{n} 条{STAGE_WORDS.get(stage, stage)}" for stage, n in pending.most_common()]
    return "现在排队/在跑：" + "，".join(parts) + "。"


def why_chosen(s: Session, cell: CellORM, place: int, total: int, cfg: FactoryConfig) -> str:
    r = cell.rank_reason or {}
    lines = []
    if cell.paper_refs:
        pattern = re.compile(rf"(?<![\w]){re.escape(cell.dataset_id)}(?![\w])", re.I)
        titles = [
            i.title
            for i in s.scalars(select(IdeaORM))
            if (i.source in PAPER_SOURCES or PAPER_TITLE.search(i.title))
            and pattern.search(f"{i.title}\n{i.body}")
        ][:4]
        lines.append(
            f"官方论文库里有 {cell.paper_refs} 篇论文点名这个数据集（×{r.get('papers')}）："
            + "；".join(f"《{_e(t)}》" for t in titles)
        )
    else:
        lines.append("没有论文点名它（×1.0）")
    lines.append(f"平台给这个类别的奖励乘数 {cell.multiplier:.2f}")
    gap = r.get("pyramid_gap", 1.0)
    lines.append(
        {
            1.15: "你在这个地区这个类别还没交过（×1.15）",
            1.2: "你在这个地区这个类别交过 1 个，再交 2 个凑满一组（×1.2）",
            1.3: "再交 1 个就凑满一组（×1.3）",
        }.get(gap, "这个地区这个类别已凑满（×1.0）")
    )
    lines.append(
        f"已有 {cell.alpha_count:,} 个 alpha 用过它，"
        f"用的人越多越难和别人不重复（×{r.get('crowding')}）"
    )
    submitted, mine = r.get("submitted_total", 0), r.get("submitted_category", 0)
    if not submitted:
        lines.append(
            "平台的金字塔记录里你本季度还没有提交（IQC 期间交的不在里面），"
            f"所以每个类别都按「缺」算（×{r.get('portfolio_gap')}）"
        )
    else:
        lines.append(
            f"本季度你交的 {submitted} 个里这个类别占 {mine} 个（×{r.get('portfolio_gap')}）"
        )
    lines.append(
        f"{cell.region} 的手续费门槛 {cfg.margin_floor(cell.region):.0f} bps"
        f"（×{r.get('cost')}），"
        f"Sharpe 门槛 {cfg.sharpe_floor(cell.region)}（×{r.get('difficulty')}）"
    )
    return (
        f"<p>在 {total:,} 个「地区 × 数据集」里排第 {place}，得分 {cell.rank_score:.2f}"
        "（各项相乘）。还没花任何回测，依据是：</p><ul>"
        + "".join(f"<li>{line}</li>" for line in lines)
        + "</ul>"
    )


def data_reading(s: Session, cell: CellORM) -> str:
    probes = list(
        s.scalars(
            select(CandidateORM).where(
                CandidateORM.cell_id == cell.cell_id, CandidateORM.stage == Stage.PROFILE
            )
        )
    )
    done = sum(c.status in ("DONE", "FAILED") for c in probes)
    fields = list(s.scalars(select(FieldORM).where(FieldORM.cell_id == cell.cell_id)))
    head = (
        f"<p>{len(fields)} 个字段，按名字相近分成 "
        f"{len({f.family for f in fields})} 组；挑最大的几组各一个代表，每个跑 6 次探测"
        f"（只看有多少股票有数据、多久更新一次、数值多大，不看赚不赚钱）。"
        f"已完成 {done}/{len(probes)} 次。</p>"
    )
    if cell.state == CellState.STUDYING:
        return head + "<p class='muted'>探测跑完后，这里会列出每组字段读出来的样子。</p>"
    reps = [f for f in fields if f.representative and f.profile]
    rows = "".join(
        f"<tr><td><code>{_e(f.field_id)}</code>"
        f"<div class='muted'>{_e(f.description[:120])}</div></td>"
        f"<td class='n'>{f.profile.get('coverage', 0):.0%}</td>"
        f"<td>{CADENCE_WORDS.get(str(f.profile.get('cadence')), '')}"
        f"（约 {f.profile.get('period_days')} 天）</td>"
        f"<td>{'在 ±1 以内' if f.profile.get('normalised') else '超出 ±1，未缩放'}</td>"
        f"<td><code>{_e(f.prep)}</code></td></tr>"
        for f in reps
    )
    return (
        head
        + "<div class='wrap'><table><tr><th>代表字段</th><th>有数据的股票</th><th>更新频率</th>"
        "<th>数值范围</th><th>所以进模版时写成</th></tr>" + rows + "</table></div>"
    )


def ideas_and_rounds(s: Session, cell: CellORM, ore: float) -> str:
    hyps = [
        h
        for h in s.scalars(select(HypothesisORM).where(HypothesisORM.cell_id == cell.cell_id))
        if h.rounds or h.status == "PROPOSED"
    ]
    if not hyps:
        return "<p class='muted'>还没有点子（读完数据后 DeepSeek 会提）。</p>"
    rounds: dict[str, dict[int, list[CandidateORM]]] = defaultdict(lambda: defaultdict(list))
    for c in s.scalars(
        select(CandidateORM).where(
            CandidateORM.cell_id == cell.cell_id, CandidateORM.hypothesis_id.is_not(None)
        )
    ):
        if c.stage != Stage.ROBUST and c.extra.get("round"):
            rounds[c.hypothesis_id or ""][int(c.extra["round"])].append(c)
    out = []
    for h in hyps:
        verdicts = {v.get("round"): v for v in h.verdicts or []}
        steps = []
        for k in sorted(rounds[h.id]):
            rows = rounds[h.id][k]
            sharpes = [c.sharpe for c in rows if c.sharpe is not None]
            waiting = sum(c.status in ("QUEUED", "RUNNING") for c in rows)
            best = max(rows, key=lambda c: abs(c.sharpe or 0), default=None)
            line = f"第 {k} 轮：{len(rows)} 条"
            if waiting:
                line += f"（还有 {waiting} 条在跑）"
            if sharpes:
                hits = sum(abs(x) >= ore for x in sharpes)
                line += (
                    f"，|Sharpe| ≥ {ore} 的有 {hits} 条，最好 {best.sharpe:+.2f}"  # type: ignore[union-attr]
                    f"：<code>{_e(best.expression if best else '')}</code>"
                )
            v = verdicts.get(k)
            if v:
                line += (
                    f"<br>→ DeepSeek 判断：<b>{VERDICT_WORDS.get(v['verdict'], v['verdict'])}</b>。"
                    f"{_e(v.get('reason'))}"
                )
            steps.append(f"<li>{line}</li>")
        out.append(
            f"<div class='tile' style='margin-bottom:10px'><b>{_e(h.title)}</b> "
            f"<span class='pill {h.status}'>{_e(h.status)}</span>"
            f"<p>{_e(h.statement)}</p>"
            f"<div class='muted'>预期方向：{_e(h.direction)} · "
            f"依据：{_e('；'.join(h.sources))}</div>"
            f"<div>模版 <code>{_e(h.template)}</code> · 窗口 {_e(h.windows)} · "
            f"设置 {_e(h.settings)}</div><ul>{''.join(steps)}</ul></div>"
        )
    return "".join(out)


def render_story(s: Session, cfg: FactoryConfig) -> str:
    cells = list(s.scalars(select(CellORM)))
    ranked = sorted(cells, key=lambda c: -c.rank_score)
    place = {c.cell_id: i + 1 for i, c in enumerate(ranked)}
    shown = [c for c in ranked if c.state in (CellState.STUDYING, CellState.RESEARCH)]
    shown += sorted(
        (c for c in cells if c.state in (CellState.EXHAUSTED, CellState.SATURATED) and c.sims_used),
        key=lambda c: c.updated_at,
        reverse=True,
    )[:3]
    parts = [
        "<h2>研究日志</h2>",
        f"<p><b>{now_line(s)}</b></p>",
        "<p class='muted'>流程：① 回测前按资料给数据集排序 → ② 用几十次探测读懂数据 → "
        "③ DeepSeek 读数据和论文提点子 → ④ 每个点子展开回测，DeepSeek 看结果决定深挖/取反/修/放弃"
        " → 连续 2 轮没矿就换下一个数据集 → ⑤ 过门槛的做稳健性检验，至多 4 条交给你确认。</p>",
    ]
    if not shown:
        parts.append("<p class='muted'>还没有开始做的数据集。</p>")
    for c in shown:
        status = STEP.get(c.state, c.state)
        if c.stop_reason:
            status += f"：{_e(c.stop_reason)}"
        parts.append(
            f"<div class='story'><h3>{_e(c.region)} · {_e(c.dataset_id)} "
            f"<span class='muted'>（{_e(c.dataset_name or c.category)}）· {status} · "
            f"已用 {c.sims_used} 次回测</span></h3>"
            f"<h4>① 为什么先做它</h4>{why_chosen(s, c, place[c.cell_id], len(cells), cfg)}"
            f"<h4>② 读数据</h4>{data_reading(s, c)}"
            f"<h4>③④ 点子和每轮结果</h4>{ideas_and_rounds(s, c, cfg.thresholds.ore_sharpe)}</div>"
        )
    return "".join(parts)
