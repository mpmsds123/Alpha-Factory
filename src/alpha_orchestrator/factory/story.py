"""The research log: what the pipeline did for each dataset, step by step, and why, in plain
words. Every sentence is built from ledger rows, so it can be checked, not trusted."""

import html
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha_orchestrator.factory import stats
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.models import CandidateORM, CellORM, CellState, FieldORM, Stage

PHASE = {
    "health_cov": "体检：先测覆盖率",
    "health_period": "体检：测更新周期",
    "survey": "普查",
    "research": "按信号浓度挖",
    "done": "已结束",
}
OUTCOME = {
    "noise": "普查判定：和噪声没有区别",
    "signal": "有信号，已挖完",
    "no_eligible": "覆盖率不够，没花更多回测",
    "no_fields": "没有可用字段",
    "budget": "用完了这个数据集的回测预算",
}
STAGE_WORDS = {
    Stage.PROFILE: "体检探针",
    Stage.SURVEY: "普查公式",
    Stage.REFINE: "挖深（结构）",
    Stage.COMBO: "挖组合",
    Stage.REPAIR: "按规则修",
    Stage.ROBUST: "稳健性检验",
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


def _health(s: Session, cell: CellORM) -> str:
    reps = list(
        s.scalars(
            select(FieldORM).where(
                FieldORM.cell_id == cell.cell_id, FieldORM.representative.is_(True)
            )
        )
    )
    plan = cell.plan or {}
    measured = [f for f in reps if (f.profile or {}).get("max_coverage") is not None]
    eligible = sum(1 for f in reps if f.eligible and f.filled_coverage is not None)
    rows = "".join(
        f"<tr><td><code>{_e(f.field_id)}</code></td><td>{_e(f.stratum)}</td>"
        f"<td class='n'>{f.profile.get('period_days', '–')}</td>"
        f"<td class='n'>{float(f.profile.get('raw_coverage') or 0):.0%}</td>"
        f"<td class='n'>{float(f.profile.get('filled_coverage') or 0):.0%}</td>"
        f"<td>{'合格' if f.eligible else '不合格'}</td></tr>"
        for f in measured[:20]
    )
    return (
        f"<p>{plan.get('family_count', len(reps))} 个家族，分成 {plan.get('strata', '?')} 层；"
        f"抽 {len(plan.get('health_sample', []))} 个家族体检，其余按所在层推断。"
        f"合格（补齐后覆盖率达标）的家族：{eligible}。</p>"
        + (
            "<div class='wrap'><table><tr><th>抽到的字段</th><th>层</th><th>更新周期（天）</th>"
            f"<th>每天覆盖</th><th>补齐后覆盖</th><th></th></tr>{rows}</table></div>"
            if rows
            else ""
        )
    )


def _survey(cell: CellORM) -> str:
    plan = cell.plan or {}
    sp = plan.get("sprt") or {}
    if not plan.get("survey_order"):
        return "<p class='muted'>还没开始。</p>"
    verdict = {"signal": "有信号", "noise": "和噪声没区别", None: "还在测"}.get(
        plan.get("verdict"), plan.get("verdict")
    )
    return (
        f"<p>抽了 {len(plan.get('survey_order', []))} 个家族，每个 4 种形态；已出结果 "
        f"{sp.get('n', 0)} 个，其中 {sp.get('hits', 0)} 个的最好 |Sharpe| 到了排队线。"
        f"纯噪声下应约 {float(plan.get('p0', 0)):.0%}。证据值 {sp.get('llr', 0)}"
        f" → <b>{_e(verdict)}</b>。</p>"
    )


def _research(s: Session, cell: CellORM, sigma: float) -> str:
    plan = cell.plan or {}
    if plan.get("phase") not in ("research", "done") or plan.get("verdict") != "signal":
        return ""
    reps = list(
        s.scalars(
            select(FieldORM).where(
                FieldORM.cell_id == cell.cell_id, FieldORM.representative.is_(True)
            )
        )
    )
    null_best = stats.expected_best_abs(sigma, 3, draws=4000)
    groups: dict[str, list[float]] = defaultdict(list)
    for f in reps:
        if f.eligible and f.survey_score is not None:
            groups[f.stratum].append(f.survey_score - null_best)
    arms = stats.shrunk_means(groups) if groups else {}
    strata_rows = "".join(
        f"<tr><td>{_e(k)}</td><td class='n'>{len(groups[k])}</td><td class='n'>{a.mean:+.2f}</td></tr>"
        for k, a in sorted(arms.items(), key=lambda kv: -kv[1].mean)[:12]
    )
    fam = plan.get("families") or {}
    fam_rows = "".join(
        f"<tr><td>{_e(name)}</td><td class='n'>{st.get('rounds', 0)}</td>"
        f"<td class='n'>{float(st.get('best', 0)):.2f}</td>"
        f"<td>{'停了' if st.get('done') else '在挖'}</td></tr>"
        for name, st in sorted(fam.items(), key=lambda kv: -float(kv[1].get("best", 0)))[:12]
    )
    arm_rows = ""
    combos = plan.get("combos") or {}
    tally: dict[str, list[float]] = {"llm": [], "random": []}
    for rows in combos.values():
        for arm, ids in rows.items():
            for cid in ids:
                c = s.get(CandidateORM, cid)
                if c is not None and c.sharpe is not None:
                    tally[arm].append(abs(c.sharpe))
    for arm, xs in tally.items():
        if xs:
            arm_rows += (
                f"<tr><td>{'DeepSeek' if arm == 'llm' else '随机'}</td><td class='n'>{len(xs)}</td>"
                f"<td class='n'>{sum(x >= 0.8 for x in xs)}</td><td class='n'>{max(xs):.2f}</td></tr>"
            )
    return (
        f"<p>已规划 {plan.get('rounds', 0)} 轮。各层信号浓度（比噪声多出的部分，样本少的往整体平均拉）：</p>"
        f"<div class='wrap'><table><tr><th>层</th><th>测过的家族</th><th>浓度</th></tr>{strata_rows}</table></div>"
        + (
            f"<p>挖深的家族：</p><div class='wrap'><table><tr><th>家族</th><th>轮数</th>"
            f"<th>最好 Sharpe</th><th></th></tr>{fam_rows}</table></div>"
            if fam_rows
            else ""
        )
        + (
            "<p>挖组合（DeepSeek 对随机）：</p><div class='wrap'><table><tr><th></th><th>条数</th>"
            f"<th>到排队线</th><th>最好</th></tr>{arm_rows}</table></div>"
            if arm_rows
            else ""
        )
    )


def render_story(s: Session, cfg: FactoryConfig) -> str:
    cells = [c for c in s.scalars(select(CellORM)) if (c.plan or {}).get("phase")]
    cells.sort(key=lambda c: c.updated_at, reverse=True)
    parts = [
        "<h2>研究日志</h2>",
        f"<p><b>{now_line(s)}</b></p>",
        "<p class='muted'>流程：抽签选数据集 → 抽样体检（更新周期、补齐后覆盖率）→ 抽样普查（4 种形态，和噪声比）"
        " → 有信号才按各层浓度分配回测（挖深、挖广、挖组合）→ 按试的次数提高门槛确认 → 至多 4 条给你。</p>",
    ]
    if not cells:
        parts.append("<p class='muted'>还没有开始做的数据集。</p>")
    for c in cells[:4]:
        plan = c.plan or {}
        status = str(PHASE.get(str(plan.get("phase")), plan.get("phase")))
        if plan.get("outcome"):
            status += "：" + OUTCOME.get(plan["outcome"], plan["outcome"])
        parts.append(
            f"<div class='story'><h3>{_e(c.region)} · {_e(c.dataset_id)} "
            f"<span class='muted'>（{_e(c.dataset_name or c.category)}）· {_e(status)} · "
            f"已用 {c.sims_used} 次回测</span></h3>"
            f"<h4>体检</h4>{_health(s, c)}<h4>普查</h4>{_survey(c)}"
            f"{'<h4>挖</h4>' + _research(s, c, cfg.thresholds.noise_sigma) if c.state != CellState.NEW else ''}"
            "</div>"
        )
    return "".join(parts)
