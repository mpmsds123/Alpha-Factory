"""`alpha-factory`: the commands you actually type."""

import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import select

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.brain.client import BrainClient
from alpha_orchestrator.brain.notify import notify
from alpha_orchestrator.brain.session import AuthenticationFailed, BrainSession, NeedBiometric
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.dashboard import render
from alpha_orchestrator.factory.ideas import import_ideas
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CellORM,
    HypothesisORM,
    HypothesisStatus,
    ReviewORM,
    ReviewStatus,
)
from alpha_orchestrator.factory.pipeline import approve_hypotheses
from alpha_orchestrator.factory.select import approve, reject

app = typer.Typer(help="Alpha 工厂：WorldQuant BRAIN 自动挖掘流水线", no_args_is_help=True)
hyp_app = typer.Typer(help="LLM 提出的假设：查看、批准、否决", no_args_is_help=True)
review_app = typer.Typer(help="待你确认的 alpha：查看、提交、放弃", no_args_is_help=True)
ideas_app = typer.Typer(help="灵感库", no_args_is_help=True)
schedule_app = typer.Typer(help="每日自动运行（macOS launchd）", no_args_is_help=True)
app.add_typer(hyp_app, name="hypotheses")
app.add_typer(review_app, name="review")
app.add_typer(ideas_app, name="ideas")
app.add_typer(schedule_app, name="schedule")

ConfigOpt = Annotated[Path, typer.Option("--config", help="factory.yaml")]
PLIST = Path("~/Library/LaunchAgents/com.local.alphafactory.plist").expanduser()


def _ctx(config_path: Path) -> tuple[FactoryConfig, Ledger]:
    config = FactoryConfig.load(config_path)
    return config, Ledger.at(Path(config.db_path))


def _api() -> BrainAPI:
    return BrainAPI(BrainClient(BrainSession()))


def _llm(config: FactoryConfig, ledger: Ledger) -> FactoryLLM | None:
    if not Path(config.llm.key_file).expanduser().exists():
        typer.echo(
            f"未找到 {config.llm.key_file}：今天只跑确定性部分（探测、扩展、修复），不调用 LLM。"
        )
        return None
    return FactoryLLM(config, ledger)


def _dashboard_path(config: FactoryConfig) -> Path:
    return Path(config.reports_dir) / "dashboard.html"


@app.command()
def login(
    then_run: Annotated[bool, typer.Option("--then-run")] = False,
    config: ConfigOpt = Path("factory.yaml"),
) -> None:
    """刷脸登录。完成后可直接开始当天的运行。"""
    session = BrainSession()
    try:
        remaining = session.ensure()
        typer.echo(f"已登录，会话剩余约 {remaining / 3600:.1f} 小时。")
    except NeedBiometric as need:
        typer.echo(f"平台要求刷脸，正在打开浏览器：\n  {need.url}")
        webbrowser.open(need.url)
        while True:
            input("在浏览器里完成验证后按回车… ")
            if session.complete_biometric(need.url):
                typer.echo("验证通过，已登录。")
                break
            typer.echo("还没通过（或链接已过期）。完成后再按回车。")
    except AuthenticationFailed as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    finally:
        session.close()
    if then_run:
        run(config=config)


@app.command()
def run(
    minutes: Annotated[float, typer.Option(help="最长运行分钟数")] = 600,
    config: ConfigOpt = Path("factory.yaml"),
) -> None:
    """跑一天：刷新目录 → 打分分配 → 探测/生成 → 回测 → 评估 → 选出待确认。"""
    from alpha_orchestrator.factory.daily import run_day

    cfg, ledger = _ctx(config)
    api = _api()
    try:
        api.client.session.ensure()
        report = run_day(cfg, api, ledger, _llm(cfg, ledger), minutes=minutes)
    except NeedBiometric:
        notify("平台要求刷脸：运行 alpha-factory login --then-run，会从断点继续")
        render(ledger, cfg, _dashboard_path(cfg))
        raise typer.Exit(2) from None
    finally:
        api.client.session.close()
    render(ledger, cfg, _dashboard_path(cfg))
    msg = (
        f"今日回测 {report.simulations}（失败 {report.failed}），过门槛 {report.ready}，"
        f"新增待确认 {len(report.reviews)}"
    )
    typer.echo(msg + f"；停止原因：{report.stopped_because or '完成'}")
    notify(msg)


@app.command()
def status(config: ConfigOpt = Path("factory.yaml")) -> None:
    """账本概况，并刷新面板。"""
    cfg, ledger = _ctx(config)
    with ledger.session() as s:
        cells = list(s.scalars(select(CellORM)))
        n_cand = len(list(s.scalars(select(CandidateORM.id))))
        pending = len(
            list(s.scalars(select(ReviewORM.id).where(ReviewORM.status == ReviewStatus.PENDING)))
        )
        hyps = len(
            list(
                s.scalars(
                    select(HypothesisORM.id).where(
                        HypothesisORM.status == HypothesisStatus.PROPOSED
                    )
                )
            )
        )
    by_state: dict[str, int] = {}
    for c in cells:
        by_state[c.state] = by_state.get(c.state, 0) + 1
    typer.echo(
        f"数据集组合 {len(cells)} {by_state}；候选 {n_cand}；待确认 {pending}；待批假设 {hyps}"
    )
    typer.echo(f"面板：{render(ledger, cfg, _dashboard_path(cfg)).resolve()}")


@app.command()
def dashboard(
    open_: Annotated[bool, typer.Option("--open")] = True, config: ConfigOpt = Path("factory.yaml")
) -> None:
    """重新生成面板并打开。"""
    cfg, ledger = _ctx(config)
    path = render(ledger, cfg, _dashboard_path(cfg)).resolve()
    typer.echo(str(path))
    if open_:
        webbrowser.open(path.as_uri())


@ideas_app.command("import")
def ideas_import(paths: list[Path], config: ConfigOpt = Path("factory.yaml")) -> None:
    """导入论文/论坛文件夹或合集 .md 文件（可重复执行，自动去重）。"""
    _, ledger = _ctx(config)
    with ledger.session() as s:
        added = import_ideas(s, paths)
    typer.echo(f"新增 {added} 篇。")


def _match(ids: list[str], pool: list[str]) -> list[str]:
    out = []
    for prefix in ids:
        hits = [p for p in pool if p.startswith(prefix)]
        if len(hits) != 1:
            typer.echo(f"ID 前缀 {prefix} 匹配到 {len(hits)} 条，已跳过")
            continue
        out.append(hits[0])
    return out


@hyp_app.command("list")
def hyp_list(config: ConfigOpt = Path("factory.yaml")) -> None:
    _, ledger = _ctx(config)
    with ledger.session() as s:
        for h in s.scalars(
            select(HypothesisORM).where(HypothesisORM.status == HypothesisStatus.PROPOSED)
        ):
            typer.echo(
                f"{h.id[:8]}  [{h.cell_id}]  {h.title}\n    {h.statement}\n"
                f"    方向：{h.direction}  字段：{', '.join(h.fields)}"
                f"  出处：{'; '.join(h.sources)}\n"
            )


@hyp_app.command("approve")
def hyp_approve(ids: list[str], config: ConfigOpt = Path("factory.yaml")) -> None:
    _, ledger = _ctx(config)
    with ledger.session() as s:
        pool = list(s.scalars(select(HypothesisORM.id)))
    typer.echo(f"已批准 {approve_hypotheses(ledger, _match(ids, pool), True)} 条。")


@hyp_app.command("reject")
def hyp_reject(ids: list[str], config: ConfigOpt = Path("factory.yaml")) -> None:
    _, ledger = _ctx(config)
    with ledger.session() as s:
        pool = list(s.scalars(select(HypothesisORM.id)))
    typer.echo(f"已否决 {approve_hypotheses(ledger, _match(ids, pool), False)} 条。")


@review_app.command("list")
def review_list(config: ConfigOpt = Path("factory.yaml")) -> None:
    _, ledger = _ctx(config)
    with ledger.session() as s:
        for r in s.scalars(select(ReviewORM).where(ReviewORM.status == ReviewStatus.PENDING)):
            c = s.get(CandidateORM, r.candidate_id)
            assert c is not None
            typer.echo(
                f"{r.id[:8]}  alpha {r.alpha_id}  [{c.cell_id}]\n    {c.expression}\n"
                f"    sharpe {c.sharpe} fitness {c.fitness} turnover {c.turnover} "
                f"score {r.score}\n    {r.description[:300]}\n"
            )


@review_app.command("approve")
def review_approve(review_id: str, config: ConfigOpt = Path("factory.yaml")) -> None:
    """提交这一条 alpha 到平台（不可撤回）。"""
    _, ledger = _ctx(config)
    with ledger.session() as s:
        pool = list(s.scalars(select(ReviewORM.id)))
    matched = _match([review_id], pool)
    if len(matched) != 1:
        raise typer.Exit(1)
    rid = matched[0]
    api = _api()
    try:
        api.client.session.ensure()
        typer.echo("已提交。" if approve(api, ledger, rid) else "提交失败，详情见面板事件。")
    finally:
        api.client.session.close()


@review_app.command("reject")
def review_reject(review_id: str, note: str = "", config: ConfigOpt = Path("factory.yaml")) -> None:
    _, ledger = _ctx(config)
    with ledger.session() as s:
        pool = list(s.scalars(select(ReviewORM.id)))
    for rid in _match([review_id], pool):
        typer.echo("已放弃。" if reject(ledger, rid, note) else "状态不是待确认。")


@schedule_app.command("install")
def schedule_install(hour: int = 0, minute: int = 37, workdir: Path | None = None) -> None:
    """每天定时运行（Mac 睡眠错过时醒来补跑）。"""
    workdir = workdir or Path.cwd()
    exe = Path(sys.executable).with_name("alpha-factory")
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    PLIST.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.local.alphafactory</string>
  <key>ProgramArguments</key><array><string>{exe}</string><string>run</string></array>
  <key>WorkingDirectory</key><string>{workdir.resolve()}</string>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>{hour}</integer>
    <key>Minute</key><integer>{minute}</integer></dict>
  <key>StandardOutPath</key><string>{Path("~/.alpha_factory/run.log").expanduser()}</string>
  <key>StandardErrorPath</key><string>{Path("~/.alpha_factory/run.log").expanduser()}</string>
</dict></plist>
""")
    subprocess.run(["launchctl", "unload", str(PLIST)], check=False, capture_output=True)
    subprocess.run(["launchctl", "load", str(PLIST)], check=True)
    typer.echo(f"已安装：每天 {hour:02d}:{minute:02d} 运行。卸载：alpha-factory schedule uninstall")


@schedule_app.command("uninstall")
def schedule_uninstall() -> None:
    subprocess.run(["launchctl", "unload", str(PLIST)], check=False, capture_output=True)
    PLIST.unlink(missing_ok=True)
    typer.echo("已卸载。")
