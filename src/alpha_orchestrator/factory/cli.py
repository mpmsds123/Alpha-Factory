"""`alpha-factory`: the commands you actually type."""

import json
import os
import plistlib
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import select

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.brain.client import (
    AmbiguousRequest,
    BrainClient,
    RateLimited,
    SessionExpired,
)
from alpha_orchestrator.brain.notify import notify
from alpha_orchestrator.brain.session import AuthenticationFailed, BrainSession, NeedBiometric
from alpha_orchestrator.factory.config import DEFAULT_CONFIG, FactoryConfig
from alpha_orchestrator.factory.dashboard import render
from alpha_orchestrator.factory.ideas import import_ideas
from alpha_orchestrator.factory.ledger import Ledger, WorkerAlreadyRunning
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CellORM,
    ReviewORM,
    ReviewStatus,
)
from alpha_orchestrator.factory.select import approve, reject

app = typer.Typer(help="Alpha 工厂：WorldQuant BRAIN 自动挖掘流水线", no_args_is_help=True)
review_app = typer.Typer(help="待你确认的 alpha：查看、提交、放弃", no_args_is_help=True)
ideas_app = typer.Typer(help="灵感库", no_args_is_help=True)
schedule_app = typer.Typer(help="每日自动运行（macOS launchd）", no_args_is_help=True)
app.add_typer(review_app, name="review")
app.add_typer(ideas_app, name="ideas")
app.add_typer(schedule_app, name="schedule")

ConfigOpt = Annotated[Path, typer.Option("--config", help="factory.yaml")]
PLIST = Path("~/Library/LaunchAgents/com.meilin.alphafactory.plist").expanduser()


def _ctx(config_path: Path) -> tuple[FactoryConfig, Ledger]:
    config = FactoryConfig.load(config_path)
    return config, Ledger.at(Path(config.db_path))


def _api() -> BrainAPI:
    return BrainAPI(BrainClient(BrainSession()))


def _llm(config: FactoryConfig, ledger: Ledger) -> FactoryLLM | None:
    if not Path(config.llm.key_file).expanduser().exists():
        typer.echo(
            f"未找到 {config.llm.key_file}：今天不用 DeepSeek（字段分组按名字分，不做组合对照）。"
        )
        return None
    return FactoryLLM(config, ledger)


def _dashboard_path(config: FactoryConfig) -> Path:
    return Path(config.reports_dir) / "dashboard.html"


@app.command()
def doctor(
    live: Annotated[bool, typer.Option("--live")] = False, config: ConfigOpt = DEFAULT_CONFIG
) -> None:
    """只读检查预算、在途和异常批次；--live 只检查登录，不登录或回测。"""
    from alpha_orchestrator.factory.health import inspect_local

    report = inspect_local(FactoryConfig.load(config))
    if live:
        session = BrainSession()
        try:
            response = session.http.get("/authentication")
            report["authentication_http_status"] = response.status_code
            report["authenticated"] = response.status_code == 200
            if response.status_code != 200:
                report["blockers"].append("authentication_required")
        except Exception as exc:
            report["authenticated"] = False
            report["blockers"].append("authentication_unreachable")
            report["connection_error"] = type(exc).__name__
        finally:
            session.close()
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if report["blockers"]:
        raise typer.Exit(1)


@app.command("recover-batch")
def recover_batch(
    batch_id: str,
    location: Annotated[str, typer.Option(help="核对后的 /simulations/... URL")],
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """把已在平台核对过的进度 URL 接回异常批次；不发起新回测。"""
    from urllib.parse import urlsplit

    from alpha_orchestrator.brain.session import API_URL
    from alpha_orchestrator.factory.models import BatchORM

    parts = urlsplit(location)
    if (parts.netloc and f"{parts.scheme}://{parts.netloc}" != API_URL) or not (
        parts.path.startswith("/simulations/") and not parts.query and not parts.fragment
    ):
        raise typer.BadParameter("必须是 BRAIN 的 /simulations/... 进度 URL")
    _, ledger = _ctx(config)
    with ledger.writer_lock(), ledger.session() as s:
        batch = s.get(BatchORM, batch_id)
        if batch is None or batch.status != "UNKNOWN":
            raise typer.BadParameter("只允许恢复 UNKNOWN 批次")
        other = s.scalar(
            select(BatchORM).where(BatchORM.location == parts.path, BatchORM.id != batch_id)
        )
        if other is not None:
            raise typer.BadParameter("该进度 URL 已属于另一批次")
        batch.location, batch.status = parts.path, "RUNNING"
        ledger.log(s, "batch_reconciled", batch=batch_id, location=parts.path)
    typer.echo("已接回原批次；下次 run 只查询原请求，返回结果必须匹配表达式及设置。")


@app.command()
def login(
    then_run: Annotated[bool, typer.Option("--then-run")] = False,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """刷脸登录。完成后可直接开始当天的运行。"""
    session = BrainSession()
    try:
        remaining = session.ensure(interactive=True)
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
    minutes: Annotated[float, typer.Option(help="最长运行分钟数")] = 720,
    llm_only: Annotated[
        bool,
        typer.Option("--llm-only", help="只生成 1 个点子和 5 个公式，不发起回测"),
    ] = False,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """跑一天（设计文档 v1）：选数据集 → 读字段 → 想点子 → 缩字段 → 测前 10 → 判断 → 改进 →
    排除运气 → 待确认。"""
    from alpha_orchestrator.factory.engine import run_day, run_llm_only
    from alpha_orchestrator.factory.record import render_record

    cfg, ledger = _ctx(config)
    llm = _llm(cfg, ledger)
    if llm is None:
        typer.echo("设计文档 v1 需要 DeepSeek 想点子、挑字段；找不到密钥就不跑。")
        raise typer.Exit(1)
    record_path = Path(cfg.reports_dir) / "record.html"

    def refresh_record() -> None:
        render_record(ledger, record_path)

    api = _api()
    try:
        api.client.session.ensure()
        if llm_only:
            preview = run_llm_only(cfg, api, ledger, llm)
            if preview is None:
                typer.echo("没有可生成点子的数据集；回测次数 = 0")
                return
            typer.echo(f"点子：{preview.title}")
            typer.echo(f"机制：{preview.mechanism}")
            for index, formula in enumerate(preview.formulas, 1):
                typer.echo(f"公式 {index}：{formula}")
            typer.echo("回测次数 = 0")
            return
        report = run_day(
            cfg,
            api,
            ledger,
            llm,
            minutes=minutes,
            on_progress=refresh_record,
        )
    except (AmbiguousRequest, WorkerAlreadyRunning) as exc:
        typer.echo(f"安全暂停：{exc}")
        raise typer.Exit(4) from None
    except NeedBiometric:
        notify("平台要求刷脸：运行 alpha-factory login --then-run，会从断点继续")
        render_record(ledger, record_path)
        raise typer.Exit(2) from None
    except (AuthenticationFailed, SessionExpired, RateLimited) as exc:
        notify(f"运行暂停：{exc}。进度已保存，稍后重跑 alpha-factory run 会接着来")
        render_record(ledger, record_path)
        raise typer.Exit(3) from None
    finally:
        api.client.session.close()
    render_record(ledger, record_path)
    msg = (
        f"今日回测 {report.simulations}（失败 {report.failed}），过门槛 {report.ready}，"
        f"新存入待挑选 {len(report.reviews)} 条"
    )
    typer.echo(f"待挑选表格：{Path(cfg.reports_dir) / '工厂_待挑选.csv'}")
    typer.echo(msg + f"；停止原因：{report.stopped_because or '完成'}")
    typer.echo(f"研究记录：{record_path}")
    notify(msg)


@app.command()
def status(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """账本概况，并刷新面板。"""
    cfg, ledger = _ctx(config)
    with ledger.session() as s:
        cells = list(s.scalars(select(CellORM)))
        n_cand = len(list(s.scalars(select(CandidateORM.id))))
        pending = len(
            list(s.scalars(select(ReviewORM.id).where(ReviewORM.status == ReviewStatus.PENDING)))
        )
    by_state: dict[str, int] = {}
    for c in cells:
        by_state[c.state] = by_state.get(c.state, 0) + 1
    typer.echo(f"数据集组合 {len(cells)} {by_state}；候选 {n_cand}；待确认 {pending}")
    typer.echo(f"面板：{render(ledger, cfg, _dashboard_path(cfg)).resolve()}")


@app.command()
def dashboard(
    open_: Annotated[bool, typer.Option("--open")] = True, config: ConfigOpt = DEFAULT_CONFIG
) -> None:
    """重新生成面板并打开。"""
    cfg, ledger = _ctx(config)
    path = render(ledger, cfg, _dashboard_path(cfg)).resolve()
    typer.echo(str(path))
    if open_:
        webbrowser.open(path.as_uri())


@ideas_app.command("import")
def ideas_import(paths: list[Path], config: ConfigOpt = DEFAULT_CONFIG) -> None:
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


@review_app.command("list")
def review_list(config: ConfigOpt = DEFAULT_CONFIG) -> None:
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
def review_approve(review_id: str, config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """提交这一条 alpha 到平台（不可撤回）。"""
    cfg, ledger = _ctx(config)
    with ledger.session() as s:
        pool = list(s.scalars(select(ReviewORM.id)))
    matched = _match([review_id], pool)
    if len(matched) != 1:
        raise typer.Exit(1)
    rid = matched[0]
    api = _api()
    try:
        api.client.session.ensure()
        typer.echo("已提交。" if approve(api, ledger, rid, cfg) else "提交失败，详情见面板事件。")
    finally:
        api.client.session.close()


@review_app.command("reject")
def review_reject(review_id: str, note: str = "", config: ConfigOpt = DEFAULT_CONFIG) -> None:
    _, ledger = _ctx(config)
    with ledger.session() as s:
        pool = list(s.scalars(select(ReviewORM.id)))
    for rid in _match([review_id], pool):
        typer.echo("已放弃。" if reject(ledger, rid, note) else "状态不是待确认。")


@schedule_app.command("install")
def schedule_install(
    hour: int = 0, minute: int = 37, workdir: Path | None = None, config: ConfigOpt = DEFAULT_CONFIG
) -> None:
    """每天定时运行；明确固定代码、配置和会话路径。"""
    from alpha_orchestrator.factory.config import REPO_ROOT

    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise typer.BadParameter("小时必须为 0-23，分钟必须为 0-59")
    workdir = (workdir or REPO_ROOT).resolve()
    cfg = FactoryConfig.load(config)
    log_path = Path(cfg.db_path).expanduser().resolve().parent / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {"PYTHONPATH": str(REPO_ROOT / "src")}
    if os.getenv("BRAIN_STATE_DIR"):
        env["BRAIN_STATE_DIR"] = str(Path(os.environ["BRAIN_STATE_DIR"]).expanduser().resolve())
    payload = {
        "Label": "com.meilin.alphafactory",
        "ProgramArguments": [
            sys.executable,
            "-c",
            "from alpha_orchestrator.factory.cli import app; app()",
            "run",
            "--config",
            str(config.expanduser().resolve()),
        ],
        "WorkingDirectory": str(workdir),
        "EnvironmentVariables": env,
        "StartCalendarInterval": {"Hour": hour, "Minute": minute},
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(log_path),
    }
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    with PLIST.open("wb") as out:
        plistlib.dump(payload, out)
    subprocess.run(["launchctl", "unload", str(PLIST)], check=False, capture_output=True)
    subprocess.run(["launchctl", "load", str(PLIST)], check=True)
    typer.echo(f"已安装：每天 {hour:02d}:{minute:02d} 运行。卸载：alpha-factory schedule uninstall")


@schedule_app.command("uninstall")
def schedule_uninstall() -> None:
    subprocess.run(["launchctl", "unload", str(PLIST)], check=False, capture_output=True)
    PLIST.unlink(missing_ok=True)
    typer.echo("已卸载。")
