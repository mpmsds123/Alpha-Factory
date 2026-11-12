"""`alpha-factory`: the commands you actually type."""

import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import select

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.brain.client import (
    BrainClient,
    RateLimited,
    SessionExpired,
)
from alpha_orchestrator.brain.session import AuthenticationFailed, BrainSession, NeedBiometric
from alpha_orchestrator.factory.config import DEFAULT_CONFIG, FactoryConfig, llm_key_available
from alpha_orchestrator.factory.ledger import Ledger, WorkerAlreadyRunning
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import (
    CandidateORM,
    ReviewORM,
    ReviewStatus,
)
from alpha_orchestrator.factory.select import approve, reject

app = typer.Typer(help="Alpha 工厂：WorldQuant BRAIN 自动挖掘流水线", no_args_is_help=True)
review_app = typer.Typer(help="待你确认的 alpha：查看、提交、放弃", no_args_is_help=True)
app.add_typer(review_app, name="review")

ConfigOpt = Annotated[Path, typer.Option("--config", help="factory.yaml")]


def _ctx(config_path: Path) -> tuple[FactoryConfig, Ledger]:
    config = FactoryConfig.load(config_path)
    return config, Ledger.at(Path(config.db_path))


# The session that just passed the face check in `login --then-run`. The run reuses this
# live object (as the scanner does) instead of rebuilding one from the cookie file, which
# came back unrecognised and forced a new password login, i.e. a new face check.
_live_session: BrainSession | None = None


def _api() -> BrainAPI:
    return BrainAPI(BrainClient(_live_session or BrainSession()))


def _llm(config: FactoryConfig, ledger: Ledger) -> FactoryLLM | None:
    if not llm_key_available(config.llm.key_file):
        typer.echo(
            f"未找到 {config.llm.key_file}：今天不用 DeepSeek（字段分组按名字分，不做组合对照）。"
        )
        return None
    return FactoryLLM(config, ledger)


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
        session.close()
        raise typer.Exit(1) from exc
    if not then_run:
        session.close()
        return
    # The run reuses this open session; closing it first made every request fail with
    # "Cannot send a request, as the client has been closed" (2026-11-03).
    global _live_session
    _live_session = session
    try:
        typer.echo("run is being rebuilt; not available in this commit")
    finally:
        _live_session = None
        session.close()


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


@review_app.command("recheck")
def review_recheck(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """复查已有候选，更新待挑选表。无需调用 LLM 或发起仿真。"""
    from alpha_orchestrator.factory.select import store_passing

    try:
        cfg, ledger = _ctx(config)
        with ledger.writer_lock():
            api = _api()
            try:
                api.client.session.ensure()
                stored = store_passing(api, ledger, cfg)
            finally:
                api.client.session.close()
    except WorkerAlreadyRunning as exc:
        typer.echo(f"安全暂停：{exc}")
        raise typer.Exit(4) from None
    except NeedBiometric:
        typer.echo("平台要求刷脸。完成 alpha-factory login 后重新执行复查。")
        raise typer.Exit(2) from None
    except (AuthenticationFailed, SessionExpired, RateLimited) as exc:
        typer.echo(f"复查暂停：{type(exc).__name__}")
        raise typer.Exit(3) from None
    typer.echo(f"新增待挑选 {len(stored)} 条。")


@review_app.command("reject")
def review_reject(review_id: str, note: str = "", config: ConfigOpt = DEFAULT_CONFIG) -> None:
    _, ledger = _ctx(config)
    with ledger.session() as s:
        pool = list(s.scalars(select(ReviewORM.id)))
    for rid in _match([review_id], pool):
        typer.echo("已放弃。" if reject(ledger, rid, note) else "状态不是待确认。")
