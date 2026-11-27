"""`alpha-factory`: the commands you type. `daily` is what the schedule runs; `run` is one
run by hand. Nothing is submitted except by `review approve`, unless submit_mode is
"auto" in factory.yaml."""

import os
import webbrowser
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from sqlalchemy import func, select

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.brain.client import (
    AmbiguousRequest,
    BrainClient,
    RateLimited,
    SessionExpired,
)
from alpha_orchestrator.brain.notify import notify
from alpha_orchestrator.brain.session import (
    SHARED_COOKIES_ENV,
    AuthenticationFailed,
    BrainSession,
    NeedBiometric,
)
from alpha_orchestrator.factory import gates, loop, memory, prompts, report
from alpha_orchestrator.factory.config import DEFAULT_CONFIG, FactoryConfig, llm_key_available
from alpha_orchestrator.factory.ledger import Ledger, WorkerAlreadyRunning
from alpha_orchestrator.factory.llm import LLMClient
from alpha_orchestrator.factory.models import (
    BatchORM,
    HStatus,
    HypothesisORM,
    PriorORM,
    Review,
    SimulationORM,
)
from alpha_orchestrator.factory.vocab import Vocabulary

app = typer.Typer(help="Hypothesis-driven alpha factory", no_args_is_help=True)
review_app = typer.Typer(help="Alphas waiting for you: list, submit, reject", no_args_is_help=True)
priors_app = typer.Typer(help="Memory A: claims from documents", no_args_is_help=True)
memory_app = typer.Typer(help="Memory B: past hypotheses", no_args_is_help=True)
app.add_typer(review_app, name="review")
app.add_typer(priors_app, name="priors")
app.add_typer(memory_app, name="memory")

ConfigOpt = Annotated[Path, typer.Option("--config", help="factory.yaml")]
SEED_DOC_CHARS = 6000  # characters of each document shown when extracting priors

# The session that just passed the face check in `login --then-run`; the run reuses this
# live object instead of rebuilding one from the cookie file (which forced a new login).
_live_session: BrainSession | None = None


SCANNER_SESSION_FILE = "brain_session.json"  # the template scanner's cookie file in its runs folder


def _share_login(config: FactoryConfig) -> None:
    """One login for the factory and the template scanner: when `scanner_runs` is set,
    the scanner's cookie file becomes a shared one (see brain/session.py)."""
    if config.scanner_runs and not os.getenv(SHARED_COOKIES_ENV):
        os.environ[SHARED_COOKIES_ENV] = str(
            Path(config.scanner_runs).expanduser() / SCANNER_SESSION_FILE
        )


def _ctx(config_path: Path) -> tuple[FactoryConfig, Ledger]:
    try:
        config = FactoryConfig.load(config_path)
        _share_login(config)
        return config, Ledger.at(Path(config.db_path))
    except WorkerAlreadyRunning as exc:
        typer.echo(f"Safe stop: {exc}")
        raise typer.Exit(4) from None


def _api() -> BrainAPI:
    return BrainAPI(BrainClient(_live_session or BrainSession()))


def _llm(config: FactoryConfig, ledger: Ledger) -> LLMClient:
    if not llm_key_available(config.llm.key_file):
        typer.echo(f"No LLM key: set ALPHA_FACTORY_LLM_API_KEY or create {config.llm.key_file}")
        raise typer.Exit(1)
    return LLMClient(config, ledger)


def _vocab(config: FactoryConfig) -> Vocabulary:
    vocab = Vocabulary.load(config.private_dir)
    if not vocab.grouping_fields:
        typer.echo(f"Warning: no platform vocabulary in {config.private_dir}/platform.yaml")
    return vocab


@app.command()
def run(
    minutes: Annotated[float, typer.Option(help="longest run, in minutes")] = 720,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Run the factory: hypotheses -> expressions -> simulations -> gates -> review."""
    cfg, ledger = _ctx(config)
    llm = _llm(cfg, ledger)
    vocab = _vocab(cfg)

    def progress(status: str) -> None:
        try:
            report.write_all(ledger, cfg, status)
        except Exception as exc:  # the CSVs must never stop a run
            typer.echo(f"Progress CSVs not written: {exc}")

    api = _api()
    progress("starting")
    try:
        with ledger.writer_lock():
            api.client.session.ensure()
            result = loop.run(
                cfg,
                api,
                ledger,
                llm,
                vocab,
                minutes=minutes,
                may_open=lambda: report.guard(ledger, cfg),
                on_progress=progress,
            )
    except (AmbiguousRequest, WorkerAlreadyRunning) as exc:
        progress(f"safe stop: {str(exc)[:80]}")
        typer.echo(f"Safe stop: {exc}")
        raise typer.Exit(4) from None
    except NeedBiometric:
        progress("stopped: face check needed")
        notify("Face check needed: run alpha-factory login --then-run (resumes where it stopped)")
        raise typer.Exit(2) from None
    except (AuthenticationFailed, SessionExpired, RateLimited) as exc:
        progress(f"stopped: {type(exc).__name__}")
        notify(f"Run paused: {exc}. Progress is saved; run again to continue.")
        raise typer.Exit(3) from None
    except Exception as exc:
        progress(f"stopped: error {type(exc).__name__}: {str(exc)[:80]}")
        raise
    finally:
        api.client.session.close()
    progress(f"done: {result.stopped_because}")
    msg = (
        f"{result.simulations} simulations ({result.failed} failed), "
        f"{result.opened} hypotheses opened, {len(result.stored)} stored for review; "
        f"stopped: {result.stopped_because}"
    )
    typer.echo(msg)
    typer.echo(f"Progress CSVs: {Path(cfg.reports_dir).resolve()}")
    notify(msg)


# AlphaForge restarts a run that stopped early after a 5-minute pause, at most 12 times a
# day; a login challenge stops everything and calls the human.
DAILY_RETRY_WAIT_S = 300
DAILY_MAX_RUNS = 12
FACE_POLL_S = 30  # while waiting for the owner's face check


def _wait_for_face_check(deadline: float) -> bool:
    """A scheduled run that hits a face check waits for it instead of quitting: it opens
    the check page once, then every 30 s asks the platform whether it was finished, or
    whether another program on the account (the scanner) already logged in. False when
    the day's deadline comes first."""
    import time

    session = BrainSession()
    try:
        url = session.store.meta().get("pending_biometric")
        if url:
            webbrowser.open(str(url))
        notify("Face check needed: the factory resumes by itself once it is done")
        while time.monotonic() < deadline:
            if (url and session.complete_biometric(str(url))) or session.adopt_shared():
                return True
            time.sleep(FACE_POLL_S)
        return False
    finally:
        session.close()


@app.command()
def daily(
    hours: Annotated[float, typer.Option(help="longest total time, in hours")] = 10,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """What the schedule runs: `run` until every daily slot is filled, the budget is spent
    or the time is up; restart after a pause (session, rate limit); stop and notify on a
    face check."""
    import time

    from alpha_orchestrator.factory import slots

    deadline = time.monotonic() + min(hours * 3600, _seconds_to_reset(config))
    for attempt in range(1, DAILY_MAX_RUNS + 1):
        left = (deadline - time.monotonic()) / 60
        if left < 5:
            break
        cfg, ledger = _ctx(config)
        with ledger.session() as s:
            if slots.next_slot(s, cfg) is None:
                typer.echo("Every daily slot is filled.")
                break
        try:
            run(minutes=left, config=config)
            break  # finished normally: budget, deadline or nothing left
        except typer.Exit as stop:
            if stop.exit_code == 2:  # face check: only the owner can do it, so wait for it
                if attempt == DAILY_MAX_RUNS or not _wait_for_face_check(deadline):
                    raise
                typer.echo("Face check done; resuming.")
                continue
            if stop.exit_code != 3 or attempt == DAILY_MAX_RUNS:
                raise
            typer.echo(f"Paused; trying again in {DAILY_RETRY_WAIT_S // 60} minutes.")
            time.sleep(DAILY_RETRY_WAIT_S)


RESET_MARGIN_S = 900  # stop 15 minutes before the quota day ends


def _seconds_to_reset(config: Path, now: datetime | None = None) -> float:
    """Seconds until 15 minutes before the quota window resets, so a daily run never
    spends the next day's simulations (the scanner's share)."""
    from datetime import timedelta

    cfg = FactoryConfig.load(config)
    instant = now or datetime.now(UTC)
    nxt = cfg.budget.window_start(instant) + timedelta(days=1)
    return max(0.0, (nxt - instant).total_seconds() - RESET_MARGIN_S)


LAUNCH_LABEL = "local.alpha-factory.daily"


def launchd_plist(program: str, repo: Path, hour: int, minute: int, log_dir: Path) -> str:
    """A macOS launchd job that runs `alpha-factory daily` every day at hour:minute."""
    from xml.sax.saxutils import escape

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LAUNCH_LABEL}</string>
  <key>ProgramArguments</key>
  <array><string>{escape(program)}</string><string>daily</string></array>
  <key>WorkingDirectory</key><string>{escape(str(repo))}</string>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>{hour}</integer><key>Minute</key><integer>{minute}</integer></dict>
  <key>StandardOutPath</key><string>{escape(str(log_dir / "daily.log"))}</string>
  <key>StandardErrorPath</key><string>{escape(str(log_dir / "daily.log"))}</string>
</dict>
</plist>
"""


@app.command()
def schedule(
    hour: Annotated[int, typer.Option(help="local hour to start, 0-23")],
    minute: Annotated[int, typer.Option(help="minute, 0-59")] = 0,
    write: Annotated[bool, typer.Option("--write", help="write the job file")] = False,
) -> None:
    """Print (or with --write, save) the macOS launchd job for the daily run. Load it with
    `launchctl load ~/Library/LaunchAgents/<label>.plist`; remove it with unload."""
    import shutil
    import sys

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise typer.BadParameter("hour 0-23, minute 0-59")
    program = shutil.which("alpha-factory") or str(Path(sys.executable).with_name("alpha-factory"))
    log_dir = Path("~/.alpha_factory/logs").expanduser()
    text = launchd_plist(program, Path.cwd(), hour, minute, log_dir)
    if not write:
        typer.echo(text)
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    target = Path("~/Library/LaunchAgents").expanduser() / f"{LAUNCH_LABEL}.plist"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    typer.echo(f"Wrote {target}. Load it with: launchctl load {target}")


@app.command()
def login(
    then_run: Annotated[bool, typer.Option("--then-run")] = False,
    minutes: Annotated[float, typer.Option(help="longest run, in minutes")] = 720,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Log in (with the face check when asked), then optionally start the run."""
    _share_login(FactoryConfig.load(config))
    session = BrainSession()
    try:
        remaining = session.ensure(interactive=True)
        typer.echo(f"Logged in; about {remaining / 3600:.1f} hours left on the session.")
    except NeedBiometric as need:
        typer.echo(f"Face check required, opening the browser:\n  {need.url}")
        webbrowser.open(need.url)
        while True:
            input("Press Enter after finishing the check in the browser... ")
            if session.complete_biometric(need.url):
                typer.echo("Verified, logged in.")
                break
            typer.echo("Not verified yet (or the link expired). Press Enter when done.")
    except AuthenticationFailed as exc:
        typer.echo(str(exc))
        session.close()
        raise typer.Exit(1) from exc
    if not then_run:
        session.close()
        return
    global _live_session
    _live_session = session
    try:
        run(minutes=minutes, config=config)
    finally:
        _live_session = None
        session.close()


@app.command()
def stats(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Rewrite the progress CSVs from the database (no login, no simulation)."""
    cfg, ledger = _ctx(config)
    typer.echo(f"CSVs: {report.write_all(ledger, cfg).resolve()}")


@app.command()
def weekly(
    write: Annotated[bool, typer.Option("--write", help="update README.md in place")] = False,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """The weekly performance table (README) and changelog lines, from the database."""
    from alpha_orchestrator.factory import weekly as wk

    cfg, ledger = _ctx(config)
    body, lines = wk.report(ledger, cfg)
    if write:
        readme = Path(config).resolve().parent / "README.md"
        changed = wk.write_readme(readme, body)
        typer.echo(f"{readme}: {'updated' if changed else 'already current'}")
    else:
        typer.echo(body)
    typer.echo("\n" + lines)


@app.command()
def status(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Counts at a glance, and whether the cost guard would let a new hypothesis open."""
    cfg, ledger = _ctx(config)
    with ledger.session() as s:
        by = dict(
            s.execute(
                select(HypothesisORM.verdict, func.count()).group_by(HypothesisORM.verdict)
            ).all()
        )
        open_n = s.scalar(
            select(func.count())
            .select_from(HypothesisORM)
            .where(HypothesisORM.status == HStatus.OPEN)
        )
        pending = ledger.pending(s)
        waiting = s.scalar(
            select(func.count())
            .select_from(SimulationORM)
            .where(SimulationORM.review == Review.PENDING)
        )
        priors = s.scalar(select(func.count()).select_from(PriorORM))
    typer.echo(f"hypotheses open {open_n}; closed by verdict {by}")
    typer.echo(f"simulations waiting/running {pending}; alphas waiting for review {waiting}")
    typer.echo(f"priors {priors}; LLM guard: {report.guard(ledger, cfg) or 'ok'}")


@app.command("recover-batch")
def recover_batch(
    batch_id: str,
    location: Annotated[str, typer.Option(help="the verified /simulations/... URL")],
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Re-attach a verified progress URL to an UNKNOWN batch; starts no simulation."""
    from urllib.parse import urlsplit

    from alpha_orchestrator.brain.session import platform_url

    parts = urlsplit(location)
    if (parts.netloc and f"{parts.scheme}://{parts.netloc}" != platform_url()) or not (
        parts.path.startswith("/simulations/") and not parts.query and not parts.fragment
    ):
        raise typer.BadParameter("must be a /simulations/... progress URL of the platform")
    _, ledger = _ctx(config)
    with ledger.writer_lock(), ledger.session() as s:
        batch = s.get(BatchORM, batch_id)
        if batch is None or batch.status != "UNKNOWN":
            raise typer.BadParameter("only an UNKNOWN batch can be recovered")
        other = s.scalar(
            select(BatchORM).where(BatchORM.location == parts.path, BatchORM.id != batch_id)
        )
        if other is not None:
            raise typer.BadParameter("this URL belongs to another batch")
        batch.location, batch.status = parts.path, "RUNNING"
    ledger.log("batch_reconciled", batch=batch_id, location=parts.path)
    typer.echo("Re-attached; the next run only polls it and matches results to requests.")


# ---------- memory ----------
@priors_app.command("seed")
def priors_seed(
    limit: Annotated[int, typer.Option(help="documents to read at most")] = 1000,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Extract priors from the official documents in the earlier ledger (LLM; capped)."""
    cfg, ledger = _ctx(config)
    vocab = _vocab(cfg)
    llm = _llm(cfg, ledger)
    docs = memory.read_documents(cfg.v7_db, list(vocab.official_doc_sources))[:limit]
    if not docs:
        typer.echo("No official documents found (check v7_db and official_doc_sources).")
        raise typer.Exit(1)
    started = datetime.now(UTC)

    def extract(doc: dict[str, str]) -> list[dict[str, Any]]:
        out, _ = llm.ask(
            prompts.PRIOR_SEED,
            prompts.PriorSeedOut,
            {"document": {"title": doc["title"], "text": doc["body"][:SEED_DOC_CHARS]}},
            hypothesis_id=None,
            retrieved={},
        )
        return [c.model_dump() for c in out.claims]

    with ledger.writer_lock():
        counts = memory.seed_priors(
            ledger.session, docs, extract, cfg.memory, lambda: llm.spent_since(started)
        )
    typer.echo(str(counts))


@priors_app.command("add")
def priors_add(
    claim: Annotated[str, typer.Option()],
    source: Annotated[str, typer.Option(help="title or reference of the source")],
    source_type: Annotated[str, typer.Option(help="official / forum / own")] = "own",
    quote: Annotated[str, typer.Option()] = "",
    category: Annotated[list[str] | None, typer.Option(help="dataset category; repeat")] = None,
    hypothesis_type: Annotated[
        list[str] | None, typer.Option(help="hypothesis type; repeat")
    ] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Add one prior by hand (e.g. a forum claim)."""
    if source_type not in ("official", "forum", "own"):
        raise typer.BadParameter("source type must be official, forum or own")
    bad = [t for t in hypothesis_type or [] if t not in prompts.HYPOTHESIS_TYPES]
    if bad:
        raise typer.BadParameter(f"unknown hypothesis types {bad}")
    cfg, ledger = _ctx(config)
    with ledger.session() as s:
        p = memory.add_prior(
            s,
            claim=claim,
            source_type=source_type,
            source_ref=f"manual:{source}",
            source_title=source,
            quote=quote,
            tags={
                "categories": [c.lower() for c in category or []],
                "hypothesis_types": list(hypothesis_type or []),
            },
            cfg=cfg.memory,
        )
        typer.echo(f"added P{p.id}" if p else "already there")


@memory_app.command("import-v7")
def memory_import_v7(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Import the earlier design's finished ideas as Memory B records (once)."""
    cfg, ledger = _ctx(config)
    with ledger.session() as s:
        n = memory.import_v7(s, cfg.v7_db)
    typer.echo(f"imported {n} records from {cfg.v7_db}")


# ---------- review ----------
@review_app.command("list")
def review_list(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Alphas that passed every gate and wait for you."""
    _, ledger = _ctx(config)
    with ledger.session() as s:
        for x in s.scalars(select(SimulationORM).where(SimulationORM.review == Review.PENDING)):
            h = s.get(HypothesisORM, x.hypothesis_id)
            typer.echo(
                f"[{x.id}] alpha {x.alpha_id}  H{x.hypothesis_id} {h.region if h else ''}\n"
                f"    {x.expression}\n"
                f"    train Sharpe {x.train_sharpe}  test Sharpe {x.test_sharpe}  "
                f"IS Sharpe {x.is_sharpe} fitness {x.is_fitness} turnover {x.is_turnover}\n"
                f"    {(h.claim if h else '')[:300]}\n"
            )


@review_app.command("approve")
def review_approve(simulation_id: int, config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Submit this alpha to the platform (cannot be undone)."""
    cfg, ledger = _ctx(config)
    api = _api()
    try:
        api.client.session.ensure()
        ok = gates.approve(api, ledger, simulation_id, cfg)
        typer.echo("Submitted." if ok else "Not submitted (not pending, or checks no longer pass).")
    finally:
        api.client.session.close()


@review_app.command("reject")
def review_reject(simulation_id: int, note: str = "", config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Drop this alpha from the review list."""
    _, ledger = _ctx(config)
    typer.echo("Rejected." if gates.reject(ledger, simulation_id, note) else "Not pending.")


if __name__ == "__main__":
    app()
