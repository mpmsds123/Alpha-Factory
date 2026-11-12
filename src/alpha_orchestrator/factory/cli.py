"""`alpha-factory`: the commands you type. Runs are started by hand; nothing schedules
them. Nothing is submitted except by `review approve`."""

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
from alpha_orchestrator.brain.session import AuthenticationFailed, BrainSession, NeedBiometric
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


def _ctx(config_path: Path) -> tuple[FactoryConfig, Ledger]:
    try:
        config = FactoryConfig.load(config_path)
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


@app.command()
def login(
    then_run: Annotated[bool, typer.Option("--then-run")] = False,
    minutes: Annotated[float, typer.Option(help="longest run, in minutes")] = 720,
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Log in (with the face check when asked), then optionally start the run."""
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
