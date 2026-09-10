"""Offline Milestone 1 command-line interface."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from alpha_orchestrator.adapters.replay import ReplayBrainAdapter
from alpha_orchestrator.config import load_limit_policy, load_routing_policy
from alpha_orchestrator.domain.enums import AdapterMode, DataType, RunStatus
from alpha_orchestrator.domain.models import RunSpecification
from alpha_orchestrator.orchestration.engine import OfflineOrchestrator
from alpha_orchestrator.reporting.report import build_run_report
from alpha_orchestrator.storage.database import Database, sqlite_url
from alpha_orchestrator.storage.repository import MissingRecord, Repository

app = typer.Typer(help="Evidence-gated alpha research orchestration.", no_args_is_help=True)
run_app = typer.Typer(help="Create and operate bounded research runs.")
app.add_typer(run_app, name="run")


def _database_path(path: Path | None) -> Path:
    if path is not None:
        return path
    env_url = os.getenv("ALPHA_LAB_DATABASE_URL")
    if env_url and env_url.startswith("sqlite:///"):
        return Path(env_url.removeprefix("sqlite:///"))
    return Path("alpha_lab.db")


def _repository(path: Path | None) -> tuple[Database, Repository]:
    database = Database(sqlite_url(_database_path(path)))
    database.create_schema()
    return database, Repository(database.session_factory)


def _policy_paths(routing_policy: Path | None, limit_policy: Path | None) -> tuple[Path, Path]:
    directory = Path.cwd() / "policies"
    if not directory.is_dir():
        directory = Path(__file__).parents[1] / "default_policies"
    return (
        routing_policy or directory / "routing.v1.yaml",
        limit_policy or directory / "limits.v1.yaml",
    )


@app.command("init")
def initialize(
    database: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    """Initialize an empty SQLite ledger."""
    db, _ = _repository(database)
    db.dispose()
    typer.echo(f"Initialized SQLite ledger at {_database_path(database).resolve()}")


@run_app.command("create")
def create_run(
    dataset: Annotated[str, typer.Option("--dataset")],
    region: Annotated[str, typer.Option("--region")],
    universe: Annotated[str, typer.Option("--universe")],
    delay: Annotated[int, typer.Option("--delay", min=0)],
    data_type: Annotated[DataType, typer.Option("--data-type")] = DataType.MATRIX,
    adapter: Annotated[AdapterMode, typer.Option("--adapter")] = AdapterMode.REPLAY,
    field_snapshot: Annotated[str, typer.Option("--field-snapshot")] = "fixture-fields.v1",
    operator_snapshot: Annotated[str, typer.Option("--operator-snapshot")] = "fixture-operators.v1",
    setting_snapshot: Annotated[str, typer.Option("--setting-snapshot")] = "fixture-settings.v1",
    routing_policy: Annotated[Path | None, typer.Option("--routing-policy")] = None,
    limit_policy: Annotated[Path | None, typer.Option("--limit-policy")] = None,
    database: Annotated[Path | None, typer.Option("--db")] = None,
) -> None:
    """Create a frozen run contract."""
    routing_path, limit_path = _policy_paths(routing_policy, limit_policy)
    routing = load_routing_policy(routing_path)
    limits = load_limit_policy(limit_path)
    specification = RunSpecification(
        dataset_id=dataset,
        region=region,
        universe=universe,
        delay=delay,
        data_type=data_type,
        field_snapshot_id=field_snapshot,
        operator_snapshot_id=operator_snapshot,
        setting_snapshot_id=setting_snapshot,
        routing_policy_version=routing.version,
        limit_policy_version=limits.version,
        generator_prompt_version="not-applicable-milestone-1",
        repair_prompt_version="not-applicable-milestone-1",
        adapter_mode=adapter,
    )
    db, repository = _repository(database)
    repository.create_run(specification, routing, limits)
    db.dispose()
    typer.echo(str(specification.run_id))


def _execute(
    run_id: str,
    adapter_mode: AdapterMode,
    fixture: Path | None,
    routing_policy: Path | None,
    limit_policy: Path | None,
    database: Path | None,
) -> None:
    if adapter_mode is AdapterMode.LIVE:
        raise typer.BadParameter("Live execution is intentionally unavailable in Milestone 1")
    if fixture is None:
        raise typer.BadParameter("Replay execution requires --fixture")
    routing_path, limit_path = _policy_paths(routing_policy, limit_policy)
    routing = load_routing_policy(routing_path)
    limits = load_limit_policy(limit_path)
    db, repository = _repository(database)
    try:
        summary = OfflineOrchestrator(repository, routing, limits).execute(
            run_id, ReplayBrainAdapter.from_path(fixture)
        )
    except MissingRecord as exc:
        db.dispose()
        raise typer.BadParameter(str(exc)) from exc
    db.dispose()
    typer.echo(json.dumps(summary.__dict__, default=str, sort_keys=True))


@run_app.command("execute")
def execute_run(
    run_id: str,
    adapter: Annotated[AdapterMode, typer.Option("--adapter")] = AdapterMode.REPLAY,
    fixture: Annotated[Path | None, typer.Option("--fixture")] = None,
    routing_policy: Annotated[Path | None, typer.Option("--routing-policy")] = None,
    limit_policy: Annotated[Path | None, typer.Option("--limit-policy")] = None,
    database: Annotated[Path | None, typer.Option("--db")] = None,
) -> None:
    """Execute a run through the replay adapter."""
    _execute(run_id, adapter, fixture, routing_policy, limit_policy, database)


@run_app.command("status")
def run_status(
    run_id: str,
    database: Annotated[Path | None, typer.Option("--db")] = None,
) -> None:
    """Show persisted run status and counts."""
    db, repository = _repository(database)
    run = repository.get_run(run_id)
    payload = {
        "run_id": run.run_id,
        "status": run.status,
        "stop_reason": run.stop_reason,
        "distinct_candidates": repository.count_candidates(run_id),
        "simulation_attempts": repository.count_attempts(run_id),
    }
    db.dispose()
    typer.echo(json.dumps(payload, sort_keys=True))


@run_app.command("stop")
def stop_run(
    run_id: str,
    reason: Annotated[str, typer.Option("--reason")] = "HUMAN_REQUESTED_STOP",
    database: Annotated[Path | None, typer.Option("--db")] = None,
) -> None:
    """Persist an explicit stop reason."""
    db, repository = _repository(database)
    repository.update_run_status(run_id, RunStatus.STOPPED, reason)
    db.dispose()
    typer.echo(f"Stopped {run_id}: {reason}")


@run_app.command("resume")
def resume_run(
    run_id: str,
    adapter: Annotated[AdapterMode, typer.Option("--adapter")] = AdapterMode.REPLAY,
    fixture: Annotated[Path | None, typer.Option("--fixture")] = None,
    routing_policy: Annotated[Path | None, typer.Option("--routing-policy")] = None,
    limit_policy: Annotated[Path | None, typer.Option("--limit-policy")] = None,
    database: Annotated[Path | None, typer.Option("--db")] = None,
) -> None:
    """Resume without repeating completed simulation attempts."""
    _execute(run_id, adapter, fixture, routing_policy, limit_policy, database)


@app.command("report")
def report(
    run_id: str,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    database: Annotated[Path | None, typer.Option("--db")] = None,
) -> None:
    """Render a deterministic Markdown report from SQLite."""
    db, repository = _repository(database)
    content = build_run_report(repository, run_id)
    db.dispose()
    if output is None:
        typer.echo(content, nl=False)
        return
    output.write_text(content, encoding="utf-8")
    typer.echo(str(output.resolve()))


if __name__ == "__main__":
    app()
