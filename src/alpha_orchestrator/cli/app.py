"""Offline-first command-line interface."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from alpha_orchestrator.adapters.base import BrainAdapter
from alpha_orchestrator.adapters.live import (
    LiveBrainAdapter,
    LiveBrainClient,
    LiveBrainSettings,
    load_session_cookies,
)
from alpha_orchestrator.adapters.replay import ReplayBrainAdapter
from alpha_orchestrator.config import (
    load_correlation_policy,
    load_limit_policy,
    load_routing_policy,
)
from alpha_orchestrator.domain.enums import (
    AdapterMode,
    DataType,
    EvidenceLevel,
    MethodCardStatus,
    RunStatus,
)
from alpha_orchestrator.domain.models import (
    AdapterFailure,
    CandidateProposal,
    RunSpecification,
)
from alpha_orchestrator.knowledge.corpus import CorpusInventory
from alpha_orchestrator.knowledge.retriever import MethodCardRetriever
from alpha_orchestrator.llm.openai_client import LLMSettings, OpenAICompatibleLLMClient
from alpha_orchestrator.llm.prompts import register_default_prompts
from alpha_orchestrator.llm.services import CandidateGenerator, RepairPlanner
from alpha_orchestrator.orchestration.engine import ResearchOrchestrator
from alpha_orchestrator.orchestration.platform_checks import PlatformCheckService
from alpha_orchestrator.reporting.report import build_run_report
from alpha_orchestrator.seeding import TemplateError, seed_from_file
from alpha_orchestrator.storage.database import Database, sqlite_url
from alpha_orchestrator.storage.repository import MissingRecord, Repository

app = typer.Typer(help="Evidence-gated alpha research orchestration.", no_args_is_help=True)
run_app = typer.Typer(help="Create and operate bounded research runs.")
corpus_app = typer.Typer(help="Inventory private Markdown without storing raw text.")
cards_app = typer.Typer(help="Review and approve paraphrased method cards.")
app.add_typer(run_app, name="run")
app.add_typer(corpus_app, name="corpus")
app.add_typer(cards_app, name="cards")


@app.command("doctor")
def doctor(
    database: Annotated[Path | None, typer.Option("--db", help="SQLite database path.")] = None,
) -> None:
    """Check local readiness without contacting an external service."""
    database_path = _database_path(database).expanduser().resolve()
    replay_fixture = Path(__file__).parents[3] / "tests" / "fixtures" / "basic_replay.json"
    routing_path, limit_path = _policy_paths(None, None)
    correlation_path = _correlation_policy_path(None)

    policy_errors: list[str] = []
    for label, loader, path in (
        ("routing", load_routing_policy, routing_path),
        ("limits", load_limit_policy, limit_path),
        ("correlation", load_correlation_policy, correlation_path),
    ):
        try:
            loader(path)
        except (OSError, ValueError) as exc:
            policy_errors.append(f"{label}: {exc}")

    llm_variables = ("ALPHA_LAB_LLM_API_KEY", "ALPHA_LAB_LLM_MODEL")
    missing_llm = [name for name in llm_variables if not os.getenv(name)]

    session_status = "not configured"
    try:
        settings = LiveBrainSettings.from_environment()
        load_session_cookies(settings.session_file)
        session_status = "configured"
    except (OSError, ValueError) as exc:
        if os.getenv("ALPHA_LAB_BRAIN_SESSION_FILE"):
            session_status = f"invalid: {exc}"

    result = {
        "offline_replay_ready": replay_fixture.is_file() and not policy_errors,
        "database": {
            "path": str(database_path),
            "initialized": database_path.is_file(),
        },
        "policies": policy_errors if policy_errors else "valid",
        "llm": "configured" if not missing_llm else {"missing": missing_llm},
        "brain_session": session_status,
        "live_research_ready": not missing_llm and session_status == "configured",
        "external_requests_made": False,
    }
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


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
    repository = Repository(database.session_factory)
    register_default_prompts(repository)
    return database, repository


def _policy_paths(routing_policy: Path | None, limit_policy: Path | None) -> tuple[Path, Path]:
    directory = Path.cwd() / "policies"
    if not directory.is_dir():
        directory = Path(__file__).parents[1] / "default_policies"
    return (
        routing_policy or directory / "routing.v2.yaml",
        limit_policy or directory / "limits.v2.yaml",
    )


def _correlation_policy_path(correlation_policy: Path | None) -> Path:
    if correlation_policy is not None:
        return correlation_policy
    directory = Path.cwd() / "policies"
    if not (directory / "correlation.v1.yaml").is_file():
        directory = Path(__file__).parents[1] / "default_policies"
    return directory / "correlation.v1.yaml"


def _simulation_settings(path: Path | None) -> dict[str, object]:
    """Load the settings block a seeded candidate declares."""
    if path is None:
        directory = Path.cwd() / "policies"
        if not (directory / "simulation-settings.v1.json").is_file():
            directory = Path(__file__).parents[1] / "default_policies"
        path = directory / "simulation-settings.v1.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"Simulation settings could not be read: {path}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"Simulation settings must be a JSON object: {path}")
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def _live_client() -> LiveBrainClient:
    try:
        return LiveBrainClient(LiveBrainSettings.from_environment())
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _live_failure(exc: AdapterFailure) -> typer.BadParameter:
    return typer.BadParameter(f"{exc.error_class}: {exc.message}")


def _llm_services(
    repository: Repository,
    *,
    base_settings: dict[str, object] | None = None,
) -> tuple[CandidateGenerator, RepairPlanner]:
    try:
        client = OpenAICompatibleLLMClient(LLMSettings.from_environment())
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    retriever = MethodCardRetriever(repository)
    return (
        CandidateGenerator(repository, retriever, client, base_settings=base_settings),
        RepairPlanner(repository, retriever, client),
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
    db, repository = _repository(database)
    try:
        if adapter is AdapterMode.LIVE:
            client = _live_client()
            try:
                capture = client.capture_metadata(
                    dataset_id=dataset,
                    region=region,
                    universe=universe,
                    delay=delay,
                    data_type=data_type,
                )
                repository.save_metadata_capture(capture)
                field_snapshot = capture.field_snapshot_id
                operator_snapshot = capture.operator_snapshot_id
                setting_snapshot = capture.setting_snapshot_id
            except AdapterFailure as exc:
                raise _live_failure(exc) from exc
            finally:
                client.close()
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
            generator_prompt_version="generator.v1",
            repair_prompt_version="repair-planner.v1",
            adapter_mode=adapter,
        )
        repository.create_run(specification, routing, limits)
    finally:
        db.dispose()
    typer.echo(str(specification.run_id))


def _execute(
    run_id: str,
    adapter_mode: AdapterMode,
    fixture: Path | None,
    routing_policy: Path | None,
    limit_policy: Path | None,
    database: Path | None,
    templates: Path | None = None,
    simulation_settings: Path | None = None,
    research_query: str | None = None,
    correlation_policy: Path | None = None,
) -> None:
    if adapter_mode is AdapterMode.REPLAY and fixture is None:
        raise typer.BadParameter("Replay execution requires --fixture")
    if adapter_mode is AdapterMode.LIVE and fixture is not None:
        raise typer.BadParameter("Live execution does not accept a replay fixture")
    if adapter_mode is not AdapterMode.LIVE and templates is not None:
        raise typer.BadParameter("Seed templates apply to live execution only")
    if templates is not None and research_query is not None:
        raise typer.BadParameter("Use either --templates or --research-query, not both")
    routing_path, limit_path = _policy_paths(routing_policy, limit_policy)
    routing = load_routing_policy(routing_path)
    limits = load_limit_policy(limit_path)
    # Loaded regardless of adapter mode, same as routing/limits: cheap, and the
    # orchestrator itself only ever consults it for a live adapter (step ⑩'s
    # platform-check + correlation gate has no meaning against a replay fixture).
    correlation = load_correlation_policy(_correlation_policy_path(correlation_policy))
    db, repository = _repository(database)
    live_client: LiveBrainClient | None = None
    try:
        selected_adapter: BrainAdapter
        if adapter_mode is AdapterMode.LIVE:
            live_client = _live_client()
            try:
                # Check the session's remaining lifetime *before* any candidate spends
                # this run's limit budget on requests a dying cookie cannot see through.
                # Without this, a session that expires mid-run fails unpredictably deep
                # inside per-candidate retry handling instead of stopping cleanly here.
                live_client.authenticate()
            except AdapterFailure as exc:
                raise _live_failure(exc) from exc
            metadata = repository.load_run_metadata(run_id)
            seeds: list[CandidateProposal] = []
            if templates is not None:
                run = repository.get_run(run_id)
                try:
                    seeds = seed_from_file(
                        templates,
                        metadata,
                        data_type=DataType(run.data_type),
                        settings=_simulation_settings(simulation_settings),
                        limit=limits.initial_candidates,
                    )
                except TemplateError as exc:
                    raise typer.BadParameter(str(exc)) from exc
            selected_adapter = LiveBrainAdapter(live_client, metadata, seed_candidates=seeds)
        else:
            assert fixture is not None
            selected_adapter = ReplayBrainAdapter.from_path(fixture)
        generator: CandidateGenerator | None = None
        planner: RepairPlanner | None = None
        if research_query is not None:
            generator, planner = _llm_services(
                repository,
                base_settings=(
                    _simulation_settings(simulation_settings)
                    if adapter_mode is AdapterMode.LIVE
                    else None
                ),
            )
        summary = ResearchOrchestrator(
            repository,
            routing,
            limits,
            candidate_generator=generator,
            repair_planner=planner,
            research_query=research_query,
            correlation_policy=correlation,
        ).execute(run_id, selected_adapter)
    except MissingRecord as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        if live_client is not None:
            live_client.close()
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
    templates: Annotated[Path | None, typer.Option("--templates")] = None,
    simulation_settings: Annotated[Path | None, typer.Option("--simulation-settings")] = None,
    research_query: Annotated[str | None, typer.Option("--research-query")] = None,
    correlation_policy: Annotated[Path | None, typer.Option("--correlation-policy")] = None,
) -> None:
    """Execute a run through the explicitly selected adapter.

    Live runs have no candidates of their own. Pass --templates to seed them from a
    researcher-authored template file, expanded deterministically and without a model.
    """
    _execute(
        run_id,
        adapter,
        fixture,
        routing_policy,
        limit_policy,
        database,
        templates=templates,
        simulation_settings=simulation_settings,
        research_query=research_query,
        correlation_policy=correlation_policy,
    )


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
    research_query: Annotated[str | None, typer.Option("--research-query")] = None,
    correlation_policy: Annotated[Path | None, typer.Option("--correlation-policy")] = None,
) -> None:
    """Resume without repeating completed simulation attempts."""
    _execute(
        run_id,
        adapter,
        fixture,
        routing_policy,
        limit_policy,
        database,
        research_query=research_query,
        correlation_policy=correlation_policy,
    )


@app.command("platform-check")
def platform_check(
    candidate_id: str,
    correlation_policy: Annotated[Path | None, typer.Option("--correlation-policy")] = None,
    database: Annotated[Path | None, typer.Option("--db")] = None,
    repair_failures: Annotated[bool, typer.Option("--repair-failures")] = False,
) -> None:
    """Request live checks and the platform's own correlation numbers for one ready
    candidate; never submit it."""
    db, repository = _repository(database)
    client: LiveBrainClient | None = None
    try:
        candidate = repository.get_candidate(candidate_id)
        metadata = repository.load_run_metadata(candidate.run_id)
        client = _live_client()
        policy = load_correlation_policy(_correlation_policy_path(correlation_policy))
        planner = _llm_services(repository)[1] if repair_failures else None
        summary = PlatformCheckService(repository, planner).check(
            candidate_id, LiveBrainAdapter(client, metadata), policy
        )
    except MissingRecord as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        if client is not None:
            client.close()
        db.dispose()
    typer.echo(
        json.dumps(
            {
                "candidate_id": summary.candidate_id,
                "candidate_status": summary.candidate_status,
                "request_attempts": summary.request_attempts,
                "checks": [check.model_dump(mode="json") for check in summary.checks],
                "repair_request_ids": summary.repair_request_ids,
                "repair_candidate_ids": summary.repair_candidate_ids,
                "repair_stop_reason": summary.repair_stop_reason,
                "automatic_submission": False,
            },
            sort_keys=True,
        )
    )


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


@corpus_app.command("inventory")
def inventory_corpus(
    path: Path,
    source_type: Annotated[EvidenceLevel, typer.Option("--source-type")] = EvidenceLevel.UNVERIFIED,
    database: Annotated[Path | None, typer.Option("--db")] = None,
) -> None:
    """Record hashes and section locators for one file or a directory, never raw text."""
    db, repository = _repository(database)
    result = CorpusInventory(repository).inventory(path, source_type)
    db.dispose()
    typer.echo(json.dumps(result.__dict__, sort_keys=True))


@corpus_app.command("extract")
def extract_cards(
    source_id: str,
    database: Annotated[Path | None, typer.Option("--db")] = None,
) -> None:
    """Extract explicit cards from a source ID or an authored Markdown file as drafts."""
    db, repository = _repository(database)
    inventory = CorpusInventory(repository)
    source_path = Path(source_id).expanduser()
    if source_path.is_file():
        cards = inventory.extract_authored_draft_cards(source_path)
    else:
        cards = inventory.extract_draft_cards(source_id)
    db.dispose()
    typer.echo(json.dumps([card.card_id for card in cards]))


@cards_app.command("list")
def list_cards(
    status: Annotated[MethodCardStatus | None, typer.Option("--status")] = None,
    database: Annotated[Path | None, typer.Option("--db")] = None,
) -> None:
    """List method cards and their human-review state."""
    db, repository = _repository(database)
    cards = repository.list_method_cards(status)
    db.dispose()
    typer.echo(
        json.dumps(
            [
                {
                    "card_id": card.card_id,
                    "version": card.version,
                    "status": card.status,
                    "title": card.title,
                    "reviewer": card.reviewer,
                }
                for card in cards
            ],
            sort_keys=True,
        )
    )


@cards_app.command("approve")
def approve_card(
    card_id: str,
    reviewer: Annotated[str, typer.Option("--reviewer")],
    database: Annotated[Path | None, typer.Option("--db")] = None,
) -> None:
    """Record explicit human approval for the latest card version."""
    db, repository = _repository(database)
    card = repository.approve_method_card(card_id, reviewer)
    db.dispose()
    typer.echo(
        json.dumps(
            {
                "card_id": card.card_id,
                "version": card.version,
                "status": card.status,
                "reviewer": card.reviewer,
            },
            sort_keys=True,
        )
    )


@cards_app.command("retire")
def retire_card(
    card_id: str,
    version: Annotated[str, typer.Option("--version")],
    reviewer: Annotated[str, typer.Option("--reviewer")],
    database: Annotated[Path | None, typer.Option("--db")] = None,
) -> None:
    """Retire one approved card version, e.g. the v1 a newly approved v2 replaces.

    Retrieval reads every APPROVED row for a card_id, not just the newest one, so an
    old version left approved keeps being handed to the LLM alongside its
    replacement -- including whatever the replacement was written to remove.
    """
    db, repository = _repository(database)
    try:
        card = repository.retire_method_card(card_id, version, reviewer)
    except MissingRecord as exc:
        raise typer.BadParameter(str(exc)) from exc
    db.dispose()
    typer.echo(
        json.dumps(
            {
                "card_id": card.card_id,
                "version": card.version,
                "status": card.status,
                "reviewer": card.reviewer,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    app()
