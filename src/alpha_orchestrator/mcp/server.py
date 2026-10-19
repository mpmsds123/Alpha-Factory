"""Narrow, offline-only MCP interface over the established service layer."""

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from alpha_orchestrator import __version__
from alpha_orchestrator.adapters.replay import ReplayBrainAdapter
from alpha_orchestrator.config import LimitPolicy, RoutingPolicy
from alpha_orchestrator.domain.enums import AdapterMode, MethodCardStatus, RunStatus
from alpha_orchestrator.domain.errors import redact_text
from alpha_orchestrator.domain.models import StrictModel
from alpha_orchestrator.orchestration.engine import ResearchOrchestrator
from alpha_orchestrator.reporting.report import build_run_report
from alpha_orchestrator.storage.database import Database, sqlite_url
from alpha_orchestrator.storage.repository import MissingRecord, Repository

TResult = TypeVar("TResult")

READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
LEDGER_WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)


class MCPServiceSettings(StrictModel):
    database_path: Path
    replay_fixture_path: Path | None = None

    @classmethod
    def from_environment(cls) -> "MCPServiceSettings":
        explicit_path = os.getenv("ALPHA_LAB_MCP_DATABASE_PATH")
        database_url = os.getenv("ALPHA_LAB_DATABASE_URL")
        if explicit_path:
            database_path = Path(explicit_path)
        elif database_url:
            if not database_url.startswith("sqlite:///"):
                raise ValueError("The MCP interface supports only SQLite database URLs")
            database_path = Path(database_url.removeprefix("sqlite:///"))
        else:
            database_path = Path("alpha_lab.db")
        fixture = os.getenv("ALPHA_LAB_MCP_REPLAY_FIXTURE")
        return cls(
            database_path=database_path.expanduser().resolve(),
            replay_fixture_path=Path(fixture).expanduser().resolve() if fixture else None,
        )


class RunStatusView(StrictModel):
    run_id: str
    status: RunStatus
    stop_reason: str | None
    adapter_mode: AdapterMode
    distinct_candidates: int = Field(ge=0)
    simulation_attempts: int = Field(ge=0)
    automatic_submission_available: bool = False


class RunReportView(StrictModel):
    run_id: str
    markdown: str
    automatic_submission_available: bool = False


class MethodCardSummaryView(StrictModel):
    card_id: str
    version: str
    status: MethodCardStatus
    title: str


class MethodCardListView(StrictModel):
    cards: list[MethodCardSummaryView]


class ReplayExecutionView(StrictModel):
    run_id: str
    status: RunStatus
    stop_reason: str | None
    distinct_candidates: int = Field(ge=0)
    simulation_attempts: int = Field(ge=0)
    adapter_mode: AdapterMode = AdapterMode.REPLAY
    automatic_submission_available: bool = False


class RunStopView(StrictModel):
    run_id: str
    status: RunStatus = RunStatus.STOPPED
    stop_reason: str = "MCP_REQUESTED_STOP"


class AlphaMCPService:
    def __init__(self, settings: MCPServiceSettings) -> None:
        self.settings = settings

    @contextmanager
    def _repository(self) -> Iterator[Repository]:
        path = self.settings.database_path
        if not path.is_file():
            raise ValueError("Configured SQLite ledger does not exist")
        database = Database(sqlite_url(path))
        try:
            yield Repository(database.session_factory)
        finally:
            database.dispose()

    def run_status(self, run_id: str) -> RunStatusView:
        with self._repository() as repository:
            run = repository.get_run(run_id)
            return RunStatusView(
                run_id=run.run_id,
                status=RunStatus(run.status),
                stop_reason=run.stop_reason,
                adapter_mode=AdapterMode(run.adapter_mode),
                distinct_candidates=repository.count_candidates(run.run_id),
                simulation_attempts=repository.count_attempts(run.run_id),
            )

    def run_report(self, run_id: str) -> RunReportView:
        with self._repository() as repository:
            return RunReportView(
                run_id=run_id,
                markdown=build_run_report(repository, run_id),
            )

    def list_method_cards(self, status: MethodCardStatus | None) -> MethodCardListView:
        with self._repository() as repository:
            cards = repository.list_method_cards(status)
            return MethodCardListView(
                cards=[
                    MethodCardSummaryView(
                        card_id=card.card_id,
                        version=card.version,
                        status=MethodCardStatus(card.status),
                        title=card.title,
                    )
                    for card in cards
                ]
            )

    def execute_replay(self, run_id: str) -> ReplayExecutionView:
        fixture_path = self.settings.replay_fixture_path
        if fixture_path is None:
            raise ValueError("ALPHA_LAB_MCP_REPLAY_FIXTURE is required for replay execution")
        if not fixture_path.is_file():
            raise ValueError("Configured replay fixture does not exist")
        with self._repository() as repository:
            run = repository.get_run(run_id)
            if AdapterMode(run.adapter_mode) is not AdapterMode.REPLAY:
                raise ValueError("The MCP interface cannot execute live runs")
            routing = RoutingPolicy.model_validate(run.routing_policy_json)
            limits = LimitPolicy.model_validate(run.limit_policy_json)
            try:
                adapter = ReplayBrainAdapter.from_path(fixture_path)
            except (OSError, ValueError) as exc:
                raise ValueError("Configured replay fixture is invalid") from exc
            summary = ResearchOrchestrator(repository, routing, limits).execute(
                run.run_id,
                adapter,
            )
            return ReplayExecutionView(
                run_id=summary.run_id,
                status=summary.status,
                stop_reason=summary.stop_reason,
                distinct_candidates=summary.distinct_candidates,
                simulation_attempts=summary.simulation_attempts,
            )

    def stop_run(self, run_id: str) -> RunStopView:
        with self._repository() as repository:
            repository.get_run(run_id)
            repository.update_run_status(
                run_id,
                RunStatus.STOPPED,
                "MCP_REQUESTED_STOP",
            )
            return RunStopView(run_id=run_id)


def _model_visible_call(operation: Callable[[], TResult]) -> TResult:
    try:
        return operation()
    except (MissingRecord, ValueError) as exc:
        safe_message = redact_text(str(exc)) or "Requested ledger operation failed"
        raise ToolError(safe_message) from exc


def build_server(settings: MCPServiceSettings) -> MCPServer[None]:
    service = AlphaMCPService(settings)
    server: MCPServer[None] = MCPServer(
        name="alpha-research-orchestrator",
        version=__version__,
        instructions=(
            "Operate an existing Alpha research SQLite ledger through a narrow offline surface. "
            "This server cannot call live BRAIN, approve methodology, invoke an LLM, or submit an "
            "Alpha."
        ),
    )

    @server.tool(annotations=READ_ONLY)
    def alpha_run_status(run_id: str) -> RunStatusView:
        """Read the persisted status and counts for one existing research run."""
        return _model_visible_call(lambda: service.run_status(run_id))

    @server.tool(annotations=READ_ONLY)
    def alpha_run_report(run_id: str) -> RunReportView:
        """Render the deterministic Markdown report for one existing run."""
        return _model_visible_call(lambda: service.run_report(run_id))

    @server.tool(annotations=READ_ONLY)
    def alpha_method_cards_list(
        status: MethodCardStatus | None = None,
    ) -> MethodCardListView:
        """List stored method-card identities and human-review states."""
        return _model_visible_call(lambda: service.list_method_cards(status))

    @server.tool(annotations=LEDGER_WRITE)
    def alpha_replay_execute(run_id: str) -> ReplayExecutionView:
        """Execute or resume one replay-mode run using the operator-configured fixture."""
        return _model_visible_call(lambda: service.execute_replay(run_id))

    @server.tool(annotations=LEDGER_WRITE)
    def alpha_run_stop(run_id: str) -> RunStopView:
        """Persist a fixed MCP-requested stop reason for one existing run."""
        return _model_visible_call(lambda: service.stop_run(run_id))

    return server


mcp = build_server(MCPServiceSettings.from_environment())


def main() -> None:
    """Run the local MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
