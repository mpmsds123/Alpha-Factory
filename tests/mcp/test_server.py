from pathlib import Path

import anyio
import pytest
from mcp import Client

from alpha_orchestrator.config import load_limit_policy, load_routing_policy
from alpha_orchestrator.domain.enums import AdapterMode, DataType
from alpha_orchestrator.domain.models import RunSpecification
from alpha_orchestrator.mcp.server import (
    AlphaMCPService,
    MCPServiceSettings,
    build_server,
)
from alpha_orchestrator.storage.database import Database, sqlite_url
from alpha_orchestrator.storage.repository import Repository

ROOT = Path(__file__).parents[2]
EXPECTED_TOOLS = {
    "alpha_method_cards_list",
    "alpha_replay_execute",
    "alpha_run_report",
    "alpha_run_status",
    "alpha_run_stop",
}


def create_run(
    database_path: Path,
    *,
    adapter_mode: AdapterMode = AdapterMode.REPLAY,
) -> RunSpecification:
    database = Database(sqlite_url(database_path))
    database.create_schema()
    repository = Repository(database.session_factory)
    routing = load_routing_policy(ROOT / "policies" / "routing.v2.yaml")
    limits = load_limit_policy(ROOT / "policies" / "limits.v2.yaml")
    specification = RunSpecification(
        dataset_id="demo-dataset",
        region="USA",
        universe="TOP1000",
        delay=1,
        data_type=DataType.MATRIX,
        field_snapshot_id="fixture-fields.v1",
        operator_snapshot_id="fixture-operators.v1",
        setting_snapshot_id="fixture-settings.v1",
        routing_policy_version=routing.version,
        limit_policy_version=limits.version,
        generator_prompt_version="generator.v1",
        repair_prompt_version="repair-planner.v1",
        adapter_mode=adapter_mode,
    )
    repository.create_run(specification, routing, limits)
    database.dispose()
    return specification


def test_mcp_lists_only_the_narrow_offline_surface_and_executes_replay(tmp_path: Path) -> None:
    database_path = tmp_path / "mcp-ledger.db"
    run = create_run(database_path)
    server = build_server(
        MCPServiceSettings(
            database_path=database_path,
            replay_fixture_path=ROOT / "tests" / "fixtures" / "basic_replay.json",
        )
    )

    async def exercise() -> None:
        async with Client(server, raise_exceptions=True) as client:
            listed = await client.list_tools()
            tools = {tool.name: tool for tool in listed.tools}
            assert set(tools) == EXPECTED_TOOLS
            assert all(
                forbidden not in name
                for name in tools
                for forbidden in ("live", "submit", "approve", "generate", "corpus")
            )
            assert tools["alpha_run_status"].annotations is not None
            assert tools["alpha_run_status"].annotations.read_only_hint is True
            assert tools["alpha_replay_execute"].annotations is not None
            assert tools["alpha_replay_execute"].annotations.read_only_hint is False
            assert set(tools["alpha_replay_execute"].input_schema["properties"]) == {"run_id"}

            initial = await client.call_tool("alpha_run_status", {"run_id": str(run.run_id)})
            assert initial.structured_content is not None
            assert initial.structured_content["status"] == "CREATED"
            assert initial.structured_content["automatic_submission_available"] is False

            executed = await client.call_tool("alpha_replay_execute", {"run_id": str(run.run_id)})
            assert executed.structured_content is not None
            assert executed.structured_content["status"] == "COMPLETED"
            assert executed.structured_content["simulation_attempts"] == 2
            assert executed.structured_content["adapter_mode"] == "replay"

            report = await client.call_tool("alpha_run_report", {"run_id": str(run.run_id)})
            assert report.structured_content is not None
            assert "# Run report" in report.structured_content["markdown"]
            assert "Automatic submission: unavailable" in report.structured_content["markdown"]

            cards = await client.call_tool("alpha_method_cards_list", {})
            assert cards.structured_content == {"cards": []}

            stopped = await client.call_tool("alpha_run_stop", {"run_id": str(run.run_id)})
            assert stopped.structured_content is not None
            assert stopped.structured_content == {
                "run_id": str(run.run_id),
                "status": "STOPPED",
                "stop_reason": "MCP_REQUESTED_STOP",
            }

    anyio.run(exercise)


def test_mcp_replay_tool_reports_configuration_and_live_run_errors(tmp_path: Path) -> None:
    database_path = tmp_path / "mcp-errors.db"
    replay_run = create_run(database_path)
    live_run = create_run(database_path, adapter_mode=AdapterMode.LIVE)

    async def exercise() -> None:
        unconfigured = build_server(MCPServiceSettings(database_path=database_path))
        async with Client(unconfigured) as client:
            result = await client.call_tool(
                "alpha_replay_execute", {"run_id": str(replay_run.run_id)}
            )
            assert result.is_error
            assert "MCP_REPLAY_FIXTURE" in result.content[0].text

        configured = build_server(
            MCPServiceSettings(
                database_path=database_path,
                replay_fixture_path=ROOT / "tests" / "fixtures" / "basic_replay.json",
            )
        )
        async with Client(configured) as client:
            result = await client.call_tool(
                "alpha_replay_execute", {"run_id": str(live_run.run_id)}
            )
            assert result.is_error
            assert "cannot execute live runs" in result.content[0].text

    anyio.run(exercise)


def test_mcp_settings_and_service_reject_unconfigured_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ALPHA_LAB_MCP_DATABASE_PATH", raising=False)
    monkeypatch.setenv("ALPHA_LAB_DATABASE_URL", "postgresql://example.invalid/lab")
    with pytest.raises(ValueError, match="only SQLite"):
        MCPServiceSettings.from_environment()

    database_path = tmp_path / "explicit.db"
    fixture_path = tmp_path / "fixture.json"
    monkeypatch.setenv("ALPHA_LAB_MCP_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("ALPHA_LAB_MCP_REPLAY_FIXTURE", str(fixture_path))
    configured = MCPServiceSettings.from_environment()
    assert configured.database_path == database_path.resolve()
    assert configured.replay_fixture_path == fixture_path.resolve()

    service = AlphaMCPService(configured)
    with pytest.raises(ValueError, match="ledger does not exist"):
        service.run_status("missing")


def test_mcp_main_uses_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    from alpha_orchestrator.mcp import server as server_module

    calls: list[str] = []
    monkeypatch.setattr(server_module.mcp, "run", lambda transport: calls.append(transport))
    server_module.main()
    assert calls == ["stdio"]
