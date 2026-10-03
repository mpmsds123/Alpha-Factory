import json
from pathlib import Path

import httpx
import respx
from typer.testing import CliRunner

from alpha_orchestrator.cli.app import app

ROOT = Path(__file__).parents[2]
runner = CliRunner()


def test_cli_exposes_platform_check_but_no_submission_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "platform-check" in result.stdout
    unavailable = runner.invoke(app, ["submit"])
    assert unavailable.exit_code != 0
    assert "No such command" in unavailable.stderr


def test_doctor_reports_readiness_without_creating_database(
    tmp_path: Path, monkeypatch: object
) -> None:
    database = tmp_path / "missing.db"
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("ALPHA_LAB_LLM_API_KEY", raising=False)
    monkeypatch.delenv("ALPHA_LAB_LLM_MODEL", raising=False)
    monkeypatch.delenv("ALPHA_LAB_BRAIN_SESSION_FILE", raising=False)

    result = runner.invoke(app, ["doctor", "--db", str(database)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["offline_replay_ready"] is True
    assert payload["database"]["initialized"] is False
    assert payload["live_research_ready"] is False
    assert payload["external_requests_made"] is False
    assert not database.exists()


def test_cli_initializes_executes_reports_and_stops(tmp_path: Path, monkeypatch: object) -> None:
    database = tmp_path / "lab.db"
    monkeypatch.chdir(ROOT)
    initialized = runner.invoke(app, ["init", "--db", str(database)])
    assert initialized.exit_code == 0
    assert database.exists()

    created = runner.invoke(
        app,
        [
            "run",
            "create",
            "--dataset",
            "demo-dataset",
            "--region",
            "USA",
            "--universe",
            "TOP1000",
            "--delay",
            "1",
            "--db",
            str(database),
        ],
    )
    assert created.exit_code == 0
    run_id = created.stdout.strip()
    executed = runner.invoke(
        app,
        [
            "run",
            "execute",
            run_id,
            "--fixture",
            str(ROOT / "tests" / "fixtures" / "basic_replay.json"),
            "--db",
            str(database),
        ],
    )
    assert executed.exit_code == 0
    assert json.loads(executed.stdout)["status"] == "COMPLETED"
    status = runner.invoke(app, ["run", "status", run_id, "--db", str(database)])
    assert json.loads(status.stdout)["simulation_attempts"] == 2
    report = runner.invoke(app, ["report", run_id, "--db", str(database)])
    assert "# Run report" in report.stdout
    stopped = runner.invoke(
        app,
        ["run", "stop", run_id, "--reason", "TEST_STOP", "--db", str(database)],
    )
    assert stopped.exit_code == 0


def test_cli_refuses_live_or_missing_fixture(tmp_path: Path, monkeypatch: object) -> None:
    database = tmp_path / "lab.db"
    monkeypatch.chdir(ROOT)
    created = runner.invoke(
        app,
        [
            "run",
            "create",
            "--dataset",
            "demo-dataset",
            "--region",
            "USA",
            "--universe",
            "TOP1000",
            "--delay",
            "1",
            "--db",
            str(database),
        ],
    )
    run_id = created.stdout.strip()
    missing = runner.invoke(app, ["run", "execute", run_id, "--db", str(database)])
    assert missing.exit_code != 0
    live = runner.invoke(
        app,
        ["run", "execute", run_id, "--adapter", "live", "--db", str(database)],
    )
    assert live.exit_code != 0


@respx.mock
def test_cli_stops_cleanly_before_spending_budget_on_a_near_expiry_session(
    tmp_path: Path, monkeypatch: object
) -> None:
    """A session too close to expiry must abort `run execute` before it ever reaches
    /simulations -- not fail unpredictably partway through a candidate's retries."""
    database = tmp_path / "lab.db"
    monkeypatch.chdir(ROOT)
    session_file = tmp_path / "brain-session.json"
    session_file.write_text(json.dumps({"cookies": {"t": "test-only"}}))
    session_file.chmod(0o600)
    monkeypatch.setenv("ALPHA_LAB_BRAIN_SESSION_FILE", str(session_file))

    created = runner.invoke(
        app,
        [
            "run",
            "create",
            "--dataset",
            "demo-dataset",
            "--region",
            "USA",
            "--universe",
            "TOP1000",
            "--delay",
            "1",
            "--db",
            str(database),
        ],
    )
    run_id = created.stdout.strip()

    authentication = respx.get("https://api.worldquantbrain.com/authentication").mock(
        return_value=httpx.Response(200, json={"token": {"expiry": 30.0}})
    )
    simulations = respx.post("https://api.worldquantbrain.com/simulations")

    live = runner.invoke(
        app,
        ["run", "execute", run_id, "--adapter", "live", "--db", str(database)],
    )
    assert live.exit_code != 0
    assert "expires in 30s" in live.output
    assert authentication.called
    assert not simulations.called
    status = runner.invoke(app, ["run", "status", run_id, "--db", str(database)])
    assert json.loads(status.stdout)["simulation_attempts"] == 0


def test_cli_corpus_and_card_approval_workflow(tmp_path: Path, monkeypatch: object) -> None:
    database = tmp_path / "lab.db"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "method.md").write_text(
        """# Method\n\n```method-card
card_id: MC-CLI-001
title: CLI method card
version: v1
applicable_failure_codes: [LOW_COVERAGE]
applicable_data_types: [MATRIX]
prerequisites: [Confirm metadata]
diagnostic_signals: [Coverage is low]
allowed_repair_actions: [Change one field]
prohibited_actions: [Do not change limits]
stop_condition: Stop after one repair.
evidence_level: training
```\n""",
        encoding="utf-8",
    )
    monkeypatch.chdir(ROOT)
    inventoried = runner.invoke(
        app,
        [
            "corpus",
            "inventory",
            str(corpus),
            "--source-type",
            "training",
            "--db",
            str(database),
        ],
    )
    assert inventoried.exit_code == 0
    source_id = json.loads(inventoried.stdout)["source_ids"][0]
    extracted = runner.invoke(app, ["corpus", "extract", source_id, "--db", str(database)])
    assert json.loads(extracted.stdout) == ["MC-CLI-001"]
    drafts = runner.invoke(app, ["cards", "list", "--status", "draft", "--db", str(database)])
    assert json.loads(drafts.stdout)[0]["status"] == "draft"
    approved = runner.invoke(
        app,
        [
            "cards",
            "approve",
            "MC-CLI-001",
            "--reviewer",
            "CLI Reviewer",
            "--db",
            str(database),
        ],
    )
    assert json.loads(approved.stdout)["status"] == "approved"
    retired = runner.invoke(
        app,
        [
            "cards",
            "retire",
            "MC-CLI-001",
            "--version",
            "v1",
            "--reviewer",
            "CLI Reviewer",
            "--db",
            str(database),
        ],
    )
    assert json.loads(retired.stdout)["status"] == "retired"
    assert (
        runner.invoke(app, ["cards", "list", "--status", "approved", "--db", str(database)]).stdout
        == "[]\n"
    )
