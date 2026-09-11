import json
from pathlib import Path

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
