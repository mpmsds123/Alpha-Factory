import plistlib
from pathlib import Path

from typer.testing import CliRunner

from alpha_orchestrator.factory import cli
from alpha_orchestrator.factory.config import REPO_ROOT


def test_scheduler_pins_code_config_and_session(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config = tmp_path / "factory.yaml"
    config.write_text(f"db_path: {tmp_path / 'factory.db'}\n")
    plist = tmp_path / "worker.plist"
    monkeypatch.setattr(cli, "PLIST", plist)
    monkeypatch.setenv("BRAIN_STATE_DIR", str(tmp_path / "session"))
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: None)
    result = CliRunner().invoke(cli.app, ["schedule", "install", "--config", str(config)])
    assert result.exit_code == 0, result.output
    data = plistlib.loads(plist.read_bytes())
    assert data["ProgramArguments"][-2:] == ["--config", str(config)]
    assert data["EnvironmentVariables"]["PYTHONPATH"] == str(REPO_ROOT / "src")
    assert data["EnvironmentVariables"]["BRAIN_STATE_DIR"] == str(tmp_path / "session")


def test_doctor_does_not_create_missing_ledger(tmp_path: Path) -> None:
    config = tmp_path / "factory.yaml"
    db = tmp_path / "missing.db"
    config.write_text(f"db_path: {db}\n")
    result = CliRunner().invoke(cli.app, ["doctor", "--config", str(config)])
    assert result.exit_code == 1 and "ledger_missing" in result.output
    assert not db.exists()
