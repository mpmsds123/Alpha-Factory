import plistlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from alpha_orchestrator.factory import cli
from alpha_orchestrator.factory.config import REPO_ROOT, FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger, WorkerAlreadyRunning


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


def test_competing_cli_does_not_probe_authentication(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    db = tmp_path / "factory.db"
    owner, contender = Ledger.at(db), Ledger.at(db)
    cfg = FactoryConfig()
    calls = []
    session = SimpleNamespace(ensure=lambda: calls.append("auth"), close=lambda: None)
    api = SimpleNamespace(client=SimpleNamespace(session=session))
    monkeypatch.setattr(cli, "_ctx", lambda _: (cfg, contender))
    monkeypatch.setattr(cli, "_llm", lambda *_: object())
    monkeypatch.setattr(cli, "_api", lambda: api)
    with owner.writer_lock():
        result = CliRunner().invoke(cli.app, ["run"])
    assert result.exit_code == 4, result.output
    assert calls == []


def test_context_lock_contention_has_explicit_safe_exit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def owned(_config):  # type: ignore[no-untyped-def]
        raise WorkerAlreadyRunning("Another worker owns this ledger")

    monkeypatch.setattr(cli, "_ctx", owned)
    result = CliRunner().invoke(cli.app, ["run"])
    assert result.exit_code == 4 and "安全暂停" in result.output


def test_cli_owns_ledger_during_auth_and_releases_after_run(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from alpha_orchestrator.factory import engine, record

    db = tmp_path / "factory.db"
    ledger = Ledger.at(db)
    cfg = FactoryConfig()
    calls = []

    def ensure() -> None:
        with pytest.raises(WorkerAlreadyRunning):
            Ledger.at(db)
        calls.append("auth")

    def run_day(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        with ledger.writer_lock():
            calls.append("engine")
        return engine.DayReport()

    session = SimpleNamespace(ensure=ensure, close=lambda: calls.append("closed"))
    monkeypatch.setattr(cli, "_ctx", lambda _: (cfg, ledger))
    monkeypatch.setattr(cli, "_llm", lambda *_: object())
    monkeypatch.setattr(
        cli, "_api", lambda: SimpleNamespace(client=SimpleNamespace(session=session))
    )
    monkeypatch.setattr(cli, "notify", lambda _: None)
    monkeypatch.setattr(record, "render_record", lambda *_: None)
    monkeypatch.setattr(engine, "run_day", run_day)
    result = CliRunner().invoke(cli.app, ["run"])
    assert result.exit_code == 0, result.output
    assert calls == ["auth", "engine", "closed"]
    with Ledger.at(db).writer_lock():
        pass
