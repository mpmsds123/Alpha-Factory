from pathlib import Path

import pytest
from typer.testing import CliRunner

from alpha_orchestrator.factory import cli
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import (
    ExpressionORM,
    HypothesisORM,
    Review,
    SimulationORM,
)

RUNNER = CliRunner()


def _config(tmp_path: Path) -> Path:
    cfg = tmp_path / "factory.yaml"
    cfg.write_text(
        f"db_path: {tmp_path / 'f.db'}\nreports_dir: {tmp_path / 'out'}\n"
        f"private_dir: {tmp_path / 'private'}\nv7_db: {tmp_path / 'none.db'}\n"
        f"llm: {{key_file: {tmp_path / 'nokey.json'}}}\n"
    )
    return cfg


def _invoke(*args: str) -> str:
    result = RUNNER.invoke(cli.app, list(args))
    assert result.exit_code == 0, result.output
    return result.output


def test_status_stats_and_priors(tmp_path: Path) -> None:
    cfg = str(_config(tmp_path))
    assert "LLM guard: ok" in _invoke("status", "--config", cfg)
    assert "CSVs:" in _invoke("stats", "--config", cfg)
    assert (tmp_path / "out" / "stages.csv").exists()
    out = _invoke(
        "priors",
        "add",
        "--claim",
        "slow data drifts",
        "--source",
        "own note",
        "--hypothesis-type",
        "change",
        "--category",
        "Cat",
        "--config",
        cfg,
    )
    assert out.strip() == "added P1"
    again = _invoke(
        "priors", "add", "--claim", "slow data drifts", "--source", "own note", "--config", cfg
    )
    assert again.strip() == "already there"
    bad = RUNNER.invoke(
        cli.app,
        [
            "priors",
            "add",
            "--claim",
            "c",
            "--source",
            "s",
            "--hypothesis-type",
            "nope",
            "--config",
            cfg,
        ],
    )
    assert bad.exit_code != 0
    assert "imported 0" in _invoke("memory", "import-v7", "--config", cfg)


def test_commands_needing_an_llm_key_stop_cleanly(tmp_path: Path) -> None:
    cfg = str(_config(tmp_path))
    for args in (["run"], ["priors", "seed"]):
        result = RUNNER.invoke(cli.app, [*args, "--config", cfg])
        assert result.exit_code == 1 and "No LLM key" in result.output


def test_review_list_and_reject(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    ledger = Ledger.at(tmp_path / "f.db")
    with ledger.session() as s:
        h = HypothesisORM(region="R1", dataset_id="d", claim="a claim about it")
        s.add(h)
        s.flush()
        e = ExpressionORM(hypothesis_id=h.id, expression="op(f)")
        s.add(e)
        s.flush()
        s.add(
            SimulationORM(
                expression_id=e.id,
                hypothesis_id=h.id,
                region="R1",
                delay=1,
                settings={},
                fingerprint="x",
                expression="op(f)",
                alpha_id="A1",
                review=Review.PENDING,
            )
        )
    ledger.engine.dispose()
    assert "alpha A1" in _invoke("review", "list", "--config", str(cfg))
    assert "Rejected." in _invoke("review", "reject", "1", "--config", str(cfg))
    assert "Not pending." in _invoke("review", "reject", "1", "--config", str(cfg))


def test_recover_batch_validates_url(tmp_path: Path) -> None:
    cfg = str(_config(tmp_path))
    result = RUNNER.invoke(
        cli.app, ["recover-batch", "b1", "--location", "https://evil/x", "--config", cfg]
    )
    assert result.exit_code != 0
    result = RUNNER.invoke(
        cli.app, ["recover-batch", "b1", "--location", "/simulations/abc", "--config", cfg]
    )
    assert result.exit_code != 0  # no such UNKNOWN batch


def test_daily_retries_pauses_and_stops_on_face_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import typer

    cfg = str(_config(tmp_path))
    calls: list[float] = []
    codes = iter([3, 3, 0])

    def fake_run(minutes: float, config: Path) -> None:
        calls.append(minutes)
        code = next(codes)
        if code:
            raise typer.Exit(code)

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr("time.sleep", lambda _: None)
    assert RUNNER.invoke(cli.app, ["daily", "--config", cfg]).exit_code == 0
    assert len(calls) == 3  # two pauses, then a normal finish

    def face(minutes: float, config: Path) -> None:
        calls.append(minutes)
        raise typer.Exit(2)

    monkeypatch.setattr(cli, "run", face)
    calls.clear()
    assert RUNNER.invoke(cli.app, ["daily", "--config", cfg]).exit_code == 2
    assert len(calls) == 1  # no retry: the face check needs the owner


def test_schedule_prints_and_writes_a_launchd_job(tmp_path: Path) -> None:
    out = _invoke("schedule", "--hour", "1", "--minute", "30")
    assert "<integer>1</integer>" in out and "<string>daily</string>" in out
    _invoke("schedule", "--hour", "1", "--write")
    assert (tmp_path / "home" / "Library" / "LaunchAgents" / f"{cli.LAUNCH_LABEL}.plist").exists()
    assert RUNNER.invoke(cli.app, ["schedule", "--hour", "25"]).exit_code != 0


def test_daily_stops_before_the_quota_reset(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    cfg = _config(tmp_path)
    # 15:00 Pacific (23:00 UTC in winter): the 21:00 reset is 6 hours away, minus 15 min
    at = datetime(2026, 11, 26, 23, 0, tzinfo=UTC)
    assert cli._seconds_to_reset(cfg, at) == 6 * 3600 - 900
