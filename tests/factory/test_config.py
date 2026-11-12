from datetime import UTC, datetime
from pathlib import Path

import pytest

from alpha_orchestrator.factory.config import Budget, FactoryConfig, Gates


def test_gates_are_the_official_delay1_numbers() -> None:
    g = Gates()
    assert (g.sharpe_min, g.fitness_min, g.turnover_min, g.turnover_max) == (1.58, 1.0, 0.01, 0.7)
    assert g.sharpe_line("CHN") == 2.08 and g.sharpe_line("USA") == 1.58


def test_gates_cannot_be_overridden_from_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "factory.yaml"
    cfg.write_text("gates: {sharpe_min: 0.5}\nprivate_dir: " + str(tmp_path / "p") + "\n")
    assert FactoryConfig.load(cfg).gates.sharpe_min == 1.58


def test_local_overrides_merge_after_yaml(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir()
    (private / "factory.local.yaml").write_text("scanner_runs: /x\nbudget: {chunk: 7}\n")
    cfg = tmp_path / "factory.yaml"
    cfg.write_text(f"private_dir: {private}\nbudget: {{daily_sim_cap: 50}}\n")
    loaded = FactoryConfig.load(cfg)
    assert loaded.scanner_runs == "/x"
    assert (loaded.budget.daily_sim_cap, loaded.budget.chunk) == (50, 7)
    assert loaded.reports_dir == str(tmp_path / "reports")


def test_llm_model_overrides_keep_other_steps(tmp_path: Path) -> None:
    cfg = tmp_path / "factory.yaml"
    cfg.write_text(f"private_dir: {tmp_path}\nllm: {{models: {{diagnosis: m2}}}}\n")
    models = FactoryConfig.load(cfg).llm.models
    assert models["diagnosis"] == "m2" and models["hypothesis"]


def test_quota_window_starts_at_reset_hour() -> None:
    b = Budget()
    start = b.window_start(datetime(2026, 11, 12, 6, 0, tzinfo=UTC))  # 22:00 PT on the 11th
    assert start == datetime(2026, 11, 12, 5, 0, tzinfo=UTC)
    with pytest.raises(ValueError):
        b.window_start(datetime(2026, 11, 12))
    with pytest.raises(ValueError):
        Budget(children_per_slot=11)
