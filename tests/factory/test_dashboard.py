import random
from pathlib import Path

import pytest

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.daily import run_day
from alpha_orchestrator.factory.dashboard import render
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM

from .world import World, analyst_fields, scripted_llm


def test_dashboard_renders_after_a_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    ledger = Ledger("sqlite://")
    cfg = FactoryConfig()
    cfg.budget.min_batch_interval_s = 0
    world = World(
        {
            "flat": analyst_fields(None, prefix="flt"),
            "est": analyst_fields("eps", signal_stats=("mean", "median", "high", "low")),
        }
    )
    run_day(
        cfg,
        world.platform(),
        ledger,
        FactoryLLM(cfg, ledger, complete=scripted_llm),
        minutes=10,
        rng=random.Random(3),
        sleep=lambda _s: None,
    )
    page = render(ledger, cfg, tmp_path / "dash.html").read_text()
    assert "研究日志" in page and "待你确认" in page and "数据集队列" in page
    assert "体检" in page and "普查" in page and "和噪声没区别" in page  # the flat dataset
    assert "各层信号浓度" in page and "DeepSeek" in page  # the dataset with signal
    assert "anl_eps" in page  # reviews show real expressions (file stays local)
