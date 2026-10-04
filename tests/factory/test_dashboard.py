import random
from pathlib import Path

import pytest

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.daily import run_day
from alpha_orchestrator.factory.dashboard import render
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM

from .fakes import FakePlatform
from .test_day import FIELDS, platform_metrics, scripted_llm


def test_dashboard_renders_after_a_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    ledger = Ledger("sqlite://")
    cfg = FactoryConfig()
    cfg.budget.min_batch_interval_s = 0
    llm = FactoryLLM(cfg, ledger, complete=scripted_llm)
    api = FakePlatform(platform_metrics, FIELDS)
    for seed in (3, 4):  # day 1 probes; day 2 proposes hypotheses for the dataset with signal
        run_day(
            cfg,
            api,
            ledger,
            llm,
            minutes=5,
            rng=random.Random(seed),  # type: ignore[arg-type]
            sleep=lambda _s: None,
        )
    page = render(ledger, cfg, tmp_path / "dash.html").read_text()
    assert "待你确认" in page and "挖掘地图" in page
    assert "goodf_fy" in page  # reviews show real expressions (file stays local)
    assert "Revision drift" in page  # hypotheses wait for approval by default
