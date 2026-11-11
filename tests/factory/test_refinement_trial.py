import json
import random
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from alpha_orchestrator.factory import catalog
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.engine import Engine
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import CandidateORM, EventORM, ResearchIdeaORM
from alpha_orchestrator.factory.record import render_record
from alpha_orchestrator.factory.research import Metadata
from alpha_orchestrator.factory.scheduler import Simulator

from .fakes import is_metrics
from .test_engine import honest_llm
from .world import World, analyst_fields

TARGET = "LOW_SUB_UNIVERSE_SHARPE"


@pytest.mark.parametrize(
    ("llm_fix", "target_result", "new_sims", "llm_state"),
    [
        ("liquidity_bucket_zscore", "PASS", 2, "observed"),
        ("liquidity_bucket_zscore", "PENDING", 2, "unusable"),
        ("cap_bucket_zscore", "PASS", 1, "observed"),
        (None, "PASS", 1, "abstained"),
    ],
)
def test_pair_survives_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    llm_fix: str | None,
    target_result: str,
    new_sims: int,
    llm_state: str,
) -> None:
    path = tmp_path / "factory.db"
    ledger = Ledger.at(path)
    cfg = FactoryConfig(design="v7", refinement_trial=True)
    cfg.budget.min_batch_interval_s = 0
    api = World({"analyst": analyst_fields(None)}).platform()
    calls = 0

    def completion(model: str, system: str, user: str) -> tuple[str, int, int]:
        nonlocal calls
        prompt = json.loads(user)
        if not prompt["task"].startswith("Choose the one next test"):
            return honest_llm(model, system, user)
        calls += 1
        assert TARGET in prompt["evidence"]["target_checks"]
        return (
            json.dumps(
                {
                    "action": "test" if llm_fix else "abstain",
                    "fix": llm_fix,
                    "hypothesis": "Prices underreact to revisions.",
                    "alternative": "The signal is a liquidity proxy.",
                    "prediction": "Liquidity grouping preserves the sign.",
                    "falsifier": "The within-liquidity spread disappears.",
                    "why": "Isolate a plausible exposure.",
                }
            ),
            10,
            10,
        )

    def metrics(expression: str, _settings: dict[str, Any]) -> dict[str, Any]:
        result = is_metrics(1.65, 1.2, fails=TARGET)
        if "bucket(rank(close*volume)" in expression:
            result["checks"][0]["result"] = target_result
        return result

    api.metrics = metrics
    metadata = Metadata.fetch(api)
    catalog.refresh(api, ledger, cfg)
    engine = Engine(
        api, ledger, cfg, metadata, FactoryLLM(cfg, ledger, completion), random.Random(3)
    )
    idea_id = engine.generate_one()
    assert idea_id
    Simulator(api, ledger, cfg, sleep=lambda _: None).run(3)
    engine.evaluate()
    engine._step_idea(idea_id)
    engine._step_idea(idea_id)
    with ledger.session() as s:
        idea = s.get(ResearchIdeaORM, idea_id)
        assert idea is not None
        plan = idea.extra["refinement_trial"]
        assert plan["choices"] == {"rule": "cap_bucket_zscore", "llm": llm_fix}
        assert len(plan["candidates"]) == new_sims
        parent = s.get(CandidateORM, plan["parent"])
        assert parent
        for cid in plan["candidates"].values():
            row = s.get(CandidateORM, cid)
            assert row and row.parent_id == parent.id and row.settings == parent.settings

    before = api.started

    def interrupt(_location: str) -> None:
        raise InterruptedError("worker stopped after the platform accepted the batch")

    with monkeypatch.context() as patch:
        patch.setattr(api, "poll", interrupt)
        with pytest.raises(InterruptedError):
            Simulator(api, ledger, cfg, sleep=lambda _: None).run(new_sims)
    assert api.started - before == new_sims
    ledger.engine.dispose()
    resumed = Ledger.at(path)
    cfg.refinement_trial = False
    engine = Engine(api, resumed, cfg, metadata, FactoryLLM(cfg, resumed, completion))
    Simulator(api, resumed, cfg, sleep=lambda _: None).run(0)
    engine.evaluate()
    engine.evaluate()
    assert api.started - before == new_sims and calls == 1
    with resumed.session() as s:
        idea = s.get(ResearchIdeaORM, idea_id)
        assert idea
        trial = idea.extra["refinement_trial"]
        assert {k: v for k, v in trial.items() if k != "result"} == plan
        assert trial["result"]["rule"]["success"] is False
        assert trial["result"]["llm"]["state"] == llm_state
        if llm_state == "observed":
            assert trial["result"]["llm"]["success"] == (llm_fix != "cap_bucket_zscore")
        assert (
            len(list(s.scalars(select(EventORM).where(EventORM.kind == "refinement_trial_queued"))))
            == 1
        )
    assert "配对改进试验" in render_record(resumed, tmp_path / "record.html").read_text()
    assert api.submitted == []
