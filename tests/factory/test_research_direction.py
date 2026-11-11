import json
import random
from pathlib import Path

import pytest
from sqlalchemy import select

from alpha_orchestrator.factory import engine, study
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM, LLMUnavailable
from alpha_orchestrator.factory.models import CellORM, EventORM, LLMCallORM, ResearchIdeaORM
from alpha_orchestrator.factory.record import render_record
from alpha_orchestrator.factory.research import Metadata

from .test_v7 import v7_llm
from .world import World, analyst_fields


def _proposal(cell_id: str) -> dict[str, str]:
    return {
        "cell_id": cell_id,
        "metadata_basis": "analyst estimates",
        "theory_id": "earnings_underreaction",
        "mechanism": "slow price adjustment to upward earnings revisions",
        "why_first": "use revisions rather than earnings level",
        "competing_explanation": "small stocks are illiquid",
        "falsifier": "no positive spread after liquidity adjustment",
    }


@pytest.mark.parametrize("arm_draw", [0.0, 0.9])
def test_trial_reaches_formula_generation_and_records_signed_result(
    arm_draw: float, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    field_specs = analyst_fields(
        "eps", strength=1.4, prefix="est", signal_stats=("mean", "median", "high", "low")
    )
    for field in field_specs.values():
        field.sign = -1
    world = World({"est": field_specs, "noise": analyst_fields(None, prefix="other")}, noise=0)
    ledger = Ledger("sqlite://")
    cfg = FactoryConfig(design="v7", direction_trial=True)
    cfg.budget.min_batch_interval_s = 0
    rng = random.Random(3)
    monkeypatch.setattr(rng, "random", lambda: arm_draw)
    # Same selected dataset in both arms isolates whether the direction reaches the prompt.
    monkeypatch.setattr(rng, "choice", lambda rows: next(r for r in rows if r["id"] == "est"))
    prompts = []

    def complete(model: str, system: str, user: str) -> tuple[str, int, int]:
        prompt = json.loads(user)
        if prompt["task"].startswith("Select the one dataset"):
            choice = next(c for c in prompt["choices"] if c["id"] == "est")
            return json.dumps(_proposal(choice["cell_id"])), 10, 10
        if prompt["task"].startswith("You are a quant researcher reading"):
            prompts.append(prompt)
        return v7_llm(model, system, user)

    api = world.platform()
    llm = FactoryLLM(cfg, ledger, complete=complete)
    engine.run_day(cfg, api, ledger, llm, minutes=1, rng=rng, sleep=lambda _: None)
    guided = arm_draw < 0.5
    assert ("research_direction" in prompts[0]["dataset"]) == guided
    with ledger.session() as s:
        assignment = s.scalar(select(EventORM).where(EventORM.kind == "direction_assigned"))
        assert assignment is not None
        assert assignment.payload["arm"] == ("guided" if guided else "random")
        assert len(assignment.payload["shortlist"]) == 2
        cell = s.get(CellORM, assignment.cell_id)
        assert cell is not None and cell.plan["direction_result"]["settled"] > 0
        idea = s.scalar(
            select(ResearchIdeaORM).where(
                ResearchIdeaORM.cell_id == cell.cell_id, ResearchIdeaORM.title == "EPS revisions"
            )
        )
        assert idea is not None
        assert idea.extra["direction_probe"]["signed_sharpe"] < 0
        assert idea.best_sharpe > 0
    record = render_record(ledger, tmp_path / "record.html").read_text()
    assert "研究方向试验" in record and "small stocks are illiquid" in record
    assert api.submitted == []


def test_bad_direction_retries_and_assignment_survives_field_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = World({"est": analyst_fields(None), "other": analyst_fields(None)}).platform()
    ledger = Ledger("sqlite://")
    cfg = FactoryConfig(design="v7", direction_trial=True)
    engine.catalog.refresh(api, ledger, cfg)
    valid = False

    def complete(_model: str, _system: str, user: str) -> tuple[str, int, int]:
        choice = json.loads(user)["choices"][0]["cell_id"]
        return json.dumps(_proposal(choice if valid else "unavailable")), 10, 10

    llm = FactoryLLM(cfg, ledger, complete=complete)
    runner = engine.Engine(api, ledger, cfg, Metadata.fetch(api), llm, random.Random(0))
    with pytest.raises(LLMUnavailable, match="outside the shortlist"):
        runner._open_dataset()
    with ledger.session() as s:
        assert list(s.scalars(select(LLMCallORM.ok))) == [False, False]
        assert not list(s.scalars(select(EventORM).where(EventORM.kind == "direction_assigned")))

    valid = True

    def fail(*_args: object) -> int:
        raise TimeoutError("metadata request interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(study, "load_fields", fail)
        with pytest.raises(TimeoutError):
            runner._open_dataset()
    valid = False
    selected = runner._open_dataset()
    with ledger.session() as s:
        assignments = list(s.scalars(select(EventORM).where(EventORM.kind == "direction_assigned")))
        assert [event.cell_id for event in assignments] == [selected]
        assert list(s.scalars(select(LLMCallORM.ok).order_by(LLMCallORM.created_at))) == [
            False,
            False,
            True,
        ]
