from typing import Any

from sqlalchemy import select

from alpha_orchestrator.factory.catalog import upsert_cells
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.generators import base_settings
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import BatchORM, CandidateORM
from alpha_orchestrator.factory.scheduler import Simulator

from .fakes import FakeBrainAPI, is_metrics

CELL = "USA|1|TOP3000|ds"


def setup(n: int) -> Ledger:
    ledger = Ledger("sqlite://")
    settings = base_settings("USA", "TOP3000", 1, "analyst")
    with ledger.session() as s:
        upsert_cells(s, [{"id": "ds", "fieldCount": 10}], "USA", 1, "TOP3000")
        for i in range(n):
            ledger.add_candidate(
                s,
                cell_id=CELL,
                stage="PROBE",
                expression=f"rank(f{i})",
                settings=settings,
                priority=float(i),
            )
    return ledger


def metrics(expr: str, _settings: dict[str, Any]) -> dict[str, Any] | None:
    if expr == "rank(f3)":
        return None
    return is_metrics(sharpe=float(expr[6:-1]) / 10, fitness=0.5)


def sim(ledger: Ledger, api: FakeBrainAPI) -> Simulator:
    cfg = FactoryConfig()
    cfg.budget.min_batch_interval_s = 0
    return Simulator(api, ledger, cfg, sleep=lambda _s: None, poll_interval=0)  # type: ignore[arg-type]


def test_batches_fill_and_results_land_on_the_right_rows() -> None:
    ledger = setup(23)
    api = FakeBrainAPI(metrics)
    stats = sim(ledger, api).run(sim_budget=100)
    assert stats.batches == 3 and stats.completed == 22 and stats.failed == 1
    with ledger.session() as s:
        rows = {c.expression: c for c in s.scalars(select(CandidateORM))}
    assert rows["rank(f12)"].sharpe == 1.2 and rows["rank(f12)"].alpha_id
    assert rows["rank(f3)"].status == "FAILED"


def test_budget_caps_simulations() -> None:
    ledger = setup(30)
    api = FakeBrainAPI(metrics)
    stats = sim(ledger, api).run(sim_budget=15)
    assert api.started == 15 and stats.stopped_because == "budget"


def test_refused_batch_returns_to_queue() -> None:
    ledger = setup(5)
    api = FakeBrainAPI(metrics, refuse_first=1)
    stats = sim(ledger, api).run(sim_budget=100)
    assert stats.completed == 4 and stats.failed == 1
    with ledger.session() as s:
        assert {b.status for b in s.scalars(select(BatchORM))} == {"REFUSED", "DONE"}


def test_inflight_batch_resumes_without_resimulating() -> None:
    ledger = setup(4)
    api = FakeBrainAPI(metrics, polls_before_done=10**6)
    simulator = sim(ledger, api)
    # launch one batch, then "crash" before it finishes
    assert simulator._launch_one(10) == 4
    api.polls_before_done = 0
    stats = sim(ledger, api).run(sim_budget=0)
    assert api.started == 4 and stats.completed == 3


class RejectingAPI(FakeBrainAPI):
    """Answers like BRAIN when one formula in a batch has an invalid setting."""

    def start_multi_simulation(self, payloads: list[dict[str, Any]]) -> str:
        if any(p["regular"] == "rank(f5)" for p in payloads):
            body = [
                '{"settings":{"decay":["Ensure this value is less than or equal to 512."]}}'
                if p["regular"] == "rank(f5)"
                else "{}"
                for p in payloads
            ]
            raise ValueError("Simulation refused (400): [" + ",".join(body) + "]")
        return super().start_multi_simulation(payloads)


def test_invalid_settings_fail_one_formula_not_the_run() -> None:
    ledger = setup(12)
    api = RejectingAPI(metrics, polls_before_done=0)
    run = sim(ledger, api).run(100)
    with ledger.session() as s:
        rows = {c.expression: c for c in s.scalars(select(CandidateORM))}
    assert rows["rank(f5)"].status == "FAILED" and "512" in (rows["rank(f5)"].error or "")
    assert rows["rank(f6)"].status == "DONE"  # its batch mates were sent again
    assert run.stopped_because != "platform kept refusing batches"
