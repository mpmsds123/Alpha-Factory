"""Failure and restart behavior, not strategy-performance assertions."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.brain.client import AmbiguousRequest, BrainClient
from alpha_orchestrator.brain.session import BrainSession, SessionStore
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger, WorkerAlreadyRunning
from alpha_orchestrator.factory.models import BatchORM, CandidateORM, ReviewORM
from alpha_orchestrator.factory.scheduler import Simulator
from alpha_orchestrator.factory.select import approve, checks_passed, store_passing

from .fakes import FakeBrainAPI, is_metrics
from .test_scheduler import metrics, setup, sim


def test_daily_allocation_survives_restart_with_unfinished_requests() -> None:
    ledger = setup(30)
    api = FakeBrainAPI(metrics, polls_before_done=1000)
    cfg = FactoryConfig()
    cfg.budget.v1_daily_cap = 13
    cfg.budget.min_batch_interval_s = 0
    first = Simulator(api, ledger, cfg, sleep=lambda _: None)
    assert first._launch_one(10) == 10
    api.polls_before_done = 0
    second = Simulator(api, ledger, cfg, sleep=lambda _: None)
    second.run(100)
    assert api.started == 13
    second.run(100)
    assert api.started == 13
    with ledger.session() as s:
        assert ledger.sims_reserved_today(s) == 13


def test_quota_tail_trims_batch_instead_of_overshooting() -> None:
    ledger = setup(20)
    api = FakeBrainAPI(metrics)
    cfg = FactoryConfig()
    cfg.budget.quota_reserve = 3
    cfg.budget.min_batch_interval_s = 0
    api.client.quota.remaining = 7

    original = api.start_multi_simulation

    def start(payloads):  # type: ignore[no-untyped-def]
        api.client.quota.remaining -= len(payloads)
        return original(payloads)

    api.start_multi_simulation = start  # type: ignore[method-assign]
    Simulator(api, ledger, cfg, sleep=lambda _: None).run(20)
    assert api.started == 4 and api.client.quota.remaining == 3


def test_lost_creation_response_is_quarantined_never_requeued() -> None:
    ledger = setup(6)
    api = FakeBrainAPI(metrics)
    original = api.start_multi_simulation

    def lost(payloads):  # type: ignore[no-untyped-def]
        original(payloads)  # Accepted by server before the response disappears.
        raise AmbiguousRequest("lost response")

    api.start_multi_simulation = lost  # type: ignore[method-assign]
    with pytest.raises(AmbiguousRequest):
        sim(ledger, api).run(100)
    with pytest.raises(AmbiguousRequest, match="Reconcile"):
        sim(ledger, api).run(100)
    assert api.started == 6
    with ledger.session() as s:
        assert ledger.sims_reserved_today(s) == 6
        assert s.scalar(select(BatchORM.status)) == "UNKNOWN"
        assert {c.status for c in s.scalars(select(CandidateORM))} == {"RUNNING"}


def test_crash_before_url_commit_does_not_recreate_batch() -> None:
    ledger = setup(5)
    api = FakeBrainAPI(metrics)
    simulator = sim(ledger, api)
    simulator._launch_one(5)
    with ledger.session() as s:
        s.scalar(select(BatchORM)).location = ""
    with pytest.raises(AmbiguousRequest):
        sim(ledger, api).run(100)
    assert api.started == 5


def test_worker_lock_blocks_separate_ledger_instances(tmp_path: Path) -> None:
    a, b = Ledger.at(tmp_path / "factory.db"), Ledger.at(tmp_path / "factory.db")
    with (
        a.writer_lock(),
        a.writer_lock(),
        pytest.raises(WorkerAlreadyRunning),
        b.writer_lock(),
    ):
        pytest.fail("second worker acquired lock")
    with b.writer_lock():
        pass


def test_deadline_preserves_inflight_work_for_resume() -> None:
    ledger = setup(5)
    api = FakeBrainAPI(metrics, polls_before_done=1000)
    now = [0.0]
    cfg = FactoryConfig()
    cfg.budget.min_batch_interval_s = 0

    def advance(seconds: float) -> None:
        now[0] += seconds

    worker = Simulator(api, ledger, cfg, clock=lambda: now[0], sleep=advance)
    result = worker.run(10, deadline=1.0)
    assert result.stopped_because == "deadline" and api.started == 5
    api.polls_before_done = 0
    result = sim(ledger, api).run(0)
    assert result.completed == 4 and result.failed == 1 and api.started == 5


def test_budget_is_charged_to_creation_day_not_completion_day() -> None:
    ledger = setup(0)
    with ledger.session() as s:
        s.add(
            BatchORM(
                region="USA",
                delay=1,
                size=10,
                status="DONE",
                created_at=datetime.now(UTC) - timedelta(days=1),
                finished_at=datetime.now(UTC),
            )
        )
        s.add(BatchORM(region="USA", delay=1, size=3, status="UNKNOWN"))
        s.add(BatchORM(region="USA", delay=1, size=7, status="REFUSED"))
    with ledger.session() as s:
        assert ledger.sims_reserved_today(s) == 3


@pytest.mark.parametrize("status", ["PENDING", "WARNING", "ERROR", "FAIL", "UNKNOWN", None])
def test_only_explicit_completed_checks_can_pass(status: str | None) -> None:
    checks = [
        {"name": "SELF_CORRELATION", "result": "PASS", "value": 0.2},
        {"name": "PROD_CORRELATION", "result": status, "value": 0.3},
    ]
    assert not checks_passed(checks, 0.7)


def test_empty_missing_nonfinite_or_excessive_correlation_blocks_promotion() -> None:
    assert not checks_passed([], 0.7)
    assert not checks_passed([{"name": "FITNESS", "result": "PASS"}], 0.7)
    for value in [None, float("nan"), float("inf"), 0.8]:
        checks = [
            {"name": "SELF_CORRELATION", "result": "PASS", "value": 0.2},
            {"name": "PROD_CORRELATION", "result": "PASS", "value": value},
        ]
        assert not checks_passed(checks, 0.7)


def test_unfinished_platform_check_is_retried_then_stored(tmp_path: Path) -> None:
    ledger = setup(1)
    api = FakeBrainAPI(metrics)
    sim(ledger, api).run(1)
    with ledger.session() as s:
        c = s.scalar(select(CandidateORM))
        c.extra = {"ready": True, "robust": {"pass": True, "ratios": {"neutralization": 0.8}}}
        alpha_id = c.alpha_id
    cfg = FactoryConfig(reports_dir=str(tmp_path))
    api.checks[alpha_id] = []
    assert store_passing(api, ledger, cfg) == []
    del api.checks[alpha_id]
    assert store_passing(api, ledger, cfg) == [alpha_id]
    with ledger.session() as s:
        rid = s.scalar(select(ReviewORM.id))
    api.checks[alpha_id] = [{"name": "PROD_CORRELATION", "result": "PENDING"}]
    assert not approve(api, ledger, rid, cfg)
    assert not api.submitted


@pytest.mark.parametrize("failure", ["timeout", "server_error", "missing_location"])
def test_mutating_request_is_never_retried_after_unknown_result(
    tmp_path: Path, failure: str
) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("response lost", request=request)
        return httpx.Response(503 if failure == "server_error" else 201)

    session = BrainSession(store=SessionStore(tmp_path), transport=httpx.MockTransport(handler))
    try:
        api = BrainAPI(BrainClient(session, sleep=lambda _: None))
        with pytest.raises(AmbiguousRequest):
            api.start_multi_simulation([{"regular": "rank(close)"}])
        assert len(calls) == 1
    finally:
        session.close()


def test_wrong_result_identity_cannot_be_attached_by_position() -> None:
    ledger = setup(2)
    api = FakeBrainAPI(lambda *_: is_metrics(sharpe=1.0, fitness=1.0))
    original = api.alpha

    def wrong(alpha_id):  # type: ignore[no-untyped-def]
        alpha = original(alpha_id)
        return {**alpha, "regular": {"code": "rank(wrong_field)"}}

    api.alpha = wrong  # type: ignore[method-assign]
    with pytest.raises(AmbiguousRequest, match="identity mismatch"):
        sim(ledger, api).run(2)
    with ledger.session() as s:
        assert s.scalar(select(BatchORM.status)) == "UNKNOWN"
        assert all(c.alpha_id is None for c in s.scalars(select(CandidateORM)))


def test_llm_failure_preserves_idea_and_retries_formulas() -> None:
    from alpha_orchestrator.factory.engine import Engine
    from alpha_orchestrator.factory.llm import FactoryLLM, LLMUnavailable
    from alpha_orchestrator.factory.models import IdeaStatus, ResearchIdeaORM
    from alpha_orchestrator.factory.research import Metadata

    from .test_engine import honest_llm
    from .world import World, analyst_fields

    ledger = Ledger("sqlite://")
    api = World({"est": analyst_fields("eps")}).platform()
    cfg = FactoryConfig()

    def disconnected(model: str, system: str, user: str):  # type: ignore[no-untyped-def]
        import json

        if json.loads(user)["task"].startswith("Write"):
            raise TimeoutError("LLM connection unavailable")
        return honest_llm(model, system, user)

    llm = FactoryLLM(cfg, ledger, complete=disconnected)
    meta = Metadata.fetch(api)
    from alpha_orchestrator.factory import catalog

    catalog.refresh(api, ledger, cfg)
    engine = Engine(api, ledger, cfg, meta, llm)
    with pytest.raises(LLMUnavailable):
        engine.generate_one()
    with ledger.session() as s:
        idea = s.scalar(select(ResearchIdeaORM))
        idea_id = idea.id
        assert idea.status == IdeaStatus.TESTING
        assert not idea.extra.get("rows") and not idea.extra.get("why_dead")
    llm._complete = honest_llm
    assert engine._step_idea(idea_id) > 0
    with ledger.session() as s:
        assert s.get(ResearchIdeaORM, idea_id).extra["rows"]


def test_llm_budget_stops_before_a_billable_request() -> None:
    from alpha_orchestrator.factory.llm import FactoryLLM, LLMUnavailable

    cfg = FactoryConfig()
    cfg.llm.max_calls_per_day = 0
    ledger = Ledger("sqlite://")

    def forbidden(*args):  # type: ignore[no-untyped-def]
        pytest.fail("called the LLM past its budget")

    with pytest.raises(LLMUnavailable, match="budget"):
        FactoryLLM(cfg, ledger, complete=forbidden).ask("test", "{}")


def test_health_check_does_not_modify_ledger(tmp_path: Path) -> None:
    from alpha_orchestrator.factory.health import inspect_local

    ledger = Ledger.at(tmp_path / "factory.db")
    with ledger.session() as s:
        s.add(BatchORM(region="USA", delay=1, size=7, status="UNKNOWN"))
    ledger.engine.dispose()
    path = tmp_path / "factory.db"
    before = path.read_bytes()
    report = inspect_local(FactoryConfig(db_path=str(path)))
    assert report["reserved_today"] == 7
    assert report["blockers"] == ["uncertain_batches"]
    assert path.read_bytes() == before


def test_full_thousand_simulation_day_and_restart_cannot_overspend() -> None:
    ledger = setup(1020)
    api = FakeBrainAPI(metrics)
    simulator = sim(ledger, api)
    simulator.run(1500)
    sim(ledger, api).run(1000)
    assert api.started == 1000
    with ledger.session() as s:
        assert ledger.sims_reserved_today(s) == 1000


def test_empty_robustness_results_never_pass() -> None:
    from alpha_orchestrator.factory.select import robust_passed

    assert not robust_passed({"robust": {"pass": True, "ratios": {}}}, 0.5)
    assert not robust_passed({"robust": {"pass": True, "ratios": {"n": float("nan")}}}, 0.5)


def test_llm_call_retains_request_and_response_provenance() -> None:
    from alpha_orchestrator.factory.llm import FactoryLLM
    from alpha_orchestrator.factory.models import LLMCallORM

    ledger = Ledger("sqlite://")
    llm = FactoryLLM(FactoryConfig(), ledger, complete=lambda *_: ('{"description":"ok"}', 12, 3))
    llm.ask("test", "{}")
    llm.ask("test", '{"x":1}')
    with ledger.session() as s:
        a, b = list(s.scalars(select(LLMCallORM).order_by(LLMCallORM.created_at)))
        assert len(a.request_hash) == len(a.response_hash) == len(a.system_hash) == 64
        assert a.request_hash != b.request_hash
        assert a.response_hash == b.response_hash and a.system_hash == b.system_hash


def test_existing_ledger_adds_provenance_without_losing_rows(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "old.db"
    ledger = Ledger.at(path)
    ledger.engine.dispose()
    with sqlite3.connect(path) as c:
        for column in ("request_hash", "response_hash", "system_hash"):
            c.execute(f"ALTER TABLE llm_calls DROP COLUMN {column}")
        c.execute(
            "INSERT INTO llm_calls "
            "(id,purpose,model,prompt_tokens,completion_tokens,cost_usd,cost_cny,ok,created_at) "
            "VALUES ('old','test','fake',0,0,0,0,1,'2026-10-26 00:00:00')"
        )
    upgraded = Ledger.at(path)
    with upgraded.session() as s:
        from alpha_orchestrator.factory.models import LLMCallORM

        row = s.get(LLMCallORM, "old")
        assert row is not None and row.request_hash == "" and row.model == "fake"


def test_budget_stopped_cell_resumes_only_with_more_allowance_and_remaining_work() -> None:
    from alpha_orchestrator.factory.engine import Engine
    from alpha_orchestrator.factory.llm import FactoryLLM
    from alpha_orchestrator.factory.models import CellORM, CellState
    from alpha_orchestrator.factory.research import Metadata
    from alpha_orchestrator.factory.study import DESIGN

    ledger = setup(0)
    cfg = FactoryConfig(only_datasets=["USA|1|TOP3000|ds"])
    cfg.budget.dataset_sim_cap = 100
    api = FakeBrainAPI(metrics)
    engine = Engine(api, ledger, cfg, Metadata(), FactoryLLM(cfg, ledger))
    with ledger.session() as s:
        cell = s.get(CellORM, cfg.only_datasets[0])
        cell.state, cell.sims_used = CellState.EXHAUSTED, 100
        cell.stop_reason = "budget spent; 3 ideas, 0 reached the regular line, 100 sims"
        cell.plan = {
            "design": DESIGN,
            "sims_at_open": 0,
            "groups": [{"name": "unused"}],
            "next_group": 0,
            "outcome": "noise",
        }
    assert engine._current_dataset() is None
    cfg.budget.dataset_sim_cap = 1000
    assert engine._current_dataset() == cfg.only_datasets[0]
    with ledger.session() as s:
        cell = s.get(CellORM, cfg.only_datasets[0])
        assert cell.sims_used == 100 and cell.plan["groups"] == [{"name": "unused"}]
        assert cell.state == CellState.RESEARCH and "outcome" not in cell.plan


def test_research_failure_is_not_reopened_by_a_budget_increase() -> None:
    from alpha_orchestrator.factory.engine import Engine
    from alpha_orchestrator.factory.llm import FactoryLLM
    from alpha_orchestrator.factory.models import CellORM, CellState
    from alpha_orchestrator.factory.research import Metadata
    from alpha_orchestrator.factory.study import DESIGN

    ledger = setup(0)
    cfg = FactoryConfig(only_datasets=["USA|1|TOP3000|ds"])
    with ledger.session() as s:
        cell = s.get(CellORM, cfg.only_datasets[0])
        cell.state = CellState.EXHAUSTED
        cell.stop_reason = "no valid new idea twice in a row"
        cell.plan = {"design": DESIGN, "groups": [{"name": "unused"}], "outcome": "noise"}
    engine = Engine(FakeBrainAPI(metrics), ledger, cfg, Metadata(), FactoryLLM(cfg, ledger))
    assert engine._current_dataset() is None


def test_duplicate_group_refill_stops_repeating_the_same_request() -> None:
    from alpha_orchestrator.factory.engine import GROUP_REFILLS, Engine
    from alpha_orchestrator.factory.llm import FactoryLLM
    from alpha_orchestrator.factory.models import CellORM, FieldORM
    from alpha_orchestrator.factory.research import Metadata

    ledger = setup(0)
    cfg = FactoryConfig()
    with ledger.session() as s:
        s.add(
            FieldORM(
                cell_id="USA|1|TOP3000|ds",
                field_id="f1",
                type="MATRIX",
                eligible=True,
                coverage=1.0,
                family="f1",
                representative=True,
            )
        )
        cell = s.get(CellORM, "USA|1|TOP3000|ds")
        cell.plan = {
            "groups": [{"name": "existing", "fields": ["f1"]}],
            "next_group": 1,
            "refills": 0,
        }
    import json

    llm = FactoryLLM(
        cfg,
        ledger,
        complete=lambda *_: (
            json.dumps({"groups": [{"name": "existing", "fields": ["f1"]}]}),
            10,
            10,
        ),
    )
    engine = Engine(FakeBrainAPI(metrics), ledger, cfg, Metadata(), llm)
    engine._groups("USA|1|TOP3000|ds", refill=True)
    with ledger.session() as s:
        assert s.get(CellORM, "USA|1|TOP3000|ds").plan["refills"] == GROUP_REFILLS


def test_ready_queue_runs_before_any_new_llm_work(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from alpha_orchestrator.factory import engine
    from alpha_orchestrator.factory.llm import FactoryLLM
    from alpha_orchestrator.factory.research import Metadata

    ledger = setup(3)
    api = FakeBrainAPI(metrics)
    cfg = FactoryConfig()
    cfg.budget.v1_daily_cap = 3
    cfg.budget.min_batch_interval_s = 0

    def forbidden(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        pytest.fail("proposal generation delayed already queued work")

    monkeypatch.setattr(engine.Engine, "advance", forbidden)
    monkeypatch.setattr("alpha_orchestrator.factory.daily.catalog_is_stale", lambda _: False)
    monkeypatch.setattr(engine, "store_passing", lambda *_: [])
    engine.run_day(cfg, api, ledger, FactoryLLM(cfg, ledger), meta=Metadata(), sleep=lambda _: None)
    assert api.started == 3


@pytest.mark.parametrize("kind", ["throttled", "expired", "biometric"])
def test_explicit_refusal_does_not_create_a_phantom_unknown_batch(kind: str) -> None:
    from alpha_orchestrator.brain.client import RateLimited, SessionExpired
    from alpha_orchestrator.brain.session import NeedBiometric

    ledger = setup(2)
    api = FakeBrainAPI(metrics)
    errors = {
        "throttled": RateLimited("429"),
        "expired": SessionExpired("401"),
        "biometric": NeedBiometric("https://example.invalid/verify"),
    }

    def refused(*_args):  # type: ignore[no-untyped-def]
        raise errors[kind]

    api.start_multi_simulation = refused  # type: ignore[method-assign]
    with pytest.raises(type(errors[kind])):
        sim(ledger, api).run(2)
    with ledger.session() as s:
        assert ledger.sims_reserved_today(s) == 0
        assert s.scalar(select(BatchORM.status)) == "REFUSED"
        assert {c.status for c in s.scalars(select(CandidateORM))} == {"QUEUED"}
