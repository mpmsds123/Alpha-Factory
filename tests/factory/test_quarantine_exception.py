from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from alpha_orchestrator.brain.client import AmbiguousRequest
from alpha_orchestrator.factory import study
from alpha_orchestrator.factory.catalog import upsert_cells
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.engine import Engine
from alpha_orchestrator.factory.models import BatchORM, CandidateORM, CellORM
from alpha_orchestrator.factory.quarantine import batch_digest, exception_scope
from alpha_orchestrator.factory.scheduler import Simulator

from .fakes import FakeBrainAPI
from .test_scheduler import CELL, metrics, setup


def fixture():
    ledger = setup(6)
    cfg = FactoryConfig()
    cfg.budget.min_batch_interval_s = 0
    now = datetime.now(UTC)
    with ledger.session() as s:
        cell = s.get(CellORM, CELL)
        cell.state, cell.plan = "RESEARCH", {"design": study.DESIGN}
        old = BatchORM(region="USA", delay=1, size=6, status="UNKNOWN", location="/simulations/old")
        s.add(old)
        s.flush()
        members = list(s.scalars(select(CandidateORM)))
        for c in members:
            c.status, c.batch_id = "RUNNING", old.id
        s.flush()
        cfg.quarantine_exception = {
            "batch_id": old.id,
            "digest": batch_digest(s, old),
            "approved_at": (now - timedelta(minutes=1)).isoformat(),
            "expires_at": (now + timedelta(minutes=30)).isoformat(),
        }
        old_id = old.id
    return ledger, cfg, old_id


def test_default_still_blocks_and_exception_preserves_old_reservations():
    ledger, cfg, old_id = fixture()
    api = FakeBrainAPI(metrics)
    with pytest.raises(AmbiguousRequest):
        Simulator(api, ledger, FactoryConfig()).assert_recoverable()
    Simulator(api, ledger, cfg).assert_recoverable()
    with ledger.session() as s:
        assert s.get(BatchORM, old_id).status == "UNKNOWN"
        assert ledger.sims_reserved_today(s) == 6
        assert all(c.status == "RUNNING" for c in s.scalars(select(CandidateORM)))


@pytest.mark.parametrize(
    "change", ["expired", "future", "too_long", "digest", "member", "new_unknown"]
)
def test_changed_expired_or_new_ambiguity_blocks(change):
    ledger, cfg, old_id = fixture()
    now = datetime.now(UTC)
    if change == "expired":
        cfg.quarantine_exception["expires_at"] = (now - timedelta(seconds=1)).isoformat()
    elif change == "future":
        cfg.quarantine_exception["approved_at"] = (now + timedelta(minutes=1)).isoformat()
    elif change == "too_long":
        cfg.quarantine_exception["expires_at"] = (now + timedelta(days=1)).isoformat()
    elif change == "digest":
        cfg.quarantine_exception["digest"] = "bad"
    elif change == "member":
        with ledger.session() as s:
            s.scalar(select(CandidateORM)).fingerprint = "changed"
    else:
        with ledger.session() as s:
            s.add(
                BatchORM(
                    region="USA", delay=1, size=1, status="UNKNOWN", location="/simulations/new"
                )
            )
    with pytest.raises(AmbiguousRequest):
        Simulator(FakeBrainAPI(metrics), ledger, cfg).assert_recoverable()
    with ledger.session() as s:
        assert s.get(BatchORM, old_id).status == "UNKNOWN"


def test_only_new_cell_runs_no_old_member_is_recreated():
    ledger, cfg, old_id = fixture()
    with ledger.session() as s:
        settings = dict(s.scalar(select(CandidateORM)).settings)
        upsert_cells(s, [{"id": "new", "fieldCount": 10}], "USA", 1, "TOP3000")
        for i in range(6, 9):
            ledger.add_candidate(
                s,
                cell_id="USA|1|TOP3000|new",
                stage="FIELD",
                expression=f"rank(f{i})",
                settings=settings,
                priority=1,
            )
        ledger.add_candidate(
            s, cell_id=CELL, stage="FIELD", expression="rank(f90)", settings=settings, priority=100
        )
    api = FakeBrainAPI(metrics)
    sim = Simulator(api, ledger, cfg, sleep=lambda _: None, poll_interval=0)
    assert sim.run(100).completed == 3
    assert api.started == 3
    with ledger.session() as s:
        assert s.get(BatchORM, old_id).status == "UNKNOWN"
        assert (
            len(list(s.scalars(select(CandidateORM).where(CandidateORM.batch_id == old_id)))) == 6
        )
        assert (
            s.scalar(select(CandidateORM.status).where(CandidateORM.expression == "rank(f90)"))
            == "QUEUED"
        )
        assert ledger.sims_reserved_today(s) == 9


def test_generation_skips_frozen_cell_and_config_round_trips(tmp_path):
    import yaml

    ledger, cfg, _ = fixture()
    with ledger.session() as s:
        upsert_cells(s, [{"id": "new", "fieldCount": 10}], "USA", 1, "TOP3000")
        cfg.only_datasets = [CELL, "USA|1|TOP3000|new"]
        assert study.choose_dataset(s, cfg) == "USA|1|TOP3000|new"
        assert exception_scope(s, cfg)[1] == {CELL}
    engine = Engine(SimpleNamespace(), ledger, cfg, SimpleNamespace(), SimpleNamespace())
    assert engine._current_dataset() is None
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"quarantine_exception": cfg.quarantine_exception}))
    assert FactoryConfig.load(path).quarantine_exception == cfg.quarantine_exception


def test_expiry_during_pacing_does_not_create_a_request():
    ledger, cfg, _ = fixture()
    cfg.budget.min_batch_interval_s = 20
    api = FakeBrainAPI(metrics)

    def expire(_):
        cfg.quarantine_exception["expires_at"] = (
            datetime.now(UTC) - timedelta(seconds=1)
        ).isoformat()

    simulator = Simulator(api, ledger, cfg, sleep=expire, clock=lambda: 0)
    simulator._last_batch_at = 0
    with pytest.raises(AmbiguousRequest):
        simulator._launch_one(10)
    assert api.started == 0


@pytest.mark.parametrize(("minutes", "max_seconds"), [(720, 5.1), (0.01, 0.7)])
def test_absolute_cutoff_cannot_be_extended_by_delayed_start(monkeypatch, minutes, max_seconds):
    import time

    from alpha_orchestrator.factory import engine as eng
    from alpha_orchestrator.factory.scheduler import RunStats

    ledger, cfg, _ = fixture()
    cfg.quarantine_exception["expires_at"] = (datetime.now(UTC) + timedelta(seconds=5)).isoformat()
    started = time.monotonic()

    class StopBeforeResearch:
        def __init__(self, *args, **kwargs):
            pass

        def assert_recoverable(self):
            pass

        def run(self, budget, deadline):
            assert 0 < deadline - started < max_seconds
            return RunStats(stopped_because="deadline")

    monkeypatch.setattr(eng, "Simulator", StopBeforeResearch)
    assert eng.run_day(cfg, None, ledger, None, minutes=minutes).stopped_because == "deadline"


@pytest.mark.parametrize("hours", [3, 22])
def test_explicit_continuation_within_one_day_keeps_quarantine(hours):
    ledger, cfg, old_id = fixture()
    cfg.quarantine_exception["expires_at"] = (
        datetime.now(UTC) + timedelta(hours=hours)
    ).isoformat()
    Simulator(FakeBrainAPI(metrics), ledger, cfg).assert_recoverable()
    with ledger.session() as s:
        assert s.get(BatchORM, old_id).status == "UNKNOWN"
        assert ledger.sims_reserved_today(s) == 6
