from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from alpha_orchestrator.factory.ledger import Ledger, WorkerAlreadyRunning, fingerprint
from alpha_orchestrator.factory.models import (
    BatchORM,
    ExpressionORM,
    HypothesisORM,
    PriorORM,
    SimStatus,
    SimulationORM,
)


def _hyp(s, region: str = "R1") -> HypothesisORM:  # type: ignore[no-untyped-def]
    h = HypothesisORM(region=region, dataset_id="d1", field_id="f1")
    s.add(h)
    s.flush()
    return h


def test_tables_and_queue_order() -> None:
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        h = _hyp(s)
        e = ExpressionORM(hypothesis_id=h.id, expression="op(f1, 5)")
        s.add(e)
        s.flush()
        for prio, expr in ((1.0, "a"), (3.0, "b")):
            s.add(
                SimulationORM(
                    expression_id=e.id,
                    hypothesis_id=h.id,
                    region="R1",
                    delay=1,
                    settings={},
                    fingerprint=fingerprint(expr, {}),
                    priority=prio,
                )
            )
    with ledger.session() as s:
        assert ledger.queue_groups(s) == [("R1", 1)]
        rows = ledger.queued(s, "R1", 1, 10)
        assert [r.priority for r in rows] == [3.0, 1.0]
        assert ledger.pending(s) == 2
        assert ledger.queued(s, "R1", 1, 10, excluded={h.id}) == []
        rows[0].status = SimStatus.DONE
    with ledger.session() as s:
        assert ledger.pending(s) == 1


def test_reserved_counts_batches_except_refusals() -> None:
    ledger = Ledger("sqlite://")
    start = datetime.now(UTC) - timedelta(hours=1)
    with ledger.session() as s:
        s.add_all(
            [
                BatchORM(region="R", delay=1, size=10),
                BatchORM(region="R", delay=1, size=10, status="REFUSED"),
                BatchORM(region="R", delay=1, size=4, status="UNKNOWN"),
            ]
        )
    with ledger.session() as s:
        assert ledger.sims_reserved(s, start) == 14


def test_prior_confidence_is_beta_mean() -> None:
    assert PriorORM(claim="c", source_type="own", a=3.0, b=1.0).confidence == 0.75


def test_fingerprint_ignores_whitespace_only() -> None:
    assert fingerprint("a( b )", {"x": 1}) == fingerprint("a(b)", {"x": 1})
    assert fingerprint("a(b)", {"x": 1}) != fingerprint("a(b)", {"x": 2})


def test_events_file_and_writer_lock(tmp_path: Path) -> None:
    ledger = Ledger.at(tmp_path / "db" / "f.db")
    ledger.log("x", n=1)
    ledger.log("y")
    assert [e["kind"] for e in ledger.events()] == ["x", "y"]
    assert ledger.events("x")[0]["n"] == 1
    with ledger.writer_lock():
        with ledger.writer_lock():  # re-entrant for the owner
            pass
        other = Ledger.at(tmp_path / "db" / "f.db")
        with pytest.raises(WorkerAlreadyRunning), other.writer_lock():
            pass


def test_memory_ledger_keeps_events_in_memory() -> None:
    ledger = Ledger("sqlite://")
    ledger.log("k", a=1)
    assert ledger.events("k") == [ledger.events()[0]]
