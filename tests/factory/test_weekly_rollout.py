from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import BatchORM, CandidateORM, CellORM, ReviewORM
from alpha_orchestrator.factory.select import quota_only, store_passing
from alpha_orchestrator.factory.study import scanner_signal_fields
from alpha_orchestrator.factory.weekly import friday_cutoff, weekly_rows, write_weekly

QUOTA = {"name": "REGULAR_SUBMISSION", "result": "FAIL", "value": 4, "limit": 4}
PASS = [
    {"name": name, "result": "PASS", "value": 0.5}
    for name in ("SELF_CORRELATION", "PROD_CORRELATION")
]


class Platform:
    def __init__(self) -> None:
        self.results = [[QUOTA], PASS]
        self.calls = 0

    def check(self, alpha_id: str) -> list[dict[str, str | float]]:
        assert alpha_id == "test-alpha"
        self.calls += 1
        return self.results.pop(0)


def test_quota_defers_until_next_window(tmp_path: Path) -> None:
    cfg = FactoryConfig(reports_dir=str(tmp_path))
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        s.add(CellORM(cell_id="test", region="USA", delay=1, universe="TOP", dataset_id="mock"))
        s.flush()
        s.add(
            CandidateORM(
                cell_id="test",
                stage="FIELD",
                status="DONE",
                fingerprint="mock",
                settings={},
                expression="rank(mock)",
                alpha_id="test-alpha",
                extra={"ready": True, "robust": {"pass": True, "ratios": {"neutralization": 0.8}}},
            )
        )
    api = Platform()
    first = datetime(2026, 11, 6, 22, tzinfo=UTC)
    assert store_passing(api, ledger, cfg, now=first) == []  # type: ignore[arg-type]
    assert store_passing(api, ledger, cfg, now=first) == []  # type: ignore[arg-type]
    assert api.calls == 1
    assert store_passing(api, ledger, cfg, now=first + timedelta(days=1)) == [  # type: ignore[arg-type]
        "test-alpha"
    ]
    with ledger.session() as s:
        assert s.scalar(select(ReviewORM.alpha_id)) == "test-alpha"
    assert not quota_only([QUOTA, {"name": "LOW_DURATION", "result": "FAIL"}])


def test_friday_snapshot_keeps_reservations_separate_and_preserves_original(tmp_path: Path) -> None:
    cfg = FactoryConfig(reports_dir=str(tmp_path))
    ledger = Ledger("sqlite://")
    end = friday_cutoff(cfg, datetime(2026, 11, 8, tzinfo=UTC))
    assert end.isoformat() == "2026-11-06T21:00:00-08:00"
    with ledger.session() as s:
        s.add(
            BatchORM(
                region="USA", delay=1, size=3, created_at=(end - timedelta(days=1)).astimezone(UTC)
            )
        )
        s.add(BatchORM(region="USA", delay=1, size=5, created_at=end.astimezone(UTC)))
    rows = dict(weekly_rows(ledger, cfg, now=end))
    assert rows["本期发起队列：配额预留"] == "3"
    assert rows["回测完成"] == "0"
    path = write_weekly(ledger, cfg, ending=end)
    first_snapshot = path.read_bytes()
    with ledger.session() as s:
        s.add(
            BatchORM(
                region="USA", delay=1, size=2, created_at=(end - timedelta(days=1)).astimezone(UTC)
            )
        )
    assert write_weekly(ledger, cfg, ending=end).read_bytes() == first_snapshot


def test_scanner_import_supports_machine_directory(tmp_path: Path) -> None:
    root = tmp_path / "机器记录"
    root.mkdir()
    (root / "USA_mock_v6_1阶.csv").write_text(
        "field,sharpe,turnover\nsynthetic,1.6,0.2\n", encoding="utf-8"
    )
    assert scanner_signal_fields(str(tmp_path), "USA", "mock") == {"synthetic"}
