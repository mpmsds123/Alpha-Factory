"""工厂_进度.csv: daily counts from the ledger, and the face-check session is reused."""

import csv
from datetime import UTC, datetime
from pathlib import Path

from alpha_orchestrator.factory import cli
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import CandidateORM, CellORM
from alpha_orchestrator.factory.progress import FILE_NAME, write_progress

PASS = [{"name": "LOW_SHARPE", "result": "PASS"}]
FAIL = [{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL"}]


def _cand(i: int, sharpe: float, checks: list, when: datetime) -> CandidateORM:  # type: ignore[type-arg]
    return CandidateORM(
        cell_id="USA|1|TOP3000|x",
        stage="FIELD",
        role="field",
        expression=f"e{i}",
        settings={},
        fingerprint=f"f{i}",
        status="DONE",
        sharpe=sharpe,
        fitness=1.2,
        turnover=0.1,
        checks=checks,
        completed_at=when,
    )


def _setup(tmp_path: Path) -> tuple[FactoryConfig, Ledger]:
    cfg = FactoryConfig()
    cfg.reports_dir = str(tmp_path / "reports")
    return cfg, Ledger.at(tmp_path / "f.db")


def test_counts_and_status(tmp_path: Path) -> None:
    cfg, ledger = _setup(tmp_path)
    now = datetime.now(UTC)
    with ledger.session() as s:
        s.add(
            CellORM(
                cell_id="USA|1|TOP3000|x",
                region="USA",
                delay=1,
                universe="TOP3000",
                dataset_id="x",
                category="analyst",
                coverage=1.0,
                multiplier=1.0,
            )
        )
        s.flush()
        s.add_all([_cand(1, 1.9, PASS, now), _cand(2, 1.9, FAIL, now), _cand(3, 0.4, PASS, now)])
    path = write_progress(ledger, cfg, "运行中")
    row = next(csv.DictReader(path.open(encoding="utf-8-sig")))
    assert path.name == FILE_NAME
    assert (row["状态"], row["回测完成"], row["过第一道门槛"], row["可提交"]) == (
        "运行中",
        "3",
        "2",
        "1",
    )


def test_past_status_kept(tmp_path: Path) -> None:
    cfg, ledger = _setup(tmp_path)
    path = Path(cfg.reports_dir) / FILE_NAME
    path.parent.mkdir(parents=True)
    path.write_text("日期,状态\n2000-01-01,今天结束：完成\n", encoding="utf-8-sig")
    write_progress(ledger, cfg, "运行中")
    rows = {r["日期"]: r["状态"] for r in csv.DictReader(path.open(encoding="utf-8-sig"))}
    assert rows["2000-01-01"] == "今天结束：完成"


def test_api_reuses_live_session(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    live = object()
    monkeypatch.setattr(cli, "_live_session", live)
    assert cli._api().client.session is live


def test_scanner_signal_fields(tmp_path: Path) -> None:
    from alpha_orchestrator.factory.study import scanner_signal_fields

    (tmp_path / "USA_ds1_v6_1阶.csv").write_text(
        "field,sharpe,turnover\nf_hit,-1.7,0.2\nf_weak,1.2,0.2\nf_churn,2.0,0.9\n",
        encoding="utf-8",
    )
    (tmp_path / "USA_ds1_v6_2阶.csv").write_text(
        "field,sharpe,turnover\nf_two,1.6,0.1\n", encoding="utf-8"
    )
    assert scanner_signal_fields(str(tmp_path), "USA", "ds1") == {"f_hit", "f_two"}
    assert scanner_signal_fields(str(tmp_path), "USA", "other") is None
