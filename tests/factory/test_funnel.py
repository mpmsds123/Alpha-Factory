"""工厂_漏斗.csv: per-day stage counts by idea cohort, and the reasons ideas were lost."""

import csv
from datetime import UTC, datetime
from pathlib import Path

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.funnel import FILE_NAME, REASONS_FILE, write_funnel
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import CandidateORM, CellORM, ResearchIdeaORM

CELL = "USA|1|TOP3000|x"


def _cand(i: int, sharpe: float | None, status: str = "DONE", **kw: object) -> CandidateORM:
    return CandidateORM(
        id=f"c{i}",
        cell_id=CELL,
        stage="FIELD",
        role="field",
        expression=f"e{i}",
        settings={},
        fingerprint=f"f{i}",
        status=status,
        sharpe=sharpe,
        completed_at=datetime.now(UTC),
        **kw,
    )


def _idea(i: int, status: str, rows: list[str], **extra: object) -> ResearchIdeaORM:
    return ResearchIdeaORM(
        id=f"i{i}",
        cell_id=CELL,
        title=f"idea {i}",
        mechanism="m",
        long_side="",
        short_side="",
        horizon_days=21,
        status=status,
        extra={"rows": rows, "formulas": [{} for _ in rows], **extra},
    )


def test_funnel_counts_and_reasons(tmp_path: Path) -> None:
    cfg = FactoryConfig()
    cfg.reports_dir = str(tmp_path / "reports")
    ledger = Ledger.at(tmp_path / "f.db")
    with ledger.session() as s:
        s.add(CellORM(cell_id=CELL, region="USA", delay=1, universe="TOP3000", dataset_id="x"))
        s.flush()
        s.add_all(
            [
                _cand(1, 0.4),
                _cand(2, None, "FAILED", error="error 42"),
                _cand(3, 1.3),
                _cand(4, 1.7, extra={"ready": True, "idea": "i2", "robust": {"pass": True}}),
            ]
        )
        s.add_all(
            [
                _idea(1, "DEAD", ["c1", "c2"], why_dead="first 2 formulas ... noise"),
                _idea(2, "DONE", ["c3", "c4"], scanner_fields=["a"], parents_from=["c3"]),
            ]
        )
    write_funnel(ledger, cfg)
    row = next(csv.DictReader((Path(cfg.reports_dir) / FILE_NAME).open(encoding="utf-8-sig")))
    got = {k: row[k] for k in ("新点子", "点子用扫描器字段", "过本地检查", "回测完成", "回测报错")}
    assert got == {
        "新点子": "2",
        "点子用扫描器字段": "1",
        "过本地检查": "4",
        "回测完成": "3",
        "回测报错": "1",
    }
    assert (row["过第一道门槛"], row["第一道门槛通过率"], row["过常规线"], row["过稳健性"]) == (
        "1",
        "50%",
        "1",
        "1",
    )
    reasons = list(
        csv.DictReader((Path(cfg.reports_dir) / REASONS_FILE).open(encoding="utf-8-sig"))
    )
    stages = {(r["阶段"], r["原因"]) for r in reasons}
    assert ("第一道门槛", "best |Sharpe| 0.3-0.5") in stages
    assert ("回测", "error ##") in stages
    assert ("平台检查", "not checked yet (waiting)") in stages
