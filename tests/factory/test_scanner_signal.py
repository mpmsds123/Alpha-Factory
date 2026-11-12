"""The scanner's master table is read for signal fields (synthetic names only)."""

from pathlib import Path

from alpha_orchestrator.factory import study
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import CellORM
from alpha_orchestrator.factory.study import MASTER_FILE, scanner_signal_fields

from .fakes import FakePlatform, is_metrics

HEADER = "地区,数据集,阶,字段,sharpe,turnover,公式\n"


def _master(tmp_path: Path, rows: list[str]) -> None:
    (tmp_path / MASTER_FILE).write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8-sig")


def test_master_table_signal_fields(tmp_path: Path) -> None:
    _master(
        tmp_path,
        [
            "USA,ds1,1阶,f_hit,1.9,0.10,x",
            "USA,ds1,1阶,reduce(f_wrapped),-1.7,0.20,x",  # sign does not matter; wrapper kept
            "USA,ds1,1阶,f_weak,1.2,0.10,x",  # below the submission line
            "USA,ds1,1阶,f_fast,2.0,0.90,x",  # turnover above 70%
            "USA,ds2,1阶,f_other,2.0,0.10,x",
        ],
    )
    got = scanner_signal_fields(str(tmp_path), "USA", "ds1")
    assert got == {"f_hit", "reduce", "f_wrapped"}
    assert scanner_signal_fields(str(tmp_path), "USA", "never_scanned") is None


def test_master_table_dataset_without_signal(tmp_path: Path) -> None:
    _master(tmp_path, ["USA,ds1,1阶,f_weak,0.4,0.10,x"])
    assert scanner_signal_fields(str(tmp_path), "USA", "ds1") == set()


def _cell(name: str) -> CellORM:
    return CellORM(
        cell_id=f"USA|1|TOP3000|{name}",
        region="USA",
        delay=1,
        universe="TOP3000",
        dataset_id=name,
        category="analyst",
        coverage=0.9,
        multiplier=1.0,
    )


def test_v7_pool_has_only_datasets_with_scanner_signal(tmp_path: Path) -> None:
    _master(tmp_path, ["USA,ds1,1阶,f_hit,1.9,0.10,x", "USA,ds2,1阶,f_weak,0.3,0.10,x"])
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        s.add_all([_cell("ds1"), _cell("ds2"), _cell("ds3")])
    config = FactoryConfig(design="v7", scanner_runs=str(tmp_path))
    with ledger.session() as s:
        assert [c.dataset_id for c in study.v7_dataset_pool(s, config, set())] == ["ds1"]
        config.scanner_runs = str(tmp_path / "missing")  # no evidence -> nothing to mine
        assert study.v7_dataset_pool(s, config, set()) == []


def test_v7_keeps_only_scanner_signal_fields(tmp_path: Path) -> None:
    _master(tmp_path, ["USA,ds1,1阶,reduce(f_hit),1.9,0.10,x", "USA,ds1,1阶,f_miss,0.2,0.1,x"])
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        s.add(_cell("ds1"))
    rows = {
        "ds1": [
            {"id": "f_hit", "type": "VECTOR", "coverage": 0.85},
            {"id": "f_miss", "type": "MATRIX", "coverage": 0.85},
        ]
    }
    api = FakePlatform(lambda e, st: is_metrics(0.0, 0.0), rows)
    config = FactoryConfig(design="v7", scanner_runs=str(tmp_path))
    assert study.load_fields(api, ledger, "USA|1|TOP3000|ds1", config) == 1
    with ledger.session() as s:
        assert [f.field_id for f in study.usable_fields(s, "USA|1|TOP3000|ds1")] == ["f_hit"]
        cell = s.get(CellORM, "USA|1|TOP3000|ds1")
        assert cell is not None and cell.plan["scanner_fields"] == ["f_hit"]


def test_v7_day_uses_scanner_fields_and_skips_scanner_formulas(tmp_path: Path) -> None:
    """End to end on the simulated world: with the scanner's evidence configured, ideas
    are built on its signal fields and formulas it already ran are not simulated again."""
    import random

    from sqlalchemy import select

    from alpha_orchestrator.factory import engine as eng
    from alpha_orchestrator.factory.llm import FactoryLLM
    from alpha_orchestrator.factory.models import CandidateORM, ResearchIdeaORM

    from .test_engine import _SizeWorld, eps_fields
    from .test_v7 import v7_llm

    def day(scanner: str) -> Ledger:
        ledger = Ledger("sqlite://")
        cfg = FactoryConfig(design="v7", scanner_runs=scanner)
        cfg.budget.min_batch_interval_s = 0
        eng.run_day(
            cfg,
            _SizeWorld({"est": eps_fields(1.1)}).platform(),
            ledger,
            FactoryLLM(cfg, ledger, complete=v7_llm),
            minutes=5,
            rng=random.Random(3),
            sleep=lambda _s: None,
        )
        return ledger

    plain = day("")
    with plain.session() as s:
        cell = s.scalars(select(CellORM)).first()
        assert cell is not None
        region, dataset = cell.region, cell.dataset_id
        firsts = [c.expression for c in s.scalars(select(CandidateORM)) if c.stage == "FIELD"]
        fields = sorted({f for c in s.scalars(select(CandidateORM)) for f in c.extra["fields"]})
    assert firsts and fields
    _master(
        tmp_path,
        [f'{region},{dataset},1阶,{f},1.9,0.10,"{firsts[0]}"' for f in fields],
    )
    ledger = day(str(tmp_path))
    with ledger.session() as s:
        ideas = list(s.scalars(select(ResearchIdeaORM)))
        assert ideas and all(i.extra.get("scanner_fields") for i in ideas)
        assert any(i.extra.get("unoriginal") for i in ideas)
        shapes = {eng._formula_shape(c.expression) for c in s.scalars(select(CandidateORM))}
        assert eng._formula_shape(firsts[0]) not in shapes
