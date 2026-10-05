"""设计文档 v1, step by step, on a fake platform whose answers are known."""

from sqlalchemy import select

from alpha_orchestrator.factory import study
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import CellORM, CellState, FieldORM

from .fakes import FakePlatform, is_metrics


def cell(name: str, coverage: float, multiplier: float = 1.0) -> CellORM:
    return CellORM(
        cell_id=f"USA|1|TOP3000|{name}",
        region="USA",
        delay=1,
        universe="TOP3000",
        dataset_id=name,
        category="analyst",
        coverage=coverage,
        multiplier=multiplier,
    )


def test_step1_skips_datasets_below_sixty_percent_coverage() -> None:
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        s.add_all([cell("thin", 0.55, multiplier=3.0), cell("wide", 0.9)])
    with ledger.session() as s:
        assert study.choose_dataset(s, FactoryConfig()) == "USA|1|TOP3000|wide"


def test_step2_keeps_numeric_fields_covering_sixty_percent() -> None:
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        s.add(cell("est", 0.9))
    rows = {
        "est": [
            {"id": "eps_mean", "type": "MATRIX", "coverage": 0.85, "description": "EPS"},
            {"id": "eps_list", "type": "VECTOR", "coverage": 0.70, "description": "EPS list"},
            {"id": "rare", "type": "MATRIX", "coverage": 0.30, "description": "thin"},
            {"id": "sector_code", "type": "GROUP", "coverage": 0.99, "description": "group"},
        ]
    }
    api = FakePlatform(lambda e, st: is_metrics(0.0, 0.0), rows)
    assert study.load_fields(api, ledger, "USA|1|TOP3000|est", FactoryConfig()) == 2
    with ledger.session() as s:
        fields = {f.field_id: f for f in s.scalars(select(FieldORM))}
        assert fields["eps_mean"].prep == "ts_backfill(eps_mean, 63)"
        assert fields["eps_list"].prep == "ts_backfill(vec_avg(eps_list), 63)"
        assert not fields["rare"].eligible and not fields["sector_code"].eligible
        assert s.get(CellORM, "USA|1|TOP3000|est").state == CellState.RESEARCH  # type: ignore[union-attr]


def test_step2_dataset_without_usable_fields_is_closed() -> None:
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        s.add(cell("thin", 0.9))
    rows = {"thin": [{"id": "x", "type": "MATRIX", "coverage": 0.1}]}
    api = FakePlatform(lambda e, st: is_metrics(0.0, 0.0), rows)
    assert study.load_fields(api, ledger, "USA|1|TOP3000|thin", FactoryConfig()) == 0
    with ledger.session() as s:
        c = s.get(CellORM, "USA|1|TOP3000|thin")
        assert c is not None and c.state == CellState.EXHAUSTED
