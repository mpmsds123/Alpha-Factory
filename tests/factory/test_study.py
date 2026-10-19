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
        s.add_all(
            [
                cell("thin", 0.55, multiplier=3.0),
                cell("unknown", 0.0, multiplier=3.0),
                cell("wide", 0.9),
            ]
        )
    with ledger.session() as s:
        assert study.choose_dataset(s, FactoryConfig()) == "USA|1|TOP3000|wide"


def test_step1_prefers_datasets_the_scanner_judged(tmp_path) -> None:
    """提案 1: the factory works first on datasets the template scanner already judged,
    so the two methods are compared on the same datasets."""
    progress = tmp_path / "数据集进度.csv"
    progress.write_text(
        "region,universe,dataset,result\n"
        "USA,TOP3000,judged,止损\n"
        "USA,TOP3000,nofields,没有可用字段\n",
        encoding="utf-8",
    )
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        s.add_all([cell("best", 0.9, multiplier=3.0), cell("judged", 0.9), cell("nofields", 0.9)])
    with ledger.session() as s:
        cfg = FactoryConfig(scanner_progress=str(progress))
        assert study.choose_dataset(s, cfg) == "USA|1|TOP3000|judged"
        assert study.choose_dataset(s, FactoryConfig()) == "USA|1|TOP3000|best"  # off by default


def test_old_verdicts_are_reset_and_stop_counting() -> None:
    from alpha_orchestrator.factory import datasets

    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        old = cell("old", 0.9)
        old.state, old.plan = CellState.EXHAUSTED, {"verdict": "noise", "outcome": "noise"}
        s.add_all([old, cell("other", 0.9)])
    assert study.reset_old_verdicts(ledger) == 1
    assert study.reset_old_verdicts(ledger) == 0  # idempotent
    with ledger.session() as s:
        c = s.get(CellORM, "USA|1|TOP3000|old")
        assert c is not None and c.state == CellState.NEW
        assert datasets.outcome_of(c) is None  # the old "noise" no longer counts


def test_ideas_judged_under_an_earlier_design_are_retired() -> None:
    from alpha_orchestrator.factory.models import IdeaStatus, ResearchIdeaORM

    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        old = cell("old", 0.9)
        old.state, old.plan = CellState.EXHAUSTED, {"design": "v1", "outcome": "noise"}
        s.add(old)
        s.flush()
        s.add(ResearchIdeaORM(cell_id=old.cell_id, title="t", status=IdeaStatus.DEAD))
    study.reset_old_verdicts(ledger)
    with ledger.session() as s:
        idea = s.scalars(select(ResearchIdeaORM)).one()
        assert idea.status == IdeaStatus.RETIRED
        c = s.get(CellORM, "USA|1|TOP3000|old")
        assert c is not None and c.state == CellState.NEW


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


def test_news_is_never_carried_forward_but_fundamentals_are() -> None:
    assert study.prep("sent", "VECTOR", "news") == "vec_avg(sent)"
    assert study.prep("px", "MATRIX", "pv") == "px"
    assert study.prep("assets", "MATRIX", "fundamental") == "ts_backfill(assets, 63)"
    assert study.prep("eps", "MATRIX", "analyst") == "ts_backfill(eps, 63)"


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
