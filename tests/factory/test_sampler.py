import random
from typing import Any

from alpha_orchestrator.factory import catalog, sampler
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import (
    CatalogORM,
    ExpressionORM,
    HStatus,
    HypothesisORM,
    Review,
    SimulationORM,
)
from alpha_orchestrator.factory.vocab import Vocabulary


def _dataset(s, region: str, did: str, category: str = "cat") -> CatalogORM:  # type: ignore[no-untyped-def]
    d = CatalogORM(
        kind="dataset", region=region, delay=1, universe="U", dataset_id=did, category=category
    )
    s.add(d)
    s.flush()
    return d


def _field(s, d: CatalogORM, fid: str, **kw: Any) -> None:  # type: ignore[no-untyped-def]
    base = {"field_type": "MATRIX", "coverage": 0.9, "description": "a quantity"}
    s.add(
        CatalogORM(
            kind="field",
            region=d.region,
            delay=1,
            universe="U",
            dataset_id=d.dataset_id,
            field_id=fid,
            **{**base, **kw},
        )
    )


def test_family_key() -> None:
    assert sampler.family_key("est_eps_fy1") == sampler.family_key("est_eps_fy2")


def test_choose_field_filters_and_prefers_untapped() -> None:
    ledger = Ledger("sqlite://")
    cfg = FactoryConfig()
    with ledger.session() as s:
        d = _dataset(s, "R1", "d1")
        _field(s, d, "low_cov", coverage=0.3)
        _field(s, d, "grp", field_type="GROUP")
        _field(s, d, "when_x", description="announcement date of the report")
        _field(s, d, "site_x", description="the cleaned domain name of the company")
        _field(s, d, "val_q1", alpha_count=50)
        _field(s, d, "val_q2", alpha_count=5)
        _field(s, d, "used_f")
        s.add(HypothesisORM(region="R1", dataset_id="d1", field_id="used_f"))
        s.flush()
        pick = sampler.choose_field(s, d, cfg, random.Random(0))
        assert pick is not None and pick.field.field_id == "val_q2"
        assert {f.field_id for f in pick.family} == {"val_q1", "val_q2"}
        assert "grp" in {f.field_id for f in pick.helpers}
        assert sampler.field_view(pick.field)["coverage"] == 0.9
        s.add(HypothesisORM(region="R1", dataset_id="d1", field_id="val_q9"))
        s.flush()
        assert sampler.choose_field(s, d, cfg, random.Random(0)) is None


def test_dataset_status_caps() -> None:
    ledger = Ledger("sqlite://")
    cfg = FactoryConfig()
    cfg.search.dataset_hypotheses_max = 3
    cfg.search.dataset_dry_streak = 2
    cfg.search.dataset_submissions_max = 1
    with ledger.session() as s:
        for sharpe in (1.5, 0.2, 0.3):
            s.add(
                HypothesisORM(
                    region="R", dataset_id="a", status=HStatus.CLOSED, best_train_sharpe=sharpe
                )
            )
        s.add(HypothesisORM(region="R", dataset_id="b", status=HStatus.CLOSED))
        s.add(HypothesisORM(region="R", dataset_id="b", status=HStatus.CLOSED))
        h = HypothesisORM(region="R", dataset_id="c")
        s.add(h)
        s.flush()
        e = ExpressionORM(hypothesis_id=h.id, expression="x")
        s.add(e)
        s.flush()
        s.add(
            SimulationORM(
                expression_id=e.id,
                hypothesis_id=h.id,
                region="R",
                delay=1,
                settings={},
                fingerprint="f",
                review=Review.PENDING,
            )
        )
        s.flush()
        assert sampler.dataset_status(s, cfg, "R", "a") == "hypothesis cap"
        assert "misses" in sampler.dataset_status(s, cfg, "R", "b")
        assert sampler.dataset_status(s, cfg, "R", "c") == "submission cap"
        assert sampler.dataset_status(s, cfg, "R", "z") == ""


def test_choose_dataset_stays_then_draws_by_arm() -> None:
    ledger = Ledger("sqlite://")
    cfg = FactoryConfig()
    with ledger.session() as s:
        good = _dataset(s, "R1", "good", "rich")
        _dataset(s, "R1", "poor", "thin")
        _dataset(s, "R9", "elsewhere", "rich")
        for _ in range(30):
            s.add(
                HypothesisORM(
                    region="R1",
                    dataset_id="x",
                    category="rich",
                    status=HStatus.CLOSED,
                    best_train_sharpe=2.0,
                    source="v7",
                )
            )
            s.add(
                HypothesisORM(
                    region="R1",
                    dataset_id="y",
                    category="thin",
                    status=HStatus.CLOSED,
                    best_train_sharpe=0.1,
                    source="v7",
                )
            )
        s.flush()
        cfg.policy.regions = ["R1"]
        wins = sum(
            sampler.choose_dataset(s, cfg, random.Random(i)).dataset_id == "good"  # type: ignore[union-attr]
            for i in range(20)
        )
        assert wins >= 18
        assert sampler.arm_counts(s, cfg)[("rich", "R1")] == (30, 0)
        s.add(HypothesisORM(region="R1", dataset_id="poor", source="factory"))
        s.flush()
        assert sampler.choose_dataset(s, cfg, random.Random(0)).dataset_id == "poor"  # type: ignore[union-attr]
        skipped = sampler.choose_dataset(s, cfg, random.Random(0), skip={("R1", "poor")})
        assert skipped is good
        assert (
            sampler.choose_dataset(s, cfg, random.Random(0), skip={("R1", "poor"), ("R1", "good")})
            is None
        )


class FakeAPI:
    def simulation_options(self) -> dict[str, Any]:
        return {
            "R1": {"universes": ["TOP500", "TOP2000", "SMALL"], "delays": [0, 1]},
            "R2": {"universes": ["ONLY"], "delays": [0]},
        }

    def datasets(self, region: str, delay: int, universe: str) -> list[dict[str, Any]]:
        return [
            {"id": "d1", "name": "D1", "category": {"id": "Cat"}, "fieldCount": 9},
            {"id": "d2", "fieldCount": 1},
        ]

    def datafields(self, *args: Any) -> list[dict[str, Any]]:
        return [{"id": "f1", "type": "matrix", "coverage": 0.8}, {"type": "x"}]


def test_catalog_refresh_and_fields() -> None:
    ledger = Ledger("sqlite://")
    assert catalog.is_stale(ledger)
    assert catalog.refresh(FakeAPI(), ledger, FactoryConfig(), Vocabulary()) == 1  # type: ignore[arg-type]
    assert not catalog.is_stale(ledger)
    assert catalog.refresh(FakeAPI(), ledger, FactoryConfig(), Vocabulary()) == 1  # type: ignore[arg-type]
    with ledger.session() as s:
        d = s.query(CatalogORM).filter_by(kind="dataset").one()
        assert (d.universe, d.category) == ("TOP2000", "cat")
    assert catalog.load_fields(FakeAPI(), ledger, d) == 1  # type: ignore[arg-type]
    with ledger.session() as s:
        f = s.query(CatalogORM).filter_by(kind="field").one()
        assert (f.field_type, f.category) == ("MATRIX", "cat")
    assert catalog.pick_universe("R", ["A", "B"], {"R": "B"}) == "B"
    assert catalog.pick_universe("R", [], {}) is None
