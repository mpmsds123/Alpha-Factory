import random

from alpha_orchestrator.factory.catalog import cell_stats, pick_universe, pyramid_counts, upsert_cells
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.scoring import (
    CellStats,
    crowding_discount,
    expected_value,
    hit_probability,
    plan_day,
    value_if_hit,
)

CFG = FactoryConfig()


def stats(cell_id: str = "USA|1|TOP3000|d", **kw):  # type: ignore[no-untyped-def]
    base = dict(cell_id=cell_id, region="USA", category="analyst", state="NEW", multiplier=1.0,
                alpha_count=10, themes=0, probes=0, hits=0, pyramid_submitted=3,
                dataset_submitted=0)
    base.update(kw)
    return CellStats(**base)


def test_prior_is_history_rate_and_data_dominates() -> None:
    assert abs(hit_probability(stats()) - 0.061) < 1e-9
    assert hit_probability(stats(probes=50, hits=0)) < 0.011
    assert hit_probability(stats(probes=50, hits=10)) > 0.17


def test_value_rewards_multiplier_gap_and_penalises_crowding_cost_difficulty() -> None:
    base = value_if_hit(stats(), CFG)
    assert value_if_hit(stats(multiplier=1.8), CFG) > base
    assert value_if_hit(stats(pyramid_submitted=2), CFG) > base
    assert value_if_hit(stats(alpha_count=5000), CFG) < base
    assert value_if_hit(stats(region="CHN"), CFG) < value_if_hit(stats(region="USA"), CFG)
    assert value_if_hit(stats(dataset_submitted=8), CFG) == 0.0


def test_crowding_is_bounded() -> None:
    assert crowding_discount(0) == 0
    assert crowding_discount(10**9) == 0.6


def test_plan_probes_new_cells_and_exploits_signal() -> None:
    cells = [stats(f"USA|1|TOP3000|n{i}") for i in range(5)]
    cells.append(stats("USA|1|TOP3000|live", state="SIGNAL", probes=50, hits=12))
    plan = plan_day(cells, CFG, budget=1000, rng=random.Random(0))
    explore = [a for a in plan if a.mode == "explore"]
    exploit = [a for a in plan if a.mode == "exploit"]
    assert all(a.budget <= CFG.budget.probe_budget for a in explore)
    assert sum(a.budget for a in explore) <= 300
    assert exploit and exploit[0].cell_id.endswith("live")
    assert sum(a.budget for a in plan) <= 1000


def test_saturated_dataset_gets_nothing() -> None:
    cells = [stats("x", state="SIGNAL", probes=50, hits=20, dataset_submitted=9)]
    assert plan_day(cells, CFG, budget=500, rng=random.Random(1)) == []
    assert expected_value(cells[0], CFG) == 0


def test_pick_universe_prefers_known_then_widest_top() -> None:
    assert pick_universe("USA", ["TOP1000", "TOP3000"]) == "TOP3000"
    assert pick_universe("XYZ", ["TOP200", "TOP800", "MINVOL1M"]) == "TOP800"


def test_pyramid_counts_accepts_both_shapes() -> None:
    nested = [{"region": "USA", "delay": 1, "category": {"id": "ANALYST"}, "alphaCount": 2}]
    flat = [{"pyramid": "EUR/D1/model", "count": 1}]
    assert pyramid_counts(nested)[("USA", 1, "analyst")] == 2
    assert pyramid_counts(flat)[("EUR", 1, "model")] == 1


def test_upsert_and_stats_round_trip() -> None:
    ledger = Ledger("sqlite://")
    row = {"id": "fundamental6", "name": "F6", "category": {"id": "fundamental"},
           "pyramidMultiplier": 1.4, "alphaCount": 12, "fieldCount": 40, "themes": ["t"]}
    with ledger.session() as s:
        assert upsert_cells(s, [row], "USA", 1, "TOP3000") == 1
    with ledger.session() as s:
        (only,) = cell_stats(s)
    assert only.cell_id == "USA|1|TOP3000|fundamental6"
    assert only.multiplier == 1.4 and only.themes == 1
