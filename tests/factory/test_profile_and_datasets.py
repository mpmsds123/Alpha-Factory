import random

from sqlalchemy import select

from alpha_orchestrator.factory import profile as prof
from alpha_orchestrator.factory import stats
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.datasets import rank
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import CellORM, IdeaORM


def counts(raw: int, r5: float, r22: float, r66: float) -> dict[str, int | None]:
    return {"raw": raw, "upd5": int(raw * r5), "upd22": int(raw * r22), "upd66": int(raw * r66)}


def test_update_period_is_read_from_the_widest_unsaturated_window() -> None:
    # a field updated every P days is non-flat in an N-day window ~min(N/P, 1) of the time
    assert prof.period_from(counts(3000, 1.0, 1.0, 1.0)) == 1.0
    assert 55 <= prof.period_from(counts(2400, 5 / 63, 22 / 63, 1.0)) <= 70
    assert 18 <= prof.period_from(counts(3000, 5 / 21, 1.0, 1.0)) <= 24
    assert prof.period_from({"raw": None}) == 999.0
    assert prof.cadence(63) == "quarterly" and prof.cadence(1) == "daily"


def test_coverage_means_after_filling_one_period() -> None:
    assert prof.fill_window(63) == 94 and prof.fill_window(1) == 5 and prof.fill_window(400) == 252
    quarterly = {"period_days": 63, "raw_coverage": 0.04, "filled_coverage": 0.8}
    daily = {"period_days": 1, "raw_coverage": 0.9, "filled_coverage": 0.9}
    assert prof.prep_for("eps", "MATRIX", quarterly) == "ts_backfill(eps, 94)"
    assert prof.prep_for("px", "MATRIX", daily) == "px"
    assert prof.prep_for("nws", "VECTOR", daily) == "vec_avg(nws)"


def test_survey_windows_follow_the_update_period() -> None:
    shapes = dict(prof.survey_shapes("x", 63))
    assert shapes["change"] == "rank(ts_delta(x, 63))"
    assert shapes["history"] == "rank(ts_zscore(x, 252))"
    assert set(shapes) == {"level", "change", "history", "peer"}
    assert dict(prof.survey_shapes("x", 1))["change"] == "rank(ts_delta(x, 5))"


def test_noise_probabilities_and_luck_bar() -> None:
    assert abs(stats.exceed_prob(1.0, 0.40) - 0.0124) < 0.001
    assert abs(stats.exceed_prob(0.8, 0.40, k=3) - 0.13) < 0.005
    assert abs(stats.luck_bar(0.40, 1500) - 1.53) < 0.01
    assert stats.luck_bar(0.40, 10**12) < 3  # grows like sqrt(log n): never explodes


def test_sprt_stops_on_the_evidence_with_preset_error_rates() -> None:
    noise = stats.Sprt(0.025, 0.074)
    while noise.verdict == "undecided":
        noise.add(False)
    assert noise.verdict == "noise" and noise.n == 31
    sig = stats.Sprt(0.025, 0.074)
    for i in range(1, 46):
        sig.add(i in (3, 7, 9))
        if sig.verdict != "undecided":
            break
    assert sig.verdict == "signal" and sig.n == 9


def test_shrinkage_and_thompson() -> None:
    arms = stats.shrunk_means({"a": [0.6, 0.7, 0.8, 0.7], "b": [1.5], "c": [0.0, 0.1]})
    # one lucky observation is pulled toward the pooled mean more than four steady ones
    assert arms["b"].mean < 1.5 and arms["b"].sd > arms["a"].sd
    picks = [stats.thompson_pick(arms, random.Random(i))[0] for i in range(300)]
    assert picks.count("c") < picks.count("a")


def _cell(cid: str, **kw: object) -> CellORM:
    base = {
        "region": "USA",
        "delay": 1,
        "universe": "TOP3000",
        "dataset_id": cid,
        "category": "analyst",
        "multiplier": 1.2,
        "alpha_count": 50,
    }
    base.update(kw)
    return CellORM(cell_id=f"{base['region']}|1|TOP3000|{cid}", **base)


def test_rank_learns_from_finished_datasets_and_papers() -> None:
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        s.add_all(
            [
                _cell("plain"),
                _cell("cited"),
                _cell("crowded", alpha_count=100_000),
                _cell("failed_elsewhere"),
                _cell(
                    "failed_elsewhere",
                    region="EUR",
                    state="EXHAUSTED",
                    plan={"verdict": "noise", "outcome": "noise"},
                ),
            ]
        )
        s.add(
            IdeaORM(
                source="Research_Papers_for_Users",
                title="Research Paper 9: revisions",
                body="Implement with the cited dataset ...",
            )
        )
        s.add(
            IdeaORM(source="中文论坛", title="research chat", body="cited is great")
        )  # not a paper
    with ledger.session() as s:
        order = [r.cell_id.split("|")[-1] for r in rank(s, FactoryConfig())]
        cited = s.scalar(select(CellORM).where(CellORM.dataset_id == "cited"))
        assert cited is not None and cited.paper_refs == 1
    assert order[0] == "cited"
    assert order.index("plain") < order.index("crowded")
    assert order.index("plain") < order.index("failed_elsewhere")  # failure fed back
    assert len(order) == 4  # the finished dataset is not offered again
