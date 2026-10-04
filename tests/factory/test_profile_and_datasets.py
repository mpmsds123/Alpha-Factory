from sqlalchemy import select

from alpha_orchestrator.factory import profile as prof
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.datasets import rank
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import CellORM, IdeaORM


def counts(cov: int, r5: float, r22: float, r66: float, big: float = 0.0) -> dict[str, int | None]:
    return {
        "cov": cov,
        "nz": cov,
        "upd5": int(cov * r5),
        "upd22": int(cov * r22),
        "upd66": int(cov * r66),
        "big": int(cov * big),
    }


def test_update_period_is_read_from_the_widest_unsaturated_window() -> None:
    # A field updated every P days is non-flat in an N-day window ~min(N/P, 1) of the time.
    assert prof.interpret(counts(3000, 1.0, 1.0, 1.0), "TOP3000")["cadence"] == "daily"
    quarterly = prof.interpret(counts(2400, 5 / 63, 22 / 63, 1.0), "TOP3000")
    assert quarterly["cadence"] == "quarterly" and 55 <= quarterly["period_days"] <= 70
    assert quarterly["coverage"] == 0.8
    monthly = prof.interpret(counts(3000, 5 / 21, 1.0, 1.0), "TOP3000")
    assert monthly["cadence"] == "monthly"
    assert prof.interpret(counts(3000, 0.0, 0.0, 0.1), "TOP3000")["cadence"] == "slow"
    assert prof.interpret({"cov": None}, "TOP3000")["coverage"] == 0.0


def test_prep_follows_the_profile() -> None:
    daily = prof.interpret(counts(3000, 1.0, 1.0, 1.0), "TOP3000")
    quarterly = prof.interpret(counts(2400, 0.08, 0.35, 1.0, big=0.9), "TOP3000")
    assert prof.prep_for("close_x", "MATRIX", "pv", daily) == "close_x"
    assert (
        prof.prep_for("eps", "MATRIX", "fundamental", quarterly)
        == "winsorize(ts_backfill(eps, 132), std=4)"
    )
    # News loses meaning when carried forward: never backfilled.
    assert prof.prep_for("nws", "VECTOR", "news", quarterly) == "winsorize(vec_avg(nws), std=4)"
    exprs = dict(prof.probe_expressions("nws", "VECTOR"))
    assert exprs["upd22"] == "ts_std_dev(vec_avg(nws), 22) != 0 ? 1 : 0"


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
    return CellORM(cell_id=f"USA|1|TOP3000|{cid}", **base)


def test_rank_uses_papers_crowding_and_skips_stopped() -> None:
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        s.add_all(
            [
                _cell("plain"),
                _cell("cited"),
                _cell("crowded", alpha_count=100_000),
                _cell("done", state="EXHAUSTED"),
                _cell("chn", region="CHN"),
            ]
        )
        s.add(
            IdeaORM(
                source="Research_Papers_for_Users",
                title="Research Paper 9: revisions",
                body="Implement with the cited dataset fields cited_rev_fy1 ...",
            )
        )
        s.add(IdeaORM(source="中文论坛", title="chat", body="cited is great"))  # not a paper
    with ledger.session() as s:
        order = [r.cell_id.split("|")[-1] for r in rank(s, FactoryConfig())]
        cited = s.scalar(select(CellORM).where(CellORM.dataset_id == "cited"))
        assert cited is not None and cited.paper_refs == 1 and cited.rank_reason["papers"] == 1.25
    assert order[0] == "cited" and "done" not in order
    assert order.index("plain") < order.index("crowded")
    assert order.index("plain") < order.index("chn")  # higher sharpe floor and costs
