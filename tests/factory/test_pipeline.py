"""The statistical pipeline on simulated datasets whose truth is known."""

import random
from typing import Any

from sqlalchemy import select

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.daily import run_day
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CellORM,
    EventORM,
    FieldORM,
    ReviewORM,
)
from alpha_orchestrator.factory.select import approve

from .world import FieldSpec, World, analyst_fields, scripted_llm


def run(world: World, *, llm: bool = True, minutes: float = 5, **cfg_kw: Any):  # type: ignore[no-untyped-def]
    ledger = Ledger("sqlite://")
    api = world.platform(**cfg_kw.pop("api", {}))
    cfg = FactoryConfig()
    cfg.budget.min_batch_interval_s = 0
    cfg.thresholds.noise_sigma = 0.40  # the older research pipeline was calibrated on 0.40
    cfg.budget.dataset_sim_cap = 2000
    for k, v in cfg_kw.items():
        setattr(cfg.budget, k, v)
    model = FactoryLLM(cfg, ledger, complete=scripted_llm) if llm else None
    report = run_day(
        cfg, api, ledger, model, minutes=minutes, rng=random.Random(5), sleep=lambda _s: None
    )
    return ledger, api, cfg, report


def cells(ledger: Ledger) -> dict[str, CellORM]:
    with ledger.session() as s:
        return {c.dataset_id: c for c in s.scalars(select(CellORM))}


def stages(ledger: Ledger, dataset: str) -> dict[str, int]:
    with ledger.session() as s:
        out: dict[str, int] = {}
        for c in s.scalars(select(CandidateORM).where(CandidateORM.cell_id.like(f"%|{dataset}"))):
            out[c.stage] = out.get(c.stage, 0) + 1
        return out


def test_signal_is_found_dug_and_handed_over() -> None:
    world = World({"est": analyst_fields("eps")})
    ledger, api, _cfg, _report = run(world)
    c = cells(ledger)["est"]
    plan = c.plan
    # health by sampling: every family probed here (24 < 40), 5 probes each
    assert plan["family_count"] == 24 and plan["strata"] == 6
    assert stages(ledger, "est")["PROFILE"] == 24 + 12 * 4  # coverage on all, period on 12
    # survey: a small dataset is surveyed whole, 4 shapes per family; signal is found
    assert plan["verdict"] == "signal"
    # the eps families (where the signal is) were dug; a noise family may be tried once
    refined = {k: v for k, v in plan["families"].items() if v.get("rounds")}
    eps = {k: v for k, v in refined.items() if "_eps_" in k}
    assert eps and max(v["best"] for v in eps.values()) > max(
        (v["best"] for k, v in refined.items() if "_eps_" not in k), default=0
    )
    with ledger.session() as s:
        ready = [x for x in s.scalars(select(CandidateORM)) if x.extra.get("ready")]
        reviews = list(s.scalars(select(ReviewORM)))
    assert ready and all("eps" in x.expression for x in ready)
    assert any((x.extra.get("robust") or {}).get("pass") for x in ready)
    assert reviews and api.submitted == []
    assert approve(api, ledger, reviews[0].id)  # type: ignore[arg-type]
    assert api.submitted == [reviews[0].alpha_id]


def test_noise_dataset_is_dropped_after_the_survey() -> None:
    world = World({"flat": analyst_fields(None)})
    ledger, _, _, _ = run(world)
    c = cells(ledger)["flat"]
    assert c.state == "EXHAUSTED" and c.plan["outcome"] == "noise"
    st = stages(ledger, "flat")
    assert "REFINE" not in st and "COMBO" not in st
    assert st["SURVEY"] <= 24 * 4  # the survey only, nothing more


def test_coverage_is_judged_after_filling_one_update_period() -> None:
    specs = analyst_fields("eps")
    # quarterly release-day data: 4% daily, 80% once filled -> kept, entered filled
    for k in [k for k in specs if k.startswith("anl_rev_")]:
        specs[k] = FieldSpec(raw=0.04, filled=0.8, period=63.0)
    # thin everywhere -> not eligible, never surveyed
    for k in [k for k in specs if k.startswith("anl_dps_")]:
        specs[k] = FieldSpec(raw=0.2, filled=0.25, period=63.0)
    world = World({"est": specs})
    ledger, *_ = run(world)
    with ledger.session() as s:
        fields = {f.field_id: f for f in s.scalars(select(FieldORM))}
    rev = fields["anl_rev_mean_fy1"]
    assert rev.eligible and rev.prep.startswith("ts_backfill(anl_rev_mean_fy1, ")
    assert 50 <= rev.profile["period_days"] <= 80
    assert not fields["anl_dps_mean_fy1"].eligible
    assert not any(
        "anl_dps" in e
        for e in world.calls
        if not e.startswith("ts_") and "dps" in e and "rank" in e
    )


def test_degenerate_fields_are_never_mined() -> None:
    specs = analyst_fields("eps")
    specs["anl_tgt_mean_fy1"] = FieldSpec(degenerate=True)
    specs["anl_tgt_mean_fy2"] = FieldSpec(degenerate=True)
    ledger, *_ = run(World({"est": specs}))
    with ledger.session() as s:
        refine = [c for c in s.scalars(select(CandidateORM)) if c.stage == "REFINE"]
        flagged = [c for c in s.scalars(select(CandidateORM)) if c.extra.get("suspicious")]
    assert not any("anl_tgt" in c.expression for c in refine)
    assert all(abs(c.sharpe or 0) > 3 or (c.turnover or 0) > 0.7 for c in flagged)


def test_llm_combinations_run_next_to_random_ones() -> None:
    # combinations only where a whole stratum is clearly above noise
    ledger, *_ = run(
        World({"est": analyst_fields("eps", signal_stats=("mean", "median", "high", "low"))})
    )
    with ledger.session() as s:
        combos = [c for c in s.scalars(select(CandidateORM)) if c.stage == "COMBO"]
    arms = {c.extra.get("arm") for c in combos}
    assert arms == {"llm", "random"}


def test_runs_without_deepseek() -> None:
    ledger, *_ = run(
        World({"est": analyst_fields("eps", signal_stats=("mean", "median", "high", "low"))}),
        llm=False,
    )
    c = cells(ledger)["est"]
    assert c.plan["verdict"] == "signal"
    assert "COMBO" in stages(ledger, "est")  # random arm still runs


def test_concurrency_follows_what_the_platform_accepts() -> None:
    ledger, _api, _cfg, report = run(
        World({"est": analyst_fields("eps")}), api={"polls_before_done": 2, "max_running": 2}
    )
    with ledger.session() as s:
        limits = [
            e.payload["batches"]
            for e in s.scalars(select(EventORM))
            if e.kind == "concurrency_limit"
        ]
    assert limits and limits[0] == 2
    assert report.simulations > 100 and report.stopped_because != "platform kept refusing batches"


def test_one_dataset_at_a_time_and_failures_feed_back() -> None:
    world = World({"flat": analyst_fields(None, prefix="flt"), "est": analyst_fields("eps")})
    ledger, *_ = run(world, minutes=10)
    with ledger.session() as s:
        opened = [
            e.cell_id
            for e in s.scalars(select(EventORM).order_by(EventORM.id))
            if e.kind == "dataset_opened"
        ]
    assert len(opened) == len(set(opened)) == 2
    states = {k: v.plan.get("outcome") for k, v in cells(ledger).items()}
    assert states == {"flat": "noise", "est": "signal"}


def sparse_fields(prefix: str, n: int = 30) -> dict[str, FieldSpec]:
    """Warrant-like data: only a few stocks ever have a value."""
    return {f"{prefix}_w{i:02d}_px": FieldSpec(raw=0.05, filled=0.07, period=1.0) for i in range(n)}


def test_a_sparse_dataset_costs_ten_probes() -> None:
    ledger, *_ = run(World({"wrnt": sparse_fields("wrnt")}))
    c = cells(ledger)["wrnt"]
    assert c.plan["outcome"] == "no_eligible"
    assert stages(ledger, "wrnt") == {"PROFILE": 10}  # first wave only, nothing else


def test_sparse_outcomes_steer_selection_away() -> None:
    world = World(
        {
            "wrnt1": sparse_fields("wa"),
            "wrnt2": sparse_fields("wb"),
            "wrnt3": sparse_fields("wc"),
            "est": analyst_fields("eps", signal_stats=("mean", "median", "high", "low")),
        }
    )
    ledger, *_ = run(world, minutes=10)
    spent = {k: v.sims_used for k, v in cells(ledger).items()}
    assert cells(ledger)["est"].plan["verdict"] == "signal"
    assert sum(v for k, v in spent.items() if k.startswith("wrnt")) <= 30


def test_catalog_coverage_below_the_floor_is_never_opened() -> None:
    from alpha_orchestrator.factory import datasets as ds

    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        for name, cov in (("thin", 0.2), ("wide", 0.9)):
            s.add(
                CellORM(
                    cell_id=f"USA|1|TOP3000|{name}",
                    region="USA",
                    delay=1,
                    universe="TOP3000",
                    dataset_id=name,
                    category="other",
                    coverage=cov,
                )
            )
    with ledger.session() as s:
        names = [r.cell_id.split("|")[-1] for r in ds.rank(s, FactoryConfig())]
    assert names == ["wide"]
