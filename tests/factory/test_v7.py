"""v7 (2026-11-03): menu refinement with Thompson sampling, random dataset order, memory of
past ideas. Same simulated world as test_engine; only the design switch differs."""

import json
import random
from collections import Counter
from typing import Any

from sqlalchemy import select

from alpha_orchestrator.factory import engine as eng
from alpha_orchestrator.factory import fixes, study
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CellORM,
    EventORM,
    IdeaStatus,
    ResearchIdeaORM,
    Stage,
)

from .test_engine import _SizeWorld, eps_fields, honest_llm
from .world import World, analyst_fields

PROMPTS: list[dict[str, Any]] = []


def v7_llm(model: str, system: str, user: str) -> tuple[str, int, int]:
    p = json.loads(user)
    PROMPTS.append(p)
    if p["task"].startswith("Choose fixes"):
        names = [m["fix"] for m in p["menu"]]
        liquid = [n for n in names if n.startswith("liquidity")]
        picks = [{"fix": n, "why": "the signal sits in small, illiquid names"} for n in liquid]
        picks.append({"fix": "not_on_the_menu", "why": "ignored"})
        return json.dumps({"decision": "change", "picks": picks}), 500, 100
    return honest_llm(model, system, user)


def run_v7(world: World, minutes: float = 30):  # type: ignore[no-untyped-def]
    ledger = Ledger("sqlite://")
    api = world.platform()
    cfg = FactoryConfig(design="v7")
    cfg.budget.min_batch_interval_s = 0
    llm = FactoryLLM(cfg, ledger, complete=v7_llm)
    report = eng.run_day(
        cfg, api, ledger, llm, minutes=minutes, rng=random.Random(3), sleep=lambda _s: None
    )
    return ledger, api, report


# ---------- the menu and the bandit ----------
def test_diagnosis_takes_hard_limits_first_then_sharpe_then_robustness() -> None:
    assert fixes.diagnose("LOW_SHARPE,HIGH_TURNOVER") == "turnover too high"
    assert fixes.diagnose("LOW_FITNESS,LOW_SHARPE") == "sharpe too low"
    assert fixes.diagnose("IS_LADDER_SHARPE,LOW_SUB_UNIVERSE_SHARPE") == "unstable over time"
    assert fixes.diagnose("LOW_GLB_EMEA_SHARPE") == "fails in part of the universe"
    assert fixes.diagnose("") == fixes.OTHER
    assert all(set(m) <= set(fixes.FIXES) for m in fixes.MENU.values())


def test_v6_enhancement_rows_are_recognized_as_menu_fixes() -> None:
    parent = "rank(x)"
    for name in ("cap_bucket_zscore", "liquidity_bucket_neutralize", "trade_on_volume_surge"):
        child = fixes.FIXES[name].shape.format(f=parent)  # type: ignore[union-attr]
        assert fixes.recognize(child, {}, parent, {}) == name
    assert fixes.recognize(parent, {"decay": 5}, parent, {"decay": 0}) == "raise_decay"
    assert (
        fixes.recognize(parent, {"neutralization": "MARKET"}, parent, {"neutralization": "SECTOR"})
        == "switch_neutralization"
    )
    assert fixes.recognize("group_rank(rank(x), subindustry)", {}, parent, {}) is None


def test_thompson_sampling_keeps_to_the_menu_and_favours_what_worked() -> None:
    problem = "fails in part of the universe"
    stats = {(problem, "cap_bucket_zscore"): [9, 1], (problem, "liquidity_bucket_zscore"): [0, 9]}
    rng = random.Random(0)
    picks = ["cap_bucket_zscore", "liquidity_bucket_zscore", "smooth_5d"]  # smooth: off-menu
    first = Counter(fixes.choose(problem, picks, stats, 1, rng)[0][0] for _ in range(400))
    assert set(first) <= {"cap_bucket_zscore", "liquidity_bucket_zscore"}
    assert first["cap_bucket_zscore"] > 300  # 9/10 beats 0/9 almost always
    chosen, source = fixes.choose(problem, [], stats, 3, rng, tried=["cap_bucket_zscore"])
    assert source == "bandit_only" and len(chosen) == 3 and "cap_bucket_zscore" not in chosen


def test_success_means_fewer_failed_checks() -> None:
    assert fixes.success("LOW_SHARPE", "LOW_SHARPE,LOW_FITNESS")
    assert not fixes.success("LOW_SHARPE", "LOW_FITNESS")
    assert fixes.score("", 1.6) > fixes.score("LOW_SHARPE", 2.5)


# ---------- end to end ----------
def test_v7_refines_from_the_menu_and_a_liquidity_fix_passes() -> None:
    world = _SizeWorld({"est": eps_fields(1.1)})
    world.bucket_gain = 0.6
    ledger, api, report = run_v7(world)
    with ledger.session() as s:
        improved = list(s.scalars(select(CandidateORM).where(CandidateORM.stage == Stage.IMPROVE)))
        assert improved, report
        assert all((c.extra or {}).get("fix") in fixes.FIXES for c in improved)
        assert all("not_on_the_menu" not in c.change for c in improved)
        assert not list(s.scalars(select(CandidateORM).where(CandidateORM.stage == Stage.ENHANCE)))
        done = [i for i in s.scalars(select(ResearchIdeaORM)) if i.status == IdeaStatus.DONE]
        assert done, "a liquidity-decile fix lifts the near miss over the line"
        hits = {s.get(CandidateORM, h) for i in done for h in i.extra["regular_hits"]}
        assert any(h is not None and h.stage == Stage.IMPROVE for h in hits)
        diag = done[0].extra["diagnoses"][0]
        assert diag["problem"] and diag["chosen"] and diag["source"] in {"llm", "bandit_only"}
    assert api.submitted == []


def eight_groups_llm(model: str, system: str, user: str) -> tuple[str, int, int]:
    p = json.loads(user)
    if p["task"].startswith("You are a quant researcher reading"):
        ids = [f["id"] for f in p["fields"]]
        groups = [
            {
                "name": f"group {k}",
                "meaning": f"meaning {k}",
                "direction": "higher is better",
                "why": "x",
                "fields": [ids[k]],
                "field_signs": {ids[k]: "high"},
                "horizon_days": 21,
                "falsifier": "top-minus-bottom quintile return is not positive",
            }
            for k in range(8)
        ]
        return json.dumps({"groups": groups}), 900, 200
    return v7_llm(model, system, user)


def test_v7_keeps_six_ideas_in_flight() -> None:
    ledger = Ledger("sqlite://")
    world = World({"est": analyst_fields(None)}, noise=0.2)
    cfg = FactoryConfig(design="v7")
    cfg.budget.min_batch_interval_s = 0
    llm = FactoryLLM(cfg, ledger, complete=eight_groups_llm)
    eng.run_day(
        cfg, world.platform(), ledger, llm, minutes=5, rng=random.Random(3), sleep=lambda _s: None
    )
    with ledger.session() as s:
        kinds = [e.kind for e in s.scalars(select(EventORM).order_by(EventORM.id))]
    first_verdict = min(
        i for i, k in enumerate(kinds) if k in ("idea_dead", "idea_to_improve", "idea_done")
    )
    assert kinds[:first_verdict].count("idea_created") == eng.V7_ACTIVE_IDEAS


def test_v7_picks_datasets_at_random_among_new_ones() -> None:
    ledger = Ledger("sqlite://")
    cfg = FactoryConfig(design="v7")
    with ledger.session() as s:
        for i in range(20):
            s.add(
                CellORM(
                    cell_id=f"USA|1|TOP3000|d{i}",
                    region="USA",
                    delay=1,
                    universe="TOP3000",
                    dataset_id=f"d{i}",
                    dataset_name=f"d{i}",
                    category="analyst",
                    description="",
                    multiplier=1.0,
                    value_score=1.0,
                    alpha_count=0,
                    user_count=0,
                    coverage=0.9,
                    field_count=10,
                    themes=[],
                    state="NEW",
                    rank_score=float(i),
                    rank_reason={},
                    paper_refs=0,
                )
            )
    with ledger.session() as s:
        picks = {study.choose_dataset(s, cfg, random.Random(k)) for k in range(30)}
    assert len(picks) > 5  # not the top-ranked one every time


def test_past_ideas_reach_idea_generation_without_formula_text() -> None:
    PROMPTS.clear()
    world = World({"est": analyst_fields(None), "est2": eps_fields()}, noise=0.2)
    ledger, _, _ = run_v7(world, minutes=30)
    with ledger.session() as s:
        past = eng._past_ideas(s, 40)
    assert past and all("rank(" not in json.dumps(p) for p in past)
    grouping = [p for p in PROMPTS if p["task"].startswith("You are a quant researcher reading")]
    assert len(grouping) >= 2  # two datasets
    assert "past_ideas_and_results" not in grouping[0]  # nothing finished yet
    assert any(p.get("past_ideas_and_results") for p in grouping[1:])


def test_weekly_report_shows_the_funnel_bottleneck_and_fix_table(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from alpha_orchestrator.factory.weekly import FUNNEL, weekly_rows, write_weekly

    world = _SizeWorld({"est": eps_fields(1.1)})
    world.bucket_gain = 0.6
    ledger, _, _ = run_v7(world)
    cfg = FactoryConfig(design="v7", reports_dir=str(tmp_path))
    rows = dict(weekly_rows(ledger, cfg))
    assert int(rows["回测完成"]) > 0
    assert int(rows[f"漏斗：{FUNNEL[0]}"]) >= int(rows[f"漏斗：{FUNNEL[-1]}"]) >= 1
    assert any(k.startswith("改法：") and "liquidity" in k for k in rows)
    assert "瓶颈（通过率最低的一步）" in rows
    assert write_weekly(ledger, cfg).exists()


def test_unlisted_robustness_checks_are_diagnosed_by_pattern() -> None:
    assert fixes.diagnose("LOW_ASI_JPN_SHARPE") == "fails in part of the universe"
    assert fixes.diagnose("LOW_ROBUST_UNIVERSE_RETURNS") == "fails in part of the universe"
    assert (
        fixes.diagnose("LOW_ROBUST_UNIVERSE_SHARPE.WITH_RATIO") == "fails in part of the universe"
    )
    assert fixes.diagnose("IS_LADDER_SHARPE,LOW_ASI_JPN_SHARPE") == "unstable over time"
    assert fixes.diagnose("LOW_DURATION") == fixes.OTHER


def test_a_wrapper_fix_is_not_applied_twice() -> None:
    once = fixes.FIXES["liquidity_bucket_neutralize"].shape.format(f="rank(x)")  # type: ignore[union-attr]
    assert fixes.already_applied("liquidity_bucket_neutralize", once)
    assert not fixes.already_applied("cap_bucket_neutralize", once)
    assert not fixes.already_applied("raise_decay", once)
