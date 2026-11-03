"""v7 refinement step: the program diagnoses, the LLM picks fixes from a fixed menu, Thompson
sampling allocates the backtests, the program applies the fix.

Modelled on R&D-Agent(Q) (NeurIPS 2025): its analysis unit diagnoses each experiment and a
Thompson-sampling bandit chooses the next action. Here the arms are refinement types and the
reward is "the variant fails fewer platform checks than its parent".

Why: in the v6 ledger the LLM wrote free-form variants 206 times and none gained 0.35 Sharpe
(mean -0.215); the near misses were blocked by platform checks, not by Sharpe.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_DECILES = "densify(bucket(rank({x}), range='0.1, 1, 0.1'))"


@dataclass(frozen=True)
class Fix:
    name: str
    what: str  # shown to the LLM
    shape: str | None  # expression with {f}; None = a simulation setting changes
    setting: str | None = None  # "decay" | "neutralization"


FIXES: dict[str, Fix] = {
    f.name: f
    for f in (
        Fix("industry_rank", "rank the signal within its industry", "group_rank({f}, industry)"),
        Fix(
            "cap_bucket_zscore",
            "z-score the signal within market-cap deciles",
            "group_zscore({f}, " + _DECILES.format(x="cap") + ")",
        ),
        Fix(
            "cap_bucket_neutralize",
            "neutralize the signal within market-cap deciles",
            "group_neutralize({f}, " + _DECILES.format(x="cap") + ")",
        ),
        Fix(
            "liquidity_bucket_zscore",
            "z-score the signal within traded-value deciles",
            "group_zscore({f}, " + _DECILES.format(x="close*volume") + ")",
        ),
        Fix(
            "liquidity_bucket_neutralize",
            "neutralize the signal within traded-value deciles",
            "group_neutralize({f}, " + _DECILES.format(x="close*volume") + ")",
        ),
        Fix(
            "trade_on_volume_surge",
            "only rebalance when 10-day volume is above 60-day volume",
            "trade_when(ts_mean(volume, 10) > ts_mean(volume, 60), {f}, -1)",
        ),
        Fix(
            "trade_on_high_volatility",
            "only rebalance in stocks whose 60-day volatility is in the top 30% of their sector",
            "trade_when(group_rank(ts_std_dev(returns, 60), sector) > 0.7, {f}, -1)",
        ),
        Fix("smooth_5d", "average the signal over 5 days", "ts_mean({f}, 5)"),
        Fix("smooth_21d", "average the signal over 21 days", "ts_mean({f}, 21)"),
        Fix("winsorize", "clip the signal at 4 standard deviations", "winsorize({f}, std=4)"),
        Fix("raise_decay", "raise the simulation decay setting", None, "decay"),
        Fix(
            "switch_neutralization",
            "switch to a neutralization setting this idea has not tried",
            None,
            "neutralization",
        ),
    )
}

# A parent's failing checks -> one problem. The first matching row is the diagnosis: hard
# limits first, then the Sharpe/Fitness lines, then robustness (what blocked the v6 near
# misses: sub-period 16 and sub-universe 11 of 22 formulas with Sharpe >= 1.58).
PROBLEM_ORDER: tuple[tuple[str, frozenset[str]], ...] = (
    ("turnover too high", frozenset({"HIGH_TURNOVER"})),
    ("turnover too low", frozenset({"LOW_TURNOVER"})),
    ("concentrated in few stocks", frozenset({"CONCENTRATED_WEIGHT"})),
    ("sharpe too low", frozenset({"LOW_SHARPE"})),
    ("fitness too low", frozenset({"LOW_FITNESS"})),
    ("unstable over time", frozenset({"LOW_2Y_SHARPE", "IS_LADDER_SHARPE"})),
    (
        "fails in part of the universe",
        frozenset(
            {
                "LOW_SUB_UNIVERSE_SHARPE",
                "LOW_GLB_AMER_SHARPE",
                "LOW_GLB_EMEA_SHARPE",
                "LOW_GLB_APAC_SHARPE",
            }
        ),
    ),
)
OTHER = "below another line (margin or an unlisted check)"

# Which fixes address which problem.
MENU: dict[str, tuple[str, ...]] = {
    "turnover too high": ("raise_decay", "smooth_5d", "smooth_21d"),
    "turnover too low": ("industry_rank", "switch_neutralization", "liquidity_bucket_zscore"),
    "concentrated in few stocks": ("winsorize", "industry_rank", "cap_bucket_zscore"),
    "sharpe too low": (
        "industry_rank",
        "switch_neutralization",
        "liquidity_bucket_neutralize",
        "cap_bucket_zscore",
        "trade_on_volume_surge",
        "trade_on_high_volatility",
    ),
    "fitness too low": (
        "liquidity_bucket_neutralize",
        "cap_bucket_neutralize",
        "trade_on_volume_surge",
        "trade_on_high_volatility",
        "raise_decay",
    ),
    "unstable over time": (
        "trade_on_volume_surge",
        "trade_on_high_volatility",
        "smooth_21d",
        "switch_neutralization",
        "liquidity_bucket_neutralize",
    ),
    "fails in part of the universe": (
        "cap_bucket_zscore",
        "cap_bucket_neutralize",
        "liquidity_bucket_zscore",
        "liquidity_bucket_neutralize",
        "switch_neutralization",
    ),
    OTHER: (
        "liquidity_bucket_neutralize",
        "cap_bucket_neutralize",
        "trade_on_volume_surge",
        "switch_neutralization",
    ),
}


def fail_set(fails: str | None) -> set[str]:
    return {x for x in (fails or "").split(",") if x}


def diagnose(fails: str | None) -> str:
    got = fail_set(fails)
    for problem, checks in PROBLEM_ORDER:
        if got & checks:
            return problem
    return OTHER


def score(fails: str | None, sharpe: float | None) -> tuple[int, float]:
    """Fewer failed checks first, then Sharpe. A variant replaces its parent when its score
    is higher; it is a bandit success when it fails fewer checks."""
    return (-len(fail_set(fails)), sharpe or 0.0)


def success(child_fails: str | None, parent_fails: str | None) -> bool:
    return len(fail_set(child_fails)) < len(fail_set(parent_fails))


def _norm(e: str) -> str:
    return re.sub(r"\s+", "", e)


def recognize(
    child_expr: str,
    child_settings: dict[str, object],
    parent_expr: str,
    parent_settings: dict[str, object],
) -> str | None:
    """Which fix turned the parent into the child; None when it is not a menu fix. This lets
    the v6 enhancement rows (same wrappers) count as prior evidence."""
    c = _norm(child_expr)
    for f in FIXES.values():
        if f.shape is not None and c == _norm(f.shape.format(f=parent_expr)):
            return f.name
    if c == _norm(parent_expr):
        if child_settings.get("neutralization") != parent_settings.get("neutralization"):
            return "switch_neutralization"
        if int(child_settings.get("decay") or 0) > int(parent_settings.get("decay") or 0):
            return "raise_decay"
    return None


Stats = dict[tuple[str, str], list[int]]  # (problem, fix) -> [successes, failures]


def tally(rows: Iterable[tuple[str, str, bool]]) -> Stats:
    out: Stats = {}
    for problem, fix, ok in rows:
        cell = out.setdefault((problem, fix), [0, 0])
        cell[0 if ok else 1] += 1
    return out


def choose(
    problem: str,
    picks: Sequence[str],
    stats: Stats,
    k: int,
    rng: random.Random,
    tried: Iterable[str] = (),
) -> tuple[list[str], str]:
    """Thompson sampling: each allowed fix draws theta ~ Beta(1 + successes, 1 + failures);
    the k largest draws run. Allowed = the LLM's picks on this problem's menu, not yet tried
    on this parent; when the LLM gave none, the whole menu ("bandit_only")."""
    menu = [f for f in MENU.get(problem, ()) if f not in set(tried)]
    allowed = [f for f in dict.fromkeys(picks) if f in menu]
    source = "llm"
    if not allowed:
        allowed, source = menu, "bandit_only"
    draws = []
    for f in allowed:
        s, n = stats.get((problem, f), [0, 0])
        draws.append((rng.betavariate(1 + s, 1 + n), f))
    draws.sort(reverse=True)
    return [f for _, f in draws[:k]], source
