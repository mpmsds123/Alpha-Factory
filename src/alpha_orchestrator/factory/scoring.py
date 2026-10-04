"""Where to mine today: expected value per simulation, with exploration.

    EV(cell) = P(hit) x value_if_hit

P(hit) is a Beta posterior over "a first-order probe reaches |sharpe| >= 1". The prior
mean 6.1% is the share of your 21,368 IQC simulations that did; its weight (10 pseudo-
observations) lets 50 real probes dominate it. value_if_hit multiplies what the platform
pays for (pyramid multiplier, open pyramid slots, themes) by what makes a hit survive
(crowding -> prod correlation, regional cost floor, regional sharpe floor).
"""

import math
import random
from dataclasses import dataclass

from alpha_orchestrator.factory.config import FactoryConfig

PRIOR_MEAN = 0.061
PRIOR_WEIGHT = 10.0


@dataclass(frozen=True)
class CellStats:
    cell_id: str
    region: str
    category: str
    state: str
    multiplier: float
    alpha_count: int
    themes: int
    probes: int
    hits: int
    pyramid_submitted: int   # your submissions this quarter in (region, delay, category)
    dataset_submitted: int   # your submissions from this exact dataset


def posterior(probes: int, hits: int) -> tuple[float, float]:
    a = PRIOR_MEAN * PRIOR_WEIGHT + hits
    b = (1 - PRIOR_MEAN) * PRIOR_WEIGHT + (probes - hits)
    return a, b


def hit_probability(stats: CellStats) -> float:
    a, b = posterior(stats.probes, stats.hits)
    return a / (a + b)


def gap_bonus(submitted: int) -> float:
    """A pyramid completes at 3 submissions; the last missing one is worth the most."""
    return {0: 1.15, 1: 1.2, 2: 1.3}.get(submitted, 1.0)


def crowding_discount(alpha_count: int) -> float:
    """10 alphas -> 0.10, 1,000 -> 0.30, capped at 0.6: crowded data rarely clears prod corr."""
    return min(0.6, math.log10(1 + max(alpha_count, 0)) / 10)


def cost_factor(margin_floor_bps: float) -> float:
    """Higher regional cost floors kill more hits after fees (5bps -> 1.0, 20bps -> 0.8)."""
    return max(0.5, 1.0 - (margin_floor_bps - 5.0) / 75.0)


def difficulty_factor(region: str, config: FactoryConfig) -> float:
    return config.thresholds.submit_sharpe / config.sharpe_floor(region)


def value_if_hit(stats: CellStats, config: FactoryConfig) -> float:
    if stats.dataset_submitted >= config.thresholds.dataset_cap:
        return 0.0
    return (
        stats.multiplier
        * gap_bonus(stats.pyramid_submitted)
        * (1.2 if stats.themes else 1.0)
        * (1 - crowding_discount(stats.alpha_count))
        * cost_factor(config.margin_floor(stats.region))
        * difficulty_factor(stats.region, config)
    )


def expected_value(stats: CellStats, config: FactoryConfig) -> float:
    return hit_probability(stats) * value_if_hit(stats, config)


@dataclass(frozen=True)
class Allocation:
    cell_id: str
    budget: int
    mode: str        # "explore" (probe) or "exploit" (expand a cell with signal)
    score: float


def plan_day(
    cells: list[CellStats],
    config: FactoryConfig,
    budget: int,
    rng: random.Random | None = None,
) -> list[Allocation]:
    """Split today's simulations between probing new cells and working cells with signal."""
    rng = rng or random.Random()
    explore_budget = int(budget * config.budget.explore_fraction)
    probe = config.budget.probe_budget

    unexplored = [c for c in cells if c.state in ("NEW", "PROFILED", "PROBING")
                  and value_if_hit(c, config) > 0]
    # Thompson sampling: draw a plausible hit rate per cell, rank by sampled EV.
    sampled = sorted(
        unexplored,
        key=lambda c: rng.betavariate(*posterior(c.probes, c.hits)) * value_if_hit(c, config),
        reverse=True,
    )
    plan: list[Allocation] = []
    remaining = explore_budget
    for c in sampled:
        need = max(probe - c.probes, 0)
        if need == 0 or remaining <= 0 or len(plan) >= config.budget.max_cells_per_day:
            continue
        take = min(need, remaining)
        plan.append(Allocation(c.cell_id, take, "explore", expected_value(c, config)))
        remaining -= take

    live = [c for c in cells if c.state == "SIGNAL"]
    scored = sorted(((expected_value(c, config), c) for c in live), key=lambda t: -t[0])
    exploit_budget = budget - (explore_budget - remaining)
    top = [(ev, c) for ev, c in scored if ev > 0][: max(config.budget.max_cells_per_day - len(plan), 0)]
    total = sum(ev for ev, _ in top)
    for ev, c in top:
        share = int(exploit_budget * ev / total) if total else 0
        if share > 0:
            plan.append(Allocation(c.cell_id, share, "exploit", ev))
    return plan
