"""Item 7: which ideas keep their budget? The program computes the numbers, DeepSeek reads the
whole pool of one dataset and proposes, the program checks every proposal and executes only
the ones that are allowed.

Hit probability of an idea is read off the ledger: ideas are put in bands by their best first
formula (below 1.0, 1.0-1.2, 1.2 and above) and each band's rate is the share of its ideas that
later reached the Sharpe line, smoothed with half a hit and half a miss (Jeffreys) so that a
band with no hit is not exactly zero.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

BAND_EDGES = (1.0, 1.2, math.inf)  # upper edges of the three bands
ACTIONS = ("continue", "stop", "enhance")


def band(first_best: float) -> int:
    """Return the index of the band a first-formula Sharpe falls in."""
    for i, edge in enumerate(BAND_EDGES):
        if first_best < edge:
            return i
    return len(BAND_EDGES) - 1


def band_rates(ideas: Iterable[tuple[float, bool]]) -> list[dict[str, float]]:
    """ideas: (best first-formula |Sharpe|, whether the idea later reached the Sharpe line)."""
    hits = [0] * len(BAND_EDGES)
    n = [0] * len(BAND_EDGES)
    for first_best, reached in ideas:
        b = band(first_best)
        n[b] += 1
        hits[b] += bool(reached)
    return [
        {"n": n[i], "hits": hits[i], "p": (hits[i] + 0.5) / (n[i] + 1)}
        for i in range(len(BAND_EDGES))
    ]


@dataclass(frozen=True)
class Decision:
    idea: str
    action: str
    why: str


def parse(
    raw: Mapping[str, Any], allowed: Mapping[str, Sequence[str]]
) -> tuple[list[Decision], list[str]]:
    """Keep the proposals that name a known idea and an action allowed for it, at most one per
    idea. `allowed` maps the full idea id to its permitted actions. Returns (accepted,
    problems). A proposal may give the first 8 characters of the id."""
    by_prefix = {k[:8]: k for k in allowed}
    out: list[Decision] = []
    problems: list[str] = []
    seen: set[str] = set()
    for d in raw.get("decisions") or []:
        if not isinstance(d, dict):
            problems.append("not an object")
            continue
        key = str(d.get("idea", ""))
        full = key if key in allowed else by_prefix.get(key[:8])
        action = str(d.get("action", "")).lower()
        if full is None:
            problems.append(f"{key}: unknown idea")
        elif full in seen:
            problems.append(f"{key}: more than one decision")
        elif action not in allowed[full]:
            problems.append(f"{key}: {action} not allowed (allowed: {', '.join(allowed[full])})")
        else:
            seen.add(full)
            out.append(Decision(full, action, str(d.get("why", ""))[:200]))
    return out, problems
