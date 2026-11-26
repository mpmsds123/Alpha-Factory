"""Local checks before the platform's own: crowded fields and self-correlation.

AlphaForge runs a short list of checks before it asks the platform anything:
  crowded fields   a private list of fields that keep failing the platform's correlation
                   with production alphas; no new hypothesis or helper may use them, and a
                   field is added automatically when it causes such a failure
  self-correlation the candidate's daily PnL against every submitted alpha's, computed
                   locally (line 0.60, under the platform's 0.7, for a margin)
  operator cap     64 operators per expression (in validate.check, via the slot limits)
"""

import json
import math
from collections.abc import Iterable
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import yaml

from alpha_orchestrator.factory.config import FactoryConfig


# ---------- crowded fields ----------
def crowded_path(config: FactoryConfig) -> Path:
    return Path(config.private_dir).expanduser() / config.checks.crowded_file


def load_crowded(config: FactoryConfig) -> set[str]:
    """Field ids on the crowded list (empty when the file is missing)."""
    path = crowded_path(config)
    if not path.is_file():
        return set()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = raw.get("fields") if isinstance(raw, dict) else raw
    out = set()
    for r in rows or []:
        out.add(str(r["field"] if isinstance(r, dict) else r))
    return out


def add_crowded(config: FactoryConfig, fields: Iterable[str], reason: str) -> list[str]:
    """Append fields to the crowded list; return the ones that were new."""
    path = crowded_path(config)
    raw: dict[str, Any] = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw = loaded if isinstance(loaded, dict) else {"fields": loaded or []}
    rows = list(raw.get("fields") or [])
    have = {str(r["field"] if isinstance(r, dict) else r) for r in rows}
    new = [f for f in dict.fromkeys(fields) if f and f not in have]
    if not new:
        return []
    today = date.today().isoformat()
    rows += [{"field": f, "added": today, "reason": reason[:200]} for f in new]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"fields": rows}, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return new


# ---------- local self-correlation ----------
class PnLSource(Protocol):
    def daily_pnl(self, alpha_id: str) -> dict[str, float]: ...

    def submitted_alphas(self) -> list[str]: ...


def _changes(pnl: dict[str, float], years: int) -> dict[str, float]:
    """Daily PnL changes over the last `years` years (the platform's PnL is cumulative)."""
    days = sorted(pnl)
    if len(days) < 2:
        return {}
    last = date.fromisoformat(days[-1][:10])
    start = (last - timedelta(days=365 * years)).isoformat()
    out = {}
    for prev, day in pairwise(days):
        if day >= start:
            out[day] = pnl[day] - pnl[prev]
    return out


def correlation(a: dict[str, float], b: dict[str, float]) -> float | None:
    """Pearson correlation on the days both series have; None with fewer than 60."""
    days = sorted(set(a) & set(b))
    if len(days) < 60:
        return None
    xs, ys = [a[d] for d in days], [b[d] for d in days]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


class SelfCorrelation:
    """Daily PnL of submitted alphas, cached on disk (they never change once submitted)."""

    def __init__(self, api: PnLSource, config: FactoryConfig) -> None:
        self.api, self.config = api, config
        self.cache = Path(config.checks.pnl_cache).expanduser()
        self._submitted: list[str] | None = None

    def _pnl(self, alpha_id: str, cache: bool) -> dict[str, float]:
        path = self.cache / f"{alpha_id}.json"
        if cache and path.is_file():
            data: dict[str, float] = json.loads(path.read_text(encoding="utf-8"))
            return data
        pnl = self.api.daily_pnl(alpha_id)
        if cache and pnl:
            self.cache.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(pnl), encoding="utf-8")
        return pnl

    def submitted(self) -> list[str]:
        if self._submitted is None:
            self._submitted = list(self.api.submitted_alphas())
        return self._submitted

    def max_with_submitted(self, alpha_id: str) -> tuple[float | None, str]:
        """(highest correlation with a submitted alpha, that alpha's id)."""
        years = self.config.checks.self_corr_years
        mine = _changes(self._pnl(alpha_id, cache=False), years)
        best: tuple[float | None, str] = (None, "")
        for other in self.submitted():
            if other == alpha_id:
                continue
            c = correlation(mine, _changes(self._pnl(other, cache=True), years))
            if c is not None and (best[0] is None or c > best[0]):
                best = (c, other)
        return best
