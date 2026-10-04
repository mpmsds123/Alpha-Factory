"""A simulated dataset world for the pipeline tests: each field has a daily coverage, a
coverage after filling, an update period, and optionally a real signal in one shape. Every
other formula returns noise with the measured spread. No network, no LLM spend."""

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from .fakes import FakePlatform, is_metrics

UNIVERSE = 3000
WINDOW = re.compile(r",\s*(\d+)\)")


@dataclass
class FieldSpec:
    raw: float = 0.9  # daily coverage
    filled: float = 0.9  # coverage after filling one period
    period: float = 1.0  # days between updates
    signal: float = 0.0  # |Sharpe| of the shape that carries it
    shape: str = "ts_delta"  # operator that reveals the signal
    sign: int = 1
    degenerate: bool = False  # tiny coverage: absurd sharpe, 100% turnover


@dataclass
class World:
    fields: dict[str, dict[str, FieldSpec]]  # dataset -> field id -> spec
    noise: float = 0.35
    seed: int = 0
    calls: list[str] = field(default_factory=list)

    def spec(self, expr: str) -> tuple[str, FieldSpec] | tuple[None, None]:
        for ds in self.fields.values():
            for fid in sorted(ds, key=len, reverse=True):
                if re.search(rf"(?<![\w]){re.escape(fid)}(?![\w])", expr):
                    return fid, ds[fid]
        return None, None

    def metrics(self, expr: str, settings: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(expr)
        _fid, sp = self.spec(expr)
        if settings.get("neutralization") == "NONE":  # health probes: read counts
            assert sp is not None
            if expr.startswith("ts_std_dev("):
                n = int(WINDOW.search(expr).group(1))  # type: ignore[union-attr]
                count = sp.raw * UNIVERSE * min(n / sp.period, 1.0)
            elif expr.startswith("ts_backfill("):
                n = int(WINDOW.search(expr).group(1))  # type: ignore[union-attr]
                count = (sp.filled if n >= sp.period else sp.raw) * UNIVERSE
            else:
                count = sp.raw * UNIVERSE
            return is_metrics(0.0, 0.0, long_count=int(count), short_count=0)
        rng = random.Random(f"{self.seed}|{expr}")
        sharpe = rng.gauss(0, self.noise)
        turnover = 0.12
        if sp is not None and sp.degenerate:
            sharpe, turnover = 9.5, 1.0
        elif sp is not None and sp.signal and sp.shape in expr:
            sharpe = sp.sign * sp.signal + rng.gauss(0, 0.1)
            if expr.startswith("multiply(-1"):
                sharpe = -sharpe
            if "subindustry" in expr:
                sharpe += 0.35 if sharpe > 0 else -0.35
            if "volume > adv20" in expr:
                sharpe += 0.2 if sharpe > 0 else -0.2
        fitness = sharpe * 0.75
        fails = "" if sharpe >= 1.58 and fitness >= 1 else "LOW_SHARPE"
        return is_metrics(sharpe, fitness, turnover=turnover, margin=0.0012, fails=fails)

    def platform(self, **kw: Any) -> FakePlatform:
        rows = {
            ds: [
                {
                    "id": fid,
                    "type": "MATRIX",
                    "coverage": sp.raw,
                    "description": fid.replace("_", " "),
                }
                for fid, sp in specs.items()
            ]
            for ds, specs in self.fields.items()
        }
        return FakePlatform(self.metrics, rows, **kw)


def analyst_fields(
    signal_stratum: str | None,
    strength: float = 1.4,
    prefix: str = "anl",
    signal_stats: tuple[str, ...] = ("mean", "median"),
) -> dict[str, FieldSpec]:
    """6 concepts x 4 statistics x 2 horizons = 48 fields, 24 families, 6 strata."""
    out = {}
    for concept in ("eps", "rev", "dps", "tgt", "ebit", "cfps"):
        for stat in ("mean", "median", "high", "low"):
            for horizon in ("fy1", "fy2"):
                sig = strength if concept == signal_stratum and stat in signal_stats else 0.0
                out[f"{prefix}_{concept}_{stat}_{horizon}"] = FieldSpec(
                    raw=0.85, filled=0.9, period=7.0, signal=sig, shape="ts_delta"
                )
    return out


def scripted_llm(model: str, system: str, user: str) -> tuple[str, int, int]:
    prompt = json.loads(user)
    task = prompt["task"]
    if task.startswith("Group these"):
        groups: dict[str, list[str]] = {}
        for f in prompt["families"]:
            groups.setdefault(f["family"].split("_")[1], []).append(f["family"])
        return json.dumps({"groups": groups}), 100, 50
    if task.startswith("Propose"):
        fields = prompt["strata"][0]["fields"]
        a, b = fields[0]["prep"], fields[-1]["prep"]
        combos = [
            {
                "expression": f"trade_when(rank({b}) < 0.3, rank(ts_delta({a}, 5)), -1)",
                "why": "revision is reliable when dispersion is low",
            },
            {"expression": f"rank(ts_delta({a}, 5)) - rank({b})", "why": "net revision"},
        ]
        return json.dumps({"combinations": combos}), 300, 120
    return json.dumps({"description": "Idea: ... Rationale: ..."}), 50, 50
