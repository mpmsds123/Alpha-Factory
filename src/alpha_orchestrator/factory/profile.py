"""Reading a dataset before researching it: the official "6 ways" (零基础课 第2课,
cnhkmcp brain-dataset-exploration).

Each probe runs with neutralization NONE and decay 0, and only the Long Count + Short
Count of the IS summary is read, never the PnL. Per representative field:

    cov    x                              how many stocks have a value
    nz     x != 0 ? 1 : 0                 real values vs. zeros filling gaps
    upd5   ts_std_dev(x, 5) != 0 ? 1 : 0  changed within a week   -> updates ~daily
    upd22  ts_std_dev(x, 22) != 0 ? 1 : 0 changed within a month  -> ~weekly/monthly
    upd66  ts_std_dev(x, 66) != 0 ? 1 : 0 changed within a quarter -> ~quarterly
    big    abs(x) > 1                     values beyond +-1: not pre-normalised

The readings decide how the field enters every template (`prep`): which vector reducer,
whether and how far to backfill, whether to winsorize. They do not score the dataset.
"""

from typing import Any

PROBES = (
    ("cov", "{x}"),
    ("nz", "{x} != 0 ? 1 : 0"),
    ("upd5", "ts_std_dev({x}, 5) != 0 ? 1 : 0"),
    ("upd22", "ts_std_dev({x}, 22) != 0 ? 1 : 0"),
    ("upd66", "ts_std_dev({x}, 66) != 0 ? 1 : 0"),
    ("big", "abs({x}) > 1"),
)
# News-like data loses meaning when carried forward (第4课: 前年的新闻对今年没用).
TIME_SENSITIVE = frozenset({"news", "socialmedia", "sentiment"})
CADENCE_LIMITS = (("daily", 5), ("weekly", 12), ("monthly", 30), ("quarterly", 95))
BACKFILL_FOR = {"daily": 5, "weekly": 22, "monthly": 66, "quarterly": 132, "slow": 252}


def base(field_id: str, field_type: str) -> str:
    return f"vec_avg({field_id})" if field_type == "VECTOR" else field_id


def probe_expressions(field_id: str, field_type: str) -> list[tuple[str, str]]:
    x = base(field_id, field_type)
    return [(name, template.format(x=x)) for name, template in PROBES]


def universe_size(universe: str) -> int | None:
    digits = "".join(ch for ch in universe if ch.isdigit())
    return int(digits) if universe.startswith("TOP") and digits else None


def interpret(counts: dict[str, int | None], universe: str) -> dict[str, Any]:
    """`counts[name]` = long + short count of that probe (None if it failed)."""
    cov = counts.get("cov") or 0
    size = universe_size(universe) or max(cov, 1)

    def share(name: str) -> float:
        value = counts.get(name)
        return round((value or 0) / cov, 3) if cov else 0.0

    # A field that changes every P days is non-flat in an N-day window about min(N/P, 1)
    # of the time, so the widest unsaturated window gives P ~ N / share.
    r5, r22, r66 = share("upd5"), share("upd22"), share("upd66")
    if r5 >= 0.9:
        period = 1.0
    elif r22 >= 0.9:
        period = 5 / max(r5, 1e-3)
    elif r66 >= 0.9:
        period = 22 / max(r22, 1e-3)
    else:
        period = 66 / r66 if r66 > 0 else 999.0
    cadence = next((name for name, limit in CADENCE_LIMITS if period <= limit), "slow")
    return {
        "coverage": round(min(cov / size, 1.0), 3),
        "nonzero": share("nz"),
        "cadence": cadence,
        "period_days": round(min(period, 999.0), 1),
        "beyond_one": share("big"),
        "normalised": share("big") < 0.02,
    }


def prep_for(field_id: str, field_type: str, category: str, profile: dict[str, Any]) -> str:
    """How the field enters templates. `{x}` in a template is replaced by this."""
    x = base(field_id, field_type)
    if not profile:
        return f"ts_backfill({x}, 66)" if category.lower() not in TIME_SENSITIVE else x
    cadence = str(profile.get("cadence", "daily"))
    sparse = float(profile.get("coverage", 1.0)) < 0.9 or cadence != "daily"
    if sparse and category.lower() not in TIME_SENSITIVE:
        x = f"ts_backfill({x}, {BACKFILL_FOR[cadence]})"
    if not profile.get("normalised", True):
        x = f"winsorize({x}, std=4)"
    return x


def describe(profile: dict[str, Any]) -> str:
    """One line for the LLM."""
    if not profile:
        return "not probed"
    return (
        f"coverage {profile['coverage']:.0%}, non-zero {profile['nonzero']:.0%}, "
        f"updates {profile['cadence']}, "
        + ("values within +-1" if profile["normalised"] else "values beyond +-1 (unscaled)")
    )
