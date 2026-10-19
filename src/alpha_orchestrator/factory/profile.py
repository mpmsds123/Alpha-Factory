"""Health check and survey formulas.

Health check (by sampling, neutralization NONE, decay 0, only long + short counts read).
First, one probe per sampled family, in waves; a dataset whose first wave all fails stops:

    maxcov  ts_backfill(x, 252)               the most stocks the field can ever cover

Then, on a few families that pass:

    raw     x                                 how many stocks have a value on a day
    upd5    ts_std_dev(x, 5) != 0 ? 1 : 0     changed within a week
    upd22   ts_std_dev(x, 22) != 0 ? 1 : 0    changed within a month
    upd66   ts_std_dev(x, 66) != 0 ? 1 : 0    changed within a quarter

"Coverage" means maxcov. A quarterly field that only has a value on its release day
shows a tiny daily coverage; carried forward it shows the real breadth.

Survey: four fixed shapes per family, windows set from the measured update period, so
every family is measured with the same ruler and compared with noise.
"""

from typing import Any

WINDOWS = (5, 21, 63, 126, 252)
CADENCE_LIMITS = (("daily", 5), ("weekly", 12), ("monthly", 30), ("quarterly", 95))


def base(field_id: str, field_type: str) -> str:
    return f"vec_avg({field_id})" if field_type == "VECTOR" else field_id


def health_probes(field_id: str, field_type: str) -> list[tuple[str, str]]:
    x = base(field_id, field_type)
    return [
        ("raw", x),
        ("upd5", f"ts_std_dev({x}, 5) != 0 ? 1 : 0"),
        ("upd22", f"ts_std_dev({x}, 22) != 0 ? 1 : 0"),
        ("upd66", f"ts_std_dev({x}, 66) != 0 ? 1 : 0"),
    ]


def coverage_probe(field_id: str, field_type: str) -> tuple[str, str]:
    """The most coverage the field can reach: its last value carried for a full year."""
    return "maxcov", f"ts_backfill({base(field_id, field_type)}, 252)"


def fill_window(period_days: float) -> int:
    return int(min(252, max(5, round(1.5 * period_days))))


def filled_probe(field_id: str, field_type: str, period_days: float) -> tuple[str, str]:
    return "filled", f"ts_backfill({base(field_id, field_type)}, {fill_window(period_days)})"


def universe_size(universe: str) -> int | None:
    digits = "".join(ch for ch in universe if ch.isdigit())
    return int(digits) if universe.startswith("TOP") and digits else None


def period_from(counts: dict[str, int | None]) -> float:
    """A field changing every P days is non-flat in an N-day window ~min(N/P, 1) of the
    time; the widest unsaturated window gives P ~ N / share."""
    raw = counts.get("raw") or 0
    if not raw:
        return 999.0

    def share(name: str) -> float:
        return (counts.get(name) or 0) / raw

    r5, r22, r66 = share("upd5"), share("upd22"), share("upd66")
    if r5 >= 0.9:
        return 1.0
    if r22 >= 0.9:
        return 5 / max(r5, 1e-3)
    if r66 >= 0.9:
        return 22 / max(r22, 1e-3)
    return 66 / r66 if r66 > 0 else 999.0


def cadence(period_days: float) -> str:
    return next((name for name, limit in CADENCE_LIMITS if period_days <= limit), "slow")


def snap(days: float) -> int:
    return min(WINDOWS, key=lambda w: abs(w - days))


def prep_for(field_id: str, field_type: str, profile: dict[str, Any]) -> str:
    """How the field enters every formula: filled to one update period when it does not
    update daily or when filling adds coverage; otherwise as is."""
    x = base(field_id, field_type)
    raw = float(profile.get("raw_coverage") or 0.0)
    filled = float(profile.get("filled_coverage") or 0.0)
    period = float(profile.get("period_days") or 66.0)
    if filled > raw * 1.1 or period > 5:
        return f"ts_backfill({x}, {fill_window(period)})"
    return x


def survey_shapes(x: str, period_days: float) -> list[tuple[str, str]]:
    change = snap(max(5.0, period_days))
    history = min(252, snap(max(21.0, 4 * period_days)))
    return [
        ("level", f"rank({x})"),
        ("change", f"rank(ts_delta({x}, {change}))"),
        ("history", f"rank(ts_zscore({x}, {history}))"),
        ("peer", f"group_rank({x}, industry)"),
    ]


def describe(profile: dict[str, Any]) -> str:
    if not profile:
        return "not checked"
    return (
        f"updates {cadence(float(profile.get('period_days', 999)))}"
        f" (~{profile.get('period_days')} days), coverage after filling "
        f"{float(profile.get('filled_coverage') or 0):.0%}"
    )
