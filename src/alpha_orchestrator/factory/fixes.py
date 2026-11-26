"""The fix table: each failed check maps to one fix.

AlphaForge does not let the model decide freely how to repair a failing alpha: each kind
of failure has one prescribed change (high turnover -> decay, weight concentration ->
spread the weight, crowding -> combine independent data, ...). The program reads the
failure from the numbers and the table gives the change; the LLM only writes it.

A fix whose change is a setting is applied by the program itself, without an LLM call.
"""

from dataclasses import dataclass

from alpha_orchestrator.factory.config import Gates
from alpha_orchestrator.factory.models import SimulationORM

DECAYS = (0, 2, 5, 10, 21, 42)  # each step roughly halves turnover
FITNESS_TURNOVER_FLOOR = 0.125  # fitness divides returns by max(turnover, 0.125)


@dataclass(frozen=True)
class Fix:
    failure: str  # what failed, in words
    change: str  # one of the mutation change types: setting/window/operator/field/rewrite
    instruction: str  # what the change must do
    decay: int | None = None  # set for a program-applied decay change


def _next_decay(current: int) -> int | None:
    return next((d for d in DECAYS if d > current), None)


def for_train(
    sim: SimulationORM,
    region: str,
    gates: Gates,
    tried: set[str],
    band: tuple[float, float] = (0.0, 1.0),
) -> Fix | None:
    """The fix for the best expression of a round, from its training numbers.
    `tried` holds the change types already used on this hypothesis, so a failure that
    persists moves down its list instead of repeating the same fix."""
    sharpe = sim.train_sharpe or 0.0
    fitness = sim.train_fitness or 0.0
    turnover = sim.train_turnover or 0.0
    decay = int((sim.settings or {}).get("decay") or 0)
    nxt = _next_decay(decay)
    line = gates.sharpe_line(region)

    too_fast = turnover > min(gates.turnover_max, band[1])  # above what the claim predicts
    if too_fast or (
        sharpe >= line and fitness < gates.fitness_min and turnover >= FITNESS_TURNOVER_FLOOR
    ):
        if nxt is not None:
            what = "turnover too high" if too_fast else "fitness too low"
            return Fix(what, "setting", f"raise decay from {decay} to {nxt}", decay=nxt)
        return Fix(
            "turnover too high at the largest decay",
            "window",
            "use longer windows or smooth the input so the signal changes more slowly",
        )
    if turnover < gates.turnover_min:
        return Fix(
            "turnover too low",
            "window",
            "use shorter windows so the signal reacts within the holding period",
        )
    if sharpe >= line and fitness < gates.fitness_min:
        return Fix(
            "fitness too low at low turnover",
            "operator",
            "make the signal more decisive: a nonlinear or group-relative transform of the "
            "same input, so each position earns more",
        )
    # Sharpe below the line: windows first (cheapest), then operator form, then add
    # information.
    for change, instruction in (
        ("window", "move the windows toward the holding period of the claim"),
        ("operator", "change the operator form that turns the field into a signal"),
        ("field", "add or swap in a field that the claim says should strengthen the signal"),
    ):
        if change not in tried:
            return Fix("Sharpe below the line", change, instruction)
    return None


# Platform check name fragments -> the fix. Matched on the check's name, in order.
PLATFORM_FIXES: tuple[tuple[str, str, str], ...] = (
    (
        "PROD_CORRELATION",
        "rewrite",
        "the signal is crowded: keep the claim but combine the field with one or more "
        "fields from different data (a product or ratio of independent signals), and do not "
        "re-use any field listed as crowded",
    ),
    (
        "SELF_CORRELATION",
        "rewrite",
        "too close to an alpha already submitted: keep the claim but change the structure "
        "(different operator form, a group-relative version, or an added independent field)",
    ),
    (
        "WEIGHT",
        "operator",
        "weight is concentrated in few stocks: rank or group-normalize the signal and fill "
        "missing values so that every covered stock gets a position",
    ),
    (
        "SUB_UNIVERSE",
        "operator",
        "the signal fails in the largest stocks: normalize within size groups or use a "
        "slower version that large, liquid stocks can follow",
    ),
    (
        "LADDER",
        "rewrite",
        "recent years are weak: add a complementary field from a second dataset so the "
        "signal does not depend on one data source",
    ),
    (
        "2Y",
        "rewrite",
        "recent years are weak: add a complementary field from a second dataset so the "
        "signal does not depend on one data source",
    ),
)


def for_platform(failed_checks: list[str]) -> Fix:
    """The fix for an alpha that passed the train and test gates but failed a platform
    check."""
    for name in failed_checks:
        for fragment, change, instruction in PLATFORM_FIXES:
            if fragment in name.upper():
                return Fix(f"platform check {name}", change, instruction)
    names = ", ".join(failed_checks) or "unknown"
    return Fix(
        f"platform check {names}",
        "rewrite",
        "keep the claim and change the implementation so the failed check passes",
    )


def is_correlation(failed_checks: list[str]) -> bool:
    return any("CORRELATION" in n.upper() for n in failed_checks)
