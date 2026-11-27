"""The pool route: how the platform's checks judge an alpha made for the pool slot.

A pool alpha (few operators, few fields) can be submitted without passing the regular
tests, as long as the pool's own tests pass. Ported from the template scanner, which has
used these rules on several hundred passes:

* The pool correlation must PASS, and no test outside the regular-only set may FAIL.
* If a regular-only test fails or is still computing, the alpha is a *pure* pool alpha: it
  counts against the account's daily pool quota and must match a current pool theme
  (theme check PASS; WARNING means off-theme, seen 2026-11-21).
* Otherwise it is pool and regular at once, outside the quota.

The platform runs the pool tests only once the alpha has a description in its
Idea / Rationale template of at least 100 characters; `description` writes one from the
hypothesis.
"""

from typing import Any

POOL_CORRELATION = "POWER_POOL_CORRELATION"
THEMES = "MATCHES_THEMES"
# Tests that only decide the regular route; failing them makes a pure pool alpha.
REGULAR_ONLY = frozenset(
    {
        "LOW_SHARPE",
        "LOW_FITNESS",
        "PROD_CORRELATION",
        "SELF_CORRELATION",
        "IS_LADDER_SHARPE",
        "LOW_2Y_SHARPE",
        "LOW_RISK_NEUTRALIZED_SHARPE",
        "LOW_INVESTABILITY_CONSTRAINED_SHARPE",
        "CLUSTER_TEST",
        "OSMOSIS_ALLOCATION",
        "MATCHES_PYRAMID",
        "REGULAR_SUBMISSION",
        "DATA_DIVERSITY",
        "MATCHES_THEMES",
        "MATCHES_COMPETITION",
        POOL_CORRELATION,
        "LOW_GLB_EMEA_SHARPE",
        "LOW_GLB_APAC_SHARPE",
    }
)
# Listed with the checks but not a test of the alpha itself.
NOT_A_TEST = frozenset(
    {"MATCHES_PYRAMID", THEMES, "MATCHES_COMPETITION", "OSMOSIS_ALLOCATION", POOL_CORRELATION}
)
MIN_DESCRIPTION = 100


def _pool_themes(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        c
        for c in checks
        if c.get("name") == THEMES
        and any("power pool" in str(t.get("name", "")).lower() for t in c.get("themes") or [])
    ]


def verdict(checks: list[dict[str, Any]]) -> tuple[bool | None, str]:
    """(True, kind) when the pool route accepts the alpha, (False, why) when it does not,
    (None, why) while the platform is still computing what decides it."""
    pending = any(c.get("result") == "PENDING" for c in checks)
    pool = [c for c in checks if c.get("name") == POOL_CORRELATION]
    if not pool:
        return (None, "pool correlation not computed yet") if pending else (False, "no pool test")
    if pool[0].get("result") == "PENDING":
        return None, "pool correlation not computed yet"
    hard = [
        str(c.get("name"))
        for c in checks
        if c.get("result") == "FAIL" and c.get("name") not in REGULAR_ONLY
    ]
    tests = REGULAR_ONLY - NOT_A_TEST
    regular_failed = [c for c in checks if c.get("result") == "FAIL" and c.get("name") in tests]
    regular_waiting = [c for c in checks if c.get("result") == "PENDING" and c.get("name") in tests]
    reported = any(c.get("name") == "LOW_SHARPE" for c in checks)
    pure = bool(regular_failed or regular_waiting or not reported)
    if pure:
        themes = _pool_themes(checks)
        if not themes:
            if pending:
                return None, "theme match not computed yet"
            return False, "a pure pool alpha must match a current pool theme"
        if any(c.get("result") == "PENDING" for c in themes):
            return None, "theme match not computed yet"
        if not any(c.get("result") == "PASS" for c in themes):
            hard.append("off theme")
    if pool[0].get("result") != "PASS":
        hard.append(f"{POOL_CORRELATION}={pool[0].get('result')}")
    if hard:
        return False, ", ".join(hard)
    return True, "pure pool (daily pool quota)" if pure else "pool and regular"


def description(claim: str, field_description: str, falsifier: str, holding_days: int) -> str:
    """The Idea / Rationale text the platform asks for, built from the hypothesis."""
    data = field_description.strip().rstrip(".") or "the field named in the hypothesis"
    text = "\n".join(
        [
            f"Idea: {claim.strip()}",
            f"Rationale for data used: {data}. The hypothesis expects the effect to play out "
            f"over about {holding_days} trading days.",
            "Rationale for operators used: the operators fill gaps, normalise the field over "
            "time and rank it across stocks, so each stock is compared with its peers rather "
            "than with its own scale. The idea would be wrong if: "
            f"{falsifier.strip().rstrip('.') or 'the backtest showed no relation'}.",
        ]
    )
    return text  # the fixed sentences alone exceed MIN_DESCRIPTION
