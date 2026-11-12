"""The program's gates, in order: train -> out-of-sample -> platform checks -> review.

train      official submission numbers, applied to the training period only
oos        the 2-year test period, which nothing upstream has seen
platform   the platform's own checks on the whole IS (ladder, sub-universe, weight,
           correlation); empty, pending or failing evidence never passes
review     stored for you; `review approve` is the only path that submits
"""

import math
from typing import Any

from sqlalchemy import select

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.factory.config import FactoryConfig, Gates, Search
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import (
    HypothesisORM,
    Review,
    SimStatus,
    SimulationORM,
)

CORRELATION_CHECKS = ("SELF_CORRELATION", "PROD_CORRELATION")


def suspicious(sim: SimulationORM, search: Search) -> bool:
    """Too good or too fast to trust: checked, never mined further."""
    return (
        abs(sim.train_sharpe or 0.0) > search.suspicious_sharpe
        or (sim.train_turnover or 0.0) > search.suspicious_turnover
    )


def train_pass(sim: SimulationORM, region: str, gates: Gates) -> bool:
    """The official submission numbers on the training period."""
    sharpe, fitness, turnover = sim.train_sharpe, sim.train_fitness, sim.train_turnover
    if sharpe is None or fitness is None or turnover is None:
        return False
    ok = (
        sharpe >= gates.sharpe_line(region)
        and fitness >= gates.fitness_min
        and gates.turnover_min <= turnover <= gates.turnover_max
    )
    if region == "CHN":
        ok = ok and (sim.train_returns or 0.0) >= gates.returns_min_chn
    return ok


def oos_pass(sim: SimulationORM, search: Search) -> bool:
    """The test period must keep the train sign and at least the OOS line."""
    if sim.test_sharpe is None or not math.isfinite(sim.test_sharpe):
        return False
    sign = 1.0 if (sim.train_sharpe or 0.0) >= 0 else -1.0
    return sign * sim.test_sharpe >= search.oos_min_sharpe


def _value(checks: list[dict[str, Any]], name: str) -> float | None:
    for c in checks:
        if c.get("name") == name and isinstance(c.get("value"), (int, float)):
            return float(c["value"])
    return None


def platform_pass(checks: list[dict[str, Any]], corr_max: float) -> bool:
    """Every check PASS and both correlations known and under the line."""
    if not checks or any(c.get("result") != "PASS" for c in checks):
        return False
    for name in CORRELATION_CHECKS:
        v = _value(checks, name)
        if v is None or not math.isfinite(v) or v > corr_max:
            return False
    return True


def quota_only(checks: list[dict[str, Any]]) -> bool:
    """The only failure is the exhausted daily submission allowance: try again later."""
    failed = [c for c in checks if c.get("result") == "FAIL"]
    return len(failed) == 1 and failed[0].get("name") == "REGULAR_SUBMISSION"


def mark_train(ledger: Ledger, config: FactoryConfig) -> int:
    """Set train_pass on finished simulations; return how many newly passed."""
    n = 0
    with ledger.session() as s:
        rows = s.scalars(
            select(SimulationORM).where(
                SimulationORM.status == SimStatus.DONE,
                SimulationORM.oos_pass.is_(None),
                SimulationORM.train_pass.is_(False),
                SimulationORM.gate_note == "",
            )
        )
        for sim in rows:
            if suspicious(sim, config.search):
                sim.gate_note = "suspicious"
            elif train_pass(sim, sim.region, config.gates):
                sim.train_pass = True
                n += 1
            else:
                sim.gate_note = "train"
    return n


def evaluate(api: BrainAPI, ledger: Ledger, config: FactoryConfig) -> list[str]:
    """Run every train-passing simulation through the OOS and platform gates; store the
    survivors for review. Return the alpha ids stored this time."""
    mark_train(ledger, config)
    with ledger.session() as s:
        todo = [
            (sim.id, sim.alpha_id)
            for sim in s.scalars(
                select(SimulationORM).where(
                    SimulationORM.train_pass.is_(True),
                    SimulationORM.review == Review.NONE,
                    SimulationORM.checks_pass.is_(None),
                )
            )
            if sim.alpha_id
        ]
    stored: list[str] = []
    for sid, alpha_id in todo:
        with ledger.session() as s:
            sim = s.get(SimulationORM, sid)
            assert sim is not None
            if sim.oos_pass is None:
                sim.oos_pass = oos_pass(sim, config.search)
            if not sim.oos_pass:
                sim.gate_note = "oos"
                sim.checks_pass = False
                continue
        checks = api.check(alpha_id)
        with ledger.session() as s:
            sim = s.get(SimulationORM, sid)
            assert sim is not None
            sim.submission_check = checks
            if quota_only(checks):
                ledger.log("platform_check_deferred", alpha=alpha_id, reason="quota")
                continue
            if any(c.get("result") == "PENDING" for c in checks) and not any(
                c.get("result") == "FAIL" for c in checks
            ):
                ledger.log("platform_check_deferred", alpha=alpha_id, reason="pending")
                continue
            sim.checks_pass = platform_pass(checks, config.gates.corr_max)
            if not sim.checks_pass:
                sim.gate_note = "platform: " + ",".join(
                    str(c.get("name")) for c in checks if c.get("result") != "PASS"
                )
                continue
            sim.review = Review.PENDING
            sim.gate_note = "stored"
        ledger.log("stored_for_review", alpha=alpha_id, simulation=sid)
        stored.append(alpha_id)
    return stored


def approve(api: BrainAPI, ledger: Ledger, sim_id: int, config: FactoryConfig) -> bool:
    """The only code path that submits. Called by `alpha-factory review approve`."""
    with ledger.session() as s:
        sim = s.get(SimulationORM, sim_id)
        if sim is None or sim.review != Review.PENDING or not sim.alpha_id:
            return False
        alpha_id = sim.alpha_id
        h = s.get(HypothesisORM, sim.hypothesis_id)
        description = h.claim if h else ""
    checks = api.check(alpha_id)
    if not platform_pass(checks, config.gates.corr_max):
        ledger.log("submission_blocked", alpha=alpha_id, checks=checks)
        return False
    if description:
        api.set_properties(alpha_id, description=description[:2000], tags=["alpha-factory"])
    ok = api.submit(alpha_id)
    with ledger.session() as s:
        sim = s.get(SimulationORM, sim_id)
        assert sim is not None
        sim.review = Review.SUBMITTED if ok else Review.FAILED
    ledger.log("submitted" if ok else "submit_failed", alpha=alpha_id)
    return ok


def reject(ledger: Ledger, sim_id: int, note: str = "") -> bool:
    """Mark a pending alpha as rejected by you."""
    with ledger.session() as s:
        sim = s.get(SimulationORM, sim_id)
        if sim is None or sim.review != Review.PENDING:
            return False
        sim.review, sim.review_note = Review.REJECTED, note
        return True
