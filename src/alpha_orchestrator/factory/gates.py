"""The program's gates, in order: train -> out-of-sample -> platform checks -> review.

train      official submission numbers, applied to the training period only
oos        the 2-year test period, which nothing upstream has seen
platform   the platform's own checks on the whole IS (ladder, sub-universe, weight,
           correlation); empty, pending or failing evidence never passes
review     stored for you; `review approve` submits it, or `auto_submit` when the
           owner has set submit_mode to "auto"

Before the platform's checks: the local self-correlation check (guards.py). A platform or
local correlation failure is written on the hypothesis so the loop can retry it once with
the fix table's instruction, and a production-correlation failure puts the expression's
fields on the crowded list.
"""

import math
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.factory import guards, notes
from alpha_orchestrator.factory.config import FactoryConfig, Gates, Search
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import (
    ExpressionORM,
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


def _failed_names(checks: list[dict[str, Any]], corr_max: float) -> list[str]:
    out = [str(c.get("name")) for c in checks if c.get("result") != "PASS"]
    for name in CORRELATION_CHECKS:
        v = _value(checks, name)
        if v is not None and v > corr_max and name not in out:
            out.append(name)
    return out


def _record_failure(
    s: Session,
    config: FactoryConfig,
    ledger: Ledger,
    sim: SimulationORM,
    failed: list[str],
    detail: str,
) -> None:
    """Write the failure on the hypothesis (for the loop's one retry); crowd the fields of
    an expression that failed the production correlation."""
    h = s.get(HypothesisORM, sim.hypothesis_id)
    if h is None:
        return
    if not (h.state or {}).get("platform_failure"):
        h.state = {
            **(h.state or {}),
            "platform_failure": {
                "simulation": sim.id,
                "expression": sim.expression,
                "checks": failed,
                "detail": detail[:300],
            },
        }
    if any("PROD_CORRELATION" in n for n in failed):
        e = s.get(ExpressionORM, sim.expression_id)
        fields = list(e.fields) if e is not None else []
        new = guards.add_crowded(config, fields, f"production correlation, alpha {sim.alpha_id}")
        if new:
            ledger.log("crowded_added", fields=new, alpha=sim.alpha_id)
            notes.add_lesson(
                config.memory_dir,
                "crowded",
                ",".join(sorted(new)),
                f"Fields {sorted(new)} failed the production correlation ({detail}); they "
                "are on the crowded list and are not drawn or offered again.",
            )


def _first_pass_lesson(s: Session, config: FactoryConfig, sim: SimulationORM) -> None:
    h = s.get(HypothesisORM, sim.hypothesis_id)
    if h is None:
        return
    slot = str((h.state or {}).get("slot") or "pool")
    notes.add_lesson(
        config.memory_dir,
        "first_pass",
        f"{slot}/{h.region}",
        f"First alpha stored for slot {slot} in region {h.region}: hypothesis H{h.id} "
        f"({h.hypothesis_type}, {h.category}), train Sharpe {sim.train_sharpe}.",
    )


def evaluate(
    api: BrainAPI,
    ledger: Ledger,
    config: FactoryConfig,
    selfcorr: guards.SelfCorrelation | None = None,
) -> list[str]:
    """Run every train-passing simulation through the OOS, local and platform gates;
    store the survivors for review. Return the alpha ids stored this time."""
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
        if selfcorr is not None:
            try:
                value, other = selfcorr.max_with_submitted(alpha_id)
            except Exception as exc:  # the platform's own check still runs
                ledger.log("local_self_corr_skipped", alpha=alpha_id, error=str(exc)[:200])
                value, other = None, ""
            if value is not None and value > config.checks.local_self_corr_max:
                with ledger.session() as s:
                    sim = s.get(SimulationORM, sid)
                    assert sim is not None
                    sim.checks_pass = False
                    sim.gate_note = f"local self-correlation {value:.2f} with {other}"
                    _record_failure(s, config, ledger, sim, ["SELF_CORRELATION"], sim.gate_note)
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
                failed = _failed_names(checks, config.gates.corr_max)
                sim.gate_note = "platform: " + ",".join(failed)
                values = {n: _value(checks, n) for n in CORRELATION_CHECKS}
                _record_failure(s, config, ledger, sim, failed, f"{failed} {values}")
                continue
            sim.review = Review.PENDING
            sim.gate_note = "stored"
            _first_pass_lesson(s, config, sim)
        ledger.log("stored_for_review", alpha=alpha_id, simulation=sid)
        stored.append(alpha_id)
    return stored


def approve(api: BrainAPI, ledger: Ledger, sim_id: int, config: FactoryConfig) -> bool:
    """The only code path that submits: `alpha-factory review approve`, or auto_submit."""
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


PAUSE_FILE = "PAUSE_SUBMIT"


def submit_mode(config: FactoryConfig) -> str:
    """'auto' only when configured and not paused by the PAUSE_SUBMIT file."""
    if config.policy.submit_mode != "auto":
        return "approve"
    if (Path(config.memory_dir).expanduser() / PAUSE_FILE).exists():
        return "approve"
    return "auto"


def submitted_today(ledger: Ledger, config: FactoryConfig, now: datetime | None = None) -> int:
    start = config.budget.window_start(now)
    return sum(
        1 for e in ledger.events("submitted") if datetime.fromisoformat(str(e["at"])) >= start
    )


def auto_submit(api: BrainAPI, ledger: Ledger, config: FactoryConfig) -> list[str]:
    """In auto mode, submit stored alphas, oldest first, up to the daily allowance."""
    if submit_mode(config) != "auto":
        return []
    room = config.policy.max_submissions_per_day - submitted_today(ledger, config)
    if room <= 0:
        return []
    with ledger.session() as s:
        pending = [
            (x.id, x.alpha_id)
            for x in s.scalars(
                select(SimulationORM)
                .where(SimulationORM.review == Review.PENDING)
                .order_by(SimulationORM.id)
            )
        ]
    blocked = {e.get("alpha") for e in ledger.events("submission_blocked")}
    pending = [(sid, a) for sid, a in pending if a not in blocked]
    done: list[str] = []
    for sid, alpha_id in pending[:room]:
        if approve(api, ledger, sid, config):
            done.append(str(alpha_id))
    return done


def reject(ledger: Ledger, sim_id: int, note: str = "") -> bool:
    """Mark a pending alpha as rejected by you."""
    with ledger.session() as s:
        sim = s.get(SimulationORM, sim_id)
        if sim is None or sim.review != Review.PENDING:
            return False
        sim.review, sim.review_note = Review.REJECTED, note
        return True
