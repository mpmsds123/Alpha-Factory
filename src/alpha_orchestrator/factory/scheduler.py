"""The simulator: consumes the simulation queue, never generates anything itself.

The research loop writes QUEUED simulations; this loop packs them into multi-simulations
(same region/delay per batch, a platform rule), paces batch creation, polls, and writes
results back. Batches in flight are rows in the database, so a crash or a face-check pause
resumes by re-polling them instead of simulating again.
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from sqlalchemy import select

from alpha_orchestrator.brain.api import BrainAPI, simulation_payload
from alpha_orchestrator.brain.client import AmbiguousRequest, RateLimited, SessionExpired
from alpha_orchestrator.brain.session import AuthenticationFailed, NeedBiometric
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import BatchORM, SimStatus, SimulationORM
from alpha_orchestrator.factory.quarantine import exception_scope
from alpha_orchestrator.factory.util import utc_now

REFUSED = -1
REJECTED = -2  # the platform said some expressions are invalid: not a capacity problem
MAX_REFUSALS = 10
REFUSAL_PAUSE_S = 60.0
# Settings the platform does not echo back in the alpha record, so they cannot be used to
# match a result to the simulation that asked for it.
NOT_ECHOED = frozenset({"testPeriod"})
PERIOD_KEYS = ("sharpe", "fitness", "turnover", "returns", "margin", "drawdown")


@dataclass
class RunStats:
    batches: int = 0
    completed: int = 0
    failed: int = 0
    stopped_because: str = ""


def rejected_children(message: str, size: int) -> set[int]:
    """Positions the platform rejected in a 400 answer (a list with one entry per child,
    non-empty where that child is invalid); empty when it is not that kind of answer."""
    match = re.search(r"\(400\): (.*)$", message, re.S)
    if not match:
        return set()
    try:
        body = json.loads(match.group(1))
    except ValueError:
        return set()
    if isinstance(body, dict):
        return {0} if body and size == 1 else set()
    if isinstance(body, list):
        return {i for i, item in enumerate(body) if item and i < size}
    return set()


def _num(block: dict[str, Any], key: str) -> float | None:
    value = block.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def metrics_from_alpha(alpha: dict[str, Any]) -> dict[str, Any]:
    """Train, test and whole-IS numbers and the IS checks of an alpha record. Without a
    test period the platform returns no train block, and train equals the whole IS."""
    is_ = alpha.get("is") or {}
    train = alpha.get("train") or is_
    test = alpha.get("test") or {}
    checks = list(is_.get("checks") or [])
    out: dict[str, Any] = {f"train_{k}": _num(train, k) for k in PERIOD_KEYS}
    out.update({f"test_{k}": _num(test, k) for k in ("sharpe", "fitness", "returns")})
    out.update({f"is_{k}": _num(is_, k) for k in ("sharpe", "fitness", "turnover", "returns")})
    out["checks"] = checks
    out["fails"] = ",".join(c.get("name", "") for c in checks if c.get("result") == "FAIL")
    return out


def same_request(code: str, actual: dict[str, Any], sim: SimulationORM) -> bool:
    """Does an alpha record answer this simulation (same text and echoed settings)?"""
    if re.sub(r"\s+", "", code) != re.sub(r"\s+", "", sim.expression):
        return False
    return all(actual.get(k) == v for k, v in sim.settings.items() if k not in NOT_ECHOED)


class Simulator:
    def __init__(
        self,
        api: BrainAPI,
        ledger: Ledger,
        config: FactoryConfig,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        poll_interval: float = 5.0,
    ) -> None:
        self.api = api
        self.ledger = ledger
        self.config = config
        self.sleep = sleep
        self.clock = clock
        self.poll_interval = poll_interval
        self._last_batch_at = -1e9
        learned = [int(e["batches"]) for e in ledger.events("concurrency_limit")]
        self.slots = min(config.budget.slots, learned[-1] if learned else config.budget.slots)

    # ---------- public ----------
    def run(self, sim_budget: int, deadline: float | None = None) -> RunStats:
        """Simulate until the queue is empty, the budget is spent, or the deadline passes."""
        with self.ledger.writer_lock():
            self.assert_recoverable()
            return self._run(sim_budget, deadline)

    def assert_recoverable(self) -> None:
        """Refuse to run while a batch is in an uncertain state."""
        with self.ledger.session() as s:
            uncertain = list(
                s.scalars(
                    select(BatchORM).where(
                        (BatchORM.status == "UNKNOWN")
                        | ((BatchORM.status == "RUNNING") & (BatchORM.location == ""))
                    )
                )
            )
            for batch in uncertain:
                batch.status = "UNKNOWN"
            permitted, _ = exception_scope(s, self.config)
            ids = [b.id for b in uncertain if b.id not in permitted]
        if ids:
            raise AmbiguousRequest(
                "Reconcile uncertain simulation batches before running: " + ", ".join(ids)
            )

    def remaining_daily(self) -> int:
        """Simulations left in this quota window's allocation."""
        with self.ledger.session() as s:
            used = self.ledger.sims_reserved(s, self.config.budget.window_start())
        return max(0, self.config.budget.daily_sim_cap - used)

    def _run(self, sim_budget: int, deadline: float | None) -> RunStats:
        stats = RunStats()
        started = 0
        refusals = 0
        while True:
            self.assert_recoverable()
            if deadline is not None and self.clock() >= deadline:
                stats.stopped_because = "deadline"
                return stats  # in-flight URLs remain durable for the next worker
            inflight = self._inflight()
            remaining_quota = self.api.client.quota.remaining
            quota_left = (
                max(0, remaining_quota - self.config.budget.quota_reserve)
                if remaining_quota is not None
                else sim_budget
            )
            capacity = min(sim_budget - started, self.remaining_daily(), quota_left)
            can_start = capacity > 0 and (deadline is None or self.clock() < deadline)
            if can_start and len(inflight) < self.slots:
                launched = self._launch_one(capacity, deadline)
                if launched > 0:
                    started += launched
                    stats.batches += 1
                    refusals = 0
                    continue
                if launched == REJECTED:
                    continue
                if launched == REFUSED and inflight:
                    # the platform's limit is what is running now: obey it from here on
                    self.slots = len(inflight)
                    self.ledger.log("concurrency_limit", batches=self.slots)
                    continue
                if launched == REFUSED:
                    refusals += 1
                    if refusals >= MAX_REFUSALS:
                        stats.stopped_because = "platform kept refusing batches"
                        return stats
                    continue
            if not inflight:
                if not can_start:
                    if started >= sim_budget or self.remaining_daily() <= 0:
                        stats.stopped_because = "budget"
                    elif remaining_quota is not None and quota_left <= 0:
                        stats.stopped_because = "quota"
                    else:
                        stats.stopped_because = "deadline"
                else:
                    stats.stopped_because = "queue empty"
                return stats
            for batch_id in inflight:
                if deadline is not None and self.clock() >= deadline:
                    stats.stopped_because = "deadline"
                    return stats
                done, failed = self._poll(batch_id)
                stats.completed += done
                stats.failed += failed
            self.sleep(self.poll_interval)

    # ---------- internals ----------
    def _inflight(self) -> list[str]:
        with self.ledger.session() as s:
            return list(s.scalars(select(BatchORM.id).where(BatchORM.status == "RUNNING")))

    def _launch_one(self, budget_left: int, deadline: float | None = None) -> int:
        wait = self.config.budget.min_batch_interval_s - (self.clock() - self._last_batch_at)
        if wait > 0:
            self.sleep(min(wait, max(0, deadline - self.clock())) if deadline else wait)
        if deadline is not None and self.clock() >= deadline:
            return 0
        self.assert_recoverable()
        budget_left = min(budget_left, self.remaining_daily())
        if budget_left <= 0:
            return 0
        with self.ledger.session() as s:
            _, excluded = exception_scope(s, self.config)
            groups = self.ledger.queue_groups(s, excluded)
            if not groups:
                return 0
            region, delay = groups[0]
            size = min(self.config.budget.children_per_slot, budget_left)
            batch = self.ledger.queued(s, region, delay, size, excluded)
            if not batch:
                return 0
            payloads = [simulation_payload(c.expression, c.settings) for c in batch]
            member_ids = [c.id for c in batch]
            row = BatchORM(region=region, delay=delay, size=len(batch))
            s.add(row)
            s.flush()
            for c in batch:
                c.status = SimStatus.RUNNING
                c.batch_id = row.id
            batch_id = row.id
        try:
            location = self.api.start_multi_simulation(payloads)
        except ValueError as exc:
            bad = rejected_children(str(exc), len(payloads))
            if bad:
                self._reject(batch_id, member_ids, bad, str(exc))
                return REJECTED
            self._release(batch_id, str(exc))
            self.sleep(REFUSAL_PAUSE_S)
            return REFUSED
        except (RateLimited, SessionExpired, AuthenticationFailed, NeedBiometric) as exc:
            # explicit refusals, not a lost create response: keep the queue for a retry
            self._release(batch_id, type(exc).__name__)
            raise
        except Exception:
            with self.ledger.session() as s:
                unknown = s.get(BatchORM, batch_id)
                assert unknown is not None
                unknown.status = "UNKNOWN"
            self.ledger.log("batch_unknown", batch=batch_id, size=len(payloads))
            raise
        self._last_batch_at = self.clock()
        with self.ledger.session() as s:
            launched = s.get(BatchORM, batch_id)
            assert launched is not None
            launched.location = location
        self.ledger.log("batch_started", batch=batch_id, size=len(payloads))
        return len(payloads)

    def _reject(self, batch_id: str, ids: list[int], bad: set[int], reason: str) -> None:
        """Invalid expressions fail with the platform's message; the rest go back in line."""
        with self.ledger.session() as s:
            row = s.get(BatchORM, batch_id)
            assert row is not None
            row.status = "REJECTED"
            row.finished_at = utc_now()
            for i, sid in enumerate(ids):
                c = s.get(SimulationORM, sid)
                assert c is not None
                c.batch_id = None
                if i in bad:
                    c.status = SimStatus.FAILED
                    c.error = f"platform rejected it: {reason[:400]}"
                    c.completed_at = utc_now()
                else:
                    c.status = SimStatus.QUEUED
        self.ledger.log("batch_rejected", batch=batch_id, invalid=sorted(bad), reason=reason[:300])

    def _release(self, batch_id: str, reason: str) -> None:
        with self.ledger.session() as s:
            row = s.get(BatchORM, batch_id)
            assert row is not None
            row.status = "REFUSED"
            row.finished_at = utc_now()
            for c in s.scalars(select(SimulationORM).where(SimulationORM.batch_id == batch_id)):
                c.status = SimStatus.QUEUED
                c.batch_id = None
        self.ledger.log("batch_refused", batch=batch_id, reason=reason[:300])

    def _quarantine(self, batch_id: str) -> None:
        with self.ledger.session() as s:
            row = s.get(BatchORM, batch_id)
            assert row is not None
            row.status = "UNKNOWN"
        self.ledger.log("batch_unknown", batch=batch_id)

    def _close_lost(self, batch_id: str, member_ids: list[int]) -> int:
        with self.ledger.session() as s:
            for mid in member_ids:
                c = s.get(SimulationORM, mid)
                assert c is not None
                c.status = SimStatus.FAILED
                c.error = "result lost: progress URL expired (404)"
                c.completed_at = utc_now()
            row = s.get(BatchORM, batch_id)
            assert row is not None
            row.status, row.finished_at = "DONE", utc_now()
        self.ledger.log("batch_closed_lost", batch=batch_id, simulations=len(member_ids))
        return len(member_ids)

    def _poll(self, batch_id: str) -> tuple[int, int]:
        with self.ledger.session() as s:
            row = s.get(BatchORM, batch_id)
            assert row is not None
            location = row.location
            members = list(
                s.scalars(
                    select(SimulationORM)
                    .where(SimulationORM.batch_id == batch_id)
                    .order_by(SimulationORM.priority.desc(), SimulationORM.id)
                )
            )
            member_ids = [m.id for m in members]
        if not location:
            raise AmbiguousRequest(f"Batch {batch_id} has no confirmed progress URL")
        try:
            progress = self.api.poll(location)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            # the platform keeps a progress address for only a few days
            return 0, self._close_lost(batch_id, member_ids)
        if not progress.done:
            return 0, 0
        self.ledger.log(
            "batch_progress_observed",
            batch=batch_id,
            status=progress.status,
            children=progress.children,
            alpha_id=progress.alpha_id,
            message=progress.message,
        )
        results: dict[int, dict[str, Any]] = {}
        unmatched: list[str] = []

        def attach(alpha_id: str) -> None:
            """Match a platform alpha back to the simulation that produced it."""
            alpha = self.api.alpha(alpha_id)
            code = (alpha.get("regular") or {}).get("code")
            actual = alpha.get("settings") or {}
            matches = [
                m.id for m in members if isinstance(code, str) and same_request(code, actual, m)
            ]
            self.ledger.log(
                "batch_result_observed",
                batch=batch_id,
                alpha_id=alpha_id,
                matched=matches,
            )
            if len(matches) > 1 or (matches and matches[0] in results):
                self._quarantine(batch_id)
                raise AmbiguousRequest(f"Batch {batch_id}: result identity mismatch")
            if not matches:
                # an older alpha of the same data under another alias: never attached
                unmatched.append(alpha_id)
                return
            results[matches[0]] = {"alpha": alpha, "alpha_id": alpha_id}

        if progress.children:
            if len(progress.children) != len(member_ids):
                self._quarantine(batch_id)
                raise AmbiguousRequest(f"Batch {batch_id}: incomplete child list")
            errors = []
            for child_id in progress.children:
                child = self.api.child(child_id)
                self.ledger.log(
                    "batch_child_observed",
                    batch=batch_id,
                    child_id=child_id,
                    status=child.status,
                    alpha_id=child.alpha_id,
                    message=child.message,
                )
                if child.alpha_id:
                    attach(child.alpha_id)
                else:
                    errors.append(child.message or child.status)
            if unmatched and not results:
                self._quarantine(batch_id)
                raise AmbiguousRequest(f"Batch {batch_id}: result identity mismatch")
            reason = " | ".join(sorted(set(errors)))
            for mid in member_ids:
                if mid not in results:
                    results[mid] = {"error": reason or "child result unavailable"}
        elif progress.alpha_id and len(member_ids) == 1:
            attach(progress.alpha_id)
        elif progress.status in {"ERROR", "CANCELLED", "FAIL", "FAILED"}:
            for mid in member_ids:
                results[mid] = {"error": progress.message or progress.status}
        else:
            self._quarantine(batch_id)
            raise AmbiguousRequest(f"Batch {batch_id}: terminal response lacks results")

        done = failed = 0
        with self.ledger.session() as s:
            for mid in member_ids:
                c = s.get(SimulationORM, mid)
                assert c is not None
                outcome = results.get(mid, {"error": "no result returned"})
                c.completed_at = utc_now()
                if "alpha" in outcome:
                    for key, value in metrics_from_alpha(outcome["alpha"]).items():
                        setattr(c, key, value)
                    c.alpha_id = outcome["alpha_id"]
                    c.status = SimStatus.DONE
                    done += 1
                else:
                    c.error = str(outcome["error"])[:500]
                    c.status = SimStatus.FAILED
                    failed += 1
            row = s.get(BatchORM, batch_id)
            assert row is not None
            row.status = "DONE"
            row.finished_at = utc_now()
        self.ledger.log("batch_done", batch=batch_id, done=done, failed=failed)
        return done, failed
