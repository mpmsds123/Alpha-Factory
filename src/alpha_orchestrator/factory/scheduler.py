"""The simulator: consumes the ledger queue, never generates anything itself.

Producers write QUEUED candidates; this loop packs them into multi-simulations (same
region/delay per batch, a platform rule), paces batch creation, polls, and writes results
back. Batches in flight are rows in the ledger, so a crash or a face-check pause resumes
by re-polling them instead of simulating again.
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
from alpha_orchestrator.factory.ledger import Ledger, fingerprint
from alpha_orchestrator.factory.models import BatchORM, CandidateORM, CandidateStatus
from alpha_orchestrator.factory.quarantine import exception_scope
from alpha_orchestrator.factory.util import utc_now

QUOTA_RESERVE = 0  # Compatibility import; actual reserve is configured per factory.
REFUSED = -1
REJECTED = -2  # the platform said some formulas are invalid: not a capacity problem
MAX_REFUSALS = 10


@dataclass
class RunStats:
    batches: int = 0
    completed: int = 0
    failed: int = 0
    stopped_because: str = ""


def rejected_children(message: str, size: int) -> set[int]:
    """Positions the platform rejected in a 400 answer, e.g.
    `[{"settings":{"decay":["..."]}},{},...]`; empty when it is not that kind of answer."""
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


def metrics_from_alpha(alpha: dict[str, Any]) -> dict[str, Any]:
    """Extract Sharpe, fitness, turnover and checks from an alpha record."""
    is_ = alpha.get("is") or {}
    checks = list(is_.get("checks") or [])

    def num(key: str) -> float | None:
        """Return the named in-sample number, or None."""
        value = is_.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return {
        "sharpe": num("sharpe"),
        "fitness": num("fitness"),
        "turnover": num("turnover"),
        "returns": num("returns"),
        "margin": num("margin"),
        "drawdown": num("drawdown"),
        "long_count": int(is_["longCount"]) if isinstance(is_.get("longCount"), int) else None,
        "short_count": int(is_["shortCount"]) if isinstance(is_.get("shortCount"), int) else None,
        "checks": checks,
        "fails": ",".join(c.get("name", "") for c in checks if c.get("result") == "FAIL"),
    }


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
        self.slots = min(config.budget.slots, self._learned_limit() or config.budget.slots)

    def _learned_limit(self) -> int | None:
        """The concurrency the platform accepted last time it refused more (kept in the
        ledger, so every later run starts within the platform's rule)."""
        from alpha_orchestrator.factory.models import EventORM

        with self.ledger.session() as s:
            event = s.scalar(
                select(EventORM)
                .where(EventORM.kind == "concurrency_limit")
                .order_by(EventORM.id.desc())
                .limit(1)
            )
            return int(event.payload["batches"]) if event else None

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

    def _remaining_daily(self) -> int:
        with self.ledger.session() as s:
            return max(
                0,
                min(self.config.budget.v1_daily_cap, self.config.budget.daily_sim_cap)
                - self.ledger.sims_reserved_today(s, self.config.budget.window_start()),
            )

    def _run(self, sim_budget: int, deadline: float | None) -> RunStats:
        stats = RunStats()
        started = 0
        refusals = 0
        while True:
            self.assert_recoverable()
            if deadline is not None and self.clock() >= deadline:
                stats.stopped_because = "deadline"
                return stats  # In-flight URLs remain durable for the next worker.
            inflight = self._inflight()
            remaining_quota = self.api.client.quota.remaining
            quota_left = (
                max(0, remaining_quota - self.config.budget.quota_reserve)
                if remaining_quota is not None
                else sim_budget
            )
            capacity = min(sim_budget - started, self._remaining_daily(), quota_left)
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
                    # The platform's limit is what is running now: obey it from here on.
                    self.slots = len(inflight)
                    with self.ledger.session() as s:
                        self.ledger.log(s, "concurrency_limit", batches=self.slots)
                    continue
                if launched == REFUSED:
                    refusals += 1
                    if refusals >= MAX_REFUSALS:
                        stats.stopped_because = "platform kept refusing batches"
                        return stats
                    continue
            if not inflight:
                if not can_start:
                    stats.stopped_because = (
                        "budget"
                        if started >= sim_budget or self._remaining_daily() <= 0
                        else (
                            "quota"
                            if remaining_quota is not None and quota_left <= 0
                            else "deadline"
                        )
                    )
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
        budget_left = min(budget_left, self._remaining_daily())
        if budget_left <= 0:
            return 0
        with self.ledger.session() as s:
            _, excluded = exception_scope(s, self.config)
            groups = sorted(self.ledger.queue_groups(s, excluded), key=lambda g: -g[3])
            if not groups:
                return 0
            region, delay, _, _ = groups[0]
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
                c.status = CandidateStatus.RUNNING
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
            self.sleep(60)
            return REFUSED
        except (RateLimited, SessionExpired, AuthenticationFailed, NeedBiometric) as exc:
            # These arise after explicit HTTP refusals, not a lost create response.
            # Preserve the queued candidates for a normal auth/throttle retry.
            self._release(batch_id, type(exc).__name__)
            raise
        except Exception:
            with self.ledger.session() as s:
                unknown = s.get(BatchORM, batch_id)
                assert unknown is not None
                unknown.status = "UNKNOWN"
                self.ledger.log(s, "batch_unknown", batch=batch_id, size=len(payloads))
            raise
        self._last_batch_at = self.clock()
        with self.ledger.session() as s:
            launched = s.get(BatchORM, batch_id)
            assert launched is not None
            launched.location = location
            self.ledger.log(s, "batch_started", batch=batch_id, size=len(payloads))
        return len(payloads)

    def _reject(self, batch_id: str, ids: list[str], bad: set[int], reason: str) -> None:
        """Invalid formulas fail with the platform's message; the rest go back in line."""
        with self.ledger.session() as s:
            row = s.get(BatchORM, batch_id)
            assert row is not None
            row.status = "REJECTED"
            row.finished_at = utc_now()
            for i, cid in enumerate(ids):
                c = s.get(CandidateORM, cid)
                assert c is not None
                c.batch_id = None
                if i in bad:
                    c.status = CandidateStatus.FAILED
                    c.error = f"platform rejected the settings: {reason[:400]}"
                    c.completed_at = utc_now()
                else:
                    c.status = CandidateStatus.QUEUED
            self.ledger.log(
                s, "batch_rejected", batch=batch_id, invalid=sorted(bad), reason=reason[:300]
            )

    def _release(self, batch_id: str, reason: str) -> None:
        with self.ledger.session() as s:
            row = s.get(BatchORM, batch_id)
            assert row is not None
            row.status = "REFUSED"
            row.finished_at = utc_now()
            for c in s.scalars(select(CandidateORM).where(CandidateORM.batch_id == batch_id)):
                c.status = CandidateStatus.QUEUED
                c.batch_id = None
            self.ledger.log(s, "batch_refused", batch=batch_id, reason=reason[:300])

    def _quarantine(self, batch_id: str) -> None:
        with self.ledger.session() as s:
            row = s.get(BatchORM, batch_id)
            assert row is not None
            row.status = "UNKNOWN"
            self.ledger.log(s, "batch_unknown", batch=batch_id)

    def _close_lost(self, batch_id: str, member_ids: list[str]) -> int:
        with self.ledger.session() as s:
            for mid in member_ids:
                c = s.get(CandidateORM, mid)
                assert c is not None
                c.status = CandidateStatus.FAILED
                c.error = "result lost: progress URL expired (404)"
                c.completed_at = utc_now()
            row = s.get(BatchORM, batch_id)
            assert row is not None
            row.status, row.finished_at = "DONE", utc_now()
            self.ledger.log(s, "batch_closed_lost", batch=batch_id, candidates=len(member_ids))
        return len(member_ids)

    def _poll(self, batch_id: str) -> tuple[int, int]:
        with self.ledger.session() as s:
            row = s.get(BatchORM, batch_id)
            assert row is not None
            location = row.location
            members = list(
                s.scalars(
                    select(CandidateORM)
                    .where(CandidateORM.batch_id == batch_id)
                    .order_by(CandidateORM.priority.desc(), CandidateORM.created_at)
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
            # The platform keeps a progress address for only a few days. Gone means the
            # results cannot be fetched this way; close the batch instead of crashing.
            return 0, self._close_lost(batch_id, member_ids)
        if not progress.done:
            return 0, 0

        # Persist reconciliation evidence before a later child/identity failure can
        # quarantine the batch. Progress endpoints can become unavailable overnight.
        with self.ledger.session() as s:
            self.ledger.log(
                s,
                "batch_progress_observed",
                batch=batch_id,
                status=progress.status,
                children=progress.children,
                alpha_id=progress.alpha_id,
                message=progress.message,
            )

        results: dict[str, dict[str, Any]] = {}
        unmatched: list[str] = []

        def attach(alpha_id: str) -> None:
            """Match a platform alpha back to the candidate that produced it."""
            alpha = self.api.alpha(alpha_id)
            code = (alpha.get("regular") or {}).get("code")
            actual = alpha.get("settings") or {}
            matches = [
                m.id
                for m in members
                if isinstance(code, str)
                and fingerprint(code, {k: actual.get(k) for k in m.settings}) == m.fingerprint
            ]
            with self.ledger.session() as s:
                self.ledger.log(
                    s,
                    "batch_result_observed",
                    batch=batch_id,
                    alpha_id=alpha_id,
                    matched_candidates=matches,
                    duplicate_match=bool(matches and matches[0] in results),
                    # Explicitly retained private fields, never an HTTP response or
                    # authentication payload. Do not export these audit events.
                    alpha={k: alpha[k] for k in ("regular", "settings", "is") if k in alpha},
                )
            if len(matches) > 1 or (matches and matches[0] in results):
                self._quarantine(batch_id)
                raise AmbiguousRequest(f"Batch {batch_id}: result identity mismatch")
            if not matches:
                # The platform can answer one child with an older alpha of the same data
                # under another field alias. It is not ours, so it is never attached; the
                # candidate it was meant for stays without a result and is marked failed.
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
                with self.ledger.session() as s:
                    self.ledger.log(
                        s,
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
            if unmatched and not results:  # nothing of ours came back: do not guess
                self._quarantine(batch_id)
                raise AmbiguousRequest(f"Batch {batch_id}: result identity mismatch")
            # Failed children do not expose an expression. Preserve their messages at
            # batch scope rather than guessing their identities from response order.
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
                c = s.get(CandidateORM, mid)
                assert c is not None
                outcome = results.get(mid, {"error": "no result returned"})
                c.completed_at = utc_now()
                if "alpha" in outcome:
                    for key, value in metrics_from_alpha(outcome["alpha"]).items():
                        setattr(c, key, value)
                    c.alpha_id = outcome["alpha_id"]
                    c.status = CandidateStatus.DONE
                    done += 1
                else:
                    c.error = str(outcome["error"])[:500]
                    c.status = CandidateStatus.FAILED
                    failed += 1
            row = s.get(BatchORM, batch_id)
            assert row is not None
            row.status = "DONE"
            row.finished_at = utc_now()
            self.ledger.log(s, "batch_done", batch=batch_id, done=done, failed=failed)
        return done, failed
