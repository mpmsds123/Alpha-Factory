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

from sqlalchemy import select

from alpha_orchestrator.brain.api import BrainAPI, simulation_payload
from alpha_orchestrator.domain.models import utc_now
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger, fingerprint
from alpha_orchestrator.factory.models import BatchORM, CandidateORM, CandidateStatus

QUOTA_RESERVE = 300  # stop before the platform's own counter runs dry
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
    is_ = alpha.get("is") or {}
    checks = list(is_.get("checks") or [])

    def num(key: str) -> float | None:
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
        stats = RunStats()
        started = 0
        refusals = 0
        while True:
            inflight = self._inflight()
            remaining_quota = self.api.client.quota.remaining
            can_start = (
                started < sim_budget
                and (remaining_quota is None or remaining_quota > QUOTA_RESERVE)
                and (deadline is None or self.clock() < deadline)
            )
            if can_start and len(inflight) < self.slots:
                launched = self._launch_one(sim_budget - started)
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
                        if started >= sim_budget
                        else (
                            "quota"
                            if remaining_quota is not None and remaining_quota <= QUOTA_RESERVE
                            else "deadline"
                        )
                    )
                else:
                    stats.stopped_because = "queue empty"
                return stats
            for batch_id in inflight:
                done, failed = self._poll(batch_id)
                stats.completed += done
                stats.failed += failed
            self.sleep(self.poll_interval)

    # ---------- internals ----------
    def _inflight(self) -> list[str]:
        with self.ledger.session() as s:
            return list(s.scalars(select(BatchORM.id).where(BatchORM.status == "RUNNING")))

    def _launch_one(self, budget_left: int) -> int:
        wait = self.config.budget.min_batch_interval_s - (self.clock() - self._last_batch_at)
        if wait > 0:
            self.sleep(wait)
        with self.ledger.session() as s:
            groups = sorted(self.ledger.queue_groups(s), key=lambda g: -g[3])
            if not groups:
                return 0
            region, delay, _, _ = groups[0]
            size = min(self.config.budget.children_per_slot, budget_left)
            batch = self.ledger.queued(s, region, delay, size)
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
            by_fp = {m.fingerprint: m.id for m in members}
        if not location:
            self._release(batch_id, "no location recorded")
            return 0, 0
        progress = self.api.poll(location)
        if not progress.done:
            return 0, 0

        results: dict[str, dict[str, Any]] = {}
        if progress.children:
            for index, child_id in enumerate(progress.children):
                child = self.api.child(child_id)
                cid = None
                if child.alpha_id:
                    alpha = self.api.alpha(child.alpha_id)
                    code = (alpha.get("regular") or {}).get("code")
                    settings = {
                        k: alpha.get("settings", {}).get(k)
                        for k in (
                            "region",
                            "universe",
                            "delay",
                            "decay",
                            "neutralization",
                            "truncation",
                            "maxTrade",
                        )
                    }
                    cid = by_fp.get(fingerprint(code, settings)) if code else None
                    results[cid or member_ids[min(index, len(member_ids) - 1)]] = {
                        "alpha": alpha,
                        "alpha_id": child.alpha_id,
                    }
                else:
                    results[member_ids[min(index, len(member_ids) - 1)]] = {
                        "error": child.message or child.status
                    }
        elif progress.alpha_id and member_ids:
            results[member_ids[0]] = {
                "alpha": self.api.alpha(progress.alpha_id),
                "alpha_id": progress.alpha_id,
            }
        else:
            for mid in member_ids:
                results[mid] = {"error": progress.message or progress.status}

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
