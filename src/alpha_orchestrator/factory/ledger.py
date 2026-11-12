"""Database access: sessions, the single-writer lock, de-duplication, the event log.

Events (batch started, concurrency learned, ...) are an operational audit trail, not
research data, so they go to a JSON-lines file next to the database.
"""

import fcntl
import json
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from alpha_orchestrator.factory.models import Base, BatchORM, SimStatus, SimulationORM
from alpha_orchestrator.factory.util import canonical_json, sha256_text, utc_now

P = ParamSpec("P")
R = TypeVar("R")


class WorkerAlreadyRunning(RuntimeError):
    """A second worker must not plan or spend from the same database."""


def ledger_locked(function: Callable[P, R]) -> Callable[P, R]:
    """Run the decorated function holding the ledger's writer lock."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        ledger = signature(function).bind(*args, **kwargs).arguments["ledger"]
        with ledger.writer_lock():
            return function(*args, **kwargs)

    return wrapped


def fingerprint(expression: str, settings: dict[str, Any]) -> str:
    """Same expression text (ignoring whitespace) with the same settings = same alpha."""
    return sha256_text(re.sub(r"\s+", "", expression) + "\x1f" + canonical_json(settings))


class Ledger:
    def __init__(self, url: str, events_path: Path | None = None) -> None:
        kwargs: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
        if url in {"sqlite://", "sqlite:///:memory:"}:
            kwargs["poolclass"] = StaticPool
        self.engine = create_engine(url, future=True, **kwargs)
        event.listen(self.engine, "connect", _sqlite_pragmas)
        self._worker_mutex = threading.Lock()
        self._worker_owner: int | None = None
        database = self.engine.url.database
        self.events_path = events_path or (
            Path(database + ".events.jsonl") if database and database != ":memory:" else None
        )
        self._memory_events: list[dict[str, Any]] = []
        Base.metadata.create_all(self.engine)
        self._sessions = sessionmaker(bind=self.engine, expire_on_commit=False)

    @classmethod
    def at(cls, path: Path) -> "Ledger":
        """Open (creating if needed) a SQLite database at path."""
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(f"sqlite:///{path.resolve()}")

    @contextmanager
    def writer_lock(self) -> Iterator[None]:
        """Single host, single writer; released by the OS even after a crash."""
        if self._worker_owner == threading.get_ident():
            yield
            return
        if not self._worker_mutex.acquire(blocking=False):
            raise WorkerAlreadyRunning("Another worker owns this database")
        try:
            database = self.engine.url.database
            if not database or database == ":memory:":
                self._worker_owner = threading.get_ident()
                try:
                    yield
                finally:
                    self._worker_owner = None
                return
            with Path(database + ".worker.lock").open("a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise WorkerAlreadyRunning("Another worker owns this database") from exc
                self._worker_owner = threading.get_ident()
                try:
                    yield
                finally:
                    self._worker_owner = None
                    fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            self._worker_mutex.release()

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield a session inside a transaction."""
        with self._sessions.begin() as s:
            yield s

    # ---------- simulation queue ----------
    def queue_groups(self, s: Session, excluded: set[int] | None = None) -> list[tuple[str, int]]:
        """(region, delay) of every non-empty queue, highest priority first."""
        stmt = (
            select(SimulationORM.region, SimulationORM.delay, func.max(SimulationORM.priority))
            .where(
                SimulationORM.status == SimStatus.QUEUED,
                SimulationORM.hypothesis_id.not_in(excluded or set()),
            )
            .group_by(SimulationORM.region, SimulationORM.delay)
            .order_by(func.max(SimulationORM.priority).desc())
        )
        return [(r, int(d)) for r, d, _ in s.execute(stmt)]

    def queued(
        self, s: Session, region: str, delay: int, limit: int, excluded: set[int] | None = None
    ) -> list[SimulationORM]:
        """Queued simulations of one (region, delay), in scheduling order."""
        stmt = (
            select(SimulationORM)
            .where(
                SimulationORM.status == SimStatus.QUEUED,
                SimulationORM.region == region,
                SimulationORM.delay == delay,
                SimulationORM.hypothesis_id.not_in(excluded or set()),
            )
            .order_by(SimulationORM.priority.desc(), SimulationORM.id)
            .limit(limit)
        )
        return list(s.scalars(stmt))

    def pending(self, s: Session) -> int:
        """Simulations waiting or running."""
        return int(
            s.scalar(
                select(func.count())
                .select_from(SimulationORM)
                .where(SimulationORM.status.in_([SimStatus.QUEUED, SimStatus.RUNNING]))
            )
            or 0
        )

    def sims_reserved(self, s: Session, start: datetime) -> int:
        """Children counted at creation (in flight and uncertain included); only a
        confirmed refusal releases them."""
        return int(
            s.scalar(
                select(func.coalesce(func.sum(BatchORM.size), 0)).where(
                    BatchORM.created_at >= start,
                    BatchORM.status.not_in(["REFUSED", "REJECTED"]),
                )
            )
            or 0
        )

    # ---------- events ----------
    def log(self, kind: str, **payload: Any) -> None:
        """Append one event to the JSON-lines audit file."""
        row = {"at": utc_now().isoformat(), "kind": kind, **payload}
        if self.events_path is None:
            self._memory_events.append(row)
            return
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def events(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Every logged event (optionally of one kind), oldest first."""
        if self.events_path is None:
            rows = list(self._memory_events)
        elif self.events_path.exists():
            with self.events_path.open(encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            rows = []
        return [r for r in rows if kind is None or r.get("kind") == kind]


def _sqlite_pragmas(dbapi_connection: Any, _: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()
