"""Ledger access: one place that opens sessions and enforces de-duplication."""

import fcntl
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from sqlalchemy import Engine, create_engine, event, func, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from alpha_orchestrator.factory.models import (
    BatchORM,
    CandidateORM,
    CandidateStatus,
    EventORM,
    FactoryBase,
)
from alpha_orchestrator.factory.util import canonical_json, sha256_text

P = ParamSpec("P")
R = TypeVar("R")


class WorkerAlreadyRunning(RuntimeError):
    """A second worker must not plan or spend from the same ledger."""


def ledger_locked(function: Callable[P, R]) -> Callable[P, R]:
    """Decorate a function so it runs holding the ledger's writer lock."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        """Call the wrapped function while holding the writer lock."""
        ledger = signature(function).bind(*args, **kwargs).arguments["ledger"]
        with ledger.writer_lock():
            return function(*args, **kwargs)

    return wrapped


def fingerprint(expression: str, settings: dict[str, Any]) -> str:
    """Same expression text (ignoring whitespace) with the same settings = same alpha."""
    return sha256_text(re.sub(r"\s+", "", expression) + "\x1f" + canonical_json(settings))


def _upgrade_empty_tables(engine: Engine) -> None:
    """Brings an older ledger up to the current layout without losing rows: every column
    the models have and the file lacks is added with its default."""
    existing = inspect(engine)
    statements = []
    for table in FactoryBase.metadata.sorted_tables:
        if not existing.has_table(table.name):
            continue
        have = {c["name"] for c in existing.get_columns(table.name)}
        for column in table.columns:
            if column.name in have:
                continue
            sql_type = column.type.compile(dialect=engine.dialect)
            default = getattr(column.default, "arg", None)
            if callable(default):
                default = default(None) if default.__name__ != "utc_now" else None
            if isinstance(default, bool):
                literal = "1" if default else "0"
            elif isinstance(default, (int, float)):
                literal = str(default)
            elif isinstance(default, (dict, list)):
                literal = "'" + ("{}" if isinstance(default, dict) else "[]") + "'"
            elif isinstance(default, str):
                literal = "'" + default.replace("'", "''") + "'"
            else:
                literal = "NULL"
            statements.append(
                f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {sql_type} '
                f"DEFAULT {literal}"
            )
    if statements:
        with engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))


def today_utc() -> str:
    """Return today's UTC date as YYYY-MM-DD."""
    return datetime.now(UTC).date().isoformat()


class Ledger:
    def __init__(self, url: str) -> None:
        kwargs: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
        if url in {"sqlite://", "sqlite:///:memory:"}:
            kwargs["poolclass"] = StaticPool
        self.engine = create_engine(url, future=True, **kwargs)
        event.listen(self.engine, "connect", _sqlite_pragmas)
        self._worker_mutex = threading.Lock()
        self._worker_owner: int | None = None
        with self.writer_lock():
            _upgrade_empty_tables(self.engine)
            FactoryBase.metadata.create_all(self.engine)
        self._sessions = sessionmaker(bind=self.engine, expire_on_commit=False)

    @contextmanager
    def writer_lock(self) -> Iterator[None]:
        """Single host, single ledger writer; released by the OS even after a crash."""
        if self._worker_owner == threading.get_ident():
            yield
            return
        if not self._worker_mutex.acquire(blocking=False):
            raise WorkerAlreadyRunning("Another worker owns this ledger")
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
                    raise WorkerAlreadyRunning("Another worker owns this ledger") from exc
                self._worker_owner = threading.get_ident()
                try:
                    yield
                finally:
                    self._worker_owner = None
                    fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            self._worker_mutex.release()

    @classmethod
    def at(cls, path: Path) -> "Ledger":
        """Open (creating if needed) a SQLite ledger at path."""
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(f"sqlite:///{path.resolve()}")

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield a database session inside a transaction."""
        with self._sessions.begin() as s:
            yield s

    # ---------- candidates ----------
    def add_candidate(self, s: Session, **values: Any) -> CandidateORM | None:
        """Insert unless the same expression+settings exists anywhere in the ledger."""
        row, created = self.get_or_add(s, **values)
        return row if created else None

    def get_or_add(self, s: Session, **values: Any) -> tuple[CandidateORM, bool]:
        """The existing row for this expression+settings (its result is reused, never
        simulated twice), or a new queued one."""
        fp = fingerprint(values["expression"], values["settings"])
        found = s.scalar(select(CandidateORM).where(CandidateORM.fingerprint == fp))
        if found is not None:
            return found, False
        candidate = CandidateORM(fingerprint=fp, **values)
        s.add(candidate)
        s.flush()
        return candidate, True

    def queued(
        self,
        s: Session,
        region: str,
        delay: int,
        limit: int,
        excluded_cells: set[str] | None = None,
    ) -> list[CandidateORM]:
        """Return queued candidates in scheduling order."""
        from alpha_orchestrator.factory.models import CellORM

        stmt = (
            select(CandidateORM)
            .join(CellORM, CellORM.cell_id == CandidateORM.cell_id)
            .where(
                CandidateORM.status == CandidateStatus.QUEUED,
                CandidateORM.cell_id.not_in(excluded_cells or set()),
                CellORM.region == region,
                CellORM.delay == delay,
            )
            .order_by(CandidateORM.priority.desc(), CandidateORM.created_at)
            .limit(limit)
        )
        return list(s.scalars(stmt))

    def queue_groups(
        self, s: Session, excluded_cells: set[str] | None = None
    ) -> list[tuple[str, int, int, float]]:
        """(region, delay, queued count, best priority) for every non-empty queue."""
        from alpha_orchestrator.factory.models import CellORM

        stmt = (
            select(CellORM.region, CellORM.delay, func.count(), func.max(CandidateORM.priority))
            .join(CellORM, CellORM.cell_id == CandidateORM.cell_id)
            .where(
                CandidateORM.status == CandidateStatus.QUEUED,
                CandidateORM.cell_id.not_in(excluded_cells or set()),
            )
            .group_by(CellORM.region, CellORM.delay)
        )
        return [(r, d, int(n), float(p or 0)) for r, d, n, p in s.execute(stmt)]

    def sims_used_today(self, s: Session) -> int:
        """Simulations that came back today (successful or failed both cost quota)."""
        start = datetime.combine(datetime.now(UTC).date(), datetime.min.time(), tzinfo=UTC)
        stmt = select(func.count()).where(
            CandidateORM.status.in_([CandidateStatus.DONE, CandidateStatus.FAILED]),
            CandidateORM.completed_at >= start,
        )
        return int(s.scalar(stmt) or 0)

    def sims_reserved_today(self, s: Session, start: datetime | None = None) -> int:
        """Count children at creation, including in-flight and uncertain requests.

        The caller supplies its allocation window. Only a confirmed refusal releases
        it. Completion dates never decide which day's allocation paid for a request.
        """
        start = start or datetime.combine(datetime.now(UTC).date(), datetime.min.time(), tzinfo=UTC)
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
    def log(self, s: Session, kind: str, cell_id: str | None = None, **payload: Any) -> None:
        """Append an event row to the ledger."""
        s.add(EventORM(kind=kind, cell_id=cell_id, payload=payload))


def _sqlite_pragmas(dbapi_connection: Any, _: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()
