"""Ledger access: one place that opens sessions and enforces de-duplication."""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, func, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from alpha_orchestrator.domain.identity import canonical_json, sha256_text
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CandidateStatus,
    EventORM,
    FactoryBase,
)


def fingerprint(expression: str, settings: dict[str, Any]) -> str:
    """Same expression text (ignoring whitespace) with the same settings = same alpha."""
    return sha256_text(re.sub(r"\s+", "", expression) + "\x1f" + canonical_json(settings))


def _upgrade_empty_tables(engine: Engine) -> None:
    """A table from an older layout is rebuilt if it holds nothing yet. One with rows is
    never touched: that needs a decision, so it stops with an explanation instead."""
    existing = inspect(engine)
    stale = []
    for table in FactoryBase.metadata.sorted_tables:
        if not existing.has_table(table.name):
            continue
        have = {c["name"] for c in existing.get_columns(table.name)}
        if {c.name for c in table.columns} - have:
            stale.append(table.name)
    if not stale:
        return
    with engine.begin() as conn:
        busy = [t for t in stale if conn.execute(text(f'SELECT 1 FROM "{t}" LIMIT 1')).first()]
        if busy:
            raise RuntimeError(
                f"ledger tables {busy} use an older layout and already hold data; "
                "back up the ledger file before upgrading it"
            )
        conn.execute(text("PRAGMA foreign_keys=OFF"))
        for t in stale:
            conn.execute(text(f'DROP TABLE "{t}"'))


def today_utc() -> str:
    return datetime.now(UTC).date().isoformat()


class Ledger:
    def __init__(self, url: str) -> None:
        kwargs: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
        if url in {"sqlite://", "sqlite:///:memory:"}:
            kwargs["poolclass"] = StaticPool
        self.engine = create_engine(url, future=True, **kwargs)
        event.listen(self.engine, "connect", _sqlite_pragmas)
        _upgrade_empty_tables(self.engine)
        FactoryBase.metadata.create_all(self.engine)
        self._sessions = sessionmaker(bind=self.engine, expire_on_commit=False)

    @classmethod
    def at(cls, path: Path) -> "Ledger":
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(f"sqlite:///{path.resolve()}")

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._sessions.begin() as s:
            yield s

    # ---------- candidates ----------
    def add_candidate(self, s: Session, **values: Any) -> CandidateORM | None:
        """Insert unless the same expression+settings exists anywhere in the ledger."""
        fp = fingerprint(values["expression"], values["settings"])
        if s.scalar(select(CandidateORM.id).where(CandidateORM.fingerprint == fp)):
            return None
        candidate = CandidateORM(fingerprint=fp, **values)
        s.add(candidate)
        s.flush()
        return candidate

    def queued(self, s: Session, region: str, delay: int, limit: int) -> list[CandidateORM]:
        from alpha_orchestrator.factory.models import CellORM

        stmt = (
            select(CandidateORM)
            .join(CellORM, CellORM.cell_id == CandidateORM.cell_id)
            .where(
                CandidateORM.status == CandidateStatus.QUEUED,
                CellORM.region == region,
                CellORM.delay == delay,
            )
            .order_by(CandidateORM.priority.desc(), CandidateORM.created_at)
            .limit(limit)
        )
        return list(s.scalars(stmt))

    def queue_groups(self, s: Session) -> list[tuple[str, int, int, float]]:
        """(region, delay, queued count, best priority) for every non-empty queue."""
        from alpha_orchestrator.factory.models import CellORM

        stmt = (
            select(CellORM.region, CellORM.delay, func.count(), func.max(CandidateORM.priority))
            .join(CellORM, CellORM.cell_id == CandidateORM.cell_id)
            .where(CandidateORM.status == CandidateStatus.QUEUED)
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

    # ---------- events ----------
    def log(self, s: Session, kind: str, cell_id: str | None = None, **payload: Any) -> None:
        s.add(EventORM(kind=kind, cell_id=cell_id, payload=payload))


def _sqlite_pragmas(dbapi_connection: Any, _: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()
