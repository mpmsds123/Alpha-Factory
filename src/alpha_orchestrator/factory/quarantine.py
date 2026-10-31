"""Explicit, expiring operator authorization for one immutable quarantined batch."""

import hashlib
import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.models import BatchORM, CandidateORM


def batch_digest(s: Session, batch: BatchORM) -> str:
    members = list(s.scalars(select(CandidateORM).where(CandidateORM.batch_id == batch.id)))
    created = batch.created_at
    if created.tzinfo is not None:
        created = created.astimezone(UTC)
    value = {
        "batch": [
            batch.id,
            batch.status,
            batch.location,
            batch.size,
            created.replace(tzinfo=None).isoformat(timespec="microseconds"),
        ],
        "members": sorted(
            [c.id, c.cell_id, c.fingerprint, c.status, c.expression, c.settings] for c in members
        ),
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def exception_scope(
    s: Session, config: FactoryConfig, now: datetime | None = None
) -> tuple[set[str], set[str]]:
    """No default exceptions. Never modifies outcomes, reservations, or candidate state."""
    spec = config.quarantine_exception
    if not spec:
        return set(), set()
    try:
        start = datetime.fromisoformat(spec["approved_at"])
        end = datetime.fromisoformat(spec["expires_at"])
        instant = now or datetime.now(UTC)
        if start.tzinfo is None or end.tzinfo is None or instant.tzinfo is None:
            return set(), set()
        if not 0 < (end - start).total_seconds() <= 86400 or not start <= instant < end:
            return set(), set()
        batch = s.get(BatchORM, spec["batch_id"])
        if batch is None or batch.status != "UNKNOWN" or not batch.location:
            return set(), set()
        if batch_digest(s, batch) != spec["digest"]:
            return set(), set()
        members = list(s.scalars(select(CandidateORM).where(CandidateORM.batch_id == batch.id)))
        if len(members) != batch.size or any(c.status != "RUNNING" for c in members):
            return set(), set()
        return {batch.id}, {c.cell_id for c in members}
    except (KeyError, TypeError, ValueError):
        return set(), set()
