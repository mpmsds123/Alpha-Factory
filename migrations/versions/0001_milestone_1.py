"""Milestone 1 auditable engine schema.

Revision ID: 0001_milestone_1
Revises:
"""

from collections.abc import Sequence

from alembic import op

from alpha_orchestrator.storage.models import (
    CandidateORM,
    DecisionORM,
    EventORM,
    LineageEdgeORM,
    PlatformCheckORM,
    RunORM,
    SimulationAttemptORM,
)

revision: str = "0001_milestone_1"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MILESTONE_1_TABLES = (
    RunORM.__table__,
    CandidateORM.__table__,
    LineageEdgeORM.__table__,
    SimulationAttemptORM.__table__,
    DecisionORM.__table__,
    EventORM.__table__,
    PlatformCheckORM.__table__,
)


def upgrade() -> None:
    bind = op.get_bind()
    for table in MILESTONE_1_TABLES:
        table.create(bind=bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    for table in reversed(MILESTONE_1_TABLES):
        table.drop(bind=bind, checkfirst=True)
