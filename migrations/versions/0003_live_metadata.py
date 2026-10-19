"""Milestone 3 live metadata snapshots.

Revision ID: 0003_live_metadata
Revises: 0002_knowledge_llm
"""

from collections.abc import Sequence

from alembic import op

from alpha_orchestrator.storage.models import MetadataSnapshotORM

revision: str = "0003_live_metadata"
down_revision: str | None = "0002_knowledge_llm"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    MetadataSnapshotORM.__table__.create(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    MetadataSnapshotORM.__table__.drop(bind=op.get_bind(), checkfirst=True)
