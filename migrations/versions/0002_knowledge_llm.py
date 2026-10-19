"""Milestone 2 knowledge and LLM ledger.

Revision ID: 0002_knowledge_llm
Revises: 0001_milestone_1
"""

from collections.abc import Sequence

from alembic import op

from alpha_orchestrator.storage.models import (
    LLMRequestORM,
    MethodCardORM,
    PromptVersionORM,
    SourceDocumentORM,
    SourceSectionORM,
)

revision: str = "0002_knowledge_llm"
down_revision: str | None = "0001_milestone_1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MILESTONE_2_TABLES = (
    SourceDocumentORM.__table__,
    SourceSectionORM.__table__,
    MethodCardORM.__table__,
    PromptVersionORM.__table__,
    LLMRequestORM.__table__,
)


def upgrade() -> None:
    bind = op.get_bind()
    for table in MILESTONE_2_TABLES:
        table.create(bind=bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    for table in reversed(MILESTONE_2_TABLES):
        table.drop(bind=bind, checkfirst=True)
