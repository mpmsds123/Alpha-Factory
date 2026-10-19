"""Store BRAIN's own operator call signatures alongside the frozen operator set.

Revision ID: 0004_operator_definitions
Revises: 0003_live_metadata

BRAIN's `/operators` response includes a `definition` string per operator (e.g.
"group_rank(x, group)"), but the frozen operator snapshot only ever hashed the bare
name set, so the generator's only source for argument counts was the LLM's own
guess. 2026-09-15: `group_rank(x)` was called with one argument (it takes two) and
went straight to a real BRAIN simulation before failing. This column is additive
and nullable -- it carries no identity weight, so no existing snapshot_id changes.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "0004_operator_definitions"
down_revision: str | None = "0003_live_metadata"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Milestone 3's own migration creates this table from the live ORM class
    # (checkfirst=True), so replaying migrations from scratch against
    # current code already has this column by the time 0003 runs -- only a
    # database that actually stopped at 0003 in the past is missing it.
    existing = {
        column["name"] for column in inspect(op.get_bind()).get_columns("metadata_snapshots")
    }
    if "operator_definitions_json" in existing:
        return
    with op.batch_alter_table("metadata_snapshots") as batch_op:
        batch_op.add_column(sa.Column("operator_definitions_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("metadata_snapshots") as batch_op:
        batch_op.drop_column("operator_definitions_json")
