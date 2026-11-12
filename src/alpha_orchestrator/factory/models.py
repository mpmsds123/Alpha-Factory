"""The factory's one SQLite database.

Five research tables (hypotheses, expressions, simulations, decisions, priors) and two
plumbing tables: `catalog` caches datasets and fields so a dataset can be sampled without
calling the platform each time, and `batches` keeps in-flight multi-simulations so a crash
or a face check resumes by polling instead of simulating again.

Integer ids are what the LLM cites: hypothesis 12 is "H12", prior 7 is "P7".
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from alpha_orchestrator.factory.util import utc_now


class Base(DeclarativeBase):
    pass


class HStatus:
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class Verdict:
    """How a hypothesis ended. Memory B and prior updates read this."""

    PASSED = "passed"  # an expression passed every gate and was stored for review
    SUPPORTED = "supported"  # signal as predicted, but no expression reached the gates
    IDEA_FAILED = "idea_failed"  # the claim did not hold
    IMPLEMENTATION_FAILED = "implementation_failed"  # the claim was never tested properly
    BUDGET = "budget"  # stopped by the program's budget or stopping rule
    INVALID = "invalid"  # no valid output from the LLM


class SimStatus:
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class Review:
    NONE = ""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class HypothesisORM(Base):
    __tablename__ = "hypotheses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("hypotheses.id"))  # revision of
    source: Mapped[str] = mapped_column(String(16), default="factory")  # or "v7" (imported)
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer, default=1)
    universe: Mapped[str] = mapped_column(String(64), default="")
    dataset_id: Mapped[str] = mapped_column(String(128))
    dataset_name: Mapped[str] = mapped_column(String(255), default="")
    category: Mapped[str] = mapped_column(String(64), default="")
    field_id: Mapped[str] = mapped_column(String(255), default="")
    field_type: Mapped[str] = mapped_column(String(16), default="")
    field_description: Mapped[str] = mapped_column(Text, default="")
    claim: Mapped[str] = mapped_column(Text, default="")  # the mechanism, in words
    direction: Mapped[int] = mapped_column(Integer, default=0)  # +1 higher value -> long
    hypothesis_type: Mapped[str] = mapped_column(String(32), default="unknown")
    holding_days: Mapped[int] = mapped_column(Integer, default=21)
    turnover_band: Mapped[str] = mapped_column(String(16), default="")
    falsifier: Mapped[str] = mapped_column(Text, default="")
    helper_fields: Mapped[list[str]] = mapped_column(JSON, default=list)
    tags: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # set by the program
    status: Mapped[str] = mapped_column(String(16), default=HStatus.OPEN)
    verdict: Mapped[str] = mapped_column(String(32), default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    lesson: Mapped[str] = mapped_column(Text, default="")
    round: Mapped[int] = mapped_column(Integer, default=0)
    dry_rounds: Mapped[int] = mapped_column(Integer, default=0)
    best_train_sharpe: Mapped[float] = mapped_column(Float, default=0.0)
    sims_used: Mapped[int] = mapped_column(Integer, default=0)
    sims_budget: Mapped[int] = mapped_column(Integer, default=12)
    state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # loop bookkeeping
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ExpressionORM(Base):
    __tablename__ = "expressions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hypothesis_id: Mapped[int] = mapped_column(ForeignKey("hypotheses.id"))
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("expressions.id"))  # lineage
    round: Mapped[int] = mapped_column(Integer, default=1)
    origin: Mapped[str] = mapped_column(String(16), default="initial")  # initial/mutation
    mutation: Mapped[str] = mapped_column(String(32), default="")  # declared change type
    reason: Mapped[str] = mapped_column(Text, default="")
    expression: Mapped[str] = mapped_column(Text)
    canonical: Mapped[str] = mapped_column(Text, default="")
    shape: Mapped[str] = mapped_column(Text, default="")
    fields: Mapped[list[str]] = mapped_column(JSON, default=list)
    operators: Mapped[int] = mapped_column(Integer, default=0)
    nodes: Mapped[int] = mapped_column(Integer, default=0)
    valid: Mapped[bool] = mapped_column(Boolean, default=True)
    rejection: Mapped[str] = mapped_column(Text, default="")
    decision_id: Mapped[int | None] = mapped_column(ForeignKey("decisions.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SimulationORM(Base):
    __tablename__ = "simulations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    expression_id: Mapped[int] = mapped_column(ForeignKey("expressions.id"))
    hypothesis_id: Mapped[int] = mapped_column(ForeignKey("hypotheses.id"))
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    expression: Mapped[str] = mapped_column(Text, default="")  # copy of the expression sent
    settings: Mapped[dict[str, Any]] = mapped_column(JSON)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    priority: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(16), default=SimStatus.QUEUED)
    batch_id: Mapped[str | None] = mapped_column(ForeignKey("batches.id"))
    alpha_id: Mapped[str | None] = mapped_column(String(32))
    # Train period: the only numbers the LLM ever sees.
    train_sharpe: Mapped[float | None] = mapped_column(Float)
    train_fitness: Mapped[float | None] = mapped_column(Float)
    train_turnover: Mapped[float | None] = mapped_column(Float)
    train_returns: Mapped[float | None] = mapped_column(Float)
    train_margin: Mapped[float | None] = mapped_column(Float)
    train_drawdown: Mapped[float | None] = mapped_column(Float)
    # Test period: read only by the program's out-of-sample gate.
    test_sharpe: Mapped[float | None] = mapped_column(Float)
    test_fitness: Mapped[float | None] = mapped_column(Float)
    test_returns: Mapped[float | None] = mapped_column(Float)
    # Whole IS period (what the platform's submission checks use).
    is_sharpe: Mapped[float | None] = mapped_column(Float)
    is_fitness: Mapped[float | None] = mapped_column(Float)
    is_turnover: Mapped[float | None] = mapped_column(Float)
    is_returns: Mapped[float | None] = mapped_column(Float)
    checks: Mapped[list[Any]] = mapped_column(JSON, default=list)
    fails: Mapped[str] = mapped_column(Text, default="")
    train_pass: Mapped[bool] = mapped_column(Boolean, default=False)
    oos_pass: Mapped[bool | None] = mapped_column(Boolean)
    checks_pass: Mapped[bool | None] = mapped_column(Boolean)
    gate_note: Mapped[str] = mapped_column(Text, default="")
    review: Mapped[str] = mapped_column(String(16), default=Review.NONE)
    review_note: Mapped[str] = mapped_column(Text, default="")
    submission_check: Mapped[list[Any]] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DecisionORM(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hypothesis_id: Mapped[int | None] = mapped_column(ForeignKey("hypotheses.id"))
    step: Mapped[str] = mapped_column(String(32))  # hypothesis/expression/diagnosis/decision
    actor: Mapped[str] = mapped_column(String(16), default="llm")  # or "program" (override)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    retrieved: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # ids shown
    output: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    cited: Mapped[list[str]] = mapped_column(JSON, default=list)
    flags: Mapped[list[str]] = mapped_column(JSON, default=list)
    prompt_version: Mapped[str] = mapped_column(String(64), default="")
    model: Mapped[str] = mapped_column(String(64), default="")
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class PriorORM(Base):
    __tablename__ = "priors"
    __table_args__ = (UniqueConstraint("claim", "source_ref"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(16))  # official / forum / own
    source_ref: Mapped[str] = mapped_column(String(255), default="")
    source_title: Mapped[str] = mapped_column(Text, default="")
    quote: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # what it applies to
    a: Mapped[float] = mapped_column(Float, default=1.0)  # Beta successes
    b: Mapped[float] = mapped_column(Float, default=1.0)  # Beta failures
    history: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    @property
    def confidence(self) -> float:
        """Mean of the Beta distribution."""
        return self.a / (self.a + self.b)


class CatalogORM(Base):
    __tablename__ = "catalog"
    __table_args__ = (
        UniqueConstraint("kind", "region", "delay", "universe", "dataset_id", "field_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(8))  # dataset / field
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    universe: Mapped[str] = mapped_column(String(64))
    dataset_id: Mapped[str] = mapped_column(String(128))
    dataset_name: Mapped[str] = mapped_column(String(255), default="")
    category: Mapped[str] = mapped_column(String(64), default="")
    field_id: Mapped[str] = mapped_column(String(255), default="")
    field_type: Mapped[str] = mapped_column(String(16), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    coverage: Mapped[float] = mapped_column(Float, default=0.0)
    alpha_count: Mapped[int] = mapped_column(Integer, default=0)
    user_count: Mapped[int] = mapped_column(Integer, default=0)
    field_count: Mapped[int] = mapped_column(Integer, default=0)
    refreshed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class BatchORM(Base):
    __tablename__ = "batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    location: Mapped[str] = mapped_column(String(512), default="")
    size: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="RUNNING")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
