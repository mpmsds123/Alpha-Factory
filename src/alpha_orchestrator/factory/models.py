"""The factory ledger. Every simulation, decision and LLM call lands here.

One SQLite file (`factory.db`) is the single source of truth: the scheduler reads its
queue from it, the planner reads cell statistics from it, the dashboard renders it.
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

from alpha_orchestrator.domain.models import utc_now


def new_id() -> str:
    return str(uuid4())


class FactoryBase(DeclarativeBase):
    pass


class CellState:
    NEW = "NEW"  # known from the catalog, nothing spent yet
    STUDYING = "STUDYING"  # health check: update period and filled coverage, by sampling
    SURVEY = "SURVEY"  # sampled survey: does this dataset carry more signal than noise?
    RESEARCH = "RESEARCH"  # signal found: budget allocated to strata by concentration
    EXHAUSTED = "EXHAUSTED"  # finished; `plan["outcome"]` says how
    SATURATED = "SATURATED"  # enough submissions from this dataset (concentration cap)


class CandidateStatus:
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"  # platform error (bad field, operator, etc.)
    INVALID = "INVALID"  # rejected locally before any simulation was spent


class Stage:
    PROFILE = "PROFILE"  # health probe (neutralization NONE, decay 0); read long/short counts
    SURVEY = "SURVEY"  # one of the 4 fixed signal shapes on a sampled family
    REFINE = "REFINE"  # a library structure around a family's best survey formula
    COMBO = "COMBO"  # two fields combined (arm "llm" or "random", compared head to head)
    IDEA = "IDEA"  # legacy: an idea's template expanded over its fields and windows
    DIG = "DIG"  # legacy: variant written by the LLM around ore it found in review
    SECOND = "SECOND"  # group operator wrapped around a survivor
    THIRD = "THIRD"  # trade_when / timing wrapped around a survivor
    REPAIR = "REPAIR"  # one-variable fix routed by the failing check
    FLIP = "FLIP"  # multiply(-1, parent)
    ROBUST = "ROBUST"  # stress variant of a candidate (rank, sign, perturbation, ...)
    FIELD = "FIELD"  # 设计文档 第 5 步: one field in an idea's simplest form
    ENHANCE = "ENHANCE"  # 加强: liquidity/size-bucket grouping, then trade_when, around a near miss
    IMPROVE = "IMPROVE"  # 设计文档 第 7 步: one-change variant of a parent


class HypothesisStatus:
    PROPOSED = "PROPOSED"  # waiting for you (only when policy.hypotheses_need_approval)
    APPROVED = "APPROVED"  # being tested
    REJECTED = "REJECTED"  # you said no
    RETIRED = "RETIRED"  # the LLM review dropped it, with a reason


class ReviewStatus:
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    SUBMITTED = "SUBMITTED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class CellORM(FactoryBase):
    __tablename__ = "cells"

    cell_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    universe: Mapped[str] = mapped_column(String(64))
    dataset_id: Mapped[str] = mapped_column(String(128))
    dataset_name: Mapped[str] = mapped_column(String(255), default="")
    category: Mapped[str] = mapped_column(String(64), default="other")
    description: Mapped[str] = mapped_column(Text, default="")
    multiplier: Mapped[float] = mapped_column(Float, default=1.0)
    value_score: Mapped[float] = mapped_column(Float, default=0.0)
    alpha_count: Mapped[int] = mapped_column(Integer, default=0)
    user_count: Mapped[int] = mapped_column(Integer, default=0)
    coverage: Mapped[float] = mapped_column(Float, default=0.0)
    field_count: Mapped[int] = mapped_column(Integer, default=0)
    themes: Mapped[list[Any]] = mapped_column(JSON, default=list)
    state: Mapped[str] = mapped_column(String(16), default=CellState.NEW)
    rank_score: Mapped[float] = mapped_column(Float, default=0.0)
    rank_reason: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    paper_refs: Mapped[int] = mapped_column(Integer, default=0)
    rounds: Mapped[int] = mapped_column(Integer, default=0)  # LLM reviews done
    rounds_without_ore: Mapped[int] = mapped_column(Integer, default=0)
    hits: Mapped[int] = mapped_column(Integer, default=0)  # results with |sharpe| >= ore
    stop_reason: Mapped[str] = mapped_column(Text, default="")
    plan: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # phase state and outcome
    sims_used: Mapped[int] = mapped_column(Integer, default=0)
    submitted: Mapped[int] = mapped_column(Integer, default=0)
    best_sharpe: Mapped[float] = mapped_column(Float, default=0.0)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    refreshed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class FieldORM(FactoryBase):
    __tablename__ = "fields"
    __table_args__ = (UniqueConstraint("cell_id", "field_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cell_id: Mapped[str] = mapped_column(ForeignKey("cells.cell_id"))
    field_id: Mapped[str] = mapped_column(String(255))
    type: Mapped[str] = mapped_column(String(16))
    description: Mapped[str] = mapped_column(Text, default="")
    coverage: Mapped[float] = mapped_column(Float, default=0.0)
    alpha_count: Mapped[int] = mapped_column(Integer, default=0)
    family: Mapped[str] = mapped_column(String(255), default="")
    representative: Mapped[bool] = mapped_column(Boolean, default=False)
    profile: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # health readings
    prep: Mapped[str] = mapped_column(Text, default="{x}")  # how the field enters a template
    stratum: Mapped[str] = mapped_column(String(255), default="")
    eligible: Mapped[bool] = mapped_column(Boolean, default=True)
    filled_coverage: Mapped[float | None] = mapped_column(Float)
    survey: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # shape -> candidate id
    survey_score: Mapped[float | None] = mapped_column(Float)  # best |Sharpe| over the shapes


class HypothesisORM(FactoryBase):
    __tablename__ = "hypotheses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    cell_id: Mapped[str] = mapped_column(ForeignKey("cells.cell_id"))
    title: Mapped[str] = mapped_column(String(255))
    statement: Mapped[str] = mapped_column(Text)
    direction: Mapped[str] = mapped_column(Text, default="")
    fields: Mapped[list[str]] = mapped_column(JSON, default=list)
    sources: Mapped[list[str]] = mapped_column(JSON, default=list)
    template: Mapped[str] = mapped_column(Text, default="")
    slots: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    windows: Mapped[list[int]] = mapped_column(JSON, default=list)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # decay, neutralization
    status: Mapped[str] = mapped_column(String(16), default=HypothesisStatus.PROPOSED)
    rounds: Mapped[int] = mapped_column(Integer, default=0)  # rounds queued
    reviewed: Mapped[int] = mapped_column(Integer, default=0)  # rounds the LLM has judged
    verdicts: Mapped[list[Any]] = mapped_column(JSON, default=list)  # review history
    best_sharpe: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class BatchORM(FactoryBase):
    __tablename__ = "batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    location: Mapped[str] = mapped_column(String(512), default="")
    size: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="RUNNING")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CandidateORM(FactoryBase):
    __tablename__ = "candidates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    cell_id: Mapped[str] = mapped_column(ForeignKey("cells.cell_id"))
    hypothesis_id: Mapped[str | None] = mapped_column(ForeignKey("hypotheses.id"))
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("candidates.id"))
    stage: Mapped[str] = mapped_column(String(16))
    role: Mapped[str] = mapped_column(String(64), default="")
    change: Mapped[str] = mapped_column(Text, default="")
    expression: Mapped[str] = mapped_column(Text)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    priority: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(16), default=CandidateStatus.QUEUED)
    batch_id: Mapped[str | None] = mapped_column(ForeignKey("batches.id"))
    alpha_id: Mapped[str | None] = mapped_column(String(32))
    sharpe: Mapped[float | None] = mapped_column(Float)
    fitness: Mapped[float | None] = mapped_column(Float)
    turnover: Mapped[float | None] = mapped_column(Float)
    returns: Mapped[float | None] = mapped_column(Float)
    margin: Mapped[float | None] = mapped_column(Float)
    drawdown: Mapped[float | None] = mapped_column(Float)
    long_count: Mapped[int | None] = mapped_column(Integer)
    short_count: Mapped[int | None] = mapped_column(Integer)
    fails: Mapped[str] = mapped_column(Text, default="")
    checks: Mapped[list[Any]] = mapped_column(JSON, default=list)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    evaluated: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ReviewORM(FactoryBase):
    __tablename__ = "reviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    candidate_id: Mapped[str] = mapped_column(ForeignKey("candidates.id"), unique=True)
    alpha_id: Mapped[str] = mapped_column(String(32))
    score: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default=ReviewStatus.PENDING)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LLMCallORM(FactoryBase):
    __tablename__ = "llm_calls"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    purpose: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(128))
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)  # unused since 10/14
    cost_cny: Mapped[float] = mapped_column(Float, default=0.0)  # 第 9 问: what DeepSeek bills
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str | None] = mapped_column(Text)
    idea_id: Mapped[str | None] = mapped_column(String(36))  # 第 9 问: which idea paid for it
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class IdeaORM(FactoryBase):
    __tablename__ = "ideas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(255))
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text, default="")
    body: Mapped[str] = mapped_column(Text)
    __table_args__ = (UniqueConstraint("source", "title"),)


class IdeaStatus:
    TESTING = "TESTING"  # first fields being simulated
    PROMISING = "PROMISING"  # a field reached the continue line; being improved
    DEAD = "DEAD"  # nothing reached the continue line
    DONE = "DONE"  # improvement finished
    RETIRED = "RETIRED"  # judged under an earlier design; ignored


class ResearchIdeaORM(FactoryBase):
    """One research idea on one dataset (设计文档 第 3 步)."""

    __tablename__ = "research_ideas"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    cell_id: Mapped[str] = mapped_column(ForeignKey("cells.cell_id"))
    title: Mapped[str] = mapped_column(Text)
    mechanism: Mapped[str] = mapped_column(Text, default="")  # who wins, who loses, why it lasts
    long_side: Mapped[str] = mapped_column(Text, default="")
    short_side: Mapped[str] = mapped_column(Text, default="")
    horizon_days: Mapped[int] = mapped_column(Integer, default=21)
    form: Mapped[str] = mapped_column(Text, default="")  # simplest expression, one {X}
    query: Mapped[str] = mapped_column(Text, default="")  # search words for narrowing fields
    status: Mapped[str] = mapped_column(String(16), default=IdeaStatus.TESTING)
    best_sharpe: Mapped[float] = mapped_column(Float, default=0.0)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class IdeaFieldORM(FactoryBase):
    """One field ranked for one idea (设计文档 第 4 步) and its first test (第 5 步)."""

    __tablename__ = "idea_fields"
    __table_args__ = (UniqueConstraint("idea_id", "field_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    idea_id: Mapped[str] = mapped_column(ForeignKey("research_ideas.id"))
    field_id: Mapped[str] = mapped_column(String(255))
    score: Mapped[int] = mapped_column(Integer, default=0)  # LLM relevance 0-10
    reason: Mapped[str] = mapped_column(Text, default="")
    measures: Mapped[str] = mapped_column(Text, default="")  # what it measures, per the LLM
    rank: Mapped[int] = mapped_column(Integer, default=0)  # 1 = most relevant
    candidate_id: Mapped[str | None] = mapped_column(ForeignKey("candidates.id"))


class EventORM(FactoryBase):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    kind: Mapped[str] = mapped_column(String(64))
    cell_id: Mapped[str | None] = mapped_column(String(255))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
