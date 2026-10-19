"""SQLAlchemy schema for the auditable offline engine."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from alpha_orchestrator.domain.enums import CandidateStatus, RunStatus
from alpha_orchestrator.domain.models import utc_now


class Base(DeclarativeBase):
    pass


class RunORM(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(255), nullable=False)
    region: Mapped[str] = mapped_column(String(64), nullable=False)
    universe: Mapped[str] = mapped_column(String(128), nullable=False)
    delay: Mapped[int] = mapped_column(Integer, nullable=False)
    data_type: Mapped[str] = mapped_column(String(16), nullable=False)
    field_snapshot_id: Mapped[str] = mapped_column(String(255), nullable=False)
    operator_snapshot_id: Mapped[str] = mapped_column(String(255), nullable=False)
    setting_snapshot_id: Mapped[str] = mapped_column(String(255), nullable=False)
    routing_policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    limit_policy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    generator_prompt_version: Mapped[str] = mapped_column(String(128), nullable=False)
    repair_prompt_version: Mapped[str] = mapped_column(String(128), nullable=False)
    adapter_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    routing_policy_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    limit_policy_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default=RunStatus.CREATED, nullable=False)
    stop_reason: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CandidateORM(Base):
    __tablename__ = "candidates"
    __table_args__ = (
        UniqueConstraint("run_id", "candidate_fingerprint", name="uq_candidate_run_fingerprint"),
        UniqueConstraint("run_id", "normalized_content_hash", name="uq_candidate_run_content"),
        CheckConstraint(
            "(parent_candidate_id IS NULL AND generation_number = 0 "
            "AND changed_dimension IS NULL AND change_reason IS NULL) OR "
            "(parent_candidate_id IS NOT NULL AND generation_number >= 1 "
            "AND changed_dimension IS NOT NULL AND change_reason IS NOT NULL)",
            name="ck_candidate_lineage_complete",
        ),
    )

    candidate_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), nullable=False, index=True)
    parent_candidate_id: Mapped[str | None] = mapped_column(ForeignKey("candidates.candidate_id"))
    generation_number: Mapped[int] = mapped_column(Integer, nullable=False)
    expression: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_expression: Mapped[str] = mapped_column(Text, nullable=False)
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    candidate_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    normalized_content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    mechanism_claim: Mapped[str] = mapped_column(Text, nullable=False)
    source_method_card_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    changed_dimension: Mapped[str | None] = mapped_column(String(128))
    change_reason: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(40), default=CandidateStatus.PROPOSED, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class LineageEdgeORM(Base):
    __tablename__ = "lineage_edges"
    __table_args__ = (UniqueConstraint("child_candidate_id", name="uq_lineage_child"),)

    edge_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), nullable=False)
    parent_candidate_id: Mapped[str] = mapped_column(
        ForeignKey("candidates.candidate_id"), nullable=False
    )
    child_candidate_id: Mapped[str] = mapped_column(
        ForeignKey("candidates.candidate_id"), nullable=False
    )
    changed_dimension: Mapped[str] = mapped_column(String(128), nullable=False)
    change_reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SimulationAttemptORM(Base):
    __tablename__ = "simulation_attempts"
    __table_args__ = (
        UniqueConstraint("candidate_id", "attempt_number", name="uq_candidate_attempt"),
    )

    attempt_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(
        ForeignKey("candidates.candidate_id"), nullable=False, index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False)
    external_simulation_id: Mapped[str | None] = mapped_column(String(255))
    external_alpha_id: Mapped[str | None] = mapped_column(String(255))
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error_class: Mapped[str | None] = mapped_column(String(64))
    error_message_redacted: Mapped[str | None] = mapped_column(Text)
    sharpe: Mapped[float | None] = mapped_column(Float)
    fitness: Mapped[float | None] = mapped_column(Float)
    turnover: Mapped[float | None] = mapped_column(Float)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    raw_response_artifact_ref: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DecisionORM(Base):
    __tablename__ = "decisions"

    decision_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(
        ForeignKey("candidates.candidate_id"), nullable=False, index=True
    )
    gate_name: Mapped[str] = mapped_column(String(64), nullable=False)
    gate_version: Mapped[str] = mapped_column(String(128), nullable=False)
    outcome: Mapped[str] = mapped_column(String(64), nullable=False)
    reason_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    metric_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    llm_request_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class EventORM(Base):
    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), nullable=False, index=True)
    candidate_id: Mapped[str | None] = mapped_column(ForeignKey("candidates.candidate_id"))
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class PlatformCheckORM(Base):
    __tablename__ = "platform_checks"

    check_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(
        ForeignKey("candidates.candidate_id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    message_redacted: Mapped[str | None] = mapped_column(Text)
    is_live_response: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SourceDocumentORM(Base):
    __tablename__ = "source_documents"

    source_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    corpus_root_locator: Mapped[str] = mapped_column(Text, nullable=False)
    relative_locator: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    excluded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    exclusion_reason: Mapped[str | None] = mapped_column(String(255))
    inventoried_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SourceSectionORM(Base):
    __tablename__ = "source_sections"
    __table_args__ = (
        UniqueConstraint("source_id", "section_locator", name="uq_source_section_locator"),
    )

    section_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(
        ForeignKey("source_documents.source_id"), nullable=False, index=True
    )
    heading: Mapped[str] = mapped_column(String(255), nullable=False)
    section_locator: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class MethodCardORM(Base):
    __tablename__ = "method_cards"
    __table_args__ = (UniqueConstraint("card_id", "version", name="uq_method_card_version"),)

    row_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    card_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    applicable_failure_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    applicable_data_types: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    prerequisites: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    diagnostic_signals: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    allowed_repair_actions: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    prohibited_actions: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    stop_condition: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_level: Mapped[str] = mapped_column(String(32), nullable=False)
    source_references: Mapped[list[dict[str, str]]] = mapped_column(JSON, nullable=False)
    reviewer: Mapped[str | None] = mapped_column(String(255))
    approval_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class PromptVersionORM(Base):
    __tablename__ = "prompt_versions"

    version: Mapped[str] = mapped_column(String(128), primary_key=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class LLMRequestORM(Base):
    __tablename__ = "llm_requests"

    request_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    prompt_version: Mapped[str] = mapped_column(
        ForeignKey("prompt_versions.version"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    context_manifest_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    output_hash: Mapped[str | None] = mapped_column(String(64))
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    validation_status: Mapped[str] = mapped_column(String(24), nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    related_run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), nullable=False)
    related_candidate_id: Mapped[str | None] = mapped_column(ForeignKey("candidates.candidate_id"))
    error_message_redacted: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MetadataSnapshotORM(Base):
    __tablename__ = "metadata_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(255), nullable=False)
    region: Mapped[str] = mapped_column(String(64), nullable=False)
    universe: Mapped[str] = mapped_column(String(128), nullable=False)
    delay: Mapped[int] = mapped_column(Integer, nullable=False)
    data_type: Mapped[str] = mapped_column(String(16), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict[str, Any] | list[Any]] = mapped_column(JSON, nullable=False)
    # Only ever populated for an OPERATORS-kind row, from BRAIN's own `/operators`
    # `definition` field (e.g. "group_rank(x, group)"). Deliberately kept out of
    # `payload_json`/`content_hash` -- those are the frozen snapshot identity, and
    # this is enrichment the generator reads, not part of what a snapshot IS.
    operator_definitions_json: Mapped[dict[str, str] | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
