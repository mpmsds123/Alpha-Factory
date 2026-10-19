"""Pydantic boundary contracts used by adapters, storage, and the CLI."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alpha_orchestrator.domain.enums import (
    AdapterMode,
    DataType,
    ErrorClass,
    PlatformCheckStatus,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunSpecification(StrictModel):
    run_id: UUID = Field(default_factory=uuid4)
    dataset_id: str = Field(min_length=1)
    region: str = Field(min_length=1)
    universe: str = Field(min_length=1)
    delay: int = Field(ge=0)
    data_type: DataType
    field_snapshot_id: str = Field(min_length=1)
    operator_snapshot_id: str = Field(min_length=1)
    setting_snapshot_id: str = Field(min_length=1)
    routing_policy_version: str = Field(min_length=1)
    limit_policy_version: str = Field(min_length=1)
    generator_prompt_version: str = Field(min_length=1)
    repair_prompt_version: str = Field(min_length=1)
    adapter_mode: AdapterMode
    created_at: datetime = Field(default_factory=utc_now)


class CandidateProposal(StrictModel):
    expression: str = Field(min_length=1)
    settings: dict[str, Any]
    mechanism_claim: str = Field(min_length=1, max_length=1_000)
    uncertainty: str | None = Field(default=None, min_length=1, max_length=1_000)
    source_method_card_ids: list[str] = Field(default_factory=list)
    parent_candidate_id: UUID | None = None
    generation_number: int = Field(default=0, ge=0)
    changed_dimension: str | None = None
    change_reason: str | None = None

    @model_validator(mode="after")
    def validate_lineage(self) -> "CandidateProposal":
        lineage_fields = (self.changed_dimension, self.change_reason)
        if self.parent_candidate_id is None:
            if any(value is not None for value in lineage_fields) or self.generation_number != 0:
                raise ValueError("Root candidates cannot declare lineage changes")
        elif not all(value and value.strip() for value in lineage_fields):
            raise ValueError("Child candidates require one changed dimension and one reason")
        elif self.generation_number < 1:
            raise ValueError("Child candidates must have generation_number >= 1")
        return self


class MetadataSnapshot(StrictModel):
    dataset_id: str
    fields: dict[str, DataType]
    operators: set[str]
    settings: dict[str, set[str | int | float | bool]]
    # BRAIN's own `/operators` response includes a `definition` string (e.g.
    # "group_rank(x, group)") for each operator, but frozen operator-snapshot
    # identity only ever hashed the bare name set -- adding the call signatures
    # here is purely additive so it does not change `operator_snapshot_id` for
    # any existing run. Without this, the generator's only source for an
    # operator's argument count was the LLM's own guess (see the 2026-09-15
    # incident: `group_rank(x)` called with 1 argument when it takes 2).
    operator_definitions: dict[str, str] = Field(default_factory=dict)


class CapturedMetadata(StrictModel):
    metadata: MetadataSnapshot
    region: str = Field(min_length=1)
    universe: str = Field(min_length=1)
    delay: int = Field(ge=0)
    data_type: DataType
    field_snapshot_id: str = Field(min_length=1)
    operator_snapshot_id: str = Field(min_length=1)
    setting_snapshot_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    captured_at: datetime = Field(default_factory=utc_now)


class SimulationMetrics(StrictModel):
    sharpe: float | None = None
    fitness: float | None = None
    turnover: float | None = None
    extra: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class CreateSimulationRequest(StrictModel):
    candidate_id: UUID
    expression: str
    settings: dict[str, Any]
    region: str
    universe: str
    delay: int
    idempotency_key: str


class CreateSimulationReceipt(StrictModel):
    external_simulation_id: str
    external_alpha_id: str | None = None


class NormalizedSimulationResult(StrictModel):
    completed: bool
    metrics: SimulationMetrics = Field(default_factory=SimulationMetrics)
    external_alpha_id: str | None = None
    error_class: ErrorClass | None = None
    error_message: str | None = None
    raw_artifact_ref: str | None = None


class PlatformCheck(StrictModel):
    name: str
    status: PlatformCheckStatus
    message: str | None = None


class AdapterFailure(Exception):
    """A normalized adapter failure that deterministic code can classify."""

    def __init__(
        self,
        error_class: ErrorClass,
        message: str,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.message = message
        self.retry_after_seconds = retry_after_seconds
