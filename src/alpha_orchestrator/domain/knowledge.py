"""Knowledge-corpus and human-approved method-card contracts."""

from datetime import datetime

from pydantic import Field, model_validator

from alpha_orchestrator.domain.enums import DataType, EvidenceLevel, MethodCardStatus
from alpha_orchestrator.domain.models import StrictModel


class SourceReference(StrictModel):
    source_id: str = Field(min_length=1)
    section_locator: str = Field(min_length=1)


class MethodCard(StrictModel):
    card_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    title: str = Field(min_length=1, max_length=240)
    version: str = Field(min_length=1, max_length=64)
    status: MethodCardStatus = MethodCardStatus.DRAFT
    applicable_failure_codes: list[str] = Field(default_factory=list)
    applicable_data_types: list[DataType] = Field(min_length=1)
    prerequisites: list[str] = Field(min_length=1)
    diagnostic_signals: list[str] = Field(min_length=1)
    allowed_repair_actions: list[str] = Field(min_length=1)
    prohibited_actions: list[str] = Field(min_length=1)
    stop_condition: str = Field(min_length=1)
    evidence_level: EvidenceLevel
    source_references: list[SourceReference] = Field(min_length=1)
    reviewer: str | None = None
    approval_timestamp: datetime | None = None

    @model_validator(mode="after")
    def approval_is_complete(self) -> "MethodCard":
        # A draft has had no human decision yet, so it must carry neither field.
        # Approved and retired both record an explicit human decision -- retiring
        # a superseded version is as much a reviewer action as approving its
        # replacement, and losing who-did-it-when there would be its own small
        # loss of auditability.
        has_reviewer_record = bool(self.reviewer and self.approval_timestamp)
        if self.status is MethodCardStatus.DRAFT and has_reviewer_record:
            raise ValueError("Draft cards must not carry a reviewer or approval timestamp")
        if self.status is not MethodCardStatus.DRAFT and not has_reviewer_record:
            raise ValueError(f"{self.status.title()} cards require reviewer and approval timestamp")
        return self

    def prompt_summary(self) -> dict[str, object]:
        """Return paraphrased card content without private source text."""
        return {
            "card_id": self.card_id,
            "title": self.title,
            "version": self.version,
            "applicable_failure_codes": self.applicable_failure_codes,
            "applicable_data_types": self.applicable_data_types,
            "prerequisites": self.prerequisites,
            "diagnostic_signals": self.diagnostic_signals,
            "allowed_repair_actions": self.allowed_repair_actions,
            "prohibited_actions": self.prohibited_actions,
            "stop_condition": self.stop_condition,
            "evidence_level": self.evidence_level,
        }


class SourceSection(StrictModel):
    section_id: str
    heading: str
    section_locator: str
    content_hash: str


class SourceDocument(StrictModel):
    source_id: str
    corpus_root_locator: str
    relative_locator: str
    title: str
    source_type: EvidenceLevel
    content_hash: str
    excluded: bool = False
    exclusion_reason: str | None = None
    sections: list[SourceSection] = Field(default_factory=list)
