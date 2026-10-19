"""Strict LLM proposal schemas; deterministic code remains authoritative."""

from typing import Any

from pydantic import Field, model_validator

from alpha_orchestrator.domain.enums import RepairAction
from alpha_orchestrator.domain.models import CandidateProposal, StrictModel


class GeneratedCandidateProposal(CandidateProposal):
    uncertainty: str = Field(min_length=1, max_length=1_000)


class CandidateProposalBatch(StrictModel):
    proposals: list[GeneratedCandidateProposal] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def generator_outputs_roots_only(self) -> "CandidateProposalBatch":
        if any(proposal.parent_candidate_id is not None for proposal in self.proposals):
            raise ValueError("CandidateGenerator may propose root candidates only")
        return self


class RepairProposal(StrictModel):
    changed_dimension: str = Field(min_length=1, max_length=128)
    change_reason: str = Field(min_length=1, max_length=1_000)
    expected_effect: str = Field(min_length=1, max_length=1_000)
    revised_expression: str = Field(min_length=1)
    revised_settings: dict[str, Any]
    source_method_card_ids: list[str] = Field(default_factory=list)
    falsification_condition: str = Field(min_length=1, max_length=1_000)


class RepairPlan(StrictModel):
    action: RepairAction
    reason: str = Field(min_length=1, max_length=1_000)
    proposals: list[RepairProposal] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def action_matches_proposals(self) -> "RepairPlan":
        if self.action is RepairAction.STOP and self.proposals:
            raise ValueError("STOP cannot include repair proposals")
        if self.action is RepairAction.REPAIR and not self.proposals:
            raise ValueError("REPAIR requires at least one proposal")
        return self
