"""An optional experiment path. The model proposes; an approved protocol owns the verdict."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from hashlib import sha256
from math import isfinite
from statistics import fmean
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from alpha_orchestrator.factory.llm import FactoryLLM

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: Text
    competing_explanation: Text
    evidence_ids: list[Text] = Field(min_length=1)
    candidate: Text
    benchmark: Text
    direction: Literal["higher", "lower"]
    falsifier: Text

    @model_validator(mode="after")
    def distinct_series(self) -> Proposal:
        if self.candidate == self.benchmark:
            raise ValueError("candidate and benchmark must be distinct")
        return self


def proposal_digest(proposal: Proposal) -> str:
    return sha256(proposal.model_dump_json().encode()).hexdigest()


class Protocol(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    proposal_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    approved_on: date
    start: date
    end: date
    min_observations: int = Field(ge=4)
    min_signed_mean: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered_dates(self) -> Protocol:
        if not self.approved_on < self.start <= self.end:
            raise ValueError("protocol must be approved before its evaluation window")
        return self


def propose(llm: FactoryLLM, evidence: Mapping[str, str], allowed_series: list[str]) -> Proposal:
    proposal = Proposal.model_validate(llm.experiment_proposal(dict(evidence), allowed_series))
    if set(proposal.evidence_ids) - evidence.keys():
        raise ValueError("proposal cited unknown evidence")
    if {proposal.candidate, proposal.benchmark} - set(allowed_series):
        raise ValueError("proposal named an unavailable series")
    return proposal


def replay(
    proposal: Proposal, protocol: Protocol, series: Mapping[str, Mapping[date, float]]
) -> dict[str, str | int | list[float]]:
    """Descriptive two-block check on paired changes in cumulative daily PnL."""
    if protocol.proposal_digest != proposal_digest(proposal):
        raise ValueError("proposal differs from the frozen protocol")
    if proposal.candidate not in series or proposal.benchmark not in series:
        return {"verdict": "insufficient_evidence", "reason": "missing series"}
    candidate, benchmark = series[proposal.candidate], series[proposal.benchmark]
    a = {day for day in candidate if protocol.start <= day <= protocol.end}
    b = {day for day in benchmark if protocol.start <= day <= protocol.end}
    # Both series need the same closing observation immediately before the window.
    prev_a = max((day for day in candidate if day < protocol.start), default=None)
    prev_b = max((day for day in benchmark if day < protocol.start), default=None)
    if a != b or prev_a is None or prev_a != prev_b or len(a) < protocol.min_observations:
        return {"verdict": "insufficient_evidence", "reason": "unmatched or too few observations"}
    sign = 1 if proposal.direction == "higher" else -1
    prior = prev_a
    effects = []
    for day in sorted(a):
        effect = sign * ((candidate[day] - candidate[prior]) - (benchmark[day] - benchmark[prior]))
        if not isfinite(effect):
            return {"verdict": "insufficient_evidence", "reason": "non-finite result"}
        effects.append(effect)
        prior = day
    midpoint = len(effects) // 2
    means = [fmean(effects[:midpoint]), fmean(effects[midpoint:])]
    if min(means) >= protocol.min_signed_mean:
        verdict = "supports"
    elif max(means) <= 0:
        verdict = "conflicts"
    else:
        verdict = "insufficient_evidence"
    return {"verdict": verdict, "observations": len(effects), "signed_block_means": means}
