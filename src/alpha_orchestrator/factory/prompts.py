"""The four LLM decision points (plus prior seeding): prompt text and answer schemas.

Prompts are platform-generic. The expression language reaches the model at run time as
the live operator list plus the private language notes. A prompt's version carries a hash
of its text, so an edited prompt is a new version in the decision log.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alpha_orchestrator.factory.llm import Cited, Prompt
from alpha_orchestrator.factory.util import sha256_text

HYPOTHESIS_TYPES = (
    "level",  # the value itself is mispriced
    "change",  # its recent change or trend carries information
    "reversal",  # extremes revert
    "surprise",  # deviation from expectation, revision
    "dispersion",  # disagreement or uncertainty
    "stability",  # quality, persistence, low variability
    "conditional",  # works only in a stated state or around an event
    "relative",  # relative to peers (industry, size group)
)
HypothesisType = Literal[
    "level", "change", "reversal", "surprise", "dispersion", "stability", "conditional", "relative"
]
Direction = Literal["higher_value_higher_return", "higher_value_lower_return"]
TurnoverBand = Literal["low", "mid", "high"]  # <10%, 10-30%, 30-70% per day
TURNOVER_BANDS = {"low": (0.0, 0.10), "mid": (0.10, 0.30), "high": (0.30, 0.70)}
Holding = Literal[5, 21, 63, 126, 252]

SYSTEM = (
    "You are a senior quantitative equity researcher. A signal is computed every day for "
    "every stock; positive values are bought, negative values sold, and the book is made "
    "market neutral. You work one hypothesis at a time: an economic claim about one data "
    "field, tested with short expressions. Every expression must follow from the claim. "
    "Never tune numbers only to raise in-sample results, never add terms only to lower "
    "correlation, never invent fields or operators: use only those given in the input. "
    "Records (ids H..) are this factory's past hypotheses; priors (ids P..) are claims "
    "from documents with a confidence the factory updates from its own results. Cite in "
    "'cites' every id you relied on; if none applied, set no_relevant_record and say why. "
    "Answer with one JSON object that matches answer_format exactly."
)


def _prompt(step: str, task: str) -> Prompt:
    return Prompt(
        step=step, version=f"{step}.v1-{sha256_text(SYSTEM + task)[:8]}", system=SYSTEM, task=task
    )


# ---------- (a) hypothesis ----------
class HypothesisOut(Cited):
    claim: str = Field(
        min_length=40,
        description="the mechanism: who misprices what, why, and why it is not arbitraged away",
    )
    direction: Direction
    hypothesis_type: HypothesisType
    holding_days: Holding = Field(description="how long the effect should take to pay")
    turnover_band: TurnoverBand = Field(description="expected daily turnover: low/mid/high")
    falsifier: str = Field(min_length=10, description="the result that would refute the claim")
    helper_fields: list[str] = Field(
        default_factory=list,
        max_length=2,
        description="at most 2 fields from 'helper_candidates' the claim needs; usually none",
    )


HYPOTHESIS = _prompt(
    "hypothesis",
    "State one testable economic hypothesis about the field in 'field'. Say which "
    "direction should earn money, the kind of effect, the holding period and the "
    "turnover that follows from it, and what result would prove it wrong. Use the "
    "records and priors: do not repeat a claim that already failed on similar data "
    "unless you say what is different.",
)


# ---------- (b) expression ----------
class ExpressionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(min_length=1)
    rationale: str = Field(min_length=10, description="how this expression tests the claim")


class ExpressionOut(Cited):
    expressions: list[ExpressionItem] = Field(min_length=1, max_length=3)


EXPRESSION = _prompt(
    "expression",
    "Write 1 to 3 expressions that test the hypothesis, simplest first. Each must use the "
    "hypothesis field, may use only the listed fields and operators, must respect the "
    "limits in 'limits', and must enter the field with the stated direction (higher "
    "value long, or higher value short). Windows must come from 'limits.windows' and "
    "should fit the holding period. Different expressions should implement the claim in "
    "different ways, not differ only in a window.",
)


# ---------- (c) diagnosis ----------
class DiagnosisOut(Cited):
    label: Literal["idea_supported", "idea_failed", "implementation_failed"]
    reason: str = Field(min_length=20)


DIAGNOSIS = _prompt(
    "diagnosis",
    "Compare the round's results with what the hypothesis predicted. 'checks' holds the "
    "program's own comparisons (sign, turnover band, noise line). Decide: "
    "idea_supported = the predicted signal is there; idea_failed = the expressions "
    "tested the claim properly and it did not hold; implementation_failed = the claim "
    "was not tested properly (wrong sign handling, data gaps, a window far from the "
    "holding period, turnover far outside the expected band). Results are from the "
    "training period only.",
)


# ---------- (d) decision ----------
class Mutation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parent: int = Field(description="expression id the change starts from")
    change: Literal["window", "operator", "field", "setting", "rewrite"]
    expression: str = Field(description="the full new expression (same as parent for 'setting')")
    decay: int | None = Field(default=None, description="only for change='setting'")
    neutralization: str | None = Field(default=None, description="only for change='setting'")
    reason: str = Field(min_length=10)


class Revision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=40)
    direction: Direction
    hypothesis_type: HypothesisType
    holding_days: Holding
    turnover_band: TurnoverBand
    falsifier: str = Field(min_length=10)


class DecisionOut(Cited):
    action: Literal["mutate", "revise", "drop"]
    reason: str = Field(min_length=10)
    mutations: list[Mutation] = Field(default_factory=list, max_length=3)
    revision: Revision | None = None
    lesson: str = Field(
        min_length=10, description="one line a future researcher should know from this"
    )

    @model_validator(mode="after")
    def _consistent(self) -> "DecisionOut":
        if self.action == "mutate" and not self.mutations:
            raise ValueError("action 'mutate' needs at least one mutation")
        if self.action == "revise" and self.revision is None:
            raise ValueError("action 'revise' needs a revision")
        return self


DECISION = _prompt(
    "decision",
    "Choose the next step for this hypothesis. 'mutate': up to the number of children in "
    "'limits', each exactly ONE declared change to a listed parent (window, operator, "
    "field from the same family, setting from 'setting_options', or rewrite), with its "
    "reason; the program rejects a child whose real change differs from the declared one. "
    "'revise': the claim itself was wrong in a way the results show; state the new claim "
    "(it gets its own, smaller budget). 'drop': nothing promising is left. Always write "
    "a one-line lesson.",
)


# ---------- prior seeding ----------
class PriorClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=20, description="a general, testable research claim")
    quote: str = Field(min_length=10, description="verbatim words from the document")
    categories: list[str] = Field(default_factory=list, description="dataset categories")
    hypothesis_types: list[HypothesisType] = Field(default_factory=list)


class PriorSeedOut(Cited):
    claims: list[PriorClaim] = Field(default_factory=list, max_length=5)


PRIOR_SEED = _prompt(
    "prior_seed",
    "Extract at most 5 general research claims from the document: statements about what "
    "kind of signal works, on which kind of data, and why. Each needs a verbatim quote "
    "from the document. No formulas, no field names. Return an empty list when the "
    "document has no such claim.",
)

ALL = {p.step: p for p in (HYPOTHESIS, EXPRESSION, DIAGNOSIS, DECISION, PRIOR_SEED)}
