"""Versioned policy loading."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StrongThresholds(PolicyModel):
    minimum_sharpe: float
    minimum_fitness: float


class WeakThresholds(PolicyModel):
    minimum_sharpe: float
    minimum_fitness: float


class DiscardThresholds(PolicyModel):
    maximum_turnover: float


class RoutingPolicy(PolicyModel):
    version: str
    strong: StrongThresholds
    weak: WeakThresholds
    # Only ever present on routing.v1 (retained for that version's own historical
    # loadability/audit trail). route_completed_result no longer reads this field
    # regardless of whether a policy happens to carry it -- see routing.v2.yaml
    # for why the turnover-based discard was removed rather than re-guessed.
    discard: DiscardThresholds | None = None


class LimitPolicy(PolicyModel):
    version: str
    initial_candidates: int = Field(gt=0)
    maximum_concurrent_simulations: int = Field(gt=0, le=10)
    children_per_parent_per_generation: int = Field(gt=0)
    maximum_generations: int = Field(gt=0)
    maximum_distinct_candidates: int = Field(gt=0)
    maximum_simulation_result_rows: int = Field(gt=0)
    maximum_pending_candidates: int = Field(gt=0)
    maximum_llm_requests: int = Field(ge=0)
    transient_retries_per_external_request: int = Field(ge=0)
    structural_repairs_per_candidate: int = Field(ge=0)
    maximum_wall_clock_seconds: int = Field(gt=0)
    crossover_enabled: bool
    automatic_submission_available: bool

    @model_validator(mode="after")
    def enforce_safety_boundary(self) -> "LimitPolicy":
        if self.crossover_enabled:
            raise ValueError("Crossover is unavailable in the MVP")
        if self.automatic_submission_available:
            raise ValueError("Automatic submission is unavailable in the MVP")
        return self


class ErrorTaxonomy(PolicyModel):
    version: str
    retryable: list[str]
    terminal: list[str]
    unresolved: list[str]


class CorrelationPolicy(PolicyModel):
    version: str
    maximum_correlation: float = Field(gt=0.0, lt=1.0)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Policy must be a mapping: {path}")
    return value


def load_routing_policy(path: Path) -> RoutingPolicy:
    return RoutingPolicy.model_validate(_load_yaml(path))


def load_limit_policy(path: Path) -> LimitPolicy:
    return LimitPolicy.model_validate(_load_yaml(path))


def load_error_taxonomy(path: Path) -> ErrorTaxonomy:
    return ErrorTaxonomy.model_validate(_load_yaml(path))


def load_correlation_policy(path: Path) -> CorrelationPolicy:
    return CorrelationPolicy.model_validate(_load_yaml(path))
