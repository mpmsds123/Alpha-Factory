from pathlib import Path

import pytest
from pydantic import ValidationError

from alpha_orchestrator.config import (
    LimitPolicy,
    load_error_taxonomy,
    load_limit_policy,
    load_routing_policy,
)
from alpha_orchestrator.domain.enums import DataType, ErrorClass, GateOutcome
from alpha_orchestrator.domain.models import MetadataSnapshot, SimulationMetrics
from alpha_orchestrator.gates.limits import LimitController, UsageSnapshot
from alpha_orchestrator.gates.routing import route_completed_result
from alpha_orchestrator.gates.structural import StructuralValidator

ROOT = Path(__file__).parents[2]


def test_versioned_policies_load() -> None:
    routing = load_routing_policy(ROOT / "policies" / "routing.v1.yaml")
    limits = load_limit_policy(ROOT / "policies" / "limits.v1.yaml")
    taxonomy = load_error_taxonomy(ROOT / "policies" / "error-taxonomy.v1.yaml")
    assert routing.version == "routing.v1"
    assert limits.maximum_pending_candidates == 20
    assert "PENDING" in taxonomy.unresolved


def test_routing_v2_drops_the_unsourced_turnover_discard() -> None:
    v1 = load_routing_policy(ROOT / "policies" / "routing.v1.yaml")
    v2 = load_routing_policy(ROOT / "policies" / "routing.v2.yaml")
    assert v1.discard is not None and v1.discard.maximum_turnover == 0.90
    assert v2.version == "routing.v2"
    assert v2.discard is None


def test_invalid_yaml_and_unsafe_limit_policy_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_limit_policy(path)
    with pytest.raises(ValidationError, match="Crossover"):
        LimitPolicy(
            version="unsafe",
            initial_candidates=1,
            maximum_concurrent_simulations=10,
            children_per_parent_per_generation=1,
            maximum_generations=1,
            maximum_distinct_candidates=1,
            maximum_simulation_result_rows=1,
            maximum_pending_candidates=1,
            maximum_llm_requests=0,
            transient_retries_per_external_request=0,
            structural_repairs_per_candidate=0,
            maximum_wall_clock_seconds=1,
            crossover_enabled=True,
            automatic_submission_available=False,
        )


def test_structural_validator_accepts_snapshot_compatible_expression(
    metadata: MetadataSnapshot,
) -> None:
    result = StructuralValidator().validate(
        "rank(ts_mean(close, 5))",
        {"neutralization": "SUBINDUSTRY", "truncation": 0.08},
        metadata,
        DataType.MATRIX,
    )
    assert result.valid
    assert result.issues == []


def test_structural_validator_accepts_group_classification_keyword(
    metadata: MetadataSnapshot,
) -> None:
    """group_neutralize's second argument is a built-in classification keyword
    (industry, sector, ...), never a dataset field -- it must not be flagged as an
    unknown field. A live batch on 2026-09-16 rejected six otherwise-valid candidates
    on exactly this false positive before this allowlist existed."""
    snapshot = metadata.model_copy(update={"operators": metadata.operators | {"group_neutralize"}})
    result = StructuralValidator().validate(
        "group_neutralize(rank(close), industry)",
        {"neutralization": "SUBINDUSTRY", "truncation": 0.08},
        snapshot,
        DataType.MATRIX,
    )
    assert result.valid
    assert result.issues == []


@pytest.mark.parametrize(
    ("expression", "settings", "data_type", "expected"),
    [
        ("rank({{field}})", {}, DataType.MATRIX, ErrorClass.INVALID_EXPRESSION),
        ("rank(close", {}, DataType.MATRIX, ErrorClass.INVALID_EXPRESSION),
        ("mystery(close)", {}, DataType.MATRIX, ErrorClass.INVALID_OPERATOR),
        ("rank(missing)", {}, DataType.MATRIX, ErrorClass.INVALID_FIELD),
        ("rank(vector_news)", {}, DataType.MATRIX, ErrorClass.INVALID_FIELD),
        ("rank(close)", {"neutralization": "BAD"}, DataType.MATRIX, ErrorClass.INVALID_SETTING),
        ("rank(close)@", {}, DataType.MATRIX, ErrorClass.INVALID_EXPRESSION),
    ],
)
def test_structural_validator_rejects_known_failure_classes(
    metadata: MetadataSnapshot,
    expression: str,
    settings: dict[str, object],
    data_type: DataType,
    expected: ErrorClass,
) -> None:
    result = StructuralValidator().validate(expression, settings, metadata, data_type)
    assert not result.valid
    assert expected in {issue.code for issue in result.issues}


def test_routing_ignores_turnover_and_is_otherwise_deterministic(routing_policy: object) -> None:
    """v2 dropped the turnover-based discard (routing.v1's discard.maximum_turnover
    had no source and was checked before sharpe/fitness were even read); a high
    turnover with strong sharpe/fitness must now route STRONG, not be discarded."""
    policy = routing_policy
    assert route_completed_result(None, policy)[0] is GateOutcome.DISCARD
    assert (
        route_completed_result(SimulationMetrics(sharpe=2.0, fitness=1.0, turnover=0.91), policy)[0]
        is GateOutcome.STRONG
    )
    assert (
        route_completed_result(SimulationMetrics(sharpe=0.5, fitness=0.3, turnover=0.2), policy)[0]
        is GateOutcome.STRONG
    )
    assert (
        route_completed_result(SimulationMetrics(sharpe=0.25, fitness=0.01, turnover=0.2), policy)[
            0
        ]
        is GateOutcome.WEAK
    )
    assert (
        route_completed_result(SimulationMetrics(sharpe=0.01, fitness=0.01, turnover=0.2), policy)[
            0
        ]
        is GateOutcome.DISCARD
    )


@pytest.mark.parametrize(
    ("usage", "reason"),
    [
        (UsageSnapshot(distinct_candidates=2), "MAXIMUM_DISTINCT_CANDIDATES_REACHED"),
        (
            UsageSnapshot(simulation_result_rows=2),
            "MAXIMUM_SIMULATION_RESULT_ROWS_REACHED",
        ),
        (UsageSnapshot(pending_candidates=2), "MAXIMUM_PENDING_CANDIDATES_REACHED"),
        (UsageSnapshot(llm_requests=2), "MAXIMUM_LLM_REQUESTS_REACHED"),
        (UsageSnapshot(elapsed_seconds=2), "MAXIMUM_WALL_CLOCK_REACHED"),
    ],
)
def test_each_limit_has_an_exact_stop_reason(
    limit_policy: LimitPolicy, usage: UsageSnapshot, reason: str
) -> None:
    policy = limit_policy.model_copy(
        update={
            "maximum_distinct_candidates": 2,
            "maximum_simulation_result_rows": 2,
            "maximum_pending_candidates": 2,
            "maximum_llm_requests": 2,
            "maximum_wall_clock_seconds": 2,
        }
    )
    assert LimitController(policy).stop_reason(usage) == reason


def test_queue_backpressure(limit_policy: LimitPolicy) -> None:
    policy = limit_policy.model_copy(update={"maximum_pending_candidates": 2})
    controller = LimitController(policy)
    assert controller.can_enqueue(1)
    assert not controller.can_enqueue(2)
    assert controller.stop_reason(UsageSnapshot()) is None
