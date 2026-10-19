from collections.abc import Callable
from pathlib import Path

import pytest

from alpha_orchestrator.config import (
    CorrelationPolicy,
    LimitPolicy,
    RoutingPolicy,
    load_correlation_policy,
    load_limit_policy,
    load_routing_policy,
)
from alpha_orchestrator.domain.enums import AdapterMode, DataType
from alpha_orchestrator.domain.models import MetadataSnapshot, RunSpecification
from alpha_orchestrator.storage.database import Database
from alpha_orchestrator.storage.repository import Repository

ROOT = Path(__file__).parents[1]


@pytest.fixture
def routing_policy() -> RoutingPolicy:
    return load_routing_policy(ROOT / "policies" / "routing.v2.yaml")


@pytest.fixture
def limit_policy() -> LimitPolicy:
    return load_limit_policy(ROOT / "policies" / "limits.v2.yaml")


@pytest.fixture
def correlation_policy() -> CorrelationPolicy:
    return load_correlation_policy(ROOT / "policies" / "correlation.v1.yaml")


@pytest.fixture
def database() -> Database:
    value = Database("sqlite://")
    value.create_schema()
    yield value
    value.dispose()


@pytest.fixture
def repository(database: Database) -> Repository:
    return Repository(database.session_factory)


@pytest.fixture
def metadata() -> MetadataSnapshot:
    return MetadataSnapshot(
        dataset_id="demo-dataset",
        fields={
            "close": DataType.MATRIX,
            "volume": DataType.MATRIX,
            "vector_news": DataType.VECTOR,
        },
        operators={"rank", "ts_mean", "subtract"},
        settings={"neutralization": {"SUBINDUSTRY"}, "truncation": {0.08}},
    )


@pytest.fixture
def make_run(
    repository: Repository,
    routing_policy: RoutingPolicy,
    limit_policy: LimitPolicy,
) -> Callable[..., RunSpecification]:
    def factory(**overrides: object) -> RunSpecification:
        selected_routing = overrides.pop("_routing_policy", routing_policy)
        selected_limits = overrides.pop("_limit_policy", limit_policy)
        assert isinstance(selected_routing, RoutingPolicy)
        assert isinstance(selected_limits, LimitPolicy)
        values: dict[str, object] = {
            "dataset_id": "demo-dataset",
            "region": "USA",
            "universe": "TOP1000",
            "delay": 1,
            "data_type": DataType.MATRIX,
            "field_snapshot_id": "fields.v1",
            "operator_snapshot_id": "operators.v1",
            "setting_snapshot_id": "settings.v1",
            "routing_policy_version": selected_routing.version,
            "limit_policy_version": selected_limits.version,
            "generator_prompt_version": "not-applicable-milestone-1",
            "repair_prompt_version": "not-applicable-milestone-1",
            "adapter_mode": AdapterMode.REPLAY,
        }
        values.update(overrides)
        specification = RunSpecification.model_validate(values)
        repository.create_run(specification, selected_routing, selected_limits)
        return specification

    return factory
