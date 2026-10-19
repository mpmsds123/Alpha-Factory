import json
import logging
from pathlib import Path
from uuid import uuid4

import pytest

from alpha_orchestrator.adapters.fake import FakeBrainAdapter
from alpha_orchestrator.adapters.replay import ReplayBrainAdapter
from alpha_orchestrator.audit_logging import AuditLogger, sanitize
from alpha_orchestrator.domain.enums import DataType, ErrorClass
from alpha_orchestrator.domain.models import (
    AdapterFailure,
    CandidateProposal,
    CreateSimulationRequest,
    MetadataSnapshot,
)

ROOT = Path(__file__).parents[2]


def request(expression: str = "rank(close)") -> CreateSimulationRequest:
    return CreateSimulationRequest(
        candidate_id=uuid4(),
        expression=expression,
        settings={},
        region="USA",
        universe="TOP1000",
        delay=1,
        idempotency_key="stable-key",
    )


def test_fake_adapter_is_idempotent_and_deterministic() -> None:
    metadata = MetadataSnapshot(
        dataset_id="demo-dataset",
        fields={"close": DataType.MATRIX},
        operators={"rank"},
        settings={},
    )
    proposal = CandidateProposal(
        expression="rank(close)", settings={}, mechanism_claim="Synthetic claim"
    )
    adapter = FakeBrainAdapter(metadata, [proposal])
    receipt = adapter.create_simulation(request())
    assert adapter.create_simulation(request()) == receipt
    assert adapter.create_call_count == 1
    assert adapter.poll_simulation(receipt.external_simulation_id).completed
    assert adapter.initial_candidates() == [proposal]
    assert adapter.request_platform_checks("candidate") == []


def test_fake_adapter_exposes_normalized_failures() -> None:
    metadata = MetadataSnapshot(dataset_id="demo-dataset", fields={}, operators=set(), settings={})
    adapter = FakeBrainAdapter(metadata, create_failures={"rank(close)": [ErrorClass.TIMEOUT]})
    with pytest.raises(AdapterFailure) as caught:
        adapter.create_simulation(request())
    assert caught.value.error_class is ErrorClass.TIMEOUT
    with pytest.raises(AdapterFailure, match="Unknown"):
        adapter.poll_simulation("missing")


def test_replay_adapter_loads_fixture_without_network() -> None:
    path = ROOT / "tests" / "fixtures" / "basic_replay.json"
    adapter = ReplayBrainAdapter.from_path(path)
    proposal = adapter.initial_candidates()[0]
    replay_request = request(proposal.expression).model_copy(update={"settings": proposal.settings})
    receipt = adapter.create_simulation(replay_request)
    assert adapter.poll_simulation(receipt.external_simulation_id).metrics.sharpe == 0.70
    assert adapter.create_simulation(replay_request) == receipt
    assert adapter.create_call_count == 1
    assert adapter.fixture_path == path
    assert adapter.request_platform_checks("missing") == []
    missing_request = request("rank(missing)").model_copy(update={"idempotency_key": "other"})
    with pytest.raises(AdapterFailure, match="missing"):
        adapter.create_simulation(missing_request)
    with pytest.raises(AdapterFailure, match="missing"):
        adapter.poll_simulation("replay-missing")


def test_audit_logger_recursively_redacts(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="alpha_orchestrator.audit")
    AuditLogger().event(
        "safe_event",
        api_key="secret-value",
        nested={"password": "another-secret", "message": "token=hidden"},
        values=("safe", "Bearer abc123"),
    )
    payload = json.loads(caplog.records[-1].message)
    assert payload["api_key"] == "[REDACTED]"
    assert payload["nested"]["password"] == "[REDACTED]"
    assert "hidden" not in payload["nested"]["message"]
    assert "abc123" not in payload["values"][1]
    assert sanitize(3) == 3
