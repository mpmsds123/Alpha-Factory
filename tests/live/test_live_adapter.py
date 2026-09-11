import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import respx

from alpha_orchestrator.adapters.live import (
    LiveBrainAdapter,
    LiveBrainClient,
    LiveBrainSettings,
    load_session_cookies,
)
from alpha_orchestrator.domain.enums import (
    AdapterMode,
    DataType,
    ErrorClass,
    PlatformCheckStatus,
)
from alpha_orchestrator.domain.models import AdapterFailure, CreateSimulationRequest


def session_file(tmp_path: Path, payload: object | None = None) -> Path:
    path = tmp_path / "brain-session.json"
    value = {"cookies": {"sessionid": "test-only"}} if payload is None else payload
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return path


def settings(tmp_path: Path, **overrides: object) -> LiveBrainSettings:
    values: dict[str, object] = {
        "session_file": session_file(tmp_path),
        "base_url": "https://test.worldquantbrain.com",
        "maximum_retry_after_seconds": 10,
    }
    values.update(overrides)
    return LiveBrainSettings.model_validate(values)


def simulation_request() -> CreateSimulationRequest:
    return CreateSimulationRequest(
        candidate_id=uuid4(),
        expression="rank(close)",
        settings={"neutralization": "SUBINDUSTRY", "truncation": 0.08},
        region="USA",
        universe="TOP1000",
        delay=1,
        idempotency_key="stable-test-key",
    )


def test_session_cookie_formats_and_permissions(tmp_path: Path) -> None:
    mapping = session_file(tmp_path)
    assert load_session_cookies(mapping) == {"sessionid": "test-only"}
    cookie_list = session_file(
        tmp_path,
        [{"name": "sessionid", "value": "list-value", "domain": "brain.example"}],
    )
    assert load_session_cookies(cookie_list) == {"sessionid": "list-value"}

    cookie_list.chmod(0o644)
    with pytest.raises(ValueError, match="group or other"):
        load_session_cookies(cookie_list)
    with pytest.raises(ValueError, match="does not exist"):
        load_session_cookies(tmp_path / "missing.json")


@pytest.mark.parametrize(
    "payload",
    [[], {"cookies": {}}, {"cookies": {"sessionid": 1}}, [{"name": "sessionid"}]],
)
def test_session_cookie_rejects_invalid_payloads(tmp_path: Path, payload: object) -> None:
    path = session_file(tmp_path, payload)
    with pytest.raises(ValueError, match=r"cookie|usable"):
        load_session_cookies(path)


def test_live_settings_require_explicit_safe_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ALPHA_LAB_BRAIN_SESSION_FILE", raising=False)
    with pytest.raises(ValueError, match="SESSION_FILE"):
        LiveBrainSettings.from_environment()
    monkeypatch.setenv("ALPHA_LAB_BRAIN_SESSION_FILE", str(session_file(tmp_path)))
    monkeypatch.setenv("ALPHA_LAB_BRAIN_BASE_URL", "https://test.worldquantbrain.com/")
    configured = LiveBrainSettings.from_environment()
    assert configured.base_url == "https://test.worldquantbrain.com"

    with pytest.raises(ValueError, match="HTTPS"):
        settings(tmp_path, base_url="http://api.worldquantbrain.com")
    with pytest.raises(ValueError, match="credentials"):
        settings(tmp_path, base_url="https://user:secret@api.worldquantbrain.com")
    with pytest.raises(ValueError, match="official"):
        settings(tmp_path, base_url="https://brain.example")
    with pytest.raises(ValueError, match="absolute paths"):
        settings(tmp_path, simulations_path="https://other.example/simulations")


@respx.mock
def test_intercepted_low_volume_live_smoke_flow(tmp_path: Path) -> None:
    authentication = respx.get("https://test.worldquantbrain.com/authentication").mock(
        return_value=httpx.Response(200, json={"authenticated": True})
    )
    fields = respx.get("https://test.worldquantbrain.com/data-fields").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"id": "close", "type": "MATRIX"},
                    {"id": "vector_news", "type": "VECTOR"},
                ]
            },
        )
    )
    operators = respx.get("https://test.worldquantbrain.com/operators").mock(
        return_value=httpx.Response(200, json={"results": [{"name": "rank"}]})
    )
    setting_route = respx.get("https://test.worldquantbrain.com/settings").mock(
        return_value=httpx.Response(
            200,
            json={"settings": {"neutralization": ["SUBINDUSTRY"], "truncation": [0.08]}},
        )
    )
    create = respx.post("https://test.worldquantbrain.com/simulations").mock(
        return_value=httpx.Response(
            201,
            headers={"Location": "https://test.worldquantbrain.com/simulations/sim-1"},
        )
    )
    poll = respx.get("https://test.worldquantbrain.com/simulations/sim-1").mock(
        return_value=httpx.Response(200, json={"status": "COMPLETE", "alpha": "alpha-1"})
    )
    alpha = respx.get("https://test.worldquantbrain.com/alphas/alpha-1").mock(
        return_value=httpx.Response(
            200,
            json={
                "is": {
                    "sharpe": 0.8,
                    "fitness": 0.5,
                    "turnover": 0.2,
                    "returns": 0.1,
                }
            },
        )
    )
    check = respx.get("https://test.worldquantbrain.com/alphas/alpha-1/check").mock(
        return_value=httpx.Response(
            200,
            json={
                "is": {
                    "checks": [
                        {"name": "LOW_SHARPE", "result": "PASS"},
                        {"name": "WEIGHT", "result": "WARNING", "message": "Review"},
                    ]
                }
            },
        )
    )

    client = LiveBrainClient(settings(tmp_path))
    capture = client.capture_metadata(
        dataset_id="demo-dataset",
        region="USA",
        universe="TOP1000",
        delay=1,
        data_type=DataType.MATRIX,
    )
    receipt = client.create_simulation(simulation_request())
    result = client.poll_simulation(receipt.external_simulation_id)
    checks = client.request_platform_checks(result.external_alpha_id or "")
    adapter = LiveBrainAdapter(client, capture.metadata)

    assert capture.field_snapshot_id.startswith("live-fields-")
    assert capture.metadata.fields["vector_news"] is DataType.VECTOR
    assert receipt.external_simulation_id == "sim-1"
    assert result.completed and result.metrics.sharpe == 0.8
    assert result.external_alpha_id == "alpha-1"
    assert [item.status for item in checks] == [
        PlatformCheckStatus.PASS,
        PlatformCheckStatus.WARNING,
    ]
    assert adapter.mode is AdapterMode.LIVE
    assert adapter.initial_candidates() == []
    assert not hasattr(client, "submit")
    assert create.calls[0].request.headers["Idempotency-Key"] == "stable-test-key"
    assert json.loads(create.calls[0].request.content)["regular"] == "rank(close)"
    assert fields.calls[0].request.url.params["dataset.id"] == "demo-dataset"
    assert all(
        route.called
        for route in (authentication, fields, operators, setting_route, poll, alpha, check)
    )
    assert len(respx.calls) == 8
    client.close()


@respx.mock
def test_rate_limit_is_classified_and_wait_is_capped(tmp_path: Path) -> None:
    respx.get("https://test.worldquantbrain.com/authentication").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.get("https://test.worldquantbrain.com/data-sets").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "120"}, json={"detail": "slow"})
    )
    waits: list[float] = []
    client = LiveBrainClient(settings(tmp_path), sleeper=waits.append)
    with pytest.raises(AdapterFailure) as caught:
        client.list_datasets(region="USA", universe="TOP1000", delay=1)
    assert caught.value.error_class is ErrorClass.RATE_LIMIT
    assert caught.value.retry_after_seconds == 120
    assert waits == [10]


@respx.mock
def test_persona_challenge_is_not_treated_as_bad_research(tmp_path: Path) -> None:
    respx.get("https://test.worldquantbrain.com/authentication").mock(
        return_value=httpx.Response(401, json={"inquiry": {"id": "test"}})
    )
    client = LiveBrainClient(settings(tmp_path))
    with pytest.raises(AdapterFailure) as caught:
        client.authenticate()
    assert caught.value.error_class is ErrorClass.HUMAN_VERIFICATION_REQUIRED


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (httpx.ReadTimeout("timed out"), ErrorClass.TIMEOUT),
        (httpx.ConnectError("offline"), ErrorClass.NETWORK),
    ],
)
def test_transport_failures_are_normalized(
    tmp_path: Path, raised: Exception, expected: ErrorClass
) -> None:
    def fail(_request: httpx.Request) -> httpx.Response:
        raise raised

    transport = httpx.MockTransport(fail)
    http_client = httpx.Client(transport=transport)
    client = LiveBrainClient(settings(tmp_path), http_client=http_client)
    with pytest.raises(AdapterFailure) as caught:
        client.authenticate()
    assert caught.value.error_class is expected


@respx.mock
def test_pending_terminal_and_unknown_results_remain_distinct(tmp_path: Path) -> None:
    respx.get("https://test.worldquantbrain.com/authentication").mock(
        return_value=httpx.Response(200, json={})
    )
    route = respx.get("https://test.worldquantbrain.com/simulations/sim-1")
    route.side_effect = [
        httpx.Response(200, json={"status": "RUNNING"}),
        httpx.Response(200, json={"status": "FAILED", "message": "invalid field"}),
        httpx.Response(200, json={"status": "NEW_STATE"}),
    ]
    client = LiveBrainClient(settings(tmp_path))
    pending = client.poll_simulation("sim-1")
    failed = client.poll_simulation("sim-1")
    unknown = client.poll_simulation("sim-1")
    assert pending.error_class is ErrorClass.PENDING
    assert failed.error_class is ErrorClass.INVALID_FIELD
    assert unknown.error_class is ErrorClass.UNKNOWN


def test_unsafe_external_identifiers_are_rejected(tmp_path: Path) -> None:
    client = LiveBrainClient(settings(tmp_path))
    with pytest.raises(AdapterFailure, match="unsafe"):
        client.poll_simulation("../authentication")
    with pytest.raises(AdapterFailure, match="unsafe"):
        client.request_platform_checks("alpha/other")
