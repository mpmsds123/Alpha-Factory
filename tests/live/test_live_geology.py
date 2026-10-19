"""Tests pinning the live adapter to sanitized request and response shapes.

These fixtures keep adapter assumptions testable without credentials, private research
data, or authenticated external calls.
"""

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
)
from alpha_orchestrator.domain.enums import AdapterMode, DataType, ErrorClass
from alpha_orchestrator.domain.models import AdapterFailure, CandidateProposal, MetadataSnapshot
from alpha_orchestrator.seeding import TemplateError, expand_templates, load_templates

BASE = "https://test.worldquantbrain.com"


def _settings(tmp_path: Path) -> LiveBrainSettings:
    path = tmp_path / "brain-session.json"
    path.write_text(json.dumps({"cookies": {"sessionid": "test-only"}}))
    path.chmod(0o600)
    return LiveBrainSettings.model_validate(
        {"session_file": path, "base_url": BASE, "maximum_retry_after_seconds": 1}
    )


def _client(tmp_path: Path) -> LiveBrainClient:
    return LiveBrainClient(_settings(tmp_path), sleeper=lambda _seconds: None)


# `OPTIONS /simulations` -> actions.POST.settings.children, keyed by API field name.
# Region choices nest under instrumentType; Universe/Delay/Neutralization nest under
# instrumentType and then region. Numeric settings publish bounds, not choices.
SIMULATION_OPTIONS = {
    "actions": {
        "POST": {
            "settings": {
                "children": {
                    "instrumentType": {
                        "label": "Instrument type",
                        "type": "choice",
                        "choices": [{"value": "EQUITY"}],
                    },
                    "region": {
                        "label": "Region",
                        "type": "choice",
                        "choices": {
                            "instrumentType": {
                                "EQUITY": [{"value": "USA"}, {"value": "EUR"}],
                            }
                        },
                    },
                    "universe": {
                        "label": "Universe",
                        "type": "choice",
                        "choices": {
                            "instrumentType": {
                                "EQUITY": {
                                    "region": {
                                        "USA": [{"value": "TOP3000"}, {"value": "TOP1000"}],
                                        "EUR": [{"value": "TOP2500"}],
                                    }
                                }
                            }
                        },
                    },
                    "neutralization": {
                        "label": "Neutralization",
                        "type": "choice",
                        "choices": {
                            "instrumentType": {
                                "EQUITY": {
                                    "region": {
                                        "USA": [{"value": "MARKET"}, {"value": "SUBINDUSTRY"}],
                                        "EUR": [{"value": "COUNTRY"}],
                                    }
                                }
                            }
                        },
                    },
                    "decay": {
                        "label": "Decay",
                        "type": "integer",
                        "min_value": 0,
                        "max_value": 512,
                    },
                    "truncation": {
                        "label": "Truncation",
                        "type": "decimal",
                        "min_value": 0.0,
                        "max_value": 0.05,
                    },
                    "visualization": {"label": "Visualization", "type": "boolean"},
                }
            }
        }
    }
}

DATA_FIELDS = {
    "count": 2,
    "results": [
        {"id": "est_eps_mean", "type": "MATRIX"},
        {"id": "last_close_price", "type": "MATRIX"},
    ],
}

OPERATORS = [
    {"name": "ts_mean"},
    {"name": "ts_delta"},
    {"name": "rank"},
    {"name": "ts_backfill"},
]


@respx.mock
def test_authentication_gets_and_reports_biometric_verification(tmp_path: Path) -> None:
    """Checking a stored session is a GET; a persona challenge is a human step, never automated."""
    route = respx.get(f"{BASE}/authentication").mock(
        return_value=httpx.Response(401, headers={"WWW-Authenticate": "persona"})
    )
    client = _client(tmp_path)
    with pytest.raises(AdapterFailure) as raised:
        client.authenticate()
    assert route.called
    assert raised.value.error_class is ErrorClass.HUMAN_VERIFICATION_REQUIRED
    assert "biometric" in raised.value.message.lower()


@respx.mock
def test_authentication_rejects_stale_cookies_separately(tmp_path: Path) -> None:
    respx.get(f"{BASE}/authentication").mock(return_value=httpx.Response(401))
    client = _client(tmp_path)
    with pytest.raises(AdapterFailure) as raised:
        client.authenticate()
    assert raised.value.error_class is ErrorClass.AUTHENTICATION


@respx.mock
def test_authentication_stops_cleanly_when_session_is_nearly_expired(tmp_path: Path) -> None:
    """A session too close to expiry must fail before work is dispatched."""
    respx.get(f"{BASE}/authentication").mock(
        return_value=httpx.Response(200, json={"token": {"expiry": 1999.9}})
    )
    client = _client(tmp_path)
    with pytest.raises(AdapterFailure) as raised:
        client.authenticate()
    assert raised.value.error_class is ErrorClass.AUTHENTICATION
    assert "expires in 1999s" in raised.value.message


@respx.mock
def test_authentication_accepts_a_session_with_ample_time_left(tmp_path: Path) -> None:
    respx.get(f"{BASE}/authentication").mock(
        return_value=httpx.Response(200, json={"token": {"expiry": 13804.15}})
    )
    client = _client(tmp_path)
    client.authenticate()  # must not raise
    assert client._authenticated is True


@respx.mock
def test_session_seconds_remaining_is_estimated_locally_after_authenticating(
    tmp_path: Path,
) -> None:
    """A long-running execute() call must be able to re-check freshness mid-run
    without spending another request on BRAIN -- this is what it reads instead."""
    respx.get(f"{BASE}/authentication").mock(
        return_value=httpx.Response(200, json={"token": {"expiry": 13804.15}})
    )
    client = _client(tmp_path)
    assert client.session_seconds_remaining() is None  # nothing to estimate yet
    client.authenticate()
    remaining = client.session_seconds_remaining()
    assert remaining is not None
    assert 13800.0 < remaining <= 13804.15


@respx.mock
def test_session_seconds_remaining_is_unknown_when_brain_reports_no_expiry(
    tmp_path: Path,
) -> None:
    respx.get(f"{BASE}/authentication").mock(return_value=httpx.Response(200, json={}))
    client = _client(tmp_path)
    client.authenticate()
    assert client.session_seconds_remaining() is None


@respx.mock
def test_metadata_capture_reads_settings_from_simulation_options(tmp_path: Path) -> None:
    """Valid settings come from OPTIONS /simulations, narrowed to the run's own cell."""
    respx.get(f"{BASE}/authentication").mock(return_value=httpx.Response(200, json={}))
    respx.get(f"{BASE}/data-fields").mock(return_value=httpx.Response(200, json=DATA_FIELDS))
    respx.get(f"{BASE}/operators").mock(return_value=httpx.Response(200, json=OPERATORS))
    options = respx.request("OPTIONS", f"{BASE}/simulations").mock(
        return_value=httpx.Response(200, json=SIMULATION_OPTIONS)
    )

    capture = _client(tmp_path).capture_metadata(
        dataset_id="analyst_base_ref",
        region="USA",
        universe="TOP3000",
        delay=1,
        data_type=DataType.MATRIX,
    )
    assert options.called
    settings = capture.metadata.settings

    # The cell is USA, so EUR-only values must not enter the snapshot.
    assert settings["universe"] == {"TOP3000", "TOP1000"}
    assert settings["neutralization"] == {"MARKET", "SUBINDUSTRY"}
    assert "COUNTRY" not in settings["neutralization"]
    assert settings["instrumentType"] == {"EQUITY"}
    assert settings["visualization"] == {True, False}

    # Numeric ladders are clipped by the platform's own bounds: truncation stops at 0.05.
    assert settings["decay"] == {0, 1, 4, 16, 64, 128, 256, 512}
    assert settings["truncation"] == {0.01, 0.02, 0.04}


@respx.mock
def test_metadata_capture_fails_loudly_without_a_settings_schema(tmp_path: Path) -> None:
    respx.get(f"{BASE}/authentication").mock(return_value=httpx.Response(200, json={}))
    respx.get(f"{BASE}/data-fields").mock(return_value=httpx.Response(200, json=DATA_FIELDS))
    respx.get(f"{BASE}/operators").mock(return_value=httpx.Response(200, json=OPERATORS))
    respx.request("OPTIONS", f"{BASE}/simulations").mock(
        return_value=httpx.Response(200, json={"actions": {}})
    )
    with pytest.raises(AdapterFailure) as raised:
        _client(tmp_path).capture_metadata(
            dataset_id="analyst_base_ref",
            region="USA",
            universe="TOP3000",
            delay=1,
            data_type=DataType.MATRIX,
        )
    assert "settings schema" in raised.value.message


@respx.mock
def test_correlations_read_both_endpoints_and_keep_absence_distinct(tmp_path: Path) -> None:
    """A correlation BRAIN has not computed is None, never zero."""
    respx.get(f"{BASE}/authentication").mock(return_value=httpx.Response(200, json={}))
    respx.get(f"{BASE}/alphas/A1/correlations/self").mock(
        return_value=httpx.Response(
            200,
            json={
                "schema": {"properties": [{"name": "min"}, {"name": "max"}]},
                "records": [[-0.1, 0.42]],
                "max": 0.42,
                "min": -0.1,
            },
        )
    )
    respx.get(f"{BASE}/alphas/A1/correlations/prod").mock(
        return_value=httpx.Response(200, json={"records": [], "schema": {"properties": []}})
    )
    correlations = _client(tmp_path).fetch_correlations("A1")
    assert correlations["self"] == pytest.approx(0.42)
    assert correlations["prod"] is None


@respx.mock
def test_running_simulation_reports_only_progress_and_is_pending(tmp_path: Path) -> None:
    """A live simulation still running answers `{"progress": 0.35}` with no status field
    at all (confirmed against the real platform 2026-09-12) -- that silence means "still
    running", not an unrecognized state."""
    respx.get(f"{BASE}/authentication").mock(return_value=httpx.Response(200, json={}))
    respx.get(f"{BASE}/simulations/SIM123").mock(
        return_value=httpx.Response(200, headers={"Retry-After": "5"}, json={"progress": 0.35})
    )
    result = _client(tmp_path).poll_simulation("SIM123")
    assert result.completed is False
    assert result.error_class is ErrorClass.PENDING


def _snapshot() -> MetadataSnapshot:
    return MetadataSnapshot(
        dataset_id="analyst_base_ref",
        fields={"est_eps_mean": DataType.MATRIX, "last_close_price": DataType.MATRIX},
        operators={"ts_mean", "ts_delta", "rank", "ts_backfill"},
        settings={"neutralization": {"MARKET"}},
    )


def test_templates_expand_deterministically_and_only_over_time_series_operators() -> None:
    proposals = expand_templates(
        ["rank(<operator>(ts_backfill(<field>, 120), 60))"],
        _snapshot(),
        data_type=DataType.MATRIX,
        settings={"neutralization": "MARKET"},
    )
    expressions = [proposal.expression for proposal in proposals]
    # 2 fields x 3 time-series operators (ts_mean, ts_delta, ts_backfill). `rank` is not a
    # ts_ operator and must not enter the operator slot, whose shape is operator(x, d).
    assert len(expressions) == 6
    assert "rank(ts_delta(ts_backfill(est_eps_mean, 120), 60))" in expressions
    assert not any("rank(rank(" in expression for expression in expressions)
    # Expansion is field-major then operator-major, so the order is reproducible.
    assert expressions[0].startswith("rank(ts_backfill(ts_backfill(est_eps_mean")

    again = expand_templates(
        ["rank(<operator>(ts_backfill(<field>, 120), 60))"],
        _snapshot(),
        data_type=DataType.MATRIX,
        settings={"neutralization": "MARKET"},
    )
    assert [proposal.expression for proposal in again] == expressions


def test_templates_accept_chinese_placeholders_and_literal_lines() -> None:
    proposals = expand_templates(
        ["ts_rank(ts_backfill(<字段>, 120), 250)", "rank(close)"],
        _snapshot(),
        data_type=DataType.MATRIX,
        settings={},
    )
    expressions = [proposal.expression for proposal in proposals]
    assert "ts_rank(ts_backfill(est_eps_mean, 120), 250)" in expressions
    assert "rank(close)" in expressions


def test_template_limit_is_respected_and_duplicates_collapse() -> None:
    proposals = expand_templates(
        ["ts_rank(ts_backfill(<field>, 120), 250)", "ts_rank(ts_backfill(<field>, 120), 250)"],
        _snapshot(),
        data_type=DataType.MATRIX,
        settings={},
        limit=1,
    )
    assert len(proposals) == 1


def test_template_file_reading_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "templates.txt"
    path.write_text("# only a comment\n\n", encoding="utf-8")
    with pytest.raises(TemplateError):
        load_templates(path)


def test_seeded_live_adapter_supplies_candidates_and_stays_seedless_by_default(
    tmp_path: Path,
) -> None:
    """A live adapter with no seeds still yields nothing: the platform never invents ideas."""
    client = _client(tmp_path)
    snapshot = _snapshot()
    assert LiveBrainAdapter(client, snapshot).initial_candidates() == []

    proposal = CandidateProposal(
        expression="ts_rank(ts_backfill(est_eps_mean, 120), 250)",
        settings={"neutralization": "MARKET"},
        mechanism_claim="seed",
    )
    seeded = LiveBrainAdapter(client, snapshot, seed_candidates=[proposal])
    assert seeded.mode is AdapterMode.LIVE
    assert [candidate.expression for candidate in seeded.initial_candidates()] == [
        proposal.expression
    ]


def test_seeded_expressions_pass_the_structural_gate() -> None:
    """Seeding is pointless if the gate rejects everything it produces."""
    from alpha_orchestrator.gates.structural import StructuralValidator

    snapshot = _snapshot()
    proposals = expand_templates(
        ["rank(<operator>(ts_backfill(<field>, 120), 60))"],
        snapshot,
        data_type=DataType.MATRIX,
        settings={"neutralization": "MARKET"},
    )
    validator = StructuralValidator()
    for proposal in proposals:
        result = validator.validate(
            proposal.expression, proposal.settings, snapshot, DataType.MATRIX
        )
        assert result.valid, result.issues


def test_simulation_receipt_reads_the_location_header(tmp_path: Path) -> None:
    """BRAIN returns the simulation id in Location, with no body at all."""
    from alpha_orchestrator.domain.models import CreateSimulationRequest

    with respx.mock:
        respx.get(f"{BASE}/authentication").mock(return_value=httpx.Response(200, json={}))
        respx.post(f"{BASE}/simulations").mock(
            return_value=httpx.Response(
                201, headers={"Location": f"{BASE}/simulations/SIM123"}, content=b""
            )
        )
        receipt = _client(tmp_path).create_simulation(
            CreateSimulationRequest(
                candidate_id=uuid4(),
                expression="rank(close)",
                settings={"neutralization": "MARKET"},
                region="USA",
                universe="TOP3000",
                delay=1,
                idempotency_key="key",
            )
        )
    assert receipt.external_simulation_id == "SIM123"
