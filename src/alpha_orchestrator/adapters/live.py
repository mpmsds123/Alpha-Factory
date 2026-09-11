"""Explicitly authorized WorldQuant BRAIN HTTP adapter with no submit capability."""

import json
import os
import re
import stat
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from alpha_orchestrator.domain.enums import (
    AdapterMode,
    DataType,
    ErrorClass,
    MetadataSnapshotKind,
    PlatformCheckStatus,
)
from alpha_orchestrator.domain.errors import redact_text
from alpha_orchestrator.domain.identity import metadata_snapshot_identity
from alpha_orchestrator.domain.models import (
    AdapterFailure,
    CandidateProposal,
    CapturedMetadata,
    CreateSimulationReceipt,
    CreateSimulationRequest,
    MetadataSnapshot,
    NormalizedSimulationResult,
    PlatformCheck,
    SimulationMetrics,
)

DEFAULT_BRAIN_BASE_URL = "https://api.worldquantbrain.com"
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._-]+$")
PENDING_STATES = frozenset({"CREATED", "PENDING", "QUEUED", "RUNNING", "IN_PROGRESS"})
COMPLETE_STATES = frozenset({"COMPLETE", "COMPLETED", "DONE", "SUCCESS", "SUCCEEDED"})
ERROR_STATES = frozenset({"ERROR", "FAILED", "FAILURE", "CANCELLED", "CANCELED"})


class LiveBrainSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_file: Path
    base_url: str = DEFAULT_BRAIN_BASE_URL
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=900.0)
    maximum_retry_after_seconds: float = Field(default=60.0, ge=0.0, le=300.0)
    authentication_path: str = "/authentication"
    datasets_path: str = "/data-sets"
    fields_path: str = "/data-fields"
    operators_path: str = "/operators"
    settings_path: str = "/settings"
    simulations_path: str = "/simulations"
    alphas_path: str = "/alphas"

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Live BRAIN base URL must use HTTPS")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Live BRAIN base URL cannot contain credentials, query, or fragment")
        hostname = (parsed.hostname or "").casefold()
        if hostname != "worldquantbrain.com" and not hostname.endswith(".worldquantbrain.com"):
            raise ValueError("Live BRAIN base URL must use an official WorldQuant BRAIN host")
        return value.rstrip("/")

    @field_validator(
        "authentication_path",
        "datasets_path",
        "fields_path",
        "operators_path",
        "settings_path",
        "simulations_path",
        "alphas_path",
    )
    @classmethod
    def validate_endpoint_path(cls, value: str) -> str:
        if not value.startswith("/") or "://" in value or ".." in value:
            raise ValueError("BRAIN endpoint paths must be absolute paths on the configured host")
        return value.rstrip("/") or "/"

    @classmethod
    def from_environment(cls) -> "LiveBrainSettings":
        session_file = os.getenv("ALPHA_LAB_BRAIN_SESSION_FILE")
        if not session_file:
            raise ValueError("ALPHA_LAB_BRAIN_SESSION_FILE is required for live BRAIN access")
        return cls(
            session_file=Path(session_file),
            base_url=os.getenv("ALPHA_LAB_BRAIN_BASE_URL", DEFAULT_BRAIN_BASE_URL),
            timeout_seconds=float(os.getenv("ALPHA_LAB_BRAIN_TIMEOUT_SECONDS", "60")),
            maximum_retry_after_seconds=float(
                os.getenv("ALPHA_LAB_BRAIN_MAX_RETRY_AFTER_SECONDS", "60")
            ),
            authentication_path=os.getenv("ALPHA_LAB_BRAIN_AUTHENTICATION_PATH", "/authentication"),
            datasets_path=os.getenv("ALPHA_LAB_BRAIN_DATASETS_PATH", "/data-sets"),
            fields_path=os.getenv("ALPHA_LAB_BRAIN_FIELDS_PATH", "/data-fields"),
            operators_path=os.getenv("ALPHA_LAB_BRAIN_OPERATORS_PATH", "/operators"),
            settings_path=os.getenv("ALPHA_LAB_BRAIN_SETTINGS_PATH", "/settings"),
            simulations_path=os.getenv("ALPHA_LAB_BRAIN_SIMULATIONS_PATH", "/simulations"),
            alphas_path=os.getenv("ALPHA_LAB_BRAIN_ALPHAS_PATH", "/alphas"),
        )


def load_session_cookies(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"BRAIN session file does not exist: {resolved}")
    if os.name != "nt" and stat.S_IMODE(resolved.stat().st_mode) & 0o077:
        raise ValueError("BRAIN session file must not be readable by group or other users")
    with resolved.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    raw_cookies: object
    if isinstance(payload, Mapping) and "cookies" in payload:
        raw_cookies = payload["cookies"]
    else:
        raw_cookies = payload
    cookies: dict[str, str] = {}
    if isinstance(raw_cookies, Mapping):
        if any(
            not isinstance(name, str) or not isinstance(value, str) or not name or not value
            for name, value in raw_cookies.items()
        ):
            raise ValueError("BRAIN cookie mappings require non-empty string names and values")
        cookies = dict(raw_cookies)
    elif isinstance(raw_cookies, list):
        for item in raw_cookies:
            if (
                not isinstance(item, Mapping)
                or not isinstance(item.get("name"), str)
                or not isinstance(item.get("value"), str)
                or not item["name"]
                or not item["value"]
            ):
                raise ValueError("BRAIN cookie-list entries require name and value")
            cookies[item["name"]] = item["value"]
    else:
        raise ValueError("BRAIN session file must contain a cookie mapping or cookie list")
    if not cookies or any(not name or not value for name, value in cookies.items()):
        raise ValueError("BRAIN session file contains no usable cookies")
    return cookies


class LiveBrainClient:
    def __init__(
        self,
        settings: LiveBrainSettings,
        *,
        http_client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._sleeper = sleeper
        self._owns_client = http_client is None
        cookies = load_session_cookies(settings.session_file)
        self._client = http_client or httpx.Client(
            timeout=settings.timeout_seconds,
            follow_redirects=False,
            headers={"Accept": "application/json", "User-Agent": "alpha-research-orchestrator"},
        )
        self._client.cookies.update(cookies)
        self._authenticated = False

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def authenticate(self) -> None:
        response = self._send("GET", self.settings.authentication_path, authenticate=False)
        payload = _optional_json(response)
        if isinstance(payload, Mapping) and payload.get("inquiry"):
            raise AdapterFailure(
                ErrorClass.HUMAN_VERIFICATION_REQUIRED,
                "BRAIN authentication requires human verification",
            )
        self._authenticated = True

    def list_datasets(self, *, region: str, universe: str, delay: int) -> list[dict[str, object]]:
        response = self._send(
            "GET",
            self.settings.datasets_path,
            params={"region": region, "universe": universe, "delay": delay},
        )
        return [dict(item) for item in _record_items(_required_json(response))]

    def capture_metadata(
        self,
        *,
        dataset_id: str,
        region: str,
        universe: str,
        delay: int,
        data_type: DataType,
    ) -> CapturedMetadata:
        params = {
            "dataset.id": dataset_id,
            "region": region,
            "universe": universe,
            "delay": delay,
        }
        fields_response = self._send("GET", self.settings.fields_path, params=params)
        operators_response = self._send("GET", self.settings.operators_path)
        settings_response = self._send("GET", self.settings.settings_path, params=params)
        fields = _parse_fields(_required_json(fields_response))
        operators = _parse_operators(_required_json(operators_response))
        valid_settings = _parse_settings(_required_json(settings_response))
        metadata = MetadataSnapshot(
            dataset_id=dataset_id,
            fields=fields,
            operators=operators,
            settings=valid_settings,
        )
        fields_payload = {name: value.value for name, value in sorted(fields.items())}
        operators_payload = sorted(operators)
        settings_payload = {
            name: sorted(values, key=repr) for name, values in sorted(valid_settings.items())
        }
        field_snapshot_id, _ = metadata_snapshot_identity(
            MetadataSnapshotKind.FIELDS,
            dataset_id,
            region,
            universe,
            delay,
            data_type,
            fields_payload,
        )
        operator_snapshot_id, _ = metadata_snapshot_identity(
            MetadataSnapshotKind.OPERATORS,
            dataset_id,
            region,
            universe,
            delay,
            data_type,
            operators_payload,
        )
        setting_snapshot_id, _ = metadata_snapshot_identity(
            MetadataSnapshotKind.SETTINGS,
            dataset_id,
            region,
            universe,
            delay,
            data_type,
            settings_payload,
        )
        return CapturedMetadata(
            metadata=metadata,
            region=region,
            universe=universe,
            delay=delay,
            data_type=data_type,
            field_snapshot_id=field_snapshot_id,
            operator_snapshot_id=operator_snapshot_id,
            setting_snapshot_id=setting_snapshot_id,
            source="worldquant-brain-live",
        )

    def create_simulation(self, request: CreateSimulationRequest) -> CreateSimulationReceipt:
        payload = {
            "type": "REGULAR",
            "settings": {
                "region": request.region,
                "universe": request.universe,
                "delay": request.delay,
                **request.settings,
            },
            "regular": request.expression,
        }
        response = self._send(
            "POST",
            self.settings.simulations_path,
            json=payload,
            headers={"Idempotency-Key": request.idempotency_key},
        )
        body = _optional_json(response)
        external_id = _response_identifier(response, body, "id", "simulationId")
        alpha_id = _mapping_identifier(body, "alpha", "alphaId")
        return CreateSimulationReceipt(
            external_simulation_id=external_id,
            external_alpha_id=alpha_id,
        )

    def poll_simulation(self, external_simulation_id: str) -> NormalizedSimulationResult:
        simulation_id = _validate_identifier(external_simulation_id, "simulation")
        response = self._send("GET", f"{self.settings.simulations_path}/{simulation_id}")
        body = _required_mapping(response)
        state = str(body.get("status") or body.get("state") or "UNKNOWN").upper()
        if state in PENDING_STATES:
            return NormalizedSimulationResult(completed=False, error_class=ErrorClass.PENDING)
        if state in ERROR_STATES:
            error_class = _classify_payload_error(body)
            return NormalizedSimulationResult(
                completed=False,
                error_class=error_class,
                error_message=_safe_payload_message(body),
            )
        if state not in COMPLETE_STATES:
            return NormalizedSimulationResult(
                completed=False,
                error_class=ErrorClass.UNKNOWN,
                error_message=f"Unrecognized BRAIN simulation state: {state[:40]}",
            )
        alpha_id = _mapping_identifier(body, "alpha", "alphaId")
        details = body
        if alpha_id is not None:
            alpha_response = self._send("GET", f"{self.settings.alphas_path}/{alpha_id}")
            details = _required_mapping(alpha_response)
        metrics = _parse_metrics(details)
        if metrics.sharpe is None and metrics.fitness is None and metrics.turnover is None:
            return NormalizedSimulationResult(
                completed=False,
                external_alpha_id=alpha_id,
                error_class=ErrorClass.UNKNOWN,
                error_message="Completed BRAIN simulation contained no normalized metrics",
            )
        return NormalizedSimulationResult(
            completed=True,
            metrics=metrics,
            external_alpha_id=alpha_id,
            raw_artifact_ref=f"brain://simulations/{simulation_id}",
        )

    def request_platform_checks(self, external_alpha_id: str) -> list[PlatformCheck]:
        alpha_id = _validate_identifier(external_alpha_id, "alpha")
        response = self._send("GET", f"{self.settings.alphas_path}/{alpha_id}/check")
        payload = _required_json(response)
        return _parse_platform_checks(payload)

    def _send(
        self,
        method: str,
        path: str,
        *,
        authenticate: bool = True,
        **kwargs: Any,
    ) -> httpx.Response:
        if authenticate and not self._authenticated:
            self.authenticate()
        url = f"{self.settings.base_url}{path}"
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise AdapterFailure(ErrorClass.TIMEOUT, "BRAIN request timed out") from exc
        except httpx.NetworkError as exc:
            raise AdapterFailure(ErrorClass.NETWORK, "BRAIN network request failed") from exc
        if response.is_success:
            return response
        error_class = _classify_http_error(response)
        if response.status_code == 401:
            self._authenticated = False
        retry_after = _retry_after_seconds(response)
        if error_class is ErrorClass.RATE_LIMIT and retry_after:
            self._sleeper(min(retry_after, self.settings.maximum_retry_after_seconds))
        raise AdapterFailure(
            error_class,
            _safe_http_message(response, error_class),
            retry_after_seconds=retry_after,
        )


class LiveBrainAdapter:
    def __init__(self, client: LiveBrainClient, metadata: MetadataSnapshot) -> None:
        self.client = client
        self._metadata = metadata

    @property
    def mode(self) -> AdapterMode:
        return AdapterMode.LIVE

    @property
    def metadata(self) -> MetadataSnapshot:
        return self._metadata

    def initial_candidates(self) -> list[CandidateProposal]:
        return []

    def create_simulation(self, request: CreateSimulationRequest) -> CreateSimulationReceipt:
        return self.client.create_simulation(request)

    def poll_simulation(self, external_simulation_id: str) -> NormalizedSimulationResult:
        return self.client.poll_simulation(external_simulation_id)

    def request_platform_checks(self, external_alpha_id: str) -> list[PlatformCheck]:
        return self.client.request_platform_checks(external_alpha_id)


def _required_json(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError as exc:
        raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN response was not valid JSON") from exc


def _optional_json(response: httpx.Response) -> object | None:
    if not response.content:
        return None
    return _required_json(response)


def _required_mapping(response: httpx.Response) -> Mapping[str, Any]:
    payload = _required_json(response)
    if not isinstance(payload, Mapping):
        raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN response must be a JSON object")
    return payload


def _record_items(payload: object) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        candidate = payload.get("results", payload.get("items", payload.get("data", [])))
    else:
        candidate = payload
    if not isinstance(candidate, list) or any(not isinstance(item, Mapping) for item in candidate):
        raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN metadata response has no record list")
    return list(candidate)


def _parse_fields(payload: object) -> dict[str, DataType]:
    fields: dict[str, DataType] = {}
    for item in _record_items(payload):
        name = item.get("id") or item.get("name")
        raw_type = item.get("type") or item.get("dataType") or item.get("data_type")
        if not isinstance(name, str) or not isinstance(raw_type, str):
            raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN field metadata is incomplete")
        try:
            fields[name] = DataType(raw_type.upper())
        except ValueError as exc:
            raise AdapterFailure(
                ErrorClass.UNKNOWN, f"Unsupported BRAIN field data type: {raw_type[:40]}"
            ) from exc
    if not fields:
        raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN returned no data fields")
    return fields


def _parse_operators(payload: object) -> set[str]:
    operators: set[str] = set()
    for item in _record_items(payload):
        name = item.get("name") or item.get("id")
        if not isinstance(name, str) or not name:
            raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN operator metadata is incomplete")
        operators.add(name)
    if not operators:
        raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN returned no operators")
    return operators


def _parse_settings(payload: object) -> dict[str, set[str | int | float | bool]]:
    raw_settings = payload.get("settings", payload) if isinstance(payload, Mapping) else payload
    settings: dict[str, set[str | int | float | bool]] = {}
    if isinstance(raw_settings, Mapping):
        for name, values in raw_settings.items():
            if name in {"count", "next", "previous", "results", "items", "data"}:
                continue
            if isinstance(values, list) and all(
                isinstance(value, str | int | float | bool) for value in values
            ):
                settings[str(name)] = set(values)
    elif isinstance(raw_settings, list):
        for item in raw_settings:
            if not isinstance(item, Mapping):
                raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN settings metadata is incomplete")
            name = item.get("name") or item.get("id")
            values = item.get("values") or item.get("options")
            if not isinstance(name, str) or not isinstance(values, list):
                raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN settings metadata is incomplete")
            if not all(isinstance(value, str | int | float | bool) for value in values):
                raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN setting values must be scalar")
            settings[name] = set(values)
    if not settings:
        raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN returned no valid settings")
    return settings


def _response_identifier(
    response: httpx.Response,
    payload: object | None,
    *keys: str,
) -> str:
    identifier = _mapping_identifier(payload, *keys)
    if identifier is None:
        location = response.headers.get("Location")
        if location:
            identifier = urlsplit(location).path.rstrip("/").rsplit("/", 1)[-1]
    if identifier is None:
        raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN create response contained no simulation ID")
    return _validate_identifier(identifier, "simulation")


def _mapping_identifier(payload: object | None, *keys: str) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            value = value.get("id")
        if isinstance(value, str) and value:
            return _validate_identifier(value, "external")
    return None


def _validate_identifier(value: str, label: str) -> str:
    if not IDENTIFIER_RE.fullmatch(value):
        raise AdapterFailure(ErrorClass.UNKNOWN, f"BRAIN {label} ID has an unsafe format")
    return value


def _parse_metrics(payload: Mapping[str, Any]) -> SimulationMetrics:
    raw = payload.get("is") or payload.get("metrics") or payload
    if not isinstance(raw, Mapping):
        raw = {}

    def number(name: str) -> float | None:
        value = raw.get(name)
        return (
            float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
        )

    extra = {
        str(name): value
        for name, value in raw.items()
        if name not in {"sharpe", "fitness", "turnover", "checks"}
        and isinstance(value, str | int | float | bool | type(None))
    }
    return SimulationMetrics(
        sharpe=number("sharpe"),
        fitness=number("fitness"),
        turnover=number("turnover"),
        extra=extra,
    )


def _parse_platform_checks(payload: object) -> list[PlatformCheck]:
    raw_checks: object
    if isinstance(payload, Mapping):
        raw_checks = payload.get("checks")
        if raw_checks is None and isinstance(payload.get("is"), Mapping):
            raw_checks = payload["is"].get("checks")
    else:
        raw_checks = payload
    if not isinstance(raw_checks, list):
        raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN platform-check response has no check list")
    checks: list[PlatformCheck] = []
    for item in raw_checks:
        if not isinstance(item, Mapping):
            raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN platform-check item is invalid")
        name = item.get("name") or item.get("id")
        raw_status = item.get("status") or item.get("result")
        if not isinstance(name, str) or not isinstance(raw_status, str):
            raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN platform-check item is incomplete")
        normalized = raw_status.upper()
        try:
            status = PlatformCheckStatus(normalized)
        except ValueError:
            status = PlatformCheckStatus.ERROR
        message = item.get("message")
        checks.append(
            PlatformCheck(
                name=name,
                status=status,
                message=redact_text(str(message)) if message is not None else None,
            )
        )
    return checks


def _classify_http_error(response: httpx.Response) -> ErrorClass:
    if response.status_code == 401:
        payload = _safe_response_json(response)
        if isinstance(payload, Mapping) and payload.get("inquiry"):
            return ErrorClass.HUMAN_VERIFICATION_REQUIRED
        return ErrorClass.AUTHENTICATION
    if response.status_code == 403:
        return ErrorClass.AUTHORIZATION
    if response.status_code == 408 or response.status_code == 504:
        return ErrorClass.TIMEOUT
    if response.status_code == 429:
        return ErrorClass.RATE_LIMIT
    if response.status_code >= 500:
        return ErrorClass.NETWORK
    if response.status_code in {400, 409, 422}:
        return _classify_error_text(_safe_payload_message(_safe_response_json(response)))
    return ErrorClass.UNKNOWN


def _classify_payload_error(payload: Mapping[str, Any]) -> ErrorClass:
    state = str(payload.get("status") or payload.get("state") or "").upper()
    if state in {"CANCELLED", "CANCELED"}:
        return ErrorClass.CANCELLED
    return _classify_error_text(_safe_payload_message(payload))


def _classify_error_text(message: str) -> ErrorClass:
    lowered = message.casefold()
    if "operator" in lowered:
        return ErrorClass.INVALID_OPERATOR
    if "data field" in lowered or "field" in lowered:
        return ErrorClass.INVALID_FIELD
    if "setting" in lowered or "neutralization" in lowered or "universe" in lowered:
        return ErrorClass.INVALID_SETTING
    if "expression" in lowered or "syntax" in lowered or "parse" in lowered:
        return ErrorClass.INVALID_EXPRESSION
    return ErrorClass.PLATFORM_TERMINAL_FAILURE


def _safe_payload_message(payload: object | None) -> str:
    if isinstance(payload, Mapping):
        for key in ("message", "detail", "error"):
            value = payload.get(key)
            if isinstance(value, str):
                return (redact_text(value) or "BRAIN request failed")[:500]
    return "BRAIN request failed"


def _safe_http_message(response: httpx.Response, error_class: ErrorClass) -> str:
    detail = _safe_payload_message(_safe_response_json(response))
    return f"BRAIN HTTP {response.status_code} ({error_class}): {detail}"


def _safe_response_json(response: httpx.Response) -> object | None:
    try:
        return response.json() if response.content else None
    except ValueError:
        return None


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None
