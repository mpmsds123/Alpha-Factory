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

from alpha_orchestrator.adapters.base import MINIMUM_SESSION_REMAINING_SECONDS
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

# BRAIN exposes valid simulation settings through `OPTIONS /simulations`, not through a
# `GET /settings` collection. Choice values for Region, Universe, Delay and Neutralization
# are nested per instrument type and region, so they must be resolved against the research
# cell before they can be frozen into a MetadataSnapshot.
SETTING_OPTIONS_METHOD = "OPTIONS"
CHOICE_DIMENSIONS = ("instrumentType", "region", "delay")

# GET /data-fields is a query BRAIN validates strictly: instrumentType, region, delay,
# universe, type, dataset.id, limit and offset must all be present or it answers
# 400 "Invalid query" with no further detail. limit/offset also paginate -- a dataset
# can hold more fields than one page, and the reference client walks every page.
FIELDS_PAGE_SIZE = 50

# Paging /data-fields is the one place this adapter issues many requests back to back:
# a dataset with 600+ fields is 13+ consecutive GETs. BRAIN answers 429 for that burst.
# The official API session states the pause between pages explicitly -- five seconds --
# so it is recorded here as a default rather than tuned by trial.
# [source: WQ官方课程字幕全文.md:125] (锚点: 避免咱们平台报429的错误)
FIELDS_PAGE_PAUSE_SECONDS = 5.0

# A 429 is a pacing signal, not a research result, so one paced retry is attempted before
# the run is stopped. When BRAIN sends no Retry-After with the 429 there is nothing to
# honour, so this floor is used instead of retrying immediately.
RATE_LIMIT_DEFAULT_PAUSE_SECONDS = 5.0

# MINIMUM_SESSION_REMAINING_SECONDS lives in adapters.base (shared with the
# orchestrator, which re-checks freshness mid-run without an extra network call).

# Numeric settings are published as bounded ranges rather than choice lists, so a snapshot
# cannot enumerate them from the platform alone. These ladders are the discrete values this
# project is willing to search; every value is intersected with the platform's own min/max
# before it enters the snapshot, so the platform still has the final say.
# The bounded search ladders are explicit and versionable.
DEFAULT_NUMERIC_LADDERS: dict[str, tuple[float, ...]] = {
    "decay": (0, 1, 4, 16, 64, 128, 256, 512),
    "truncation": (0.01, 0.02, 0.04, 0.06, 0.08),
}


class LiveBrainSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_file: Path
    base_url: str = DEFAULT_BRAIN_BASE_URL
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=900.0)
    maximum_retry_after_seconds: float = Field(default=60.0, ge=0.0, le=300.0)
    maximum_rate_limit_attempts: int = Field(default=5, ge=1, le=5)
    fields_page_pause_seconds: float = Field(default=FIELDS_PAGE_PAUSE_SECONDS, ge=0.0, le=60.0)
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
            fields_page_pause_seconds=float(
                os.getenv(
                    "ALPHA_LAB_BRAIN_FIELDS_PAGE_PAUSE_SECONDS",
                    str(FIELDS_PAGE_PAUSE_SECONDS),
                )
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
        self._session_deadline_monotonic: float | None = None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def authenticate(self) -> None:
        # A fresh BRAIN login is POST with Basic Auth credentials -- but this adapter
        # never holds credentials (that is the whole point: it only ever receives a
        # session file someone else produced with a human biometric step already done).
        # Given only cookies, POST /authentication is a bare login attempt with no
        # credentials attached and BRAIN always answers 401, regardless of whether the
        # cookies are still good. Checking an *existing* session is a GET instead, the
        # same call the reference client uses to read remaining token lifetime. A 401
        # here means the session file is genuinely stale: get a new one, it does not
        # mean a persona challenge is in progress (that only happens on a POST login).
        response = self._request("GET", self.settings.authentication_path)
        if response.is_success:
            self._throttle_for_remaining_budget(response)
        if response.status_code == 401:
            scheme = response.headers.get("WWW-Authenticate", "").strip().casefold()
            body = _optional_json(response)
            challenged = scheme.startswith("persona") or (
                isinstance(body, Mapping) and bool(body.get("inquiry"))
            )
            if challenged:
                raise AdapterFailure(
                    ErrorClass.HUMAN_VERIFICATION_REQUIRED,
                    "BRAIN requires biometric verification: complete it in a browser, "
                    "then export a fresh session cookie file",
                )
            raise AdapterFailure(
                ErrorClass.AUTHENTICATION,
                "BRAIN rejected the supplied session cookies",
            )
        if not response.is_success:
            raise self._failure_for(response)
        payload = _optional_json(response)
        if isinstance(payload, Mapping) and payload.get("inquiry"):
            raise AdapterFailure(
                ErrorClass.HUMAN_VERIFICATION_REQUIRED,
                "BRAIN authentication requires human verification",
            )
        remaining = _session_remaining_seconds(payload)
        if remaining is not None and remaining < MINIMUM_SESSION_REMAINING_SECONDS:
            raise AdapterFailure(
                ErrorClass.AUTHENTICATION,
                f"BRAIN session expires in {int(remaining)}s, below the "
                f"{int(MINIMUM_SESSION_REMAINING_SECONDS)}s minimum: stopping cleanly "
                "instead of risking a mid-run expiry. Run tools/取会话.py for a fresh one.",
            )
        # Recorded as a deadline on the monotonic clock, not the raw `remaining`
        # value, so later callers (mid-run, no new request) can tell how much is
        # left without re-asking BRAIN. Unknown (BRAIN reported no expiry) stays
        # unknown rather than being treated as "plenty of time".
        self._session_deadline_monotonic = (
            time.monotonic() + remaining if remaining is not None else None
        )
        self._authenticated = True

    def session_seconds_remaining(self) -> float | None:
        if self._session_deadline_monotonic is None:
            return None
        return self._session_deadline_monotonic - time.monotonic()

    def list_datasets(self, *, region: str, universe: str, delay: int) -> list[dict[str, object]]:
        response = self._send(
            "GET",
            self.settings.datasets_path,
            params={"region": region, "universe": universe, "delay": delay},
        )
        return [dict(item) for item in _record_items(_required_json(response))]

    def _fetch_all_fields(self, params: dict[str, object]) -> list[dict[str, Any]]:
        page_params = dict(params)
        offset = 0
        records: list[dict[str, Any]] = []
        page_index = 0
        while True:
            if page_index:
                self._sleeper(self.settings.fields_page_pause_seconds)
            page_index += 1
            page_params["limit"] = FIELDS_PAGE_SIZE
            page_params["offset"] = offset
            response = self._send("GET", self.settings.fields_path, params=page_params)
            payload = _required_mapping(response)
            results = payload.get("results")
            if not isinstance(results, list):
                raise AdapterFailure(
                    ErrorClass.UNKNOWN, "BRAIN data-fields response has no record list"
                )
            records.extend(dict(item) for item in results if isinstance(item, Mapping))
            offset += len(results)
            total = payload.get("count")
            if not results or not isinstance(total, int) or offset >= total:
                return records

    def capture_metadata(
        self,
        *,
        dataset_id: str,
        region: str,
        universe: str,
        delay: int,
        data_type: DataType,
        instrument_type: str = "EQUITY",
    ) -> CapturedMetadata:
        params: dict[str, object] = {
            "instrumentType": instrument_type,
            "region": region,
            "universe": universe,
            "delay": delay,
            "type": data_type.value,
            "dataset.id": dataset_id,
        }
        field_records = self._fetch_all_fields(params)
        operators_response = self._send("GET", self.settings.operators_path)
        settings_response = self._send(SETTING_OPTIONS_METHOD, self.settings.simulations_path)
        fields = _parse_fields(field_records)
        operators_json = _required_json(operators_response)
        operators = _parse_operators(operators_json)
        operator_definitions = _parse_operator_definitions(operators_json)
        valid_settings = _parse_setting_options(
            _required_json(settings_response),
            instrument_type=instrument_type,
            region=region,
            delay=delay,
        )
        metadata = MetadataSnapshot(
            dataset_id=dataset_id,
            fields=fields,
            operators=operators,
            operator_definitions=operator_definitions,
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
        raw_state = body.get("status") or body.get("state")
        if raw_state is None:
            # A running simulation reports only `{"progress": <0..1>}` with no status
            # field at all -- BRAIN's silence here means "still running", not "unknown".
            # Retry-After on this same response says how soon to poll again.
            return NormalizedSimulationResult(completed=False, error_class=ErrorClass.PENDING)
        state = str(raw_state).upper()
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

    def fetch_correlations(self, external_alpha_id: str) -> dict[str, float | None]:
        """Read `/correlations/self` and `/correlations/prod` for one alpha.

        Expression fingerprints cannot tell two spellings of one signal apart; these are the
        platform's own answers about redundancy. A correlation the platform has not computed
        yet comes back as None rather than as a zero.
        """
        alpha_id = _validate_identifier(external_alpha_id, "alpha")
        correlations: dict[str, float | None] = {}
        for kind in ("self", "prod"):
            response = self._send(
                "GET", f"{self.settings.alphas_path}/{alpha_id}/correlations/{kind}"
            )
            correlations[kind] = _parse_max_correlation(_optional_json(response))
        return correlations

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Send one request and return the response, successful or not."""
        url = f"{self.settings.base_url}{path}"
        try:
            return self._client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise AdapterFailure(ErrorClass.TIMEOUT, "BRAIN request timed out") from exc
        except httpx.TransportError as exc:
            # Every transport-layer failure is infrastructure, never a research result.
            # Catching the whole TransportError family (proxy errors, protocol errors,
            # unsupported schemes) keeps them classified instead of escaping as a traceback.
            raise AdapterFailure(
                ErrorClass.NETWORK,
                f"BRAIN request failed at the transport layer: {type(exc).__name__}",
            ) from exc

    def _throttle_for_remaining_budget(self, response: httpx.Response) -> None:
        """Pace ourselves when BRAIN's own rate-limit headers say we are close to
        the edge, instead of waiting to actually draw a 429.

        Retry-after-429 handling in `_failure_for` is reactive; this check is
        proactive and runs on every successful response. It does nothing if the
        platform does not send rate-limit headers.
        """
        try:
            remaining_second = int(response.headers["x-ratelimit-remaining-second"])
            remaining_minute = int(response.headers["x-ratelimit-remaining-minute"])
        except (KeyError, TypeError, ValueError):
            return
        if remaining_second < 1:
            self._sleeper(1.0)
        if remaining_minute <= 1:
            self._sleeper(60.0)

    def _failure_for(self, response: httpx.Response) -> AdapterFailure:
        error_class = _classify_http_error(response)
        if response.status_code == 401:
            self._authenticated = False
        retry_after = _retry_after_seconds(response)
        if error_class is ErrorClass.RATE_LIMIT:
            pause = retry_after if retry_after else RATE_LIMIT_DEFAULT_PAUSE_SECONDS
            self._sleeper(min(pause, self.settings.maximum_retry_after_seconds))
        return AdapterFailure(
            error_class,
            _safe_http_message(response, error_class),
            retry_after_seconds=retry_after,
        )

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
        for attempt in range(self.settings.maximum_rate_limit_attempts):
            response = self._request(method, path, **kwargs)
            if response.is_success:
                self._throttle_for_remaining_budget(response)
                return response
            failure = self._failure_for(response)
            if failure.error_class is not ErrorClass.RATE_LIMIT:
                raise failure
            if attempt + 1 == self.settings.maximum_rate_limit_attempts:
                raise failure
        raise AssertionError("rate-limit loop must return or raise")


class LiveBrainAdapter:
    def __init__(
        self,
        client: LiveBrainClient,
        metadata: MetadataSnapshot,
        seed_candidates: list[CandidateProposal] | None = None,
    ) -> None:
        self.client = client
        self._metadata = metadata
        # Live runs have no candidates of their own: the platform supplies data, never
        # ideas. Seeds come from an explicit template file so a first run can be executed
        # and audited without involving a model.
        self._seed_candidates = list(seed_candidates or [])

    @property
    def mode(self) -> AdapterMode:
        return AdapterMode.LIVE

    @property
    def metadata(self) -> MetadataSnapshot:
        return self._metadata

    def initial_candidates(self) -> list[CandidateProposal]:
        return list(self._seed_candidates)

    def fetch_correlations(self, external_alpha_id: str) -> dict[str, float | None]:
        """Read the platform's own self and production correlation for one alpha."""
        return self.client.fetch_correlations(external_alpha_id)

    def create_simulation(self, request: CreateSimulationRequest) -> CreateSimulationReceipt:
        return self.client.create_simulation(request)

    def poll_simulation(self, external_simulation_id: str) -> NormalizedSimulationResult:
        return self.client.poll_simulation(external_simulation_id)

    def request_platform_checks(self, external_alpha_id: str) -> list[PlatformCheck]:
        return self.client.request_platform_checks(external_alpha_id)

    def session_seconds_remaining(self) -> float | None:
        return self.client.session_seconds_remaining()


def _required_json(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError as exc:
        raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN response was not valid JSON") from exc


def _optional_json(response: httpx.Response) -> object | None:
    if not response.content:
        return None
    return _required_json(response)


def _session_remaining_seconds(payload: object) -> float | None:
    """Seconds left on the session, per `GET /authentication`'s own `token.expiry`."""
    if not isinstance(payload, Mapping):
        return None
    token = payload.get("token")
    if not isinstance(token, Mapping):
        return None
    expiry = token.get("expiry")
    if isinstance(expiry, int | float) and not isinstance(expiry, bool):
        return float(expiry)
    return None


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


def _parse_operator_definitions(payload: object) -> dict[str, str]:
    """Read each operator's call signature (e.g. "group_rank(x, group)") straight
    from BRAIN's own `/operators` response -- the one place this information
    exists. A name missing its own `definition` is just omitted, not fatal: the
    name set itself is still validated separately by `_parse_operators`.
    """
    definitions: dict[str, str] = {}
    for item in _record_items(payload):
        name = item.get("name") or item.get("id")
        definition = item.get("definition")
        if isinstance(name, str) and name and isinstance(definition, str) and definition:
            definitions[name] = definition
    return definitions


def _parse_max_correlation(payload: object) -> float | None:
    """Read the highest correlation BRAIN reports, or None when it reports none.

    The endpoint answers `{"schema": {...}, "records": [...], "max": ..., "min": ...}` and
    publishes `max` either at the top level or inside `schema`. An empty `records` list means
    the platform has not computed a correlation yet, which is not the same as zero.
    """
    if not isinstance(payload, Mapping):
        return None
    for holder in (payload.get("schema"), payload):
        if isinstance(holder, Mapping):
            value = holder.get("max")
            if isinstance(value, int | float) and not isinstance(value, bool):
                return float(value)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return None
    columns = []
    schema = payload.get("schema")
    if isinstance(schema, Mapping):
        properties = schema.get("properties")
        if isinstance(properties, list):
            columns = [item.get("name") for item in properties if isinstance(item, Mapping)]
    try:
        index = columns.index("max")
    except ValueError:
        return None
    observed = [
        float(row[index])
        for row in records
        if isinstance(row, list)
        and len(row) > index
        and isinstance(row[index], int | float)
        and not isinstance(row[index], bool)
    ]
    return max(observed) if observed else None


def _choice_values(node: object, context: Mapping[str, object]) -> set[str | int | float | bool]:
    """Collect `value` entries from a BRAIN choice tree, narrowed by the research cell.

    Region, Universe, Delay and Neutralization publish their choices nested by instrument
    type and then by region. When the cell pins one of those dimensions the matching branch
    is followed; otherwise every branch is unioned so nothing is silently dropped.
    """
    if isinstance(node, list):
        values: set[str | int | float | bool] = set()
        for item in node:
            if isinstance(item, Mapping):
                value = item.get("value")
                if isinstance(value, str | int | float | bool):
                    values.add(value)
        return values
    if isinstance(node, Mapping):
        values = set()
        for key, branch in node.items():
            selected: object | None = None
            if key in CHOICE_DIMENSIONS and key in context and isinstance(branch, Mapping):
                pinned = context[key]
                for lookup in (pinned, str(pinned)):
                    if lookup in branch:
                        selected = branch[lookup]
                        break
            values |= _choice_values(selected if selected is not None else branch, context)
        return values
    return set()


def _numeric_bounds(child: Mapping[str, Any]) -> tuple[float | None, float | None]:
    minimum = child.get("min_value", child.get("minValue", child.get("min")))
    maximum = child.get("max_value", child.get("maxValue", child.get("max")))
    lower = float(minimum) if isinstance(minimum, int | float) else None
    upper = float(maximum) if isinstance(maximum, int | float) else None
    return lower, upper


def _parse_setting_options(
    payload: object,
    *,
    instrument_type: str,
    region: str,
    delay: int,
    numeric_ladders: Mapping[str, tuple[float, ...]] | None = None,
) -> dict[str, set[str | int | float | bool]]:
    """Read `OPTIONS /simulations` into the settings a candidate may legally declare."""
    ladders = DEFAULT_NUMERIC_LADDERS if numeric_ladders is None else numeric_ladders
    children: object = None
    if isinstance(payload, Mapping):
        actions = payload.get("actions")
        if isinstance(actions, Mapping):
            post = actions.get("POST")
            if isinstance(post, Mapping):
                block = post.get("settings")
                if isinstance(block, Mapping):
                    children = block.get("children")
    if not isinstance(children, Mapping):
        raise AdapterFailure(
            ErrorClass.UNKNOWN,
            "BRAIN simulation options contained no settings schema",
        )
    context: dict[str, object] = {
        "instrumentType": instrument_type,
        "region": region,
        "delay": delay,
    }
    settings: dict[str, set[str | int | float | bool]] = {}
    for name, child in children.items():
        if not isinstance(child, Mapping):
            continue
        kind = str(child.get("type", "")).casefold()
        if "choices" in child:
            values = _choice_values(child["choices"], context)
            if values:
                settings[str(name)] = values
            continue
        if kind == "boolean":
            settings[str(name)] = {True, False}
            continue
        ladder = ladders.get(str(name))
        if ladder is None:
            continue
        lower, upper = _numeric_bounds(child)
        allowed = {
            value
            for value in ladder
            if (lower is None or value >= lower) and (upper is None or value <= upper)
        }
        if allowed:
            settings[str(name)] = {
                int(value) if float(value).is_integer() else float(value) for value in allowed
            }
    if not settings:
        raise AdapterFailure(ErrorClass.UNKNOWN, "BRAIN returned no valid settings")
    return settings


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
    if "required attribute" in lowered or "must have a value" in lowered:
        return ErrorClass.INVALID_OPERATOR
    # First real evidence for this branch: a live 2026-09-15 run's `group_rank(x)`
    # (called with 1 argument; it takes 2) came back as "Invalid number of inputs :
    # 1, should be exactly 2 input(s)." -- no "operator" substring, so it fell
    # through to PLATFORM_TERMINAL_FAILURE and never reached RepairPlanner even
    # though a wrong-arity operator call is exactly what MC-INVALID-OPERATOR-017
    # is for.
    if "number of input" in lowered:
        return ErrorClass.INVALID_OPERATOR
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
