"""Sanitized JSON replay adapter; it performs no network I/O."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from pydantic import Field

from alpha_orchestrator.domain.enums import AdapterMode, ErrorClass
from alpha_orchestrator.domain.models import (
    AdapterFailure,
    CandidateProposal,
    CreateSimulationReceipt,
    CreateSimulationRequest,
    MetadataSnapshot,
    NormalizedSimulationResult,
    PlatformCheck,
    StrictModel,
)


class ReplayScenario(StrictModel):
    expression: str
    create_errors: list[ErrorClass] = Field(default_factory=list)
    result: NormalizedSimulationResult
    platform_checks: list[PlatformCheck] = Field(default_factory=list)
    correlations: dict[str, float | None] = Field(default_factory=dict)


class ReplayFixture(StrictModel):
    fixture_version: str
    metadata: MetadataSnapshot
    candidates: list[CandidateProposal]
    scenarios: list[ReplayScenario]
    regression_cases: list[str] = Field(default_factory=list)


class ReplayBrainAdapter:
    """Replays only explicit, sanitized fixtures from disk."""

    def __init__(self, fixture: ReplayFixture, fixture_path: Path | None = None) -> None:
        self.fixture = fixture
        self.fixture_path = fixture_path
        self._scenarios = {scenario.expression: scenario for scenario in fixture.scenarios}
        self._create_calls: dict[str, int] = defaultdict(int)
        self._receipts: dict[str, CreateSimulationReceipt] = {}
        self._external_to_expression: dict[str, str] = {}
        self._candidate_to_expression: dict[str, str] = {}
        for expression in self._scenarios:
            external_id = self._external_id(expression)
            self._external_to_expression[external_id] = expression

    @classmethod
    def from_path(cls, path: Path) -> "ReplayBrainAdapter":
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        return cls(ReplayFixture.model_validate(payload), path)

    @property
    def metadata(self) -> MetadataSnapshot:
        return self.fixture.metadata

    @property
    def mode(self) -> AdapterMode:
        return AdapterMode.REPLAY

    @property
    def create_call_count(self) -> int:
        return sum(self._create_calls.values())

    def initial_candidates(self) -> list[CandidateProposal]:
        return list(self.fixture.candidates)

    @staticmethod
    def _external_id(expression: str) -> str:
        digest = hashlib.sha256(expression.encode()).hexdigest()[:20]
        return f"replay-{digest}"

    def create_simulation(self, request: CreateSimulationRequest) -> CreateSimulationReceipt:
        self._candidate_to_expression[str(request.candidate_id)] = request.expression
        existing = self._receipts.get(request.idempotency_key)
        if existing is not None:
            return existing
        scenario = self._scenarios.get(request.expression)
        if scenario is None:
            raise AdapterFailure(ErrorClass.UNKNOWN, "Replay scenario is missing")
        call_index = self._create_calls[request.expression]
        self._create_calls[request.expression] += 1
        if call_index < len(scenario.create_errors):
            error_class = scenario.create_errors[call_index]
            raise AdapterFailure(error_class, f"Sanitized replay error: {error_class}")
        receipt = CreateSimulationReceipt(
            external_simulation_id=self._external_id(request.expression)
        )
        self._receipts[request.idempotency_key] = receipt
        return receipt

    def poll_simulation(self, external_simulation_id: str) -> NormalizedSimulationResult:
        expression = self._external_to_expression.get(external_simulation_id)
        if expression is None:
            raise AdapterFailure(ErrorClass.UNKNOWN, "Replay simulation ID is missing")
        return self._scenarios[expression].result

    def request_platform_checks(self, external_alpha_id: str) -> list[PlatformCheck]:
        expression = self._candidate_to_expression.get(external_alpha_id, external_alpha_id)
        scenario = self._scenarios.get(expression)
        return list(scenario.platform_checks) if scenario else []

    def fetch_correlations(self, external_alpha_id: str) -> dict[str, float | None]:
        expression = self._candidate_to_expression.get(external_alpha_id, external_alpha_id)
        scenario = self._scenarios.get(expression)
        return dict(scenario.correlations) if scenario else {}

    def session_seconds_remaining(self) -> float | None:
        return None
