"""Deterministic in-memory adapter for focused unit tests."""

from collections import defaultdict

from alpha_orchestrator.domain.enums import AdapterMode, ErrorClass
from alpha_orchestrator.domain.models import (
    AdapterFailure,
    CandidateProposal,
    CreateSimulationReceipt,
    CreateSimulationRequest,
    MetadataSnapshot,
    NormalizedSimulationResult,
    PlatformCheck,
    SimulationMetrics,
)


class FakeBrainAdapter:
    def __init__(
        self,
        metadata: MetadataSnapshot,
        candidates: list[CandidateProposal] | None = None,
        results: dict[str, NormalizedSimulationResult] | None = None,
        create_failures: dict[str, list[ErrorClass]] | None = None,
    ) -> None:
        self._metadata = metadata
        self._candidates = candidates or []
        self._results = results or {}
        self._create_failures = create_failures or {}
        self._calls: dict[str, int] = defaultdict(int)
        self._receipts: dict[str, CreateSimulationReceipt] = {}
        self._expressions: dict[str, str] = {}

    @property
    def metadata(self) -> MetadataSnapshot:
        """Return the synthetic metadata snapshot."""
        return self._metadata

    @property
    def mode(self) -> AdapterMode:
        """Report replay mode; the fake never contacts the platform."""
        return AdapterMode.REPLAY

    @property
    def create_call_count(self) -> int:
        """Return how many simulations were created (for idempotency tests)."""
        return sum(self._calls.values())

    def initial_candidates(self) -> list[CandidateProposal]:
        """Return the configured seed candidates."""
        return list(self._candidates)

    def create_simulation(self, request: CreateSimulationRequest) -> CreateSimulationReceipt:
        """Create a synthetic simulation, reusing the receipt for a repeated key."""
        existing = self._receipts.get(request.idempotency_key)
        if existing is not None:
            return existing
        expression = request.expression
        call_index = self._calls[expression]
        self._calls[expression] += 1
        failures = self._create_failures.get(expression, [])
        if call_index < len(failures):
            error_class = failures[call_index]
            raise AdapterFailure(error_class, f"Synthetic {error_class} failure")
        external_id = f"fake-{request.candidate_id}"
        receipt = CreateSimulationReceipt(external_simulation_id=external_id)
        self._receipts[request.idempotency_key] = receipt
        self._expressions[external_id] = expression
        return receipt

    def poll_simulation(self, external_simulation_id: str) -> NormalizedSimulationResult:
        """Return the scripted result for a synthetic simulation."""
        expression = self._expressions.get(external_simulation_id)
        if expression is None:
            raise AdapterFailure(ErrorClass.UNKNOWN, "Unknown synthetic simulation")
        return self._results.get(
            expression,
            NormalizedSimulationResult(
                completed=True,
                metrics=SimulationMetrics(sharpe=0.0, fitness=0.0, turnover=0.0),
            ),
        )

    def request_platform_checks(self, external_alpha_id: str) -> list[PlatformCheck]:
        """Return no platform checks."""
        del external_alpha_id
        return []

    def fetch_correlations(self, external_alpha_id: str) -> dict[str, float | None]:
        """Return no correlations."""
        del external_alpha_id
        return {}

    def session_seconds_remaining(self) -> float | None:
        """Return None: the fake has no session expiry."""
        return None
