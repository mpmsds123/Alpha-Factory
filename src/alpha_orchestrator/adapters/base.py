"""The narrow service boundary used by the orchestrator."""

from typing import Protocol

from alpha_orchestrator.domain.enums import AdapterMode
from alpha_orchestrator.domain.models import (
    CandidateProposal,
    CreateSimulationReceipt,
    CreateSimulationRequest,
    MetadataSnapshot,
    NormalizedSimulationResult,
    PlatformCheck,
)

# This adapter never holds credentials and cannot auto-relogin. A session nearing
# expiry stops cleanly before new work is dispatched. The shared cutoff lets the
# orchestrator apply the same safety rule without importing a concrete adapter.
MINIMUM_SESSION_REMAINING_SECONDS = 2000.0


class BrainAdapter(Protocol):
    @property
    def mode(self) -> AdapterMode: ...

    @property
    def metadata(self) -> MetadataSnapshot: ...

    def initial_candidates(self) -> list[CandidateProposal]: ...

    def create_simulation(self, request: CreateSimulationRequest) -> CreateSimulationReceipt: ...

    def poll_simulation(self, external_simulation_id: str) -> NormalizedSimulationResult: ...

    def request_platform_checks(self, external_alpha_id: str) -> list[PlatformCheck]: ...

    def fetch_correlations(self, external_alpha_id: str) -> dict[str, float | None]: ...

    def session_seconds_remaining(self) -> float | None:
        """Seconds left on this adapter's session, or None if unknown or not
        applicable (replay/fake adapters have no session to expire). Estimated
        locally from the wall clock -- calling this must never itself reach the
        network, or checking freshness would spend the very budget it protects.
        """
        ...
