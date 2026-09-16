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

# This adapter never holds credentials, so it can never auto-relogin (that is a red
# line: forced relogins have locked this account before). A session that is about to
# expire must therefore stop the run cleanly, before candidates spend the run's limit
# budget on requests a dying cookie cannot see through -- not fail unpredictably
# partway in. 2000 seconds is not a guess: it is the exact cutoff the reference client
# (`ace_lib.py`, 1,115 real simulations run through it) uses to decide a cached session
# is still worth reusing rather than treated as needing a fresh one. Shared here (not
# just in the live adapter) so the orchestrator can apply the same cutoff mid-run
# without importing a concrete adapter implementation.
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
