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


class BrainAdapter(Protocol):
    @property
    def mode(self) -> AdapterMode: ...

    @property
    def metadata(self) -> MetadataSnapshot: ...

    def initial_candidates(self) -> list[CandidateProposal]: ...

    def create_simulation(self, request: CreateSimulationRequest) -> CreateSimulationReceipt: ...

    def poll_simulation(self, external_simulation_id: str) -> NormalizedSimulationResult: ...

    def request_platform_checks(self, external_alpha_id: str) -> list[PlatformCheck]: ...
