"""LLM client protocol and response metadata."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMCompletion:
    raw_text: str
    model: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMClient(Protocol):
    @property
    def model(self) -> str:
        """Return the model name."""

    @property
    def temperature(self) -> float:
        """Return the sampling temperature."""

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> LLMCompletion:
        """Return a completion constrained to the given JSON schema."""
