"""Explicitly configured OpenAI-compatible chat-completion client."""

import json
import os
from time import monotonic
from typing import Any, cast

import httpx2 as httpx
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from alpha_orchestrator.llm.base import LLMCompletion


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    api_key: SecretStr
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=900.0)

    @classmethod
    def from_environment(cls) -> "LLMSettings":
        api_key = os.getenv("ALPHA_LAB_LLM_API_KEY")
        model = os.getenv("ALPHA_LAB_LLM_MODEL")
        if not api_key or not model:
            raise ValueError("ALPHA_LAB_LLM_API_KEY and ALPHA_LAB_LLM_MODEL are required")
        return cls(
            api_key=SecretStr(api_key),
            model=model,
            base_url=os.getenv("ALPHA_LAB_LLM_BASE_URL") or None,
            temperature=float(os.getenv("ALPHA_LAB_LLM_TEMPERATURE", "0")),
            timeout_seconds=float(os.getenv("ALPHA_LAB_LLM_TIMEOUT_SECONDS", "60")),
        )


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        settings: LLMSettings,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings
        self._client = OpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=0,
            http_client=http_client,
        )

    @property
    def model(self) -> str:
        return self.settings.model

    @property
    def temperature(self) -> float:
        return self.settings.temperature

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> LLMCompletion:
        schema_instruction = json.dumps(json_schema, sort_keys=True, separators=(",", ":"))
        started = monotonic()
        completions = cast(Any, self._client.chat.completions)
        response = completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": f"{system_prompt}\n\nRequired JSON Schema:\n{schema_instruction}",
                },
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM response contained no text")
        usage = response.usage
        return LLMCompletion(
            raw_text=content,
            model=response.model,
            latency_ms=max(0, int((monotonic() - started) * 1_000)),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )
