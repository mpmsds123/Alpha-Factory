import json

import httpx2 as httpx
import pytest
from pydantic import SecretStr

from alpha_orchestrator.llm.openai_client import LLMSettings, OpenAICompatibleLLMClient


def test_openai_compatible_client_uses_json_mode_and_normalizes_usage() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture-model-response",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"value":1}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 4,
                    "total_tokens": 15,
                },
            },
        )

    settings = LLMSettings(
        model="fixture-model",
        api_key=SecretStr("EXAMPLE_NOT_A_REAL_KEY"),
        base_url="https://llm.invalid/v1",
        temperature=0.1,
    )
    client = OpenAICompatibleLLMClient(
        settings, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    completion = client.complete(
        system_prompt="Return JSON.",
        user_prompt='{"input":1}',
        json_schema={"type": "object"},
    )
    assert client.model == "fixture-model"
    assert client.temperature == 0.1
    assert completion.raw_text == '{"value":1}'
    assert completion.model == "fixture-model-response"
    assert completion.total_tokens == 15
    assert requests
    captured = json.loads(requests[0].content)
    assert captured["response_format"] == {"type": "json_object"}
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert "Required JSON Schema" in messages[0]["content"]


def test_llm_settings_load_only_from_explicit_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "ALPHA_LAB_LLM_API_KEY",
        "ALPHA_LAB_LLM_MODEL",
        "ALPHA_LAB_LLM_BASE_URL",
        "ALPHA_LAB_LLM_TEMPERATURE",
        "ALPHA_LAB_LLM_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="required"):
        LLMSettings.from_environment()
    monkeypatch.setenv("ALPHA_LAB_LLM_API_KEY", "EXAMPLE_NOT_A_REAL_KEY")
    monkeypatch.setenv("ALPHA_LAB_LLM_MODEL", "fixture-model")
    monkeypatch.setenv("ALPHA_LAB_LLM_BASE_URL", "https://llm.invalid/v1")
    monkeypatch.setenv("ALPHA_LAB_LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("ALPHA_LAB_LLM_TIMEOUT_SECONDS", "12")
    settings = LLMSettings.from_environment()
    assert settings.model == "fixture-model"
    assert settings.temperature == 0.2
    assert settings.timeout_seconds == 12
    assert "EXAMPLE_NOT_A_REAL_KEY" not in repr(settings)
    monkeypatch.setenv("ALPHA_LAB_LLM_BASE_URL", "")
    assert LLMSettings.from_environment().base_url is None


def test_openai_client_rejects_empty_response_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-empty",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": None},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client = OpenAICompatibleLLMClient(
        LLMSettings(model="fixture", api_key=SecretStr("EXAMPLE")),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ValueError, match="no text"):
        client.complete(system_prompt="x", user_prompt="y", json_schema={})
