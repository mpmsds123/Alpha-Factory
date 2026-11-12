"""The LLM layer's only door: strict JSON in, schema-checked JSON out, every call a
`decisions` row.

Rules enforced here, not in prompts:
  * the reply must validate against the step's schema (unknown keys rejected); one retry
    with the exact errors, then the step fails
  * every id the reply cites must be one the program retrieved for this call; an invented
    id is an error, an empty citation list is allowed but flagged "uncited"
  * no test-period number may appear in a request (the out-of-sample gate stays blind)
  * the day's dollar cap is checked before every call
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import func, select

from alpha_orchestrator.factory.config import FactoryConfig, load_llm_key
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import DecisionORM

TOKENS_PER_PRICE_UNIT = 1_000_000
WEEKDAYS = 5
FORBIDDEN_KEY_PREFIX = "test_"  # test-period numbers are the program's alone


class LLMUnavailable(RuntimeError):
    """Network, quota or provider failure: keep the research state and stop for now."""


class LLMInvalid(RuntimeError):
    """The model answered twice without a valid reply for this step."""

    def __init__(self, message: str, decision_id: int) -> None:
        super().__init__(message)
        self.decision_id = decision_id


class CostCapReached(RuntimeError):
    """Today's LLM dollar cap is spent."""


class Cited(BaseModel):
    """Every LLM answer says which retrieved records and priors it used."""

    model_config = ConfigDict(extra="forbid")
    cites: list[str] = Field(default_factory=list, description="ids like H12 or P7")
    no_relevant_record: bool = Field(
        default=False, description="true when nothing retrieved applied; then explain"
    )
    why_uncited: str = ""


@dataclass(frozen=True)
class Prompt:
    step: str
    version: str
    system: str
    task: str


S = TypeVar("S", bound=Cited)
Completion = Callable[[str, str, str, int], tuple[str, int, int]]


def _forbidden_keys(value: Any, path: str = "") -> list[str]:
    if isinstance(value, Mapping):
        out = []
        for k, v in value.items():
            if str(k).lower().startswith(FORBIDDEN_KEY_PREFIX):
                out.append(f"{path}.{k}")
            out.extend(_forbidden_keys(v, f"{path}.{k}"))
        return out
    if isinstance(value, list | tuple):
        return [x for i, v in enumerate(value) for x in _forbidden_keys(v, f"{path}[{i}]")]
    return []


class LLMClient:
    def __init__(
        self, config: FactoryConfig, ledger: Ledger, complete: Completion | None = None
    ) -> None:
        self.config = config
        self.ledger = ledger
        self._complete = complete or self._provider

    # ---------- money ----------
    def is_peak(self, now: datetime) -> bool:
        """Provider peak hours (UTC, Monday-Friday)."""
        t = now.astimezone(UTC)
        return t.weekday() < WEEKDAYS and any(
            lo <= t.hour < hi for lo, hi in self.config.llm.peak_hours_utc
        )

    def cost(self, model: str, prompt: int, completion: int, now: datetime | None = None) -> float:
        """US dollars at the provider's price for the hour of the call."""
        price = self.config.llm.prices_usd_peak.get(model)
        if not price:
            return 0.0
        factor = 1.0 if self.is_peak(now or datetime.now(UTC)) else 0.5
        return factor * (prompt * price[0] + completion * price[1]) / TOKENS_PER_PRICE_UNIT

    def spent_since(self, start: datetime) -> float:
        """Dollars spent on LLM calls since `start`."""
        with self.ledger.session() as s:
            return float(
                s.scalar(
                    select(func.coalesce(func.sum(DecisionORM.cost_usd), 0.0)).where(
                        DecisionORM.created_at >= start
                    )
                )
                or 0.0
            )

    def spent_today(self) -> float:
        """Dollars spent in the current quota window."""
        return self.spent_since(self.config.budget.window_start())

    # ---------- transport ----------
    def _provider(
        self, model: str, system: str, user: str, max_tokens: int
    ) -> tuple[str, int, int]:
        from openai import OpenAI

        client = OpenAI(
            api_key=load_llm_key(self.config.llm.key_file),
            base_url=self.config.llm.base_url,
            timeout=self.config.llm.timeout_s,
            max_retries=0,
        )
        with client:
            response = client.chat.completions.create(
                model=model,
                temperature=self.config.llm.temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_body={
                    "thinking": {"type": "enabled" if self.config.llm.thinking else "disabled"}
                },
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        usage = response.usage
        choice = response.choices[0]
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        content = choice.message.content or ""
        if choice.finish_reason != "stop":
            # a truncated reply is a bad answer, not an outage: it goes to the retry
            return content + "\x00truncated", prompt, completion
        return content, prompt, completion

    # ---------- the one entry point ----------
    def ask(
        self,
        prompt: Prompt,
        schema: type[S],
        payload: dict[str, Any],
        *,
        hypothesis_id: int | None,
        retrieved: dict[str, list[str]],
    ) -> tuple[S, int]:
        """Ask one decision; return the validated answer and its decisions row id."""
        leaked = _forbidden_keys(payload)
        if leaked:
            raise ValueError(f"test-period values may not reach the LLM: {leaked}")
        if self.spent_today() >= self.config.cost.daily_usd_cap:
            raise CostCapReached(f"daily LLM cap ${self.config.cost.daily_usd_cap} reached")
        step = prompt.step
        model = self.config.llm.models[step]
        max_tokens = self.config.llm.max_tokens.get(step, 1000)
        allowed = sorted({i for ids in retrieved.values() for i in ids})
        request = {
            "task": prompt.task,
            "input": payload,
            "allowed_citations": allowed,
            "answer_format": schema.model_json_schema(),
        }
        user = json.dumps(request, ensure_ascii=False, default=str)
        p_tok = c_tok = 0
        cost = 0.0
        errors: list[str] = []
        for attempt in (1, 2):
            message = user
            if errors:
                message += "\n\nYour previous answer was rejected: " + errors[-1][:1500]
            try:
                text, p, c = self._complete(model, prompt.system, message, max_tokens)
            except Exception as exc:  # network, auth, quota at the provider
                self._record(
                    prompt, model, hypothesis_id, payload, retrieved, {}, [], ["transport"],
                    p_tok, c_tok, cost, attempt, f"{type(exc).__name__}: {exc}"[:500],
                )  # fmt: skip
                raise LLMUnavailable(f"{step}: {type(exc).__name__}: {exc}") from exc
            p_tok, c_tok = p_tok + p, c_tok + c
            cost += self.cost(model, p, c)
            try:
                if text.endswith("\x00truncated"):
                    raise ValueError("the answer was cut off; answer more briefly")
                answer = schema.model_validate_json(text)
                bad = [i for i in answer.cites if i not in allowed]
                if bad:
                    raise ValueError(f"cited ids that were not provided: {bad}")
            except (ValidationError, ValueError) as exc:
                errors.append(str(exc))
                continue
            flags = []
            if not answer.cites:
                flags.append("uncited")
            if attempt > 1:
                flags.append("retried")
            decision_id = self._record(
                prompt, model, hypothesis_id, payload, retrieved, answer.model_dump(),
                list(answer.cites), flags, p_tok, c_tok, cost, attempt, None,
            )  # fmt: skip
            return answer, decision_id
        decision_id = self._record(
            prompt, model, hypothesis_id, payload, retrieved, {}, [], ["invalid"],
            p_tok, c_tok, cost, 2, " | ".join(errors)[:2000],
        )  # fmt: skip
        raise LLMInvalid(f"{step}: no valid answer after 2 tries", decision_id)

    def _record(
        self,
        prompt: Prompt,
        model: str,
        hypothesis_id: int | None,
        payload: dict[str, Any],
        retrieved: dict[str, list[str]],
        output: dict[str, Any],
        cited: list[str],
        flags: list[str],
        p_tok: int,
        c_tok: int,
        cost: float,
        attempts: int,
        error: str | None,
    ) -> int:
        with self.ledger.session() as s:
            row = DecisionORM(
                hypothesis_id=hypothesis_id,
                step=prompt.step,
                inputs=payload,
                retrieved=retrieved,
                output=output,
                cited=cited,
                flags=flags,
                prompt_version=prompt.version,
                model=model,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                cost_usd=cost,
                attempts=attempts,
                ok=error is None,
                error=error,
            )
            s.add(row)
            s.flush()
            return row.id


def record_program_decision(
    ledger: Ledger,
    hypothesis_id: int | None,
    step: str,
    output: dict[str, Any],
    inputs: dict[str, Any] | None = None,
) -> int:
    """A decision the program made itself (an override or a rule), in the same log."""
    with ledger.session() as s:
        row = DecisionORM(
            hypothesis_id=hypothesis_id,
            step=step,
            actor="program",
            inputs=inputs or {},
            output=output,
            prompt_version="rule",
        )
        s.add(row)
        s.flush()
        return row.id
