"""DeepSeek's three jobs, each one a decision plain templates cannot make.

1. group_fields()  sort a dataset's field families into economic concepts (strata) by
                   reading names and descriptions, so sampling has meaningful layers.
                   Cheap model; only used when field names do not show the structure.
2. combinations()  inside the strata that showed signal, pick which pairs of fields to
                   combine and how, by what they mean. Millions of pairs exist and none
                   has been tested, so statistics alone cannot choose. Every LLM combination
                   is simulated next to the same number of random ones; if it does not
                   beat random, it stops getting budget.
3. describe()      the submission description.

Code validates, simulates, gates and decides budgets; nothing is submitted without you.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from alpha_orchestrator.factory.config import FactoryConfig, load_llm_key
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import LLMCallORM
from alpha_orchestrator.factory.util import canonical_json, sha256_text

SYSTEM = (
    "You are a senior quantitative researcher on WorldQuant BRAIN writing FASTEXPR alphas. "
    "An alpha is a daily cross-sectional signal; positive values go long, negative short, and "
    "the platform neutralizes and scales it. Good alphas have an economic story explainable in "
    "one minute, use one dataset (ATOM) where possible, at most 3 data fields and 8 operators, "
    "and only meaningful windows (5, 21, 63, 126, 252 days). Never combine weak signals with "
    "fitted weights, never add terms only to lower correlation, never invent field names, "
    "never tune parameters just to raise in-sample sharpe. "
    "Answer with a single JSON object only."
)


# The provider prices by Beijing time (UTC+8): weekdays 09-12 and 14-18 are peak hours.
PRICING_UTC_OFFSET_HOURS = 8
PEAK_HOUR_RANGES = ((9, 12), (14, 18))
WEEKDAYS = 5  # Monday-Friday (datetime.weekday() 0-4)
TOKENS_PER_PRICE_UNIT = 1_000_000  # prices are quoted per million tokens
# Default acceptance requirements shown to the refinement prompt when the caller passes
# none; they mirror the platform's regular submission line.
DEFAULT_ACCEPTANCE = {
    "sharpe_min": 1.58,
    "fitness_min": 1.0,
    "turnover_min": 0.01,
    "turnover_max": 0.70,
}


def is_peak(now: datetime) -> bool:
    """Return True during the provider's peak-price hours (Beijing time, weekdays)."""
    bj = now.astimezone(UTC) + timedelta(hours=PRICING_UTC_OFFSET_HOURS)
    return bj.weekday() < WEEKDAYS and any(lo <= bj.hour < hi for lo, hi in PEAK_HOUR_RANGES)


class LLMUnavailable(RuntimeError):
    """Infrastructure/output failure; preserve the research state for a retry."""


class IncompleteOutput(ValueError):
    def __init__(self, text: str, prompt: int, completion: int, reason: str) -> None:
        super().__init__(f"Incomplete LLM output: {reason}")
        self.text, self.prompt_tokens, self.completion_tokens = text, prompt, completion


Completion = Callable[[str, str, str], tuple[str, int, int]]


@dataclass
class FieldInfo:
    id: str
    type: str
    description: str
    profile: str = ""
    prep: str = ""


class FactoryLLM:
    def __init__(
        self, config: FactoryConfig, ledger: Ledger, complete: Completion | None = None
    ) -> None:
        self.config = config
        self.ledger = ledger
        self._complete = complete or self._deepseek
        # Design Q9: set by the engine so each call is billed to an idea
        self.idea_id: str | None = None
        self.unbilled: list[str] = []  # calls made while no idea existed yet (idea generation)

    def cost(
        self, model: str, prompt_tokens: int, completion_tokens: int, now: datetime | None = None
    ) -> float:
        """Design Q9: yuan, tokens x the provider's official price at the hour of the call."""
        table = self.config.llm.prices_cny.get(model)
        if not table:
            return 0.0
        price = table["peak"] if is_peak(now or datetime.now(UTC)) else table["off"]
        return (prompt_tokens * price[0] + completion_tokens * price[1]) / TOKENS_PER_PRICE_UNIT

    # ---------- transport ----------
    def _deepseek(self, model: str, system: str, user: str) -> tuple[str, int, int]:
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
                max_tokens=self.config.llm.max_completion_tokens,
                response_format={"type": "json_object"},
                extra_body={
                    "thinking": {"type": "enabled" if self.config.llm.thinking else "disabled"}
                },
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            usage = response.usage
            choice = response.choices[0]
            content = choice.message.content or ""
            if choice.finish_reason != "stop" or not content.strip():
                raise IncompleteOutput(
                    content,
                    getattr(usage, "prompt_tokens", 0) or 0,
                    getattr(usage, "completion_tokens", 0) or 0,
                    str(choice.finish_reason),
                )
            return (
                content,
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
            )

    def ask(
        self,
        purpose: str,
        user: str,
        *,
        strong: bool = False,
        validate: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Send one budgeted, ledgered JSON request to the LLM and return its reply."""
        model = self.config.llm.strong_model if strong else self.config.llm.model
        provenance = {
            "request_hash": sha256_text(
                canonical_json(
                    {
                        "model": model,
                        "system": SYSTEM,
                        "user": user,
                        "temperature": self.config.llm.temperature,
                        "max_tokens": self.config.llm.max_completion_tokens,
                        "thinking": self.config.llm.thinking,
                    }
                )
            ),
            "system_hash": sha256_text(SYSTEM),
        }
        last_error = ""
        for _ in range(2):
            start = datetime.combine(datetime.now(UTC).date(), datetime.min.time(), tzinfo=UTC)
            with self.ledger.session() as s:
                used = int(
                    s.scalar(
                        select(func.count())
                        .select_from(LLMCallORM)
                        .where(LLMCallORM.created_at >= start)
                    )
                    or 0
                )
            if used >= self.config.llm.max_calls_per_day:
                raise LLMUnavailable("Daily LLM call budget exhausted")
            text = ""
            p_tok, c_tok = 0, 0
            try:
                text, p_tok, c_tok = self._complete(model, SYSTEM, user)
                data = json.loads(text)
                if not isinstance(data, dict):
                    raise ValueError("top-level JSON is not an object")
                if not data:
                    raise ValueError("empty JSON object")
                if purpose == "diagnose":
                    if any(
                        not isinstance(data.get(k), str) or not data[k].strip()
                        for k in ("problem", "evidence", "decision", "why")
                    ):
                        raise ValueError("diagnosis is missing decision or evidence")
                    if data["decision"] not in {"change", "give_up"}:
                        raise ValueError("invalid diagnosis decision")
                    if not isinstance(data.get("variants"), list):
                        raise ValueError("diagnosis variants must be a list")
                if purpose == "pick_fixes":
                    if data.get("decision") not in {"change", "give_up"}:
                        raise ValueError("invalid pick_fixes decision")
                    if not isinstance(data.get("picks", []), list):
                        raise ValueError("pick_fixes picks must be a list")
                    if data["decision"] == "give_up" and (
                        not isinstance(data.get("why"), str) or not data["why"].strip()
                    ):
                        raise ValueError("pick_fixes must explain a decision to give up")
                if validate is not None:
                    validate(data)
            except Exception as exc:  # network, quota, malformed JSON: all recorded
                if isinstance(exc, IncompleteOutput):
                    text, p_tok, c_tok = exc.text, exc.prompt_tokens, exc.completion_tokens
                last_error = f"{type(exc).__name__}: {exc}"[:500]
                with self.ledger.session() as s:
                    s.add(
                        LLMCallORM(
                            purpose=purpose,
                            model=model,
                            ok=False,
                            prompt_tokens=p_tok,
                            completion_tokens=c_tok,
                            cost_cny=self.cost(model, p_tok, c_tok),
                            error=last_error,
                            idea_id=self.idea_id,
                            **provenance,
                            response_hash=sha256_text(text) if text else "",
                        )
                    )
                continue
            with self.ledger.session() as s:
                row = LLMCallORM(
                    purpose=purpose,
                    model=model,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    cost_cny=self.cost(model, p_tok, c_tok),
                    idea_id=self.idea_id,
                    **provenance,
                    response_hash=sha256_text(text),
                )
                s.add(row)
                s.flush()
                if self.idea_id is None:
                    self.unbilled.append(row.id)
            return data
        raise LLMUnavailable(f"LLM {purpose} failed twice: {last_error}")

    # ---------- jobs ----------
    def describe(self, alpha: dict[str, Any]) -> str:
        """Ask the LLM to write an alpha's submission description."""
        prompt = {
            "task": "Write the alpha description for submission: 'Idea:' then 'Rationale:', "
            "at least 100 words in English, plain and specific, no hype, no metrics.",
            "alpha": alpha,
            "output": {"description": "..."},
        }
        data = self.ask("description", json.dumps(prompt, ensure_ascii=False))
        return str(data.get("description", "")).strip()
