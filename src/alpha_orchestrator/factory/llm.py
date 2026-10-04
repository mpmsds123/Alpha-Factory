"""The LLM's four jobs: hypotheses, expressions, alignment review, descriptions.

It proposes; code validates, simulates and gates. Every hypothesis must cite the ideas it
came from, every expression must name the hypothesis it implements, and every call is
recorded in the ledger with its token usage.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from alpha_orchestrator.factory.config import FactoryConfig, load_llm_key
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import LLMCallORM

SYSTEM = (
    "You are a senior quantitative researcher on WorldQuant BRAIN writing FASTEXPR alphas. "
    "An alpha is a daily cross-sectional signal; positive values go long, negative short, and "
    "the platform neutralizes and scales it. Good alphas have an economic story explainable in "
    "one minute, use one dataset (ATOM) where possible, at most 3 data fields and 8 operators, "
    "and only meaningful windows (5, 21, 63, 126, 252 days). Never combine weak signals with "
    "fitted weights, never add terms only to lower correlation, never invent field names. "
    "Answer with a single JSON object only."
)

Completion = Callable[[str, str, str], tuple[str, int, int]]


@dataclass
class FieldInfo:
    id: str
    type: str
    description: str


class FactoryLLM:
    def __init__(
        self, config: FactoryConfig, ledger: Ledger, complete: Completion | None = None
    ) -> None:
        self.config = config
        self.ledger = ledger
        self._complete = complete or self._deepseek

    # ---------- transport ----------
    def _deepseek(self, model: str, system: str, user: str) -> tuple[str, int, int]:
        from openai import OpenAI

        client = OpenAI(
            api_key=load_llm_key(self.config.llm.key_file),
            base_url=self.config.llm.base_url,
            timeout=self.config.llm.timeout_s,
        )
        response = client.chat.completions.create(
            model=model,
            temperature=self.config.llm.temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        usage = response.usage
        return (
            response.choices[0].message.content or "{}",
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )

    def ask(self, purpose: str, user: str, *, strong: bool = False) -> dict[str, Any]:
        model = self.config.llm.strong_model if strong else self.config.llm.model
        last_error = ""
        for _ in range(2):
            try:
                text, p_tok, c_tok = self._complete(model, SYSTEM, user)
                data = json.loads(text)
                if not isinstance(data, dict):
                    raise ValueError("top-level JSON is not an object")
            except Exception as exc:  # network, quota, malformed JSON: all recorded
                last_error = f"{type(exc).__name__}: {exc}"[:500]
                with self.ledger.session() as s:
                    s.add(LLMCallORM(purpose=purpose, model=model, ok=False, error=last_error))
                continue
            with self.ledger.session() as s:
                s.add(
                    LLMCallORM(
                        purpose=purpose, model=model, prompt_tokens=p_tok, completion_tokens=c_tok
                    )
                )
            return data
        raise RuntimeError(f"LLM {purpose} failed twice: {last_error}")

    # ---------- jobs ----------
    def propose_hypotheses(
        self,
        cell: dict[str, Any],
        fields: Sequence[FieldInfo],
        ideas: Sequence[dict[str, str]],
        n: int = 4,
    ) -> list[dict[str, Any]]:
        prompt = {
            "task": f"Propose {n} distinct, testable economic hypotheses for this dataset.",
            "dataset": cell,
            "fields": [f.__dict__ for f in fields],
            "related_ideas": ideas,
            "rules": [
                "Each hypothesis must be grounded in one or more related_ideas (cite their "
                "titles in sources) or in the field descriptions; no free invention.",
                "State the predicted direction: which stocks should outperform and why.",
                "Use only field ids from `fields`; 1-3 fields per hypothesis.",
            ],
            "output": {
                "hypotheses": [
                    {
                        "title": "short name",
                        "statement": "one paragraph",
                        "direction": "high X -> long",
                        "fields": ["field_id"],
                        "sources": ["idea title"],
                    }
                ]
            },
        }
        data = self.ask("hypotheses", json.dumps(prompt, ensure_ascii=False), strong=True)
        known = {f.id for f in fields}
        out = []
        for h in data.get("hypotheses") or []:
            used = [f for f in h.get("fields") or [] if f in known]
            if h.get("title") and h.get("statement") and used:
                out.append({**h, "fields": used, "sources": list(h.get("sources") or [])})
        return out

    def write_expressions(
        self,
        hypothesis: dict[str, Any],
        fields: Sequence[FieldInfo],
        operators: Sequence[str],
        history: Sequence[dict[str, Any]],
        n: int = 6,
    ) -> list[dict[str, str]]:
        prompt = {
            "task": f"Write {n} FASTEXPR expressions that implement this hypothesis.",
            "hypothesis": hypothesis,
            "fields": [f.__dict__ for f in fields],
            "operators": list(operators),
            "previous_attempts": list(history),
            "rules": [
                "Only the listed operators and field ids. VECTOR fields must be reduced with a "
                "vec_* operator first. Sparse fundamentals/analyst data need ts_backfill(x, 63) "
                "first.",
                "Each expression must differ structurally (operator skeleton or field), not only "
                "in window length.",
                "If previous_attempts exist, learn from their metrics and failing checks; do not "
                "repeat an expression; do not tune parameters of an attempt with sharpe < 1.4.",
                "Prefer cross-sectional comparison inside groups (group_rank/group_zscore by "
                "industry or subindustry) when the field has strong industry structure.",
            ],
            "output": {"expressions": [{"expression": "...", "rationale": "one sentence"}]},
        }
        data = self.ask("expressions", json.dumps(prompt, ensure_ascii=False))
        return [
            e
            for e in data.get("expressions") or []
            if isinstance(e, dict) and isinstance(e.get("expression"), str)
        ]

    def aligned(self, hypothesis: dict[str, Any], expression: str) -> tuple[bool, str]:
        prompt = {
            "task": "Does this expression actually implement the hypothesis, with the stated "
            "direction? Be strict: a missing component or reversed sign is a no.",
            "hypothesis": hypothesis,
            "expression": expression,
            "output": {"consistent": True, "reason": "one sentence"},
        }
        data = self.ask("alignment", json.dumps(prompt, ensure_ascii=False))
        return bool(data.get("consistent")), str(data.get("reason", ""))

    def describe(self, alpha: dict[str, Any]) -> str:
        prompt = {
            "task": "Write the alpha description for submission: 'Idea:' then 'Rationale:', "
            "at least 100 words in English, plain and specific, no hype, no metrics.",
            "alpha": alpha,
            "output": {"description": "..."},
        }
        data = self.ask("description", json.dumps(prompt, ensure_ascii=False))
        return str(data.get("description", "")).strip()
