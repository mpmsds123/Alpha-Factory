"""The LLM as researcher: it proposes ideas, then reads each round of results and decides
what happens next. Plus the submission description.

1. ideas()   dataset understanding (field descriptions + 6-way profile) + papers and
             forum/homework templates retrieved for this dataset -> a few ideas, each a
             mechanism with a direction, a template, the fields it applies to, windows
             and a decay that matches the horizon.
2. review()  every idea's results for the round -> dig near ore (writing variants), flip
             a reversed direction, fix a failing check, or drop; and whether the dataset
             deserves new ideas or is exhausted.

It proposes; code validates, expands, simulates, enforces budgets and stop rules, and
nothing is submitted without you. Every call is recorded with its token usage.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from alpha_orchestrator.factory.config import FactoryConfig, load_llm_key
from alpha_orchestrator.factory.generators import PLACEHOLDER
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import LLMCallORM

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
VERDICTS = frozenset({"dig", "flip", "fix", "drop"})
DECISIONS = frozenset({"continue", "new_ideas", "exhausted"})

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
    def ideas(
        self,
        dataset: dict[str, Any],
        fields: Sequence[FieldInfo],
        references: Sequence[dict[str, str]],
        operators: Sequence[str],
        tried: Sequence[dict[str, Any]],
        n: int = 4,
    ) -> list[dict[str, Any]]:
        prompt = {
            "task": f"Propose {n} distinct, testable alpha ideas for this dataset.",
            "dataset": dataset,
            "fields": [f.__dict__ for f in fields],
            "references": list(references),
            "operators": list(operators),
            "already_tried": list(tried),
            "rules": [
                "Start from the references: a paper naming this dataset, or a template that "
                "worked on similar data, adapted to these fields. Cite reference titles in "
                "sources; 'field descriptions' is allowed only with a concrete mechanism.",
                "mechanism: what economic behaviour the signal captures and why it predicts "
                "returns; direction: which stocks go long.",
                "template: a FASTEXPR expression where {A}, {B}, {C} stand for fields and "
                "{w} for a window. Code substitutes each field already prepared (its `prep`: "
                "backfill, vector reduction, winsorize), so do not wrap placeholders in "
                "ts_backfill or vec_* yourself. At most 3 placeholders, 8 operators.",
                "slots: for each placeholder the field ids it may take, all from `fields`, "
                "chosen because the mechanism applies to them (not every field).",
                "windows: from 5, 21, 63, 126, 252, never shorter than the field's update "
                "period. decay: consistent with the horizon of the template (a 5-day signal "
                "with decay 20 pulls in information the idea never used).",
                "neutralization: one of MARKET, SECTOR, INDUSTRY, SUBINDUSTRY.",
                "Do not repeat already_tried ideas; learn from why they were dropped.",
            ],
            "output": {
                "ideas": [
                    {
                        "title": "short name",
                        "mechanism": "one paragraph",
                        "direction": "high X -> long",
                        "sources": ["reference title"],
                        "template": "group_rank(ts_delta({A}, {w}), industry)",
                        "slots": {"A": ["field_id"]},
                        "windows": [21, 63],
                        "decay": 0,
                        "neutralization": "INDUSTRY",
                    }
                ]
            },
        }
        data = self.ask("ideas", json.dumps(prompt, ensure_ascii=False), strong=True)
        known = {f.id for f in fields}
        out = []
        for idea in data.get("ideas") or []:
            parsed = _parse_idea(idea, known)
            if parsed:
                out.append(parsed)
        return out

    def review(
        self,
        dataset: dict[str, Any],
        ideas: Sequence[dict[str, Any]],
        operators: Sequence[str],
        fields: Sequence[FieldInfo],
        dig_per_idea: int,
    ) -> dict[str, Any]:
        prompt = {
            "task": "Read this round's results and decide the next step for every idea, "
            "then for the dataset.",
            "dataset": dataset,
            "ideas": list(ideas),
            "fields": [f.__dict__ for f in fields],
            "operators": list(operators),
            "verdicts": {
                "dig": "ore found (|sharpe| at or above the ore line with a sensible "
                f"turnover): write up to {dig_per_idea} variants near the best expressions "
                "that test the mechanism more precisely (another related field, a group "
                "comparison, a conditioning event with trade_when, a residual against a "
                "control) - not window tuning. Use raw field ids with the same preparation "
                "the best expressions show.",
                "flip": "results are consistently and strongly negative: the direction was "
                "reversed; code will negate the best ones.",
                "fix": "positive signal blocked by one check; name it in `fix`: turnover, "
                "concentration or sub_universe.",
                "drop": "no signal or no consistent sign: say what that tells us.",
            },
            "dataset_decision": {
                "continue": "keep working the ideas that dig/flip/fix",
                "new_ideas": "the ideas are dropped but the data still looks promising: ask "
                "for new ideas",
                "exhausted": "stop this dataset; say why",
            },
            "output": {
                "ideas": [
                    {
                        "id": "idea id",
                        "verdict": "dig|flip|fix|drop",
                        "reason": "one or two sentences citing the numbers",
                        "fix": "turnover|concentration|sub_universe",
                        "expressions": [{"expression": "...", "why": "one sentence"}],
                    }
                ],
                "dataset": {"decision": "continue|new_ideas|exhausted", "reason": "..."},
            },
        }
        data = self.ask("review", json.dumps(prompt, ensure_ascii=False), strong=True)
        verdicts = []
        for v in data.get("ideas") or []:
            if not isinstance(v, dict) or v.get("verdict") not in VERDICTS:
                continue
            exprs = [
                e
                for e in v.get("expressions") or []
                if isinstance(e, dict) and isinstance(e.get("expression"), str)
            ][:dig_per_idea]
            verdicts.append(
                {
                    "id": str(v.get("id", "")),
                    "verdict": str(v["verdict"]),
                    "reason": str(v.get("reason", "")),
                    "fix": str(v.get("fix", "")),
                    "expressions": exprs,
                }
            )
        ds_raw = data.get("dataset")
        ds: dict[str, Any] = ds_raw if isinstance(ds_raw, dict) else {}
        decision = ds.get("decision") if ds.get("decision") in DECISIONS else "continue"
        return {
            "ideas": verdicts,
            "dataset": {"decision": decision, "reason": str(ds.get("reason", ""))},
        }

    def describe(self, alpha: dict[str, Any]) -> str:
        prompt = {
            "task": "Write the alpha description for submission: 'Idea:' then 'Rationale:', "
            "at least 100 words in English, plain and specific, no hype, no metrics.",
            "alpha": alpha,
            "output": {"description": "..."},
        }
        data = self.ask("description", json.dumps(prompt, ensure_ascii=False))
        return str(data.get("description", "")).strip()


def _parse_idea(idea: Any, known: set[str]) -> dict[str, Any] | None:
    """Keeps an idea only if every placeholder has at least one real field."""
    if not isinstance(idea, dict) or not idea.get("title") or not idea.get("mechanism"):
        return None
    template = str(idea.get("template") or "")
    raw_slots = idea.get("slots")
    raw: dict[str, Any] = raw_slots if isinstance(raw_slots, dict) else {}
    needed = {p for p in PLACEHOLDER.findall(template) if p != "w"}
    slots = {p: [f for f in raw.get(p) or [] if isinstance(f, str) and f in known] for p in needed}
    if not needed or any(not v for v in slots.values()):
        return None
    windows = [int(w) for w in idea.get("windows") or [] if str(w).isdigit() and int(w) > 0]
    try:
        decay = int(idea.get("decay") or 0)
    except (TypeError, ValueError):
        decay = 0
    return {
        "title": str(idea["title"])[:255],
        "mechanism": str(idea["mechanism"]),
        "direction": str(idea.get("direction", "")),
        "sources": [str(x) for x in idea.get("sources") or []],
        "template": template,
        "slots": slots,
        "windows": windows or [63],
        "decay": max(0, min(decay, 30)),
        "neutralization": str(idea.get("neutralization") or "").upper(),
    }
