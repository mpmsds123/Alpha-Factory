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
    "fitted weights, never add terms only to lower correlation, never invent field names, "
    "never tune parameters just to raise in-sample sharpe. "
    "Answer with a single JSON object only."
)

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
    def group_fields(
        self, dataset: dict[str, Any], families: Sequence[dict[str, str]], max_groups: int = 40
    ) -> dict[str, str]:
        """family -> stratum name. Families the model leaves out keep no stratum."""
        prompt = {
            "task": f"Group these field families of one dataset into at most {max_groups} "
            "economic concepts (for example: EPS estimate level, EPS revisions, forecast "
            "dispersion, number of analysts, target price). Families measuring the same "
            "thing in different units or horizons belong together.",
            "dataset": dataset,
            "families": list(families),
            "output": {"groups": {"concept name": ["family"]}},
        }
        data = self.ask("group_fields", json.dumps(prompt, ensure_ascii=False))
        known = {f["family"] for f in families}
        out: dict[str, str] = {}
        raw = data.get("groups")
        groups: dict[str, Any] = raw if isinstance(raw, dict) else {}
        for name, members in groups.items():
            for fam in members if isinstance(members, list) else []:
                if isinstance(fam, str) and fam in known and fam not in out:
                    out[fam] = str(name)[:120]
        return out

    def combinations(
        self,
        dataset: dict[str, Any],
        strata: Sequence[dict[str, Any]],
        operators: Sequence[str],
        n: int,
    ) -> list[dict[str, str]]:
        """Up to n combinations of fields from the strata that showed signal."""
        prompt = {
            "task": f"Propose {n} combinations of two or three fields from these strata "
            "that could predict returns better than each field alone.",
            "dataset": dataset,
            "strata": list(strata),
            "operators": list(operators),
            "rules": [
                "Each strata entry lists its fields with `prep`, the exact form a field must "
                "be written in. Use those forms verbatim; do not invent fields.",
                "Ground each combination in what the fields mean and in the survey results "
                "(which shape worked, which sign). Typical reasons: one field conditions "
                "when another is reliable; a ratio expresses a meaningful quantity; a "
                "grouping by one field sharpens the other.",
                "At most 8 operators. No fitted weights, no constants tuned for sharpe.",
            ],
            "output": {
                "combinations": [{"expression": "...", "why": "one sentence on the mechanism"}]
            },
        }
        data = self.ask("combinations", json.dumps(prompt, ensure_ascii=False), strong=True)
        return [
            {"expression": c["expression"], "why": str(c.get("why", ""))}
            for c in (data.get("combinations") or [])[:n]
            if isinstance(c, dict) and isinstance(c.get("expression"), str)
        ]

    # ---------- 设计文档 v1 ----------
    # No example expression appears in any prompt: the model copies examples (v2 wrote only
    # group_rank, v1 copied rank(ts_delta({X}, 21)) into 11 of 36 ideas).
    def idea(
        self,
        dataset: dict[str, Any],
        fields: Sequence[dict[str, str]],
        papers: Sequence[dict[str, str]],
        tried: Sequence[str],
        dead_ends: Sequence[str] = (),
        worked: Sequence[str] = (),
    ) -> dict[str, Any]:
        """第 3 步: one research idea grounded in this dataset's fields. No formula yet.
        Dead ends and what worked (the knowledge base) keep it off old paths."""
        prompt = {
            "task": "Propose ONE new research idea for an alpha on this dataset.",
            "dataset": dataset,
            "fields": list(fields),
            "papers_naming_this_dataset": list(papers),
            "ideas_already_tried_here": list(tried),
            "dead_ends_elsewhere": list(dead_ends),
            "what_worked_elsewhere": list(worked),
            "questions_to_ask_of_the_fields": [
                "Which quantities are stable, and which change over time?",
                "What would an anomalous value mean?",
                "What should a value be compared with: its own history, peers, another field?",
                "Who wins and who loses when this quantity moves, and why does it persist?",
            ],
            "rules": [
                "The idea must be something these fields can measure.",
                "Do not repeat or rephrase an idea already tried here.",
                "Do not write any formula.",
            ],
            "output_fields": {
                "title": "string: short name",
                "mechanism": "string: who wins, who loses, why it persists",
                "long": "string: which stocks to buy",
                "short": "string: which stocks to sell",
                "horizon_days": "integer: how many trading days the effect takes",
                "query": "string: 5-15 English keywords describing the fields the idea needs",
            },
        }
        return self.ask("idea", json.dumps(prompt, ensure_ascii=False), strong=True)

    def score_fields(
        self, idea: dict[str, Any], fields: Sequence[dict[str, str]]
    ) -> dict[str, tuple[int, str, str]]:
        """第 4 步: relevance 0-10, a one-line reason, and what the field measures (so
        versions of the same quantity can be told apart)."""
        prompt = {
            "task": "Rate how well each field can carry this idea, 0 (unrelated) to 10 "
            "(exactly what the idea needs). Also name what the field measures in 2-4 "
            "English words; fields measuring the same quantity from different sources or "
            "versions must get exactly the same name.",
            "idea": idea,
            "fields": list(fields),
            "output_fields": {
                "scores": "list of objects with id (string), score (integer 0-10), "
                "reason (string, one line), measures (string, 2-4 words)"
            },
        }
        data = self.ask("score_fields", json.dumps(prompt, ensure_ascii=False))
        known = {f["id"] for f in fields}
        out: dict[str, tuple[int, str, str]] = {}
        for row in data.get("scores") or []:
            if not isinstance(row, dict) or row.get("id") not in known:
                continue
            try:
                score = max(0, min(10, int(row.get("score", 0))))
            except (TypeError, ValueError):
                continue
            measures = str(row.get("measures") or row["id"]).strip().lower()[:80]
            out[str(row["id"])] = (score, str(row.get("reason", ""))[:300], measures)
        return out

    def forms(
        self,
        idea: dict[str, Any],
        fields: Sequence[dict[str, str]],
        operators: Sequence[str],
        n: int,
    ) -> list[dict[str, str]]:
        """第 5 步: n different expressions of the idea, each with one {X} for a field."""
        prompt = {
            "task": f"Write {n} DIFFERENT expressions of this idea. Each uses exactly one "
            "placeholder {X} that will be replaced by one of the fields listed, already "
            "cleaned (filled and averaged). Each expression must express the idea in a "
            "different way: for example a change over time, a comparison with the stock's "
            "own history, a comparison with peers, the level itself, a change of the "
            "change. Choose windows from the idea's horizon.",
            "idea": idea,
            "fields_it_will_be_applied_to": list(fields),
            "operators": list(operators),
            "rules": [
                "Only listed operators. At most 4 operators per expression.",
                "Exactly one {X} per expression; no other field names.",
                "No two expressions may differ only by a window length.",
            ],
            "output_fields": {
                "forms": "list of objects with form (string, the expression with {X}) "
                "and way (string: which way of expressing the idea it is)"
            },
        }
        data = self.ask("forms", json.dumps(prompt, ensure_ascii=False), strong=True)
        return [
            {"form": f["form"], "way": str(f.get("way", ""))[:120]}
            for f in (data.get("forms") or [])
            if isinstance(f, dict) and isinstance(f.get("form"), str)
        ]

    def formulas(
        self,
        idea: dict[str, Any],
        fields: Sequence[dict[str, str]],
        operators: Sequence[str],
        n: int,
    ) -> list[dict[str, str]]:
        """Main arm: n formulas that state the idea itself, written with the idea's own
        fields (one to three per formula). Not a template: no placeholder."""
        prompt = {
            "task": f"Write {n} DIFFERENT formulas that each state this idea's mechanism "
            "directly. Use the fields by their exact id; a formula may combine one to three "
            "of them when the mechanism involves a relation between quantities. Each "
            "formula must express the mechanism in a different way.",
            "idea": idea,
            "fields": list(fields),
            "operators": list(operators),
            "rules": [
                "Only listed operators and listed field ids. At most 4 operators.",
                "Each formula must follow from the mechanism; say how in `why`.",
                "No two formulas may differ only by a window length or a sign.",
                "Fields are already cleaned: do not add filling or averaging.",
            ],
            "output_fields": {
                "formulas": "list of objects with formula (string) and why (string, one "
                "sentence linking it to the mechanism)"
            },
        }
        data = self.ask("formulas", json.dumps(prompt, ensure_ascii=False), strong=True)
        return [
            {"form": f["formula"], "way": str(f.get("why", ""))[:200]}
            for f in (data.get("formulas") or [])
            if isinstance(f, dict) and isinstance(f.get("formula"), str)
        ]

    def improve(
        self, idea: dict[str, Any], parents: Sequence[dict[str, Any]], changes: Sequence[str]
    ) -> list[dict[str, str]]:
        """第 7 步: up to 3 one-change variants per parent."""
        prompt = {
            "task": "For each parent expression propose up to 3 variants. Each variant "
            "makes exactly ONE change, chosen from the allowed changes, aimed at the "
            "parent's weakness (failing checks, low fitness, high turnover).",
            "idea": idea,
            "parents": list(parents),
            "allowed_changes": list(changes),
            "rules": [
                "Keep the same fields. Do not invent fields or operators.",
                "Change only one thing per variant and name it in `change`.",
                "`decay` is the simulation setting (0-512); set it only when the change "
                "is the decay change, otherwise omit it.",
            ],
            "output_fields": {
                "variants": "list of objects with parent (string id), expression (string), "
                "change (string), and decay (integer, only for the decay change)"
            },
        }
        data = self.ask("improve", json.dumps(prompt, ensure_ascii=False), strong=True)
        out = []
        for v in data.get("variants") or []:
            if isinstance(v, dict) and isinstance(v.get("expression"), str) and v.get("parent"):
                item = {
                    "parent": str(v["parent"]),
                    "expression": v["expression"],
                    "change": str(v.get("change", ""))[:200],
                }
                if isinstance(v.get("decay"), int):
                    item["decay"] = str(v["decay"])
                out.append(item)
        return out

    def describe(self, alpha: dict[str, Any]) -> str:
        prompt = {
            "task": "Write the alpha description for submission: 'Idea:' then 'Rationale:', "
            "at least 100 words in English, plain and specific, no hype, no metrics.",
            "alpha": alpha,
            "output": {"description": "..."},
        }
        data = self.ask("description", json.dumps(prompt, ensure_ascii=False))
        return str(data.get("description", "")).strip()
