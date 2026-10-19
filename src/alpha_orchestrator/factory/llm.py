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
from datetime import UTC, datetime, timedelta
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


def is_peak(now: datetime) -> bool:
    """DeepSeek peak hours: Beijing (UTC+8) Monday-Friday 9-12 and 14-18."""
    bj = now.astimezone(UTC) + timedelta(hours=8)
    return bj.weekday() < 5 and (9 <= bj.hour < 12 or 14 <= bj.hour < 18)


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
        self.idea_id: str | None = (
            None  # 第 9 问: set by the engine so each call is billed to an idea
        )
        self.unbilled: list[str] = []  # calls made while no idea existed yet (想点子)

    def cost(
        self, model: str, prompt_tokens: int, completion_tokens: int, now: datetime | None = None
    ) -> float:
        """第 9 问: yuan, tokens x DeepSeek's official price at the hour of the call."""
        table = self.config.llm.prices_cny.get(model)
        if not table:
            return 0.0
        price = table["peak"] if is_peak(now or datetime.now(UTC)) else table["off"]
        return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000

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
                    s.add(
                        LLMCallORM(
                            purpose=purpose,
                            model=model,
                            ok=False,
                            error=last_error,
                            idea_id=self.idea_id,
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
                )
                s.add(row)
                s.flush()
                if self.idea_id is None:
                    self.unbilled.append(row.id)
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
        cards: dict[str, str],
        documents: Sequence[dict[str, str]],
        tried: Sequence[str],
        dead_ends: Sequence[str] = (),
        worked: Sequence[str] = (),
        rejected: str = "",
    ) -> dict[str, Any]:
        """第 3 步 + 第 2 问: one research idea grounded in this dataset's fields, citing
        exactly one piece of evidence the program can check. No formula yet."""
        prompt: dict[str, Any] = {
            "task": "Propose ONE new research idea for an alpha on this dataset.",
            "dataset": dataset,
            "fields": list(fields),
            "method_cards": cards,
            "official_documents": list(documents),
            "ideas_already_tried_here": list(tried),
            "dead_ends_elsewhere": list(dead_ends),
            "what_worked_elsewhere": list(worked),
            "rules": [
                "The idea must be something these fields can measure.",
                "Do not repeat or rephrase an idea already tried here.",
                "Do not write any formula.",
                "Cite exactly one piece of evidence: either one method card by its id, or one "
                "official document by its id with a quote of 20-400 characters copied word "
                "for word from that document's text. The quote must support the mechanism.",
            ],
            "output_fields": {
                "title": "string: short name",
                "mechanism": "string: who wins, who loses, why it persists",
                "long": "string: which stocks to buy",
                "short": "string: which stocks to sell",
                "horizon_days": "integer: how many trading days the effect takes",
                "query": "string: 5-15 English keywords describing the fields the idea needs",
                "evidence": "object: {type: card, id} or {type: paper, id, quote}",
            },
        }
        if rejected:
            prompt["your_last_answer_was_rejected_because"] = rejected
        return self.ask("idea", json.dumps(prompt, ensure_ascii=False), strong=True)

    def supports(self, mechanism: str, quote: str) -> tuple[bool, str]:
        """第 2 问 决定 B: does the quoted passage support the stated mechanism?"""
        prompt = {
            "task": "Judge whether the quoted passage supports the claim. A passage about a "
            "different outcome (for example risk instead of returns) does not support it.",
            "claim": mechanism,
            "quote": quote,
            "output_fields": {"supports": "boolean", "why": "string, one sentence"},
        }
        data = self.ask("supports", json.dumps(prompt, ensure_ascii=False))
        return bool(data.get("supports") is True), str(data.get("why", ""))[:300]

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
                "Do not multiply three or more terms together; products of several "
                "ranked signals overfit and collapse out of sample.",
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
        self, idea: dict[str, Any], parents: Sequence[dict[str, Any]], steps: dict[str, str]
    ) -> list[dict[str, str]]:
        """第 7 步 + 第 5 问: up to 3 one-change variants per parent, each naming the step of
        the logic chain it applies (the change's evidence)."""
        prompt = {
            "task": "For each parent expression propose up to 3 variants. Each variant "
            "makes exactly ONE change, aimed at the parent's weakness (failing checks, low "
            "fitness, high turnover), and applies one step of the logic chain.",
            "idea": idea,
            "parents": list(parents),
            "logic_chain": steps,
            "rules": [
                "Keep the same fields. Do not invent fields or operators.",
                "Change only one thing per variant; name the step id in `step` and the "
                "change in `change`.",
                "`decay` is the simulation setting (0-512); set it only when the change "
                "is the decay change, otherwise omit it.",
            ],
            "output_fields": {
                "variants": "list of objects with parent (string id), expression (string), "
                "step (string, a logic_chain id), change (string), and decay (integer, only "
                "for the decay change)"
            },
        }
        data = self.ask("improve", json.dumps(prompt, ensure_ascii=False), strong=True)
        out = []
        for v in data.get("variants") or []:
            if not (isinstance(v, dict) and isinstance(v.get("expression"), str)):
                continue
            step = str(v.get("step", "")).strip().upper()
            if not v.get("parent") or step not in steps:
                continue  # a change without a step on the list has no evidence
            item = {
                "parent": str(v["parent"]),
                "expression": v["expression"],
                "change": f"{step}: {str(v.get('change', ''))[:180]}",
            }
            if isinstance(v.get("decay"), int):
                item["decay"] = str(v["decay"])
            out.append(item)
        return out

    # ---------- prompt guidance ----------
    @staticmethod
    def guide(name: str) -> str:
        """Load the repository's independently written, public prompt guidance."""
        from alpha_orchestrator.factory.config import REPO_ROOT

        path = REPO_ROOT / "prompts" / "guides" / f"{name}.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def field_groups(
        self,
        dataset: dict[str, Any],
        fields: Sequence[dict[str, str]],
        n: int,
        tried: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """第 2 步: read every field description; return up to n groups, each one economic
        meaning a researcher would test, with its expected direction and up to 3 fields."""
        prompt = {
            "task": f"You are a quant researcher reading a new dataset. Read the field "
            f"descriptions and pick up to {n} field groups worth testing as stock-selection "
            "signals. Each group is ONE economic meaning (e.g. 'number of analysts covering "
            "the stock'), with the expected direction and why it should predict returns.",
            "dataset": dataset,
            "fields": list(fields),
            "rules": [
                "Use field ids exactly as listed. At most 3 fields per group; prefer high "
                "coverage.",
                "Groups must differ in meaning, not just in which version of a field. At "
                "most 2 groups of the same kind (e.g. two 'estimate revision' groups); "
                "cover different kinds of meaning.",
                "direction: 'higher is better' or 'lower is better'.",
            ],
            "output_fields": {
                "groups": "list of objects with name, meaning, direction, why (one sentence), "
                "fields (list of up to 3 field ids)"
            },
        }
        if tried:
            prompt["groups_already_tried_do_not_repeat"] = list(tried)
        data = self.ask("field_groups", json.dumps(prompt, ensure_ascii=False), strong=True)
        return [g for g in (data.get("groups") or []) if isinstance(g, dict)][:n]

    def first_formulas(
        self,
        dataset: dict[str, Any],
        group: dict[str, Any],
        fields: Sequence[dict[str, str]],
        operators: Sequence[str],
        n: int,
        rejected: str = "",
    ) -> list[dict[str, str]]:
        """第 3 步: n formulas stating one group's idea from n different angles."""
        prompt = {
            "task": f"Write {n} formulas that test this idea, each from a different angle "
            "(for example: the level, the change over time, relative to the industry).",
            "dataset": dataset,
            "idea": group,
            "fields": list(fields),
            "operators": list(operators),
            "guide_single_factor_logic_chain": self.guide("logic_chain"),
            "guide_when_to_use_each_operator": self.guide("operators"),
            "rules": [
                "Use the two guides to choose operators for each angle.",
                "Group operators need a grouping input: industry, sector, subindustry, "
                "market, or bucket(...).",
                "Only listed operators and listed field ids. At most 3 fields and 5 "
                "operators per formula.",
                "Positional inputs exactly as each operator's definition; optional inputs "
                "only by name, e.g. winsorize(x, std=4).",
                "Higher formula value = buy. Follow the idea's direction.",
                "Fields are already cleaned: do not add filling.",
            ],
            "output_fields": {
                "formulas": "list of objects with formula (string) and why (one sentence)"
            },
        }
        if rejected:
            prompt["your_last_answer_was_rejected_because"] = rejected
        # writing formulas is mechanical next to judging results: the cheap model
        data = self.ask("first_formulas", json.dumps(prompt, ensure_ascii=False), strong=False)
        return [
            {"form": f["formula"], "way": str(f.get("why", ""))[:200]}
            for f in (data.get("formulas") or [])
            if isinstance(f, dict) and isinstance(f.get("formula"), str)
        ]

    def diagnose(
        self,
        idea: dict[str, Any],
        history: Sequence[dict[str, Any]],
        best: dict[str, Any],
        fields: Sequence[dict[str, str]],
        operators: Sequence[str],
        problems: Sequence[str],
        n: int,
        rejected: str = "",
    ) -> dict[str, Any]:
        """第 5+6 步: diagnose the best formula from the WHOLE history of this idea, then
        give up or propose n one-change variants of the best one."""
        prompt = {
            "task": "Diagnose why the best formula of this idea does not pass the "
            "submission line (Sharpe >= 1.58, Fitness >= 1, turnover 1%-70%), using the "
            f"whole history below. Then either give up, or write {n} new formulas, each "
            "changing exactly ONE thing in the best formula to fix the diagnosed problem.",
            "idea": idea,
            "best": best,
            "history_all_tries": list(history),
            "fields": list(fields),
            "operators": list(operators),
            "problems_to_choose_from": list(problems),
            "guide_single_factor_logic_chain": self.guide("logic_chain"),
            "rules": [
                "Pick ONE main problem from the list and quote the numbers that show it.",
                "Never repeat a formula in the history; learn from changes that made it worse.",
                "Give up when the history shows the idea cannot reach the line.",
                "Positional inputs exactly as each operator's definition; optional inputs "
                "only by name. At most 8 operators and 3 fields.",
                "`decay` is the simulation setting (0-512); set it only for a decay change.",
                "`neutralization` is the simulation setting, one of "
                "best.neutralization_options; set it only for a neutralization change.",
                "Changing only a window length or a number is not a change: change the "
                "structure (e.g. a step of the logic chain) or a setting.",
            ],
            "output_fields": {
                "problem": "one item from problems_to_choose_from",
                "evidence": "the numbers that show it",
                "decision": "'give_up' or 'change'",
                "why": "one sentence",
                "variants": "list of objects with expression, change (what and why), "
                "decay (integer, only for a decay change) and neutralization (only for a "
                "neutralization change); empty when giving up",
            },
        }
        if rejected:
            prompt["your_last_answer_was_rejected_because"] = rejected
        return self.ask("diagnose", json.dumps(prompt, ensure_ascii=False), strong=True)

    def describe(self, alpha: dict[str, Any]) -> str:
        prompt = {
            "task": "Write the alpha description for submission: 'Idea:' then 'Rationale:', "
            "at least 100 words in English, plain and specific, no hype, no metrics.",
            "alpha": alpha,
            "output": {"description": "..."},
        }
        data = self.ask("description", json.dumps(prompt, ensure_ascii=False))
        return str(data.get("description", "")).strip()
