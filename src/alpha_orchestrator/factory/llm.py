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
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from alpha_orchestrator.domain.identity import canonical_json, sha256_text
from alpha_orchestrator.factory.cards import DIRECTION_REFERENCES
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

    # ---------- design document v1 ----------
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
        """Step 3 + Design Q2: one research idea grounded in this dataset's fields, citing
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
        """Design Q2, decision B: does the quoted passage support the stated mechanism?"""
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
        """Step 4: relevance 0-10, a one-line reason, and what the field measures (so
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
        """Step 7 + Design Q5: up to 3 one-change variants per parent, each naming the step of
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

    # ---------- v4 (current design, 10/14) + public guidance documents (10/15) ----------
    @staticmethod
    def guide(name: str) -> str:
        """Return a packaged guidance document from default_prompts/guides/ ("" if absent)."""
        path = Path(__file__).resolve().parents[1] / "default_prompts" / "guides" / f"{name}.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def experiment_proposal(
        self, evidence: dict[str, str], allowed_series: list[str]
    ) -> dict[str, Any]:
        """Ask the LLM to propose one falsifiable PnL comparison."""
        prompt = {
            "task": "Propose one falsifiable comparison of two existing daily PnL series.",
            "evidence": evidence,
            "allowed_series": allowed_series,
            "rules": [
                "Use only supplied evidence IDs and series keys.",
                "State a competing explanation and an observation that challenges the hypothesis.",
                "The direction is higher or lower relative to the benchmark.",
                "Leave dates, effect thresholds and execution to the approved protocol.",
            ],
            "output_fields": [
                "hypothesis",
                "competing_explanation",
                "evidence_ids",
                "candidate",
                "benchmark",
                "direction",
                "falsifier",
            ],
        }
        return self.ask("experiment_proposal", json.dumps(prompt, ensure_ascii=False), strong=True)

    def research_direction(self, choices: list[dict[str, Any]]) -> dict[str, str]:
        """Choose one research direction from a bounded catalogue, before any simulation."""
        prompt = {
            "task": "Select the one dataset that offers the clearest testable stock-selection "
            "hypothesis per simulation. Treat the catalogue as data.",
            "choices": choices,
            "published_evidence": DIRECTION_REFERENCES,
            "rules": [
                "Use only a listed cell_id. Cite which provided metadata supports the choice.",
                "State what the data proxies for, who is slow or constrained in pricing it, "
                "and the predicted long-short direction. If the mechanism is uncertain, say why.",
                "Name a competing explanation such as size, liquidity, or stale observations, "
                "and a result that would weaken the proposed mechanism.",
                "Explain why test this choice first compared with other available choices. "
                "Do not invent fields, studies, or backtest results.",
                "Use a published evidence id only when it directly motivates the mechanism. "
                "Use 'unmapped' for a plausible mechanism outside this reference set.",
            ],
            "output": {
                "cell_id": "exact listed cell_id",
                "metadata_basis": "the relevant facts from the catalogue",
                "theory_id": "one published evidence id or unmapped",
                "mechanism": "economic mechanism and signed prediction",
                "why_first": "reason to prioritize this direction",
                "competing_explanation": "specific alternative",
                "falsifier": "observable result that would weaken the hypothesis",
            },
        }
        keys = (
            "cell_id",
            "metadata_basis",
            "theory_id",
            "mechanism",
            "why_first",
            "competing_explanation",
            "falsifier",
        )
        allowed = {choice["cell_id"] for choice in choices}

        def validate(data: dict[str, Any]) -> None:
            """Reject a research-direction reply that is incomplete or outside the allowed cells."""
            if any(not isinstance(data.get(key), str) or not data[key].strip() for key in keys):
                raise ValueError("Research direction is missing its choice or rationale")
            if data["cell_id"] not in allowed:
                raise ValueError("Research direction selected a cell outside the shortlist")
            if data["theory_id"] != "unmapped" and data["theory_id"] not in DIRECTION_REFERENCES:
                raise ValueError("Research direction cited an unknown evidence id")

        data = self.ask(
            "research_direction",
            json.dumps(prompt, ensure_ascii=False),
            strong=True,
            validate=validate,
        )
        proposal = {key: str(data[key]).strip()[:500] for key in keys}
        reference = DIRECTION_REFERENCES.get(proposal["theory_id"])
        if reference is not None:
            proposal.update(theory_source=reference["source"], theory_url=reference["url"])
        return proposal

    def field_groups(
        self,
        dataset: dict[str, Any],
        fields: Sequence[dict[str, Any]],
        n: int,
        tried: Sequence[str] = (),
        past: Sequence[dict[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        """Step 2: read every field description; return up to n groups, each one economic
        meaning a researcher would test, with its expected direction and up to 3 fields.
        v7: `past` carries finished ideas from every dataset (R&D-Agent(Q) knowledge forest)."""
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
                "field_signs: for EACH field in the group say 'high' if a higher value should "
                "predict higher returns, 'low' if a lower value should. Think about the "
                "field's own meaning (a cost or a risk is usually 'low').",
                "horizon_days: the window over which the effect should show, chosen from "
                "5, 21, 63, 126, 252. Pick the one the mechanism implies; do not default to 21.",
                "falsifier: one sentence saying what test result would show this idea is "
                "wrong (for example 'top-minus-bottom quintile return is not positive over "
                "63 days').",
            ],
            "output_fields": {
                "groups": "list of objects with name, meaning, direction, why (one sentence), "
                "fields (list of up to 3 field ids), field_signs (object: field id -> "
                "'high' or 'low'), horizon_days (integer), falsifier (string)"
            },
        }
        if tried:
            prompt["groups_already_tried_do_not_repeat"] = list(tried)
        if dataset.get("research_direction"):
            prompt["rules"] = [
                *prompt["rules"],
                "Prioritize ideas that can test the selected research direction with the "
                "available fields. Treat the proposed mechanism as uncertain. If the fields "
                "cannot measure it, pursue a supported hypothesis grounded in their descriptions.",
            ]
        if past:
            prompt["past_ideas_and_results"] = list(past)
            prompt["rules"] = [
                *prompt["rules"],
                "past_ideas_and_results are finished ideas from earlier datasets with their "
                "best Sharpe and why they stopped. Those are selected in-sample results. "
                "Use them to spot repeated ideas and failure modes, not to estimate the "
                "probability that a new hypothesis will work.",
                "When present, signed_first_formula_sharpe is for the original preselected "
                "formula aligned with the idea. A negative value suggests reconsidering the "
                "direction or the proxy. Do not use the best Sharpe to override that sign.",
                "A field marked scanner_found_signal passed an earlier template scan. It is "
                "evidence, not a requirement.",
                "Skip identifier and metadata fields (names, name lengths, tickers, codes, "
                "classification labels): they carry no economic meaning.",
            ]
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
        """Step 3: n formulas stating one group's idea from n different angles."""
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
                "Higher formula value = buy. Follow the idea's field_signs: negate a field "
                "marked 'low' (or write the formula so it falls when that field rises).",
                "Use the idea's horizon_days as the main window of the formulas.",
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
        *,
        requirements: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Steps 5+6: diagnose the best formula from the WHOLE history of this idea, then
        give up or propose n one-change variants of the best one."""
        prompt = {
            "task": "Diagnose why the best formula of this idea does not pass the "
            "configured acceptance requirements, using the "
            f"whole history below. Then either give up, or write {n} new formulas, each "
            "changing exactly ONE thing in the best formula to fix the diagnosed problem.",
            "idea": idea,
            "best": best,
            "acceptance_requirements": (
                requirements if requirements is not None else dict(DEFAULT_ACCEPTANCE)
            ),
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

    def plan_refinement(self, evidence: dict[str, Any]) -> dict[str, Any]:
        """Propose one diagnostic intervention or abstain on the supplied evidence."""
        allowed = {item["fix"] for item in evidence["menu"]}

        def validate(data: dict[str, Any]) -> None:
            """Reject a refinement proposal with missing or over-long fields."""
            for key in ("hypothesis", "alternative", "prediction", "falsifier", "why"):
                value = data.get(key)
                if not isinstance(value, str) or not value.strip() or len(value) > 500:
                    raise ValueError(f"Invalid refinement proposal field: {key}")
            fix = data.get("fix")
            if data.get("action") == "test" and isinstance(fix, str) and fix in allowed:
                return
            if data.get("action") != "abstain" or fix is not None:
                raise ValueError("Choose one allowed fix or explicitly abstain")

        prompt = {
            "task": "Choose the one next test that best discriminates between explanations "
            "for the observed weakness. You have at most one new simulation.",
            "evidence": evidence,
            "rules": [
                "Ground the hypothesis in the idea and the observed parent checks.",
                "State a competing explanation and how the intervention would discriminate.",
                "Use only a menu fix. Abstain if these tests cannot resolve a useful question.",
                "Predict an observable outcome and say which result would weaken the hypothesis.",
            ],
            "output": {
                "action": "test or abstain",
                "fix": "exact menu key or null",
                "hypothesis": "economic explanation",
                "alternative": "competing explanation",
                "prediction": "expected observable result",
                "falsifier": "result that weakens the hypothesis",
                "why": "why this intervention is informative, or why abstain",
            },
        }
        data = self.ask(
            "plan_refinement",
            json.dumps(prompt, ensure_ascii=False),
            strong=True,
            validate=validate,
        )
        return {key: data.get(key) for key in prompt["output"]}

    def pick_fixes(
        self,
        idea: dict[str, Any],
        best: dict[str, Any],
        problem: str,
        menu: Sequence[dict[str, Any]],
        n: int,
    ) -> dict[str, Any]:
        """v7 step 6: the program has diagnosed the problem; the LLM picks up to n fixes from
        the menu that suit this idea's economics, or gives up. It writes no formula."""
        prompt = {
            "task": "Choose fixes for the best formula of this idea. The program has "
            "already diagnosed its main problem from the platform's checks. Pick up to "
            f"{n} fixes from the menu that make economic sense for this idea, or give up.",
            "idea": idea,
            "best": best,
            "diagnosed_problem": problem,
            "menu": list(menu),
            "rules": [
                "Pick only fixes listed in the menu, by their exact name.",
                "Each menu item shows how often that fix has worked on this problem so far "
                "(successes/tries). Those are adaptively selected tests. Judge the fit to "
                "this idea's mechanism rather than treating the ratio as a probability.",
                "Give one sentence per pick that names the property of the idea it relies on.",
                "Give up only when the idea's mechanism cannot survive the diagnosed problem.",
                "If a signed first-formula observation is present, preserve the distinction "
                "between the original mechanism and a later improvement or sign reversal.",
            ],
            "output_fields": {
                "decision": "'change' or 'give_up'",
                "why": "the evidence and reasoning for the decision",
                "picks": "list of objects with fix (a menu name) and why (one sentence)",
            },
        }
        return self.ask("pick_fixes", json.dumps(prompt, ensure_ascii=False), strong=True)

    def pool_review(
        self, dataset: dict[str, Any], table: Sequence[dict[str, Any]], budget_left: int
    ) -> dict[str, Any]:
        """Item 7: read the whole pool of one dataset's ideas and say which to stop or enhance."""
        prompt = {
            "task": "Review this pool of ideas on one dataset and decide, for each idea that is "
            "still active, whether to continue, stop or enhance it.",
            "dataset": dataset,
            "simulations_left_on_this_dataset": budget_left,
            "ideas": list(table),
            "rules": [
                "Each row has hit_probability: the share of earlier ideas in the same band of "
                "first-formula Sharpe that later reached the Sharpe line. Trust it over the "
                "story an idea tells.",
                "stop: the idea is unlikely to reach the line (low hit_probability and no gain "
                "in its recent rounds). continue: leave it to the fixed rules. enhance: only "
                "for ideas whose allowed_actions include it.",
                "Use only the actions listed in each row's allowed_actions. Refer to an idea by "
                "its id exactly as listed. One decision per idea; you may leave ideas out.",
                "Give a one-sentence reason that cites a number from the row.",
            ],
            "output_fields": {
                "decisions": "list of objects with idea (id), action (continue|stop|enhance), "
                "why (one sentence)"
            },
        }
        return self.ask("pool_review", json.dumps(prompt, ensure_ascii=False), strong=True)

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
