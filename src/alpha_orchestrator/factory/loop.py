"""The research loop: one hypothesis at a time, searched by LLM-guided mutation.

    pick field (program) -> (a) hypothesis -> (b) 1-3 expressions -> simulate
      -> (c) diagnosis -> stopping rule (program) -> (d) mutate / revise / drop -> ...

The program owns every budget and stop; the LLM proposes inside them. A child must differ
from its parent by exactly the change it declares. A hypothesis closes with a verdict that
Memory B keeps and that updates the priors it cited.
"""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.factory import catalog, gates, memory, prompts, sampler, tree
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger, fingerprint, ledger_locked
from alpha_orchestrator.factory.llm import (
    CostCapReached,
    LLMClient,
    LLMInvalid,
    LLMUnavailable,
    record_program_decision,
)
from alpha_orchestrator.factory.models import (
    CatalogORM,
    DecisionORM,
    ExpressionORM,
    HStatus,
    HypothesisORM,
    Review,
    SimStatus,
    SimulationORM,
    Verdict,
)
from alpha_orchestrator.factory.scheduler import Simulator
from alpha_orchestrator.factory.util import utc_now
from alpha_orchestrator.factory.validate import check, signature
from alpha_orchestrator.factory.vocab import Vocabulary

TRUNCATION = 0.08  # accepted by the platform on 1,000+ of this account's simulations
MAX_DECAY = 512  # the platform's largest decay
FIELD_DRAWS = 20  # datasets tried per new hypothesis before giving up for now
DESCRIPTION_CHARS = 500
LABEL_TO_VERDICT = {
    "idea_supported": Verdict.SUPPORTED,
    "idea_failed": Verdict.IDEA_FAILED,
    "implementation_failed": Verdict.IMPLEMENTATION_FAILED,
}
SCANNER_FILE = "alpha总表.csv"  # the template scanner's master table
SCANNER_COLUMNS = ("地区", "数据集", "公式")  # region, dataset, formula


@dataclass
class Meta:
    """Account facts read once per run from the platform."""

    operators: set[str] = field(default_factory=set)
    definitions: dict[str, str] = field(default_factory=dict)
    signatures: dict[str, tuple[int, int]] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fetch(cls, api: BrainAPI) -> "Meta":
        ops = [o for o in api.operators() if "REGULAR" in (o.get("scope") or ["REGULAR"])]
        return cls(
            operators={str(o["name"]).lower() for o in ops},
            definitions={
                str(o["name"]).lower(): str(o.get("definition") or o["name"])[:160] for o in ops
            },
            signatures={
                str(o["name"]).lower(): sig
                for o in ops
                if (
                    sig := signature(
                        str(o.get("definition") or ""), str(o.get("description") or "")
                    )
                )
            },
            options=api.simulation_options(),
        )

    def neutralizations(self, region: str) -> list[str]:
        return list(self.options.get(region, {}).get("neutralizations", []))


def scanner_shapes(folder: str, region: str, dataset: str, vocab: Vocabulary) -> set[str]:
    """Shapes of every formula the template scanner already ran on this dataset."""
    import csv

    path = Path(folder).expanduser() / SCANNER_FILE
    if not folder or not path.is_file():
        return set()
    col_region, col_dataset, col_formula = SCANNER_COLUMNS
    out = set()
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if (r.get(col_region), r.get(col_dataset)) != (region, dataset):
                continue
            try:
                out.add(tree.shape(tree.parse(r.get(col_formula) or ""), vocab))
            except ValueError:
                continue
    return out


class Factory:
    def __init__(
        self,
        api: BrainAPI,
        ledger: Ledger,
        config: FactoryConfig,
        llm: LLMClient,
        vocab: Vocabulary,
        meta: Meta,
        rng: random.Random | None = None,
        may_open: Callable[[], str] = lambda: "",
    ) -> None:
        self.api, self.ledger, self.config, self.llm = api, ledger, config, llm
        self.vocab, self.meta = vocab, meta
        self.rng = rng or random.Random()
        self.may_open = may_open
        self._scanner: dict[tuple[str, str], set[str]] = {}

    # ======================= driving =======================
    def advance(self) -> dict[str, int]:
        """Step every open hypothesis whose simulations are back; open new ones while
        there is room in the queue and the cost guard allows."""
        moved = {"stepped": 0, "opened": 0}
        with self.ledger.session() as s:
            open_ids = list(
                s.scalars(select(HypothesisORM.id).where(HypothesisORM.status == HStatus.OPEN))
            )
        for hid in open_ids:
            if self._pending(hid) == 0:
                self.step(hid)
                moved["stepped"] += 1
        b = self.config.budget
        while self._open_count() < b.active_hypotheses:
            with self.ledger.session() as s:
                if self.ledger.pending(s) >= b.queue_cap:
                    break
            reason = self.may_open()
            if reason:
                moved["guard"] = 1
                self.ledger.log("open_blocked", reason=reason)
                break
            if self.open_new() is None:
                break
            moved["opened"] += 1
        return moved

    def _open_count(self) -> int:
        with self.ledger.session() as s:
            return len(
                list(
                    s.scalars(select(HypothesisORM.id).where(HypothesisORM.status == HStatus.OPEN))
                )
            )

    def _pending(self, hid: int) -> int:
        with self.ledger.session() as s:
            return len(
                list(
                    s.scalars(
                        select(SimulationORM.id).where(
                            SimulationORM.hypothesis_id == hid,
                            SimulationORM.status.in_([SimStatus.QUEUED, SimStatus.RUNNING]),
                        )
                    )
                )
            )

    # ======================= (a) hypothesis =======================
    def open_new(self) -> int | None:
        """Pick a field, ask for a hypothesis, then for its first expressions."""
        pick = self._pick()
        if pick is None:
            return None
        with self.ledger.session() as s:
            h = HypothesisORM(
                region=pick.dataset.region,
                delay=pick.dataset.delay,
                universe=pick.dataset.universe,
                dataset_id=pick.dataset.dataset_id,
                dataset_name=pick.dataset.dataset_name,
                category=pick.dataset.category,
                field_id=pick.field.field_id,
                field_type=pick.field.field_type,
                field_description=pick.field.description,
                sims_budget=self.config.search.hypothesis_budget,
                state={
                    "family": [f.field_id for f in pick.family],
                    "helpers": {f.field_id: f.field_type for f in pick.helpers},
                    "family_types": {f.field_id: f.field_type for f in pick.family},
                },
            )
            s.add(h)
            s.flush()
            hid = h.id
            tags = memory.tags_of(h)
            records = memory.retrieve_records(s, tags, self.config.memory, exclude={hid})
            priors = memory.retrieve_priors(s, tags, self.config.memory)
            payload = {
                "field": sampler.field_view(pick.field),
                "dataset": {
                    "name": pick.dataset.dataset_name,
                    "category": pick.dataset.category,
                    "description": pick.dataset.description[:DESCRIPTION_CHARS],
                },
                "region": pick.dataset.region,
                "helper_candidates": [sampler.field_view(f) for f in pick.helpers],
                "records": [memory.record_view(r) for r in records],
                "priors": [memory.prior_view(p) for p in priors],
            }
            retrieved = {
                "records": [f"H{r.id}" for r in records],
                "priors": [f"P{p.id}" for p in priors],
            }
        self.ledger.log("hypothesis_opened", hypothesis=hid, field=pick.field.field_id)
        try:
            out, _ = self.llm.ask(
                prompts.HYPOTHESIS,
                prompts.HypothesisOut,
                payload,
                hypothesis_id=hid,
                retrieved=retrieved,
            )
        except LLMInvalid as exc:
            self._close(hid, Verdict.INVALID, str(exc))
            return hid
        with self.ledger.session() as s:
            found = s.get(HypothesisORM, hid)
            assert found is not None
            h = found
            helpers = dict(h.state.get("helpers") or {})
            h.claim = out.claim
            h.direction = 1 if out.direction == "higher_value_higher_return" else -1
            h.hypothesis_type = out.hypothesis_type
            h.holding_days = out.holding_days
            h.turnover_band = out.turnover_band
            h.falsifier = out.falsifier
            h.helper_fields = [f for f in out.helper_fields if f in helpers][:2]
            h.tags = memory.tags_of(h)
        self.expressions(hid, self.config.search.first_round)
        return hid

    def _pick(self) -> sampler.Pick | None:
        skip: set[tuple[str, str]] = set()
        for _ in range(FIELD_DRAWS):
            with self.ledger.session() as s:
                d = sampler.choose_dataset(s, self.config, self.rng, skip)
                if d is None:
                    return None
                has_fields = bool(sampler.dataset_fields(s, d))
                key = (d.region, d.dataset_id)
            if not has_fields:
                catalog.load_fields(self.api, self.ledger, d)
            with self.ledger.session() as s:
                d2 = s.get(CatalogORM, d.id)
                assert d2 is not None
                pick = sampler.choose_field(s, d2, self.config, self.rng)
                if pick is not None:
                    s.expunge_all()
                    return pick
            skip.add(key)
        return None

    # ======================= (b) expressions =======================
    def _allowed_fields(self, h: HypothesisORM, with_family: bool) -> dict[str, str]:
        allowed = {h.field_id: h.field_type}
        helpers = h.state.get("helpers") or {}
        allowed.update({f: helpers[f] for f in h.helper_fields if f in helpers})
        if with_family:
            allowed.update(h.state.get("family_types") or {})
        return allowed

    def _settings(
        self, h: HypothesisORM, decay: int = 0, neutralization: str = ""
    ) -> dict[str, Any]:
        options = self.meta.neutralizations(h.region)
        default = self.vocab.default_neutralization.get(h.region) or (options[0] if options else "")
        out: dict[str, Any] = {
            "region": h.region,
            "universe": h.universe,
            "delay": h.delay,
            "decay": decay,
            "truncation": TRUNCATION,
            "testPeriod": self.config.search.test_period,
            "maxTrade": "ON" if h.region in self.vocab.max_trade_on else "OFF",
        }
        if neutralization or default:
            out["neutralization"] = neutralization or default
        if self.vocab.expression_language:
            out["language"] = self.vocab.expression_language
        return out

    def _hypothesis_view(self, h: HypothesisORM) -> dict[str, Any]:
        return {
            "id": f"H{h.id}",
            "field": h.field_id,
            "field_description": h.field_description[:300],
            "claim": h.claim,
            "direction": "higher value -> long" if h.direction > 0 else "higher value -> short",
            "hypothesis_type": h.hypothesis_type,
            "holding_days": h.holding_days,
            "turnover_band": h.turnover_band,
            "falsifier": h.falsifier,
            "dataset_category": h.category,
            "region": h.region,
        }

    def _limits(self, count: int, direction: int = 1) -> dict[str, Any]:
        sc = self.config.search
        out: dict[str, Any] = {
            "count": count,
            "windows": list(sc.windows),
            "max_operators": sc.max_operators,
            "max_fields": sc.max_fields,
        }
        if direction < 0:
            # 2026-11-13 run: 44% of short hypotheses closed with no valid expression
            # because the field was entered unnegated; the check was right, the rule unstated.
            out["direction"] = (
                "this hypothesis is 'higher value -> short': negate the whole expression, "
                f"written as {self.vocab.negate_template.format(x='<expression>')}, so that a "
                "higher value of the hypothesis field gives a lower alpha value"
            )
        if self.vocab.vector_reducer_prefix:
            out["vector_fields"] = (
                "a field of type VECTOR must first be reduced by an operator whose name "
                f"starts with '{self.vocab.vector_reducer_prefix}'"
            )
        return out

    def _context(self, s: Session, h: HypothesisORM) -> tuple[dict[str, Any], dict[str, list[str]]]:
        tags = memory.tags_of(h)
        records = memory.retrieve_records(s, tags, self.config.memory, exclude={h.id})
        priors = memory.retrieve_priors(s, tags, self.config.memory)
        return (
            {
                "records": [memory.record_view(r) for r in records],
                "priors": [memory.prior_view(p) for p in priors],
            },
            {"records": [f"H{r.id}" for r in records], "priors": [f"P{p.id}" for p in priors]},
        )

    def _known_shapes(self, s: Session, h: HypothesisORM) -> set[str]:
        key = (h.region, h.dataset_id)
        if key not in self._scanner:
            self._scanner[key] = scanner_shapes(
                self.config.scanner_runs, h.region, h.dataset_id, self.vocab
            )
        mine = {
            e.shape
            for e, sim in s.execute(
                select(ExpressionORM, SimulationORM)
                .join(SimulationORM, SimulationORM.expression_id == ExpressionORM.id)
                .where(SimulationORM.review.in_([Review.PENDING, Review.SUBMITTED]))
            )
        }
        return self._scanner[key] | mine

    def _vet(
        self, s: Session, h: HypothesisORM, expression: str, with_family: bool
    ) -> tuple[Any, list[str]]:
        """Local checks + direction + originality + no repeat within the hypothesis."""
        sc = self.config.search
        v = check(
            expression,
            operators=self.meta.operators,
            signatures=self.meta.signatures,
            fields=self._allowed_fields(h, with_family),
            vocab=self.vocab,
            windows=sc.windows,
            max_operators=sc.max_operators,
            max_fields=sc.max_fields,
        )
        issues = list(v.issues)
        if v.node is not None:
            if h.field_id not in v.fields_used and not (
                with_family and set(v.fields_used) & set(h.state.get("family") or [])
            ):
                issues.append("does not use the hypothesis field")
            fid = h.field_id if h.field_id in v.fields_used else next(iter(v.fields_used), "")
            if fid and tree.alignment(v.node, self.vocab, fid, h.direction) == "opposite":
                issues.append("enters the field against the hypothesis direction")
            if v.shape in self._known_shapes(s, h):
                issues.append("same structure as a formula already run by the scanner or submitted")
            seen = {
                e.canonical
                for e in s.scalars(
                    select(ExpressionORM).where(
                        ExpressionORM.hypothesis_id == h.id, ExpressionORM.valid.is_(True)
                    )
                )
            }
            if v.canonical in seen:
                issues.append("already tried in this hypothesis")
        return v, issues

    def expressions(self, hid: int, count: int) -> int:
        """(b): ask for `count` expressions (one more ask with the errors when some are
        invalid); queue the valid ones as round `h.round + 1`."""
        queued = 0
        rejected: list[str] = []
        for _ in range(2):
            with self.ledger.session() as s:
                h = s.get(HypothesisORM, hid)
                assert h is not None
                ctx, retrieved = self._context(s, h)
                fields = [
                    {
                        "id": f,
                        "type": t,
                        "role": "hypothesis field" if f == h.field_id else "helper",
                    }
                    for f, t in self._allowed_fields(h, False).items()
                ]
                payload = {
                    "hypothesis": self._hypothesis_view(h),
                    "fields": fields,
                    "grouping_identifiers": sorted(self.vocab.grouping_fields),
                    "operators": self.meta.definitions,
                    "language_notes": self.vocab.language_notes[: self.config.llm.notes_chars],
                    "limits": self._limits(count - queued, h.direction),
                    "rejected_before": rejected,
                    **ctx,
                }
            try:
                out, did = self.llm.ask(
                    prompts.EXPRESSION,
                    prompts.ExpressionOut,
                    payload,
                    hypothesis_id=hid,
                    retrieved=retrieved,
                )
            except LLMInvalid:
                break
            with self.ledger.session() as s:
                h = s.get(HypothesisORM, hid)
                assert h is not None
                round_no = h.round + 1
                for item in out.expressions[: count - queued]:
                    v, issues = self._vet(s, h, item.expression, with_family=False)
                    e = self._store_expression(
                        s,
                        h,
                        item.expression,
                        v,
                        issues,
                        round_no,
                        "initial",
                        "",
                        item.rationale,
                        did,
                    )
                    if issues:
                        rejected.append(f"{item.expression}: {'; '.join(issues)}")
                        continue
                    if self._queue(s, h, e, self._settings(h)):
                        queued += 1
                if queued:
                    h.round = round_no
            if queued >= count or (queued and not rejected):
                break
        if not queued:
            self._close(hid, Verdict.INVALID, "no valid expression after 2 tries")
        return queued

    def _store_expression(
        self,
        s: Session,
        h: HypothesisORM,
        expression: str,
        v: Any,
        issues: list[str],
        round_no: int,
        origin: str,
        mutation: str,
        reason: str,
        decision_id: int | None,
        parent_id: int | None = None,
    ) -> ExpressionORM:
        e = ExpressionORM(
            hypothesis_id=h.id,
            parent_id=parent_id,
            round=round_no,
            origin=origin,
            mutation=mutation,
            reason=reason,
            expression=expression,
            canonical=v.canonical,
            shape=v.shape,
            fields=v.fields_used,
            operators=v.operators,
            nodes=v.nodes,
            valid=not issues,
            rejection="; ".join(issues)[:1000],
            decision_id=decision_id,
        )
        s.add(e)
        s.flush()
        return e

    def _queue(
        self, s: Session, h: HypothesisORM, e: ExpressionORM, settings: dict[str, Any]
    ) -> bool:
        if h.sims_used >= h.sims_budget:
            e.valid, e.rejection = False, "hypothesis budget spent"
            return False
        fp = fingerprint(e.expression, settings)
        if s.scalar(select(SimulationORM.id).where(SimulationORM.fingerprint == fp)):
            e.valid, e.rejection = False, "identical simulation exists"
            return False
        s.add(
            SimulationORM(
                expression_id=e.id,
                hypothesis_id=h.id,
                region=h.region,
                delay=h.delay,
                expression=e.expression,
                settings=settings,
                fingerprint=fp,
                priority=float(e.round),  # later rounds are closer to a result
            )
        )
        h.sims_used += 1
        return True

    # ======================= (c) + (d) =======================
    def _rows(self, s: Session, hid: int) -> list[tuple[ExpressionORM, SimulationORM]]:
        stmt = (
            select(ExpressionORM, SimulationORM)
            .join(SimulationORM, SimulationORM.expression_id == ExpressionORM.id)
            .where(ExpressionORM.hypothesis_id == hid)
            .order_by(SimulationORM.id)
        )
        return [(e, x) for e, x in s.execute(stmt)]

    @staticmethod
    def _result_view(e: ExpressionORM, sim: SimulationORM) -> dict[str, Any]:
        """Train-period numbers only."""
        return {
            "expression_id": e.id,
            "expression": e.expression,
            "parent": e.parent_id,
            "change": e.mutation or e.origin,
            "settings": {k: sim.settings.get(k) for k in ("decay", "neutralization")},
            "train_sharpe": sim.train_sharpe,
            "train_fitness": sim.train_fitness,
            "train_turnover": sim.train_turnover,
            "train_returns": sim.train_returns,
            "train_margin": sim.train_margin,
            "error": sim.error,
        }

    def step(self, hid: int) -> None:
        """All simulations of the current round are back: diagnose, apply the stopping
        rule, then decide."""
        sc = self.config.search
        gates.mark_train(self.ledger, self.config)
        with self.ledger.session() as s:
            h = s.get(HypothesisORM, hid)
            assert h is not None
            if h.status != HStatus.OPEN:
                return
            rows = self._rows(s, hid)
            resume = not rows and h.round == 0 and bool(h.claim)
            if not rows and not resume:
                self._close(hid, Verdict.INVALID, "nothing was simulated", s=s)
                return
        if resume:  # the LLM stopped between the hypothesis and its expressions
            self.expressions(hid, self.config.search.first_round)
            return
        with self.ledger.session() as s:
            h = s.get(HypothesisORM, hid)
            assert h is not None
            done = [(e, x) for e, x in rows if x.status == SimStatus.DONE]
            usable = [(e, x) for e, x in done if not gates.suspicious(x, sc)]
            current = [(e, x) for e, x in rows if e.round == h.round]
            before = h.best_train_sharpe
            best = max((x.train_sharpe or 0.0 for _, x in usable), default=0.0)
            worst = min((x.train_sharpe or 0.0 for _, x in usable), default=0.0)
            h.best_train_sharpe = best
            gain = best - before
            h.dry_rounds = 0 if (h.round == 1 or gain >= sc.min_gain) else h.dry_rounds + 1
            passed = any(x.train_pass for _, x in rows)
            band = prompts.TURNOVER_BANDS.get(h.turnover_band, (0.0, 1.0))
            checks = {
                "noise_line": sc.first_gate,
                "best_train_sharpe": round(best, 3),
                "cleared_noise_line": best >= sc.first_gate,
                "sign": (
                    "as predicted"
                    if best >= sc.first_gate
                    else "opposite to prediction" if worst <= -sc.first_gate else "no clear signal"
                ),
                "turnover_in_expected_band": {
                    str(e.id): (band[0] <= (x.train_turnover or 0.0) <= band[1])
                    for e, x in current
                    if x.status == SimStatus.DONE
                },
                "failed_simulations": sum(1 for _, x in current if x.status == SimStatus.FAILED),
            }
            ctx, retrieved = self._context(s, h)
            diag_payload = {
                "hypothesis": self._hypothesis_view(h),
                "round": h.round,
                "results": [self._result_view(e, x) for e, x in current],
                "best_so_far": [
                    self._result_view(e, x)
                    for e, x in sorted(usable, key=lambda r: -(r[1].train_sharpe or 0.0))[:3]
                ],
                "checks": checks,
                **ctx,
            }
        if passed:
            self._close(hid, Verdict.SUPPORTED, "an expression passed the train gate")
            return
        try:
            diag, _ = self.llm.ask(
                prompts.DIAGNOSIS,
                prompts.DiagnosisOut,
                diag_payload,
                hypothesis_id=hid,
                retrieved=retrieved,
            )
            label, why = diag.label, diag.reason
        except LLMInvalid:
            label, why = (
                "idea_supported" if best >= sc.first_gate else "idea_failed",
                "no valid diagnosis; label set by the program from the noise line",
            )
        with self.ledger.session() as s:
            h = s.get(HypothesisORM, hid)
            assert h is not None
            h.state = {**h.state, "label": label, "diagnosis": why}
            budget_left = h.sims_budget - h.sims_used
            impl_retry_used = bool(h.state.get("impl_retry_used"))
            can_revise = h.parent_id is None and sc.max_revisions > 0
        # ---- program stopping rules ----
        allowed: list[str] = ["mutate", "drop"]
        children = min(sc.children, budget_left)
        stop = ""
        if budget_left <= 0:
            stop = "simulation budget spent"
        elif h.round == 1 and best < sc.first_gate:
            if worst <= -sc.first_gate and can_revise:
                allowed = ["revise", "drop"]
            elif label == "implementation_failed" and not impl_retry_used:
                children = min(sc.impl_retry, budget_left)
                with self.ledger.session() as s:
                    hh = s.get(HypothesisORM, hid)
                    assert hh is not None
                    hh.state = {**hh.state, "impl_retry_used": True}
            else:
                stop = f"best train Sharpe {best:.2f} below the round-1 gate {sc.first_gate}"
        elif h.round > sc.max_rounds:
            stop = f"{sc.max_rounds} improvement rounds done"
        elif h.dry_rounds >= sc.dry_rounds and best < sc.near_miss:
            stop = f"{h.dry_rounds} rounds without a gain of {sc.min_gain}"
        if stop:
            record_program_decision(
                self.ledger, hid, "decision", {"action": "drop", "reason": stop}, {"label": label}
            )
            self._close(hid, LABEL_TO_VERDICT[label], f"{stop}; {why}"[:500], lesson=why)
            return
        if can_revise and "revise" not in allowed and label == "idea_failed":
            allowed.append("revise")
        with self.ledger.session() as s:
            h = s.get(HypothesisORM, hid)
            assert h is not None
            parents = sorted(usable, key=lambda r: -(r[1].train_sharpe or 0.0))[:3]
            payload = {
                "hypothesis": self._hypothesis_view(h),
                "diagnosis": {"label": label, "reason": why},
                "allowed_actions": allowed,
                "parents": [self._result_view(e, x) for e, x in parents],
                "history": [self._result_view(e, x) for e, x in rows][-12:],
                "fields": [
                    {
                        "id": f,
                        "type": t,
                        "same_family_as_hypothesis_field": f in (h.state.get("family") or []),
                    }
                    for f, t in self._allowed_fields(h, True).items()
                ],
                "grouping_identifiers": sorted(self.vocab.grouping_fields),
                "operators": self.meta.definitions,
                "setting_options": {
                    "decay": [0, 2, 5, 10, 21],
                    "neutralization": self.meta.neutralizations(h.region),
                },
                "limits": self._limits(children, h.direction),
                "budget_left": budget_left,
                **ctx,
            }
        try:
            dec, did = self.llm.ask(
                prompts.DECISION,
                prompts.DecisionOut,
                payload,
                hypothesis_id=hid,
                retrieved=retrieved,
            )
        except LLMInvalid as exc:
            self._close(hid, LABEL_TO_VERDICT[label], f"no valid decision: {exc}"[:500])
            return
        action = dec.action if dec.action in allowed else "drop"
        if action != dec.action:
            record_program_decision(
                self.ledger,
                hid,
                "decision",
                {"action": "drop", "reason": f"'{dec.action}' not allowed here", "overrides": did},
            )
        if action == "drop":
            self._close(hid, LABEL_TO_VERDICT[label], dec.reason, lesson=dec.lesson)
        elif action == "revise":
            self._revise(hid, dec, did)
        elif self._mutate(hid, dec, did, {e.id for e, _ in parents}, children) == 0:
            self._close(hid, LABEL_TO_VERDICT[label], "no valid child", lesson=dec.lesson)

    def _mutate(
        self, hid: int, dec: prompts.DecisionOut, did: int, parent_ids: set[int], children: int
    ) -> int:
        queued = 0
        with self.ledger.session() as s:
            h = s.get(HypothesisORM, hid)
            assert h is not None
            family = set(h.state.get("family") or []) | {h.field_id}
            round_no = h.round + 1
            for m in dec.mutations[:children]:
                parent = s.get(ExpressionORM, m.parent)
                psim = s.scalar(
                    select(SimulationORM).where(SimulationORM.expression_id == m.parent)
                )
                if parent is None or m.parent not in parent_ids or psim is None:
                    v, _ = self._vet(s, h, m.expression, with_family=True)
                    self._store_expression(
                        s,
                        h,
                        m.expression,
                        v,
                        ["parent is not one of the offered parents"],
                        round_no,
                        "mutation",
                        m.change,
                        m.reason,
                        did,
                        m.parent if parent else None,
                    )
                    continue
                settings = dict(psim.settings)
                expression = m.expression
                issues: list[str] = []
                if m.change == "setting":
                    expression = parent.expression
                    if m.decay is not None:
                        if not 0 <= m.decay <= MAX_DECAY:
                            issues.append(f"decay {m.decay} outside 0-{MAX_DECAY}")
                        settings["decay"] = m.decay
                    if m.neutralization:
                        if m.neutralization not in self.meta.neutralizations(h.region):
                            issues.append(f"neutralization {m.neutralization} not offered")
                        settings["neutralization"] = m.neutralization
                    if settings == psim.settings:
                        issues.append("setting change changes nothing")
                    v, _ = self._vet(s, h, expression, with_family=True)
                else:
                    v, vet_issues = self._vet(s, h, expression, with_family=True)
                    issues += vet_issues
                    if v.node is not None:
                        actual = tree.change_type(tree.parse(parent.expression), v.node, family)
                        declared = m.change
                        if actual == "same":
                            issues.append("identical to its parent")
                        elif declared != "rewrite" and actual != declared:
                            issues.append(
                                f"declared a {declared} change but made a {actual} change"
                            )
                origin = "rewrite" if m.change == "rewrite" else "mutation"
                e = self._store_expression(
                    s,
                    h,
                    expression,
                    v,
                    issues,
                    round_no,
                    origin,
                    m.change,
                    m.reason,
                    did,
                    parent.id,
                )
                if not issues and self._queue(s, h, e, settings):
                    queued += 1
            if queued:
                h.round = round_no
        return queued

    def _revise(self, hid: int, dec: prompts.DecisionOut, did: int) -> None:
        r = dec.revision
        assert r is not None
        with self.ledger.session() as s:
            old = s.get(HypothesisORM, hid)
            assert old is not None
            new = HypothesisORM(
                parent_id=old.id,
                region=old.region,
                delay=old.delay,
                universe=old.universe,
                dataset_id=old.dataset_id,
                dataset_name=old.dataset_name,
                category=old.category,
                field_id=old.field_id,
                field_type=old.field_type,
                field_description=old.field_description,
                claim=r.claim,
                direction=1 if r.direction == "higher_value_higher_return" else -1,
                hypothesis_type=r.hypothesis_type,
                holding_days=r.holding_days,
                turnover_band=r.turnover_band,
                falsifier=r.falsifier,
                helper_fields=list(old.helper_fields),
                sims_budget=self.config.search.revision_budget,
                state={k: old.state.get(k) for k in ("family", "helpers", "family_types")},
            )
            s.add(new)
            s.flush()
            new.tags = memory.tags_of(new)
            new_id = new.id
        self._close(
            hid, Verdict.IDEA_FAILED, f"revised into H{new_id}: {dec.reason}", lesson=dec.lesson
        )
        self.ledger.log("hypothesis_revised", old=hid, new=new_id, decision=did)
        self.expressions(new_id, self.config.search.first_round)

    # ======================= closing =======================
    def _close(
        self, hid: int, verdict: str, reason: str, lesson: str = "", s: Session | None = None
    ) -> None:
        if s is None:
            with self.ledger.session() as s2:
                self._close(hid, verdict, reason, lesson, s2)
            return
        h = s.get(HypothesisORM, hid)
        assert h is not None
        if h.status == HStatus.CLOSED:
            return
        h.status, h.verdict, h.reason = HStatus.CLOSED, verdict, reason[:1000]
        h.lesson = (lesson or reason)[:500]
        h.closed_at = utc_now()
        for sim in s.scalars(
            select(SimulationORM).where(
                SimulationORM.hypothesis_id == hid, SimulationORM.status == SimStatus.QUEUED
            )
        ):
            s.delete(sim)  # never simulated: nothing spent, nothing to keep
            h.sims_used -= 1
        changed = memory.update_priors(s, h)
        self.ledger.log("hypothesis_closed", hypothesis=hid, verdict=verdict, priors=changed)

    def promote_passed(self) -> int:
        """Hypotheses whose expression was stored for review become 'passed'."""
        n = 0
        with self.ledger.session() as s:
            for sim in s.scalars(
                select(SimulationORM).where(
                    SimulationORM.review.in_([Review.PENDING, Review.SUBMITTED])
                )
            ):
                h = s.get(HypothesisORM, sim.hypothesis_id)
                if h is not None and h.verdict != Verdict.PASSED:
                    h.verdict = Verdict.PASSED
                    n += 1
        return n


# ======================= one run =======================
@dataclass
class RunReport:
    simulations: int = 0
    failed: int = 0
    stored: list[str] = field(default_factory=list)
    opened: int = 0
    stopped_because: str = ""


@ledger_locked
def run(
    config: FactoryConfig,
    api: BrainAPI,
    ledger: Ledger,
    llm: LLMClient,
    vocab: Vocabulary,
    *,
    minutes: float = 720.0,
    rng: random.Random | None = None,
    meta: Meta | None = None,
    sleep: Callable[[float], None] = time.sleep,
    may_open: Callable[[], str] = lambda: "",
    on_progress: Callable[[str], None] | None = None,
) -> RunReport:
    """Run the factory until the daily budget, the deadline or nothing left to do."""
    report = RunReport()
    deadline = time.monotonic() + minutes * 60
    simulator = Simulator(api, ledger, config, sleep=sleep)
    simulator.assert_recoverable()
    recovered = simulator.run(0, deadline=deadline)  # drain batches left in flight
    report.simulations += recovered.completed
    report.failed += recovered.failed
    if catalog.is_stale(ledger):
        catalog.refresh(api, ledger, config, vocab)
    factory = Factory(api, ledger, config, llm, vocab, meta or Meta.fetch(api), rng, may_open)
    idle = 0
    llm_stopped = ""
    while time.monotonic() < deadline:
        if simulator.remaining_daily() <= 0:
            report.stopped_because = "daily simulation budget spent"
            break
        moved: dict[str, int] = {}
        if not llm_stopped:
            try:
                moved = factory.advance()
            except CostCapReached as exc:
                llm_stopped = str(exc)  # queued work still runs; open work resumes later
            except LLMUnavailable as exc:
                llm_stopped = f"LLM unavailable: {exc}"[:300]
        report.opened += moved.get("opened", 0)
        stats = simulator.run(config.budget.chunk, deadline=deadline)
        report.simulations += stats.completed
        report.failed += stats.failed
        report.stored += gates.evaluate(api, ledger, config)
        factory.promote_passed()
        if on_progress is not None:
            on_progress("running")
        if stats.stopped_because in ("quota", "deadline", "platform kept refusing batches"):
            report.stopped_because = stats.stopped_because
            break
        if stats.batches == 0 and not any(moved.values()):
            idle += 1
            if idle >= 3:
                report.stopped_because = llm_stopped or "nothing left to do"
                break
        else:
            idle = 0
    else:
        report.stopped_because = "deadline"
    ledger.log(
        "run_done",
        simulations=report.simulations,
        failed=report.failed,
        stored=report.stored,
        reason=report.stopped_because,
        at=datetime.now(UTC).isoformat(),
    )
    return report


def decisions_of(s: Session, hid: int) -> list[DecisionORM]:
    """Every decision made for a hypothesis, in order."""
    return list(
        s.scalars(
            select(DecisionORM).where(DecisionORM.hypothesis_id == hid).order_by(DecisionORM.id)
        )
    )
