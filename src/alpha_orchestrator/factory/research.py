"""One dataset, from first look to finish, with every selection made by statistics.

    structure   field ids -> families (same thing, different horizon) -> strata (economic
                concept; from names, or DeepSeek groups them when names do not show it)
    health      sample families per stratum; on one field each: daily coverage, change
                within 5/22/66 days (-> update period), coverage after filling one period.
                Inferred for the whole stratum from its sample. Coverage after filling
                below the threshold -> family not eligible.
    survey      sample eligible families per stratum; 4 fixed shapes each. A family is a
                "hit" when its best |Sharpe| reaches the screening line (noise does that
                13% of the time). A sequential test decides signal / noise for the dataset.
    research    signal: each round, strata are drawn by their shrunk excess over noise
                (Thompson sampling) and get: library structures on their best formula
                (deepen, structures also drawn by their record), untested families
                (broaden), DeepSeek vs random combinations (combine). A family stops when
                two rounds in a row gain less than the noise between variants, or after
                its round limit. The dataset stops when no stratum has work left or its
                budget is spent.
    confirm     a result counts only above the luck bar for the number of tries in this
                dataset (sigma * sqrt(2 ln n)), and must pass robustness and correlation.

Results with |Sharpe| above the suspicious line or turnover above its limit are never mined.
"""

import random
import re
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from alpha_orchestrator.brain.api import BrainAPI
from alpha_orchestrator.domain.models import utc_now
from alpha_orchestrator.factory import generators as gen
from alpha_orchestrator.factory import profile as prof
from alpha_orchestrator.factory import stats
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import (
    CandidateORM,
    CandidateStatus,
    CellORM,
    CellState,
    FieldORM,
    Stage,
)
from alpha_orchestrator.factory.validate import check

PENDING = (CandidateStatus.QUEUED, CandidateStatus.RUNNING)
STAGE_WEIGHT = {
    Stage.PROFILE: 3.0,
    Stage.SURVEY: 2.5,
    Stage.ROBUST: 3.0,
    Stage.REPAIR: 2.0,
    Stage.REFINE: 2.0,
    Stage.COMBO: 1.8,
}
FIXABLE = {"HIGH_TURNOVER", "CONCENTRATED_WEIGHT", "LOW_SUB_UNIVERSE_SHARPE"}
SURVEY_ALL_BELOW = 40  # eligible families; fewer than this are surveyed whole
FIRST_WAVE = 10  # coverage probes before deciding a dataset is too sparse to work
PERIOD_SAMPLE = 12  # families whose update period is measured (4 probes each)
TOKEN = re.compile(r"[a-z0-9]+")


from alpha_orchestrator.factory.validate import signature  # noqa: E402


@dataclass
class Metadata:
    """Account-level facts refreshed once per run."""

    operators: set[str] = field(default_factory=set)
    definitions: dict[str, str] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    signatures: dict[str, tuple[int, int]] = field(default_factory=dict)

    @classmethod
    def fetch(cls, api: BrainAPI) -> "Metadata":
        ops = [o for o in api.operators() if "REGULAR" in (o.get("scope") or ["REGULAR"])]
        return cls(
            operators={str(o["name"]).lower() for o in ops},
            definitions={str(o["name"]): str(o.get("definition") or o["name"]) for o in ops},
            options=api.simulation_options(),
            signatures={
                str(o["name"]).lower(): sig
                for o in ops
                if (
                    sig := signature(
                        str(o.get("definition") or ""), str(o.get("description") or "")
                    )
                )
            },
        )

    def neutralizations(self, region: str) -> list[str]:
        return list(self.options.get(region, {}).get("neutralizations", []))

    def universes(self, region: str) -> list[str]:
        return list(self.options.get(region, {}).get("universes", []))


def name_stratum(family: str, dataset_id: str) -> str:
    """Fallback concept from the name: the family minus its last token (the statistic),
    after removing a dataset prefix. anl4_eps_mean, anl4_eps_median -> eps."""
    tokens = TOKEN.findall(family.lower())
    prefix = TOKEN.findall(dataset_id.lower())
    if tokens and prefix and len(tokens) > 1 and tokens[0][:3] == prefix[0][:3]:
        tokens = tokens[1:]
    return "_".join(tokens[:-1]) if len(tokens) > 2 else "_".join(tokens) or family


class Factory:
    def __init__(
        self,
        api: BrainAPI,
        ledger: Ledger,
        config: FactoryConfig,
        meta: Metadata,
        llm: FactoryLLM | None,
        rng: random.Random | None = None,
    ) -> None:
        self.api = api
        self.ledger = ledger
        self.config = config
        self.meta = meta
        self.llm = llm
        self.rng = rng or random.Random()
        t = config.thresholds
        self.sigma = t.noise_sigma
        self.p0 = stats.exceed_prob(t.screen_line, self.sigma, k=3)  # 4 correlated shapes ~ 3
        self.null_best = stats.expected_best_abs(self.sigma, 3)

    # ======================= structure + health =======================
    def start(self, cell_id: str) -> int:
        with self.ledger.session() as s:
            cell = _cell(s, cell_id)
            region, delay, universe, dataset = (
                cell.region,
                cell.delay,
                cell.universe,
                cell.dataset_id,
            )
            brief = _brief(cell)
        rows = [
            r
            for r in self.api.datafields(region, delay, universe, dataset)
            if r.get("type") in ("MATRIX", "VECTOR")
        ]
        families = gen.group_families(rows)
        strata = self._strata(brief, families)
        with self.ledger.session() as s:
            cell = _cell(s, cell_id)
            if not rows:
                self._finish(s, cell, "no_fields", "no MATRIX or VECTOR fields")
                return 0
            existing = {
                f.field_id: f
                for f in s.scalars(select(FieldORM).where(FieldORM.cell_id == cell_id))
            }
            for fam, members in families.items():
                for i, r in enumerate(members):
                    f = existing.get(str(r["id"])) or FieldORM(
                        cell_id=cell_id, field_id=str(r["id"])
                    )
                    f.type = str(r["type"])
                    f.description = str(r.get("description") or "")[:1000]
                    f.coverage = float(r.get("coverage") or 0)
                    f.alpha_count = int(r.get("alphaCount") or 0)
                    f.family = fam
                    f.representative = i == 0
                    f.stratum = strata.get(fam, fam)
                    f.eligible = True
                    f.survey, f.survey_score, f.profile = {}, None, {}
                    s.add(f)
            s.flush()
            sample = self._health_sample(s, cell_id)
            first = sample[:FIRST_WAVE]
            queued = self._queue_coverage(s, cell, first)
            cell.state = CellState.STUDYING
            cell.plan = {
                "phase": "health_cov",
                "health_sample": sample,
                "cov_queued": first,
                "family_count": len(families),
                "strata": len(set(strata.values())),
            }
            self.ledger.log(
                s,
                "dataset_started",
                cell_id,
                fields=len(rows),
                families=len(families),
                strata=cell.plan["strata"],
                health_sample=len(sample),
            )
        return queued

    def _strata(self, brief: dict[str, Any], families: dict[str, list[Any]]) -> dict[str, str]:
        by_name = {fam: name_stratum(fam, brief["id"]) for fam in families}
        distinct = len(set(by_name.values()))
        # Names show structure when they collapse families into a manageable number of
        # groups; otherwise (every family alone, or one giant group) ask DeepSeek.
        if self.llm is None or len(families) <= 12 or 3 <= distinct <= len(families) // 3:
            return by_name
        listing = [
            {
                "family": fam,
                "example": str(members[0]["id"]),
                "description": str(members[0].get("description") or "")[:160],
            }
            for fam, members in families.items()
        ]
        out: dict[str, str] = {}
        for i in range(0, len(listing), 150):
            try:
                out.update(self.llm.group_fields(brief, listing[i : i + 150]))
            except RuntimeError:
                break
        return {fam: out.get(fam, by_name[fam]) for fam in families}

    def _health_sample(self, s: Session, cell_id: str) -> list[str]:
        """Families to health-check: round-robin over strata (largest first), so every
        stratum gets one before any gets two, up to the sample size."""
        by_stratum: dict[str, list[FieldORM]] = defaultdict(list)
        for f in self._reps(s, cell_id):
            by_stratum[f.stratum].append(f)
        for members in by_stratum.values():
            self.rng.shuffle(members)
        order = sorted(by_stratum, key=lambda k: -len(by_stratum[k]))
        out: list[str] = []
        depth = 0
        limit = self.config.budget.health_sample
        while len(out) < limit and any(len(by_stratum[k]) > depth for k in order):
            for k in order:
                if len(out) >= limit:
                    break
                if len(by_stratum[k]) > depth:
                    out.append(by_stratum[k][depth].family)
            depth += 1
        return out

    def _queue_coverage(self, s: Session, cell: CellORM, families: Sequence[str]) -> int:
        """One probe per family: coverage once filled for a year, the most the field can
        reach. Cheap, and it decides whether anything else is worth measuring."""
        settings = {**self._settings(cell), "neutralization": "NONE", "decay": 0}
        added = 0
        wanted = set(families)
        for f in self._reps(s, cell.cell_id):
            if f.family not in wanted:
                continue
            name, expr = prof.coverage_probe(f.field_id, f.type)
            row, created = self._add(s, cell, expr, settings, Stage.PROFILE, name)
            f.profile = {"probes": {name: row.id} if row else {}}
            added += created
        return added

    def _coverage_of(self, s: Session, cell: CellORM, f: FieldORM) -> float | None:
        cid = ((f.profile or {}).get("probes") or {}).get("maxcov")
        count = _count(s, cid)
        if count is None:
            return None
        size = prof.universe_size(cell.universe) or max(count, 1)
        return round(min(count / size, 1.0), 3)

    def _health_coverage(self, s: Session, cell: CellORM) -> None:
        """Coverage wave finished. First wave all below the floor -> stop the dataset;
        otherwise measure the rest of the sample, then the update period of the ones that
        pass."""
        floor = self.config.thresholds.min_filled_coverage
        plan = dict(cell.plan)
        reps = {f.family: f for f in self._reps(s, cell.cell_id)}
        queued = plan.get("cov_queued", [])
        covs = {fam: self._coverage_of(s, cell, reps[fam]) for fam in queued if fam in reps}
        for fam, cov in covs.items():
            reps[fam].profile = {**reps[fam].profile, "max_coverage": cov}
        passing = [fam for fam, cov in covs.items() if cov is not None and cov >= floor]
        rest = [fam for fam in plan.get("health_sample", []) if fam not in queued]
        if not passing and (len(queued) >= FIRST_WAVE or not rest):
            best = max((c for c in covs.values() if c is not None), default=0.0)
            self._finish(
                s,
                cell,
                "no_eligible",
                f"{len(covs)} sampled families: coverage even when filled for a year at most "
                f"{best:.0%}, below {floor:.0%}",
            )
            return
        if rest:
            self._queue_coverage(s, cell, rest)
            cell.plan = {**plan, "cov_queued": [*queued, *rest]}
            return
        # update period on a stratified few of the families that pass
        by_stratum: dict[str, list[str]] = defaultdict(list)
        for fam in passing:
            by_stratum[reps[fam].stratum].append(fam)
        chosen: list[str] = []
        while len(chosen) < PERIOD_SAMPLE and any(by_stratum.values()):
            for k in list(by_stratum):
                if by_stratum[k] and len(chosen) < PERIOD_SAMPLE:
                    chosen.append(by_stratum[k].pop(0))
        settings = {**self._settings(cell), "neutralization": "NONE", "decay": 0}
        for fam in chosen:
            f = reps[fam]
            ids = dict((f.profile or {}).get("probes") or {})
            for name, expr in prof.health_probes(f.field_id, f.type):
                row, _ = self._add(s, cell, expr, settings, Stage.PROFILE, name)
                if row is not None:
                    ids[name] = row.id
            f.profile = {**f.profile, "probes": ids}
        cell.plan = {**plan, "phase": "health_period", "period_sample": chosen}

    def _finish_health(self, s: Session, cell: CellORM) -> None:
        """Coverage measured on the sample, update period on a few; inferred for every
        other family from its stratum (or the whole dataset when its stratum was not hit)."""
        size = prof.universe_size(cell.universe) or 0
        reps = self._reps(s, cell.cell_id)
        cov_by: dict[str, list[float]] = defaultdict(list)
        period_by: dict[str, list[float]] = defaultdict(list)
        for f in reps:
            prof_ = f.profile or {}
            cov = prof_.get("max_coverage")
            if cov is not None:
                cov_by[f.stratum].append(float(cov))
            ids = prof_.get("probes") or {}
            if "upd66" in ids:
                counts = {n: _count(s, ids.get(n)) for n in ("raw", "upd5", "upd22", "upd66")}
                period = prof.period_from(counts)
                raw = (counts.get("raw") or 0) / (size or max(counts.get("raw") or 0, 1))
                f.profile = {
                    **prof_,
                    "period_days": round(min(period, 999.0), 1),
                    "raw_coverage": round(min(raw, 1.0), 3),
                }
                period_by[f.stratum].append(min(period, 999.0))
        all_cov = [c for cs in cov_by.values() for c in cs]
        all_period = [p for ps in period_by.values() for p in ps]
        floor = self.config.thresholds.min_filled_coverage
        eligible = 0
        for f in reps:
            prof_ = dict(f.profile or {})
            if prof_.get("max_coverage") is not None:
                cov, how = float(prof_["max_coverage"]), "measured"
            elif cov_by.get(f.stratum):
                cov, how = statistics.median(cov_by[f.stratum]), "stratum"
            else:
                cov, how = (statistics.median(all_cov) if all_cov else 0.0), "dataset"
            if "period_days" not in prof_:
                pool = period_by.get(f.stratum) or all_period
                prof_["period_days"] = round(statistics.median(pool), 1) if pool else 66.0
            prof_.setdefault("raw_coverage", round(f.coverage, 3))
            prof_.update(filled_coverage=round(cov, 3), coverage_from=how)
            f.profile = prof_
            f.filled_coverage = cov
            f.eligible = cov >= floor
            f.prep = prof.prep_for(f.field_id, f.type, f.profile)
            eligible += f.eligible
            for m in s.scalars(
                select(FieldORM).where(
                    FieldORM.cell_id == cell.cell_id,
                    FieldORM.family == f.family,
                    FieldORM.representative.is_(False),
                )
            ):
                m.profile = {k: v for k, v in f.profile.items() if k != "probes"}
                m.filled_coverage, m.eligible = f.filled_coverage, f.eligible
                m.prep = prof.prep_for(m.field_id, m.type, m.profile)
        cell.plan = {**cell.plan, "eligible_families": eligible}
        self.ledger.log(
            s,
            "health_done",
            cell.cell_id,
            coverage_measured=len(all_cov),
            period_measured=len(all_period),
            families=len(reps),
            eligible=eligible,
        )
        if eligible == 0:
            self._finish(s, cell, "no_eligible", "no family keeps enough coverage after filling")
            return
        self._start_survey(s, cell)

    # ======================= survey =======================
    def _start_survey(self, s: Session, cell: CellORM) -> None:
        by_stratum: dict[str, list[FieldORM]] = defaultdict(list)
        for f in self._reps(s, cell.cell_id):
            if f.eligible:
                by_stratum[f.stratum].append(f)
        picked: list[FieldORM] = []
        everything = sum(len(m) for m in by_stratum.values())
        per = self.config.budget.survey_per_stratum
        for members in by_stratum.values():
            self.rng.shuffle(members)
            # a small dataset is surveyed whole: sampling would leave too few to judge
            picked += members if everything <= SURVEY_ALL_BELOW else members[:per]
        self.rng.shuffle(picked)
        picked = picked[: self.config.budget.survey_max_families]
        for f in picked:
            self._queue_shapes(s, cell, f)
        cell.state = CellState.SURVEY
        t = self.config.thresholds
        cell.plan = {
            **cell.plan,
            "phase": "survey",
            "survey_order": [f.family for f in picked],
            "survey_seen": [],
            "sprt": {"llr": 0.0, "n": 0, "hits": 0},
            "p0": round(self.p0, 4),
            "p1": round(min(self.p0 + t.sprt_extra_hits, 0.95), 4),
            "verdict": None,
        }
        self.ledger.log(
            s, "survey_started", cell.cell_id, families=len(picked), p0=round(self.p0, 4)
        )

    def _queue_shapes(self, s: Session, cell: CellORM, f: FieldORM) -> None:
        settings = self._settings(cell)
        ids = {}
        period = float((f.profile or {}).get("period_days") or 66.0)
        for shape, expr in prof.survey_shapes(f.prep, period):
            row, _ = self._add(
                s, cell, expr, settings, Stage.SURVEY, shape, extra={"family": f.family}
            )
            if row is not None:
                ids[shape] = row.id
        f.survey = ids

    def _score_family(self, s: Session, f: FieldORM) -> tuple[float | None, dict[str, Any]]:
        """Best |Sharpe| over the shapes, ignoring suspicious results. None until done."""
        best, best_id, best_sign = 0.0, None, 1
        shapes = {k: v for k, v in (f.survey or {}).items() if k not in ("best_id", "sign")}
        if not shapes:
            return None, {}
        for cid in shapes.values():
            c = s.get(CandidateORM, cid)
            if c is None:
                continue
            if c.status in PENDING:
                return None, {}
            if c.sharpe is None or self._suspicious(c):
                continue
            if abs(c.sharpe) > best:
                best, best_id, best_sign = abs(c.sharpe), c.id, (1 if c.sharpe > 0 else -1)
        return best, {"best_id": best_id, "sign": best_sign}

    def _update_survey(self, s: Session, cell: CellORM) -> None:
        plan = dict(cell.plan)
        seen = list(plan.get("survey_seen") or [])
        reps = {f.family: f for f in self._reps(s, cell.cell_id)}
        t = self.config.thresholds
        test = stats.Sprt(plan["p0"], plan["p1"], t.sprt_alpha, t.sprt_beta, **plan["sprt"])
        for fam in plan.get("survey_order", []):
            if fam in seen or fam not in reps:
                continue
            score, info = self._score_family(s, reps[fam])
            if score is None:
                continue
            f = reps[fam]
            f.survey_score = score
            f.survey = {**f.survey, **{k: v for k, v in info.items() if v is not None}}
            seen.append(fam)
            if plan.get("verdict") is None:
                test.add(score >= t.screen_line)
        plan["survey_seen"] = seen
        if plan.get("verdict") is None:
            plan["sprt"] = {"llr": round(test.llr, 4), "n": test.n, "hits": test.hits}
            # Two tests, either one can call signal (each at half the error rate):
            # breadth = more families reach the screening line than noise would (SPRT);
            # peak    = one family beats the luck bar for every survey formula so far.
            scored = [reps[f] for f in seen if reps[f].survey_score is not None]
            formulas = 4 * len(scored)
            peak = max((f.survey_score or 0 for f in scored), default=0.0)
            peak_bar = stats.luck_bar(self.sigma, max(formulas, 2) * 2)
            plan["peak"] = {"best": round(peak, 3), "bar": round(peak_bar, 3)}
            verdict = test.verdict
            if peak >= peak_bar:
                verdict = "signal"
            all_done = len(seen) >= len(plan.get("survey_order", []))
            every_stratum = {reps[f].stratum for f in plan.get("survey_order", [])} <= {
                reps[f].stratum for f in seen
            }
            if verdict == "noise" and not every_stratum:
                verdict = "undecided"  # a concentrated signal may sit in an unmeasured stratum
            if verdict == "undecided" and all_done and test.n:
                tail = stats.binomial_tail(test.hits, test.n, plan["p0"])
                verdict = "signal" if tail < t.sprt_alpha / 2 else "noise"
                plan["binomial_p"] = tail
            if verdict in ("signal", "noise"):
                plan["verdict"] = verdict
                self.ledger.log(s, "survey_verdict", cell.cell_id, verdict=verdict, **plan["sprt"])
        cell.plan = plan
        if plan.get("verdict") == "noise" and cell.state != CellState.EXHAUSTED:
            sp = plan["sprt"]
            self._finish(
                s,
                cell,
                "noise",
                f"survey: {sp['hits']}/{sp['n']} families reached the screening line, "
                f"noise alone gives {plan['p0']:.0%}",
            )
        elif plan.get("verdict") == "signal" and cell.state == CellState.SURVEY:
            cell.state = CellState.RESEARCH
            cell.plan = {**plan, "phase": "research", "rounds": 0, "families": {}, "combos": {}}

    # ======================= research rounds =======================
    def _arms(self, s: Session, cell: CellORM) -> dict[str, stats.Arm]:
        groups: dict[str, list[float]] = defaultdict(list)
        for f in self._reps(s, cell.cell_id):
            if f.eligible and f.survey_score is not None:
                groups[f.stratum].append(f.survey_score - self.null_best)
        return stats.shrunk_means(groups) if groups else {}

    def _plan_round(self, s: Session, cell: CellORM) -> int:
        """Spend one round of simulations where the evidence points."""
        budget = self.config.budget.round_size
        arms = self._arms(s, cell)
        plan = dict(cell.plan)
        fam_state: dict[str, Any] = {k: dict(v) for k, v in (plan.get("families") or {}).items()}
        combos: dict[str, Any] = dict(plan.get("combos") or {})
        by_stratum: dict[str, list[FieldORM]] = defaultdict(list)
        for f in self._reps(s, cell.cell_id):
            if f.eligible:
                by_stratum[f.stratum].append(f)
        spent = 0
        closed: set[str] = set()
        while spent < budget and len(closed) < len(arms):
            open_arms = {k: v for k, v in arms.items() if k not in closed}
            stratum = stats.thompson_pick(open_arms, self.rng)[0]
            work = self._stratum_work(
                s, cell, stratum, by_stratum[stratum], fam_state, combos, arms[stratum]
            )
            if work == 0:
                closed.add(stratum)
            spent += work
        plan = dict(cell.plan)  # _stratum_work may extend the survey order
        plan["families"], plan["combos"] = fam_state, combos
        plan["rounds"] = int(plan.get("rounds", 0)) + 1
        cell.plan = plan
        self.ledger.log(
            s,
            "round_planned",
            cell.cell_id,
            round=plan["rounds"],
            queued=spent,
            strata_with_work=len(arms) - len(closed),
        )
        return spent

    def _stratum_work(
        self,
        s: Session,
        cell: CellORM,
        stratum: str,
        members: list[FieldORM],
        fam_state: dict[str, Any],
        combos: dict[str, Any],
        arm: stats.Arm,
    ) -> int:
        line = self.config.thresholds.screen_line
        # deepen: the strongest families here that still have rounds and nothing pending
        live = sorted(
            (
                f
                for f in members
                if (f.survey_score or 0) >= line and (f.survey or {}).get("best_id")
            ),
            key=lambda f: -(f.survey_score or 0),
        )
        for f in live:
            st = fam_state.setdefault(f.family, {})
            if st.get("done") or st.get("pending"):
                continue
            n = self._queue_refine(s, cell, f, st)
            if n:
                return n
        # broaden: a family of this stratum the survey has not measured yet
        untested = [f for f in members if f.survey_score is None and not f.survey]
        if untested:
            f = self.rng.choice(untested)
            self._queue_shapes(s, cell, f)
            plan = dict(cell.plan)
            plan["survey_order"] = [*plan.get("survey_order", []), f.family]
            cell.plan = plan
            return len(f.survey)
        # combine: once per stratum that is clearly above noise
        if arm.mean - 2 * arm.sd > 0 and stratum not in combos and len(members) >= 2:
            return self._queue_combos(s, cell, stratum, members, combos)
        return 0

    def _queue_refine(self, s: Session, cell: CellORM, f: FieldORM, st: dict[str, Any]) -> int:
        t = self.config.thresholds
        if st.get("rounds", 0) >= t.refine_max_rounds or st.get("dry", 0) >= 2:
            st["done"] = True
            return 0
        if not st.get("parent"):
            parent = s.get(CandidateORM, f.survey["best_id"])
            if parent is None or parent.sharpe is None:
                st["done"] = True
                return 0
            expr = (
                parent.expression
                if f.survey.get("sign", 1) > 0
                else f"multiply(-1, {parent.expression})"
            )
            st.update(
                parent=expr, parent_settings=parent.settings, best=abs(parent.sharpe), tried=[]
            )
        records = self._structure_record(s, cell.category)
        options = [n for n in gen.STRUCTURES if n not in st.get("tried", [])]
        if not options:
            st["done"] = True
            return 0
        draws = sorted(
            options, key=lambda n: -stats.beta_draw(*records.get(n, (0.0, 0.0)), self.rng)
        )[: self.config.budget.refine_structures]
        ids = []
        for name in draws:
            row, _ = self._add(
                s,
                cell,
                gen.structure(name, st["parent"]),
                st["parent_settings"],
                Stage.REFINE,
                name,
                extra={"family": f.family, "structure": name, "parent_best": st["best"]},
            )
            if row is not None:
                ids.append(row.id)
        st["tried"] = [*st.get("tried", []), *draws]
        st["pending"] = ids
        st["rounds"] = st.get("rounds", 0) + 1
        if not ids:
            st["done"] = True
        return len(ids)

    def _queue_combos(
        self,
        s: Session,
        cell: CellORM,
        stratum: str,
        members: list[FieldORM],
        combos: dict[str, Any],
    ) -> int:
        n = self.config.budget.combo_count
        settings = self._settings(cell)
        strong = sorted(members, key=lambda f: -(f.survey_score or 0))[:6]
        rows: dict[str, list[str]] = {"llm": [], "random": []}
        for p in gen.random_combos([f.prep for f in strong], n, self.rng, settings):
            row, _ = self._add(
                s,
                cell,
                p.expression,
                settings,
                Stage.COMBO,
                "random",
                change=p.change,
                extra={"stratum": stratum, "arm": "random"},
            )
            if row is not None:
                rows["random"].append(row.id)
        if self.llm is not None:
            summary = [
                {
                    "stratum": stratum,
                    "fields": [
                        {
                            "id": f.field_id,
                            "prep": f.prep,
                            "description": f.description[:200],
                            "survey_best_abs_sharpe": round(f.survey_score or 0, 2),
                            "best_sign": (f.survey or {}).get("sign"),
                        }
                        for f in strong
                    ],
                }
            ]
            ops = sorted(f"{k}: {v}" for k, v in self.meta.definitions.items())
            try:
                proposals = self.llm.combinations(_brief(cell), summary, ops, n)
            except RuntimeError:
                proposals = []
            for c in proposals:
                row, _ = self._add(
                    s,
                    cell,
                    c["expression"],
                    settings,
                    Stage.COMBO,
                    "llm",
                    change=c["why"],
                    extra={"stratum": stratum, "arm": "llm"},
                )
                if row is not None:
                    rows["llm"].append(row.id)
        combos[stratum] = rows
        return len(rows["llm"]) + len(rows["random"])

    def _settle_refines(self, s: Session, cell: CellORM) -> None:
        plan = dict(cell.plan)
        fam_state = {k: dict(v) for k, v in (plan.get("families") or {}).items()}
        gain_min = self.config.thresholds.refine_min_gain
        changed = False
        for st in fam_state.values():
            ids = st.get("pending") or []
            if not ids:
                continue
            rows = [s.get(CandidateORM, i) for i in ids]
            if any(r is not None and r.status in PENDING for r in rows):
                continue
            best_row = max(
                (
                    r
                    for r in rows
                    if r is not None and r.sharpe is not None and not self._suspicious(r)
                ),
                key=lambda r: r.sharpe or 0,
                default=None,
            )
            gain = (best_row.sharpe or 0) - st["best"] if best_row else -1.0
            if best_row is not None and gain >= gain_min:
                st.update(
                    parent=best_row.expression,
                    parent_settings=best_row.settings,
                    best=best_row.sharpe,
                    dry=0,
                )
            else:
                st["dry"] = st.get("dry", 0) + 1
            st["pending"] = []
            if (
                st.get("dry", 0) >= 2
                or st.get("rounds", 0) >= self.config.thresholds.refine_max_rounds
            ):
                st["done"] = True
            changed = True
        if changed:
            plan["families"] = fam_state
            cell.plan = plan

    def _structure_record(self, s: Session, category: str) -> dict[str, tuple[float, float]]:
        """Per structure, across all datasets of this category: (beat parent, did not)."""
        gain_min = self.config.thresholds.refine_min_gain
        out: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        rows = s.execute(
            select(CandidateORM.extra, CandidateORM.sharpe)
            .join(CellORM, CellORM.cell_id == CandidateORM.cell_id)
            .where(
                CandidateORM.stage == Stage.REFINE,
                CandidateORM.status == CandidateStatus.DONE,
                CellORM.category == category,
            )
        )
        for extra, sharpe in rows:
            name = (extra or {}).get("structure")
            if not name or sharpe is None:
                continue
            won = sharpe >= float((extra or {}).get("parent_best", 0)) + gain_min
            out[name][0 if won else 1] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}

    # ======================= driving =======================
    def advance(self) -> dict[str, int]:
        """Moves every open dataset one step when its current step is finished."""
        moved = {"health": 0, "survey": 0, "rounds": 0, "finished": 0}
        with self.ledger.session() as s:
            ids = [
                c.cell_id
                for c in s.scalars(
                    select(CellORM).where(
                        CellORM.state.in_(
                            [CellState.STUDYING, CellState.SURVEY, CellState.RESEARCH]
                        )
                    )
                )
            ]
        for cell_id in ids:
            with self.ledger.session() as s:
                cell = _cell(s, cell_id)
                if cell.sims_used >= self.config.budget.dataset_sim_cap:
                    self._finish(s, cell, "budget", "dataset simulation budget spent")
                    moved["finished"] += 1
                    continue
                phase = (cell.plan or {}).get("phase")
                pending = self._pending(s, cell_id)
                if phase == "health_cov" and not pending:
                    self._health_coverage(s, cell)
                    moved["health"] += 1
                    if cell.state == CellState.EXHAUSTED:
                        moved["finished"] += 1
                elif phase == "health_period" and not pending:
                    self._finish_health(s, cell)
                    moved["health"] += 1
                elif phase in ("survey", "research"):
                    before = (cell.state, len((cell.plan or {}).get("survey_seen") or []))
                    self._update_survey(s, cell)
                    after = (cell.state, len((cell.plan or {}).get("survey_seen") or []))
                    moved["survey"] += before != after
                    if cell.state == CellState.EXHAUSTED:
                        moved["finished"] += 1
                    elif cell.state == CellState.RESEARCH:
                        self._settle_refines(s, cell)
                        if self._pending(s, cell_id) < self.config.budget.round_size // 2:
                            if self._plan_round(s, cell) == 0:
                                if not self._pending(s, cell_id):
                                    self._finish(
                                        s,
                                        cell,
                                        "signal",
                                        "every stratum with signal has been worked",
                                    )
                                    moved["finished"] += 1
                            else:
                                moved["rounds"] += 1
        return moved

    # ======================= results =======================
    def evaluate(self) -> dict[str, int]:
        """Records every new result; queues robustness and repairs. Idempotent."""
        counts = {"evaluated": 0, "ready": 0}
        with self.ledger.session() as s:
            fresh = list(
                s.scalars(
                    select(CandidateORM).where(
                        CandidateORM.status.in_([CandidateStatus.DONE, CandidateStatus.FAILED]),
                        CandidateORM.evaluated.is_(False),
                    )
                )
            )
            for c in fresh:
                c.evaluated = True
                counts["evaluated"] += 1
                cell = _cell(s, c.cell_id)
                cell.sims_used += 1
                cell.updated_at = utc_now()
                if c.sharpe is None or c.stage in (Stage.PROFILE, Stage.ROBUST):
                    continue
                if self._suspicious(c):
                    c.extra = {**c.extra, "suspicious": True}
                    continue
                cell.best_sharpe = max(cell.best_sharpe, abs(c.sharpe))
                cell.hits += abs(c.sharpe) >= self.config.thresholds.screen_line
                if self._passes_is(c, cell):
                    c.extra = {**c.extra, "ready": True}
                    counts["ready"] += 1
                elif c.stage != Stage.REPAIR:
                    self._repair(s, cell, c)
            self._schedule_robustness(s)
            self._settle_robustness(s)
            for cell in s.scalars(
                select(CellORM).where(
                    CellORM.submitted >= self.config.thresholds.dataset_cap,
                    CellORM.state != CellState.SATURATED,
                )
            ):
                cell.state = CellState.SATURATED
                cell.stop_reason = f"{cell.submitted} submissions from this dataset"
        return counts

    def tries(self, s: Session, cell_id: str) -> int:
        return int(
            s.scalar(
                select(func.count())
                .select_from(CandidateORM)
                .where(
                    CandidateORM.cell_id == cell_id,
                    CandidateORM.stage.not_in([Stage.PROFILE, Stage.ROBUST]),
                    CandidateORM.status == CandidateStatus.DONE,
                )
            )
            or 0
        )

    def luck_bar(self, s: Session, cell_id: str) -> float:
        return stats.luck_bar(self.sigma, self.tries(s, cell_id))

    def _passes_is(self, c: CandidateORM, cell: CellORM) -> bool:
        t = self.config.thresholds
        if c.fails or c.sharpe is None or c.fitness is None or c.turnover is None:
            return False
        margin_bps = (c.margin or 0.0) * 10_000
        return (
            c.sharpe >= self.config.sharpe_floor(cell.region)
            and c.fitness >= t.submit_fitness
            and t.turnover_min <= c.turnover <= t.turnover_max
            and margin_bps >= self.config.margin_floor(cell.region)
        )

    def _suspicious(self, c: CandidateORM) -> bool:
        t = self.config.thresholds
        return abs(c.sharpe or 0) > t.suspicious_sharpe or (c.turnover or 0) > t.suspicious_turnover

    def _repair(self, s: Session, cell: CellORM, c: CandidateORM) -> None:
        """Close to the floor and blocked only by fixable checks: one rule-based fix per
        kind, never a fix of a fix."""
        fails = set((c.fails or "").split(",")) - {""}
        floor = self.config.sharpe_floor(cell.region)
        if not fails & FIXABLE or not fails <= FIXABLE | {"LOW_FITNESS", "LOW_SHARPE"}:
            return
        if (c.sharpe or 0) < floor - 0.2:
            return
        for p in gen.repair_proposals(
            c.expression,
            c.settings,
            sorted(fails),
            c.turnover,
            self.meta.neutralizations(cell.region),
            may_finetune=False,
        ):
            self._add(
                s, cell, p.expression, p.settings, Stage.REPAIR, p.role, change=p.change, parent=c
            )

    def _schedule_robustness(self, s: Session) -> None:
        """Only results above the luck bar of their dataset are stress-tested: per dataset
        the best two distinct field sets by fitness, at most twice the daily slots a day."""
        start = datetime.combine(datetime.now(UTC).date(), datetime.min.time(), tzinfo=UTC)
        tested_today = {
            c.parent_id
            for c in s.scalars(
                select(CandidateORM).where(
                    CandidateORM.stage == Stage.ROBUST, CandidateORM.created_at >= start
                )
            )
        }
        room = 2 * self.config.policy.max_submissions_per_day - len(tested_today)
        if room <= 0:
            return
        waiting = [
            c
            for c in s.scalars(select(CandidateORM).where(CandidateORM.status == "DONE"))
            if c.extra.get("ready") and not c.extra.get("robust_queued")
        ]
        waiting.sort(key=lambda c: -(c.fitness or 0))
        bars: dict[str, float] = {}
        taken: dict[str, set[tuple[str, ...]]] = {}
        for c in waiting:
            if room <= 0:
                break
            bar = bars.setdefault(c.cell_id, self.luck_bar(s, c.cell_id))
            if (c.sharpe or 0) < bar:
                c.extra = {**c.extra, "below_luck_bar": round(bar, 2)}
                continue
            signature = tuple(c.extra.get("fields") or [])
            seen = taken.setdefault(c.cell_id, set())
            if signature in seen or len(seen) >= 2:
                continue
            seen.add(signature)
            cell = _cell(s, c.cell_id)
            for p in gen.robustness_proposals(
                c.expression,
                c.settings,
                self.meta.neutralizations(cell.region),
                self.meta.universes(cell.region),
            ):
                self._add(
                    s,
                    cell,
                    p.expression,
                    p.settings,
                    Stage.ROBUST,
                    p.role,
                    parent=c,
                    extra={"robust_of": c.id},
                )
            c.extra = {**c.extra, "robust_queued": True, "luck_bar": round(bar, 2)}
            room -= 1

    def _settle_robustness(self, s: Session) -> None:
        t = self.config.thresholds
        ready = [
            c
            for c in s.scalars(select(CandidateORM).where(CandidateORM.status == "DONE"))
            if c.extra.get("robust_queued") and "robust" not in c.extra
        ]
        for parent in ready:
            kids = list(
                s.scalars(
                    select(CandidateORM).where(
                        CandidateORM.parent_id == parent.id, CandidateORM.stage == Stage.ROBUST
                    )
                )
            )
            if not kids or any(k.status in PENDING for k in kids):
                continue
            base = parent.sharpe or 0.0
            results: dict[str, float] = {}
            for k in kids:
                ratio = (k.sharpe or 0.0) / base if base else 0.0
                results[k.role] = min(results.get(k.role, 9.9), ratio)
            verdict = all(r >= t.robust_keep_ratio for r in results.values())
            parent.extra = {**parent.extra, "robust": {"pass": verdict, "ratios": results}}
            self.ledger.log(
                s, "robustness", parent.cell_id, candidate=parent.id, passed=verdict, ratios=results
            )

    # ======================= helpers =======================
    def _finish(self, s: Session, cell: CellORM, outcome: str, reason: str) -> None:
        cell.state = CellState.EXHAUSTED
        cell.stop_reason = reason
        cell.plan = {**(cell.plan or {}), "phase": "done", "outcome": outcome}
        for q in s.scalars(
            select(CandidateORM).where(
                CandidateORM.cell_id == cell.cell_id,
                CandidateORM.status == CandidateStatus.QUEUED,
                CandidateORM.stage != Stage.ROBUST,
            )
        ):
            q.status = CandidateStatus.INVALID
            q.error = f"dataset finished: {reason}"
        self.ledger.log(
            s, "dataset_stopped", cell.cell_id, outcome=outcome, reason=reason, sims=cell.sims_used
        )

    def _add(
        self,
        s: Session,
        cell: CellORM,
        expression: str,
        settings: dict[str, Any],
        stage: str,
        role: str,
        *,
        change: str = "",
        parent: CandidateORM | None = None,
        extra: dict[str, Any] | None = None,
    ) -> tuple[CandidateORM | None, bool]:
        """Validated, deduplicated (an identical earlier row is returned and its result
        reused), within the dataset budget."""
        fields = {
            f.field_id: f.type
            for f in s.scalars(select(FieldORM).where(FieldORM.cell_id == cell.cell_id))
        }
        verdict = check(expression, operators=self.meta.operators, fields=fields)
        if not verdict.ok:
            self.ledger.log(
                s, "invalid", cell.cell_id, expression=expression, issues=verdict.issues[:3]
            )
            return None, False
        if stage != Stage.ROBUST:
            room = (
                self.config.budget.dataset_sim_cap - cell.sims_used - self._pending(s, cell.cell_id)
            )
            if room <= 0:
                return None, False
        row, created = self.ledger.get_or_add(
            s,
            cell_id=cell.cell_id,
            parent_id=parent.id if parent else None,
            stage=stage,
            role=role,
            change=change,
            expression=expression,
            settings=settings,
            priority=100 * max(cell.rank_score, 1e-3) * STAGE_WEIGHT.get(stage, 1.0),
            extra={
                **(extra or {}),
                "atom": verdict.atom,
                "operators": verdict.operator_count,
                "fields": sorted(verdict.fields_used),
            },
        )
        if not created and row.status == CandidateStatus.INVALID:
            row.status = CandidateStatus.QUEUED  # cancelled by an earlier run: run it now
            row.error = None
        return row, created

    @staticmethod
    def _pending(s: Session, cell_id: str) -> int:
        return int(
            s.scalar(
                select(func.count())
                .select_from(CandidateORM)
                .where(
                    CandidateORM.cell_id == cell_id,
                    CandidateORM.status.in_(PENDING),
                    CandidateORM.stage != Stage.ROBUST,
                )
            )
            or 0
        )

    @staticmethod
    def _reps(s: Session, cell_id: str) -> list[FieldORM]:
        return list(
            s.scalars(
                select(FieldORM).where(
                    FieldORM.cell_id == cell_id, FieldORM.representative.is_(True)
                )
            )
        )

    def _settings(self, cell: CellORM) -> dict[str, Any]:
        return gen.base_settings(
            cell.region,
            cell.universe,
            cell.delay,
            cell.category.lower(),
            self.meta.neutralizations(cell.region),
        )


def _count(s: Session, cid: str | None) -> int | None:
    if not cid:
        return None
    c = s.get(CandidateORM, cid)
    if c is None or c.status != CandidateStatus.DONE:
        return None
    return (c.long_count or 0) + (c.short_count or 0)


def _brief(cell: CellORM) -> dict[str, Any]:
    return {
        "id": cell.dataset_id,
        "name": cell.dataset_name,
        "category": cell.category,
        "region": cell.region,
        "universe": cell.universe,
        "delay": cell.delay,
        "description": cell.description[:1500],
    }


def _cell(s: Session, cell_id: str) -> CellORM:
    cell = s.get(CellORM, cell_id)
    if cell is None:
        raise KeyError(cell_id)
    return cell


def reset_legacy(ledger: Ledger) -> int:
    """Datasets stopped or left open by the earlier logic (no plan) go back to the queue;
    their simulations stay and are reused when the same formula comes up again."""
    n = 0
    with ledger.session() as s:
        legacy: set[str] = set()
        for c in s.scalars(
            select(CellORM).where(
                CellORM.state.in_([CellState.EXHAUSTED, CellState.STUDYING, CellState.RESEARCH])
            )
        ):
            if not (c.plan or {}).get("phase"):
                c.state, c.stop_reason = CellState.NEW, ""
                legacy.add(c.cell_id)
                n += 1
        for q in s.scalars(
            select(CandidateORM).where(CandidateORM.status == CandidateStatus.QUEUED)
        ):
            if q.cell_id in legacy:
                q.status, q.error = CandidateStatus.INVALID, "left by the earlier logic"
    return n
