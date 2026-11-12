import json
import random
from collections.abc import Callable
from typing import Any

from sqlalchemy import select

from alpha_orchestrator.factory import loop
from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import LLMClient
from alpha_orchestrator.factory.models import (
    DecisionORM,
    ExpressionORM,
    HStatus,
    HypothesisORM,
    PriorORM,
    Review,
    SimulationORM,
    Verdict,
)
from alpha_orchestrator.factory.validate import signature
from tests.factory.fakes import OPERATORS, VOCAB
from tests.factory.platform import FakeAPI

HYP = {
    "claim": "Stocks with a high value of this quantity are underpriced because investors "
    "ignore it, and the effect lasts about a month.",
    "direction": "higher_value_higher_return",
    "hypothesis_type": "level",
    "holding_days": 21,
    "turnover_band": "mid",
    "falsifier": "no positive train Sharpe at all",
    "cites": [],
    "no_relevant_record": True,
}
GOOD = {"sharpe": 1.8, "fitness": 1.2, "turnover": 0.2, "returns": 0.1, "test_sharpe": 0.9}
MID = {"sharpe": 1.2, "fitness": 0.7, "turnover": 0.2, "test_sharpe": 0.5}
LOW = {"sharpe": 0.3, "fitness": 0.1, "turnover": 0.2, "test_sharpe": 0.1}
NEG = {"sharpe": -1.4, "fitness": -0.5, "turnover": 0.2, "test_sharpe": -0.4}


def scripted(answers: dict[str, list[Any]]) -> tuple[Callable[..., Any], list[dict[str, Any]]]:
    """An LLM whose answer depends on the step; each answer may be a function of the
    request."""
    seen: list[dict[str, Any]] = []

    def complete(model: str, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        request = json.loads(user.split("\n\nYour previous answer")[0])
        task = request["task"]
        step = next(k for k in answers if k in task)
        seen.append({"step": step, **request})
        queue = answers[step]
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if callable(answer):
            answer = answer(request)
        return json.dumps(answer), 500, 100

    return complete, seen


STEP = {
    "hypothesis": "testable economic hypothesis",
    "expression": "Write 1 to 3 expressions",
    "diagnosis": "Compare the round",
    "decision": "Choose the next step",
}


def _factory(
    answers: dict[str, list[Any]],
    score: Callable[[str, dict[str, Any]], dict[str, Any]],
    cfg: FactoryConfig | None = None,
) -> tuple[loop.Factory, Ledger, FakeAPI, list[dict[str, Any]]]:
    cfg = cfg or FactoryConfig()
    cfg.budget.min_batch_interval_s = 0
    ledger = Ledger("sqlite://")
    api = FakeAPI(score, OPERATORS)
    complete, seen = scripted({STEP[k]: v for k, v in answers.items()})
    llm = LLMClient(cfg, ledger, complete)
    meta = loop.Meta(
        operators=set(OPERATORS),
        definitions=dict(OPERATORS),
        signatures={k: s for k, d in OPERATORS.items() if (s := signature(d))},
        options=api.simulation_options(),
    )
    from alpha_orchestrator.factory import catalog

    catalog.refresh(api, ledger, cfg, VOCAB)  # type: ignore[arg-type]
    factory = loop.Factory(api, ledger, cfg, llm, VOCAB, meta, random.Random(1))  # type: ignore[arg-type]
    return factory, ledger, api, seen


def _expr(*items: str) -> dict[str, Any]:
    return {
        "expressions": [{"expression": x, "rationale": "tests the claim directly"} for x in items],
        "cites": [],
    }


def _field(req: dict[str, Any]) -> str:
    return str(req["input"]["hypothesis"]["field"])


def _drain(factory: loop.Factory, ledger: Ledger, api: FakeAPI, rounds: int = 6) -> None:
    from alpha_orchestrator.factory import gates
    from alpha_orchestrator.factory.scheduler import Simulator

    sim = Simulator(api, ledger, factory.config, sleep=lambda _: None)  # type: ignore[arg-type]
    for _ in range(rounds):
        sim.run(100)
        gates.evaluate(api, ledger, factory.config)  # type: ignore[arg-type]
        factory.promote_passed()
        with ledger.session() as s:
            open_ids = list(
                s.scalars(select(HypothesisORM.id).where(HypothesisORM.status == HStatus.OPEN))
            )
        for hid in open_ids:
            if factory._pending(hid) == 0:
                factory.step(hid)


def test_hypothesis_to_review_with_lineage_and_logged_retrieval() -> None:
    def score(e: str, st: dict[str, Any]) -> dict[str, Any]:
        if "w_avg" in e and ", 21)" in e:
            return GOOD
        return MID if "w_avg" in e else LOW

    answers = {
        "hypothesis": [HYP],
        "expression": [lambda r: _expr(f"w_avg({_field(r)}, 5)", f"order({_field(r)})")],
        "diagnosis": [{"label": "idea_supported", "reason": "the predicted sign shows up clearly"}],
        "decision": [
            lambda r: {
                "action": "mutate",
                "reason": "a monthly window fits the holding period",
                "lesson": "monthly windows fit monthly claims",
                "mutations": [
                    {
                        "parent": r["input"]["parents"][0]["expression_id"],
                        "change": "window",
                        "expression": r["input"]["parents"][0]["expression"].replace("5)", "21)"),
                        "reason": "match the 21-day holding period",
                    }
                ],
            }
        ],
    }
    factory, ledger, api, seen = _factory(answers, score)
    with ledger.session() as s:
        s.add(
            HypothesisORM(
                region="R1",
                dataset_id="other",
                category="cat",
                field_type="MATRIX",
                hypothesis_type="level",
                status=HStatus.CLOSED,
                verdict=Verdict.IDEA_FAILED,
                source="v7",
            )
        )
    hid = factory.open_new()
    assert hid is not None
    _drain(factory, ledger, api)
    with ledger.session() as s:
        h = s.get(HypothesisORM, hid)
        assert h is not None
        assert (h.status, h.verdict, h.sims_used, h.round) == (HStatus.CLOSED, Verdict.PASSED, 3, 2)
        assert h.tags["hypothesis_type"] == "level" and h.direction == 1
        child = s.scalar(select(ExpressionORM).where(ExpressionORM.parent_id.is_not(None)))
        assert child is not None and child.mutation == "window" and child.round == 2
        stored = s.scalar(select(SimulationORM).where(SimulationORM.review == Review.PENDING))
        assert stored is not None and stored.expression_id == child.id
        steps = [d.step for d in loop.decisions_of(s, hid)]
        assert steps == ["hypothesis", "expression", "diagnosis", "decision"]
        first = loop.decisions_of(s, hid)[0]
        assert first.retrieved["records"] == ["H1"] and first.flags == ["uncited"]
    assert all("test_sharpe" not in json.dumps(r) for r in seen)
    assert seen[0]["input"]["field"]["id"] in ("sig_a", "sig_b")


def test_round_one_below_gate_is_closed_by_the_program() -> None:
    answers = {
        "hypothesis": [HYP],
        "expression": [lambda r: _expr(f"order({_field(r)})")],
        "diagnosis": [{"label": "idea_failed", "reason": "no signal in either direction at all"}],
        "decision": [{"action": "drop", "reason": "x" * 10, "lesson": "y" * 10}],
    }
    factory, ledger, api, seen = _factory(answers, lambda e, st: LOW)
    with ledger.session() as s:
        p = PriorORM(claim="c", source_type="official", a=3.0, b=2.0)
        s.add(p)
    hid = factory.open_new()
    assert hid is not None
    # the diagnosis cites the prior: it should lose confidence when the idea fails
    answers_diag = {
        "label": "idea_failed",
        "reason": "no signal in either direction",
        "cites": ["P1"],
    }
    factory.llm._complete, _ = scripted(
        {STEP["diagnosis"]: [answers_diag], STEP["decision"]: answers["decision"]}
    )
    _drain(factory, ledger, api)
    with ledger.session() as s:
        h = s.get(HypothesisORM, hid)
        assert h is not None and (h.verdict, h.sims_used) == (Verdict.IDEA_FAILED, 1)
        last = loop.decisions_of(s, hid)[-1]
        assert (last.actor, last.output["action"]) == ("program", "drop")
        prior = s.get(PriorORM, 1)
        assert prior is not None and prior.b == 3.0
    assert not any(r["step"] == STEP["decision"] for r in seen)


def test_child_must_make_the_change_it_declares() -> None:
    answers = {
        "hypothesis": [HYP],
        "expression": [lambda r: _expr(f"w_avg({_field(r)}, 5)")],
        "diagnosis": [{"label": "idea_supported", "reason": "the predicted sign shows up"}],
        "decision": [
            lambda r: {
                "action": "mutate",
                "reason": "try a different window",
                "lesson": "declared changes are checked",
                "mutations": [
                    {
                        "parent": r["input"]["parents"][0]["expression_id"],
                        "change": "window",
                        "expression": r["input"]["parents"][0]["expression"].replace(
                            "w_avg", "w_chg"
                        ),
                        "reason": "says window, changes operator",
                    }
                ],
            }
        ],
    }
    factory, ledger, api, _ = _factory(answers, lambda e, st: MID)
    hid = factory.open_new()
    _drain(factory, ledger, api)
    with ledger.session() as s:
        bad = s.scalar(select(ExpressionORM).where(ExpressionORM.parent_id.is_not(None)))
        assert bad is not None and not bad.valid
        assert "declared a window change but made a operator change" in bad.rejection
        h = s.get(HypothesisORM, hid)
        assert h is not None and h.status == HStatus.CLOSED and h.reason == "no valid child"


def test_strong_opposite_sign_allows_a_revision_with_smaller_budget() -> None:
    answers = {
        "hypothesis": [HYP],
        "expression": [lambda r: _expr(f"order({_field(r)})")],
        "diagnosis": [{"label": "idea_failed", "reason": "the sign is reversed and strong"}],
        "decision": [
            {
                "action": "revise",
                "reason": "the effect runs the other way",
                "lesson": "high values were overpriced",
                "revision": {
                    **{k: HYP[k] for k in ("claim", "hypothesis_type")},
                    "direction": "higher_value_lower_return",
                    "holding_days": 21,
                    "turnover_band": "mid",
                    "falsifier": "no negative relation",
                },
            }
        ],
    }

    def score(e: str, st: dict[str, Any]) -> dict[str, Any]:
        return {**NEG, "sharpe": 1.3} if "flipit" in e else NEG

    factory, ledger, api, _ = _factory(answers, score)
    hid = factory.open_new()
    factory.llm._complete, _ = scripted(
        {
            STEP["diagnosis"]: answers["diagnosis"],
            STEP["decision"]: answers["decision"] * 1
            + [{"action": "drop", "reason": "x" * 10, "lesson": "y" * 10}],
            STEP["expression"]: [lambda r: _expr(f"flipit({_field(r)})")],
        }
    )
    _drain(factory, ledger, api, rounds=2)
    with ledger.session() as s:
        old = s.get(HypothesisORM, hid)
        new = s.scalar(select(HypothesisORM).where(HypothesisORM.parent_id == hid))
        assert old is not None and old.verdict == Verdict.IDEA_FAILED
        assert new is not None and (new.direction, new.sims_budget) == (-1, 6)
        assert "revised into" in old.reason


def test_invalid_and_opposite_expressions_are_rejected() -> None:
    answers = {
        "hypothesis": [HYP],
        "expression": [lambda r: _expr(f"flipit({_field(r)})", "nosuch(sig_a)", "order(zz)")],
    }
    factory, ledger, _, seen = _factory(answers, lambda e, st: MID)
    hid = factory.open_new()
    with ledger.session() as s:
        h = s.get(HypothesisORM, hid)
        assert h is not None and (h.verdict, h.status) == (Verdict.INVALID, HStatus.CLOSED)
        reasons = [e.rejection for e in s.scalars(select(ExpressionORM))]
    assert any("against the hypothesis direction" in r for r in reasons)
    assert any("unknown operator" in r for r in reasons)
    assert any("not allowed" in r for r in reasons)
    second = [r for r in seen if r["step"] == STEP["expression"]][1]
    assert second["input"]["rejected_before"]


def test_run_end_to_end_stops_when_nothing_is_left() -> None:
    answers = {
        "hypothesis": [HYP],
        "expression": [lambda r: _expr(f"order({_field(r)})")],
        "diagnosis": [{"label": "idea_failed", "reason": "no signal in either direction at all"}],
        "decision": [{"action": "drop", "reason": "x" * 10, "lesson": "y" * 10}],
    }
    factory, ledger, api, _ = _factory(answers, lambda e, st: LOW)
    progress: list[str] = []
    report = loop.run(
        factory.config,
        api,
        ledger,
        factory.llm,
        VOCAB,
        minutes=1,  # type: ignore[arg-type]
        rng=random.Random(0),
        meta=factory.meta,
        sleep=lambda _: None,
        on_progress=progress.append,
    )
    assert report.opened == 2 and report.stopped_because == "nothing left to do"
    assert progress and report.simulations == 2
    with ledger.session() as s:
        assert len(list(s.scalars(select(DecisionORM)))) >= 6


def test_cost_guard_blocks_new_hypotheses() -> None:
    factory, ledger, _, _ = _factory({"hypothesis": [HYP]}, lambda e, st: LOW)
    factory.may_open = lambda: "too expensive"
    assert factory.advance() == {"stepped": 0, "opened": 0, "guard": 1}
    assert ledger.events("open_blocked")[0]["reason"] == "too expensive"
