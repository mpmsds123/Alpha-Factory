import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import (
    Cited,
    CostCapReached,
    LLMClient,
    LLMInvalid,
    LLMUnavailable,
    Prompt,
    record_program_decision,
)
from alpha_orchestrator.factory.models import DecisionORM

PROMPT = Prompt(step="diagnosis", version="diagnosis.v1", system="sys", task="judge")


class Answer(Cited):
    label: str


def _client(replies: list[str], cfg: FactoryConfig | None = None) -> tuple[LLMClient, list]:
    seen: list[tuple[str, str]] = []

    def complete(model: str, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        seen.append((model, user))
        reply = replies.pop(0)
        if reply == "BOOM":
            raise ConnectionError("down")
        return reply, 1000, 100

    return LLMClient(cfg or FactoryConfig(), Ledger("sqlite://"), complete), seen


def _rows(client: LLMClient) -> list[DecisionORM]:
    with client.ledger.session() as s:
        return list(s.scalars(select(DecisionORM).order_by(DecisionORM.id)))


def test_valid_answer_is_logged_with_citations() -> None:
    client, seen = _client([json.dumps({"label": "x", "cites": ["H1"]})])
    answer, did = client.ask(
        PROMPT, Answer, {"a": 1}, hypothesis_id=None, retrieved={"records": ["H1"], "priors": []}
    )
    assert answer.label == "x" and did == 1
    row = _rows(client)[0]
    assert (row.cited, row.flags, row.model, row.prompt_version) == (
        ["H1"],
        [],
        "deepseek-flash",
        "diagnosis.v1",
    )
    assert row.retrieved == {"records": ["H1"], "priors": []} and row.inputs == {"a": 1}
    assert row.cost_usd > 0
    assert json.loads(seen[0][1])["allowed_citations"] == ["H1"]


def test_uncited_answer_is_flagged_not_rejected() -> None:
    client, _ = _client([json.dumps({"label": "x", "no_relevant_record": True})])
    client.ask(PROMPT, Answer, {}, hypothesis_id=None, retrieved={})
    assert _rows(client)[0].flags == ["uncited"]


def test_invented_citation_gets_one_retry_then_fails() -> None:
    bad = json.dumps({"label": "x", "cites": ["H99"]})
    client, seen = _client([bad, bad])
    with pytest.raises(LLMInvalid) as err:
        client.ask(PROMPT, Answer, {}, hypothesis_id=None, retrieved={"records": ["H1"]})
    assert "previous answer was rejected" in seen[1][1]
    row = _rows(client)[-1]
    assert row.id == err.value.decision_id and not row.ok and row.flags == ["invalid"]
    assert row.attempts == 2


def test_schema_violation_then_valid_answer_is_flagged_retried() -> None:
    client, _ = _client(
        ["not json", json.dumps({"label": "y", "extra": 1}), json.dumps({"label": "z"})]
    )
    with pytest.raises(LLMInvalid):
        client.ask(PROMPT, Answer, {}, hypothesis_id=None, retrieved={})
    client2, _ = _client([json.dumps({"label": 1}), json.dumps({"label": "ok"})])
    answer, _ = client2.ask(PROMPT, Answer, {}, hypothesis_id=None, retrieved={})
    assert answer.label == "ok" and _rows(client2)[0].flags == ["uncited", "retried"]


def test_truncated_reply_is_retried() -> None:
    client, _ = _client(['{"label": "x"\x00truncated', json.dumps({"label": "x"})])
    answer, _ = client.ask(PROMPT, Answer, {}, hypothesis_id=None, retrieved={})
    assert answer.label == "x"


def test_transport_failure_is_logged_and_raised() -> None:
    client, _ = _client(["BOOM"])
    with pytest.raises(LLMUnavailable):
        client.ask(PROMPT, Answer, {}, hypothesis_id=None, retrieved={})
    assert _rows(client)[0].flags == ["transport"]


def test_test_period_values_never_reach_the_llm() -> None:
    client, seen = _client([])
    with pytest.raises(ValueError, match="test-period"):
        client.ask(PROMPT, Answer, {"rows": [{"test_sharpe": 1}]}, hypothesis_id=None, retrieved={})
    assert seen == []


def test_daily_cap_blocks_calls() -> None:
    cfg = FactoryConfig()
    cfg.cost.daily_usd_cap = 0.0
    client, _ = _client([], cfg)
    with pytest.raises(CostCapReached):
        client.ask(PROMPT, Answer, {}, hypothesis_id=None, retrieved={})


def test_peak_pricing() -> None:
    client, _ = _client([])
    peak = datetime(2026, 11, 12, 2, tzinfo=UTC)  # Thursday 02:00 UTC
    off = datetime(2026, 11, 14, 2, tzinfo=UTC)  # Saturday
    assert client.cost("deepseek-flash", 1_000_000, 0, peak) == pytest.approx(0.30)
    assert client.cost("deepseek-flash", 1_000_000, 0, off) == pytest.approx(0.15)
    assert client.cost("unknown", 10, 10) == 0.0


def test_program_decisions_share_the_log() -> None:
    ledger = Ledger("sqlite://")
    did = record_program_decision(ledger, None, "decision", {"action": "drop"})
    with ledger.session() as s:
        row = s.get(DecisionORM, did)
        assert row is not None and row.actor == "program" and row.prompt_version == "rule"
