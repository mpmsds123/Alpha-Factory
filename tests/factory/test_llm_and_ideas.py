import json
from pathlib import Path

import pytest
from sqlalchemy import select

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ideas import IdeaIndex, import_ideas, tokens
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM, FieldInfo
from alpha_orchestrator.factory.models import LLMCallORM


def scripted(*replies: str):  # type: ignore[no-untyped-def]
    queue = list(replies)
    return lambda model, system, user: (queue.pop(0), 100, 50)


def test_hypotheses_drop_invented_fields_and_are_ledgered() -> None:
    ledger = Ledger("sqlite://")
    reply = json.dumps({"hypotheses": [
        {"title": "Revision", "statement": "s", "fields": ["eps_rev", "made_up"], "sources": ["P"]},
        {"title": "Ghost", "statement": "s", "fields": ["nope"]},
    ]})
    llm = FactoryLLM(FactoryConfig(), ledger, complete=scripted(reply))
    out = llm.propose_hypotheses({"id": "ds"}, [FieldInfo("eps_rev", "MATRIX", "d")], [])
    assert [h["title"] for h in out] == ["Revision"] and out[0]["fields"] == ["eps_rev"]
    with ledger.session() as s:
        (call,) = s.scalars(select(LLMCallORM))
    assert call.purpose == "hypotheses" and call.prompt_tokens == 100


def test_bad_json_is_retried_then_raises() -> None:
    ledger = Ledger("sqlite://")
    llm = FactoryLLM(FactoryConfig(), ledger, complete=scripted("not json", '{"consistent": true}'))
    assert llm.aligned({"title": "t"}, "rank(x)") == (True, "")
    llm = FactoryLLM(FactoryConfig(), ledger, complete=scripted("x", "y"))
    with pytest.raises(RuntimeError):
        llm.describe({})
    with ledger.session() as s:
        assert len([c for c in s.scalars(select(LLMCallORM)) if not c.ok]) == 3


def test_tokens_mix_words_and_cjk_bigrams() -> None:
    assert tokens("Analyst 新闻情绪") == ["analyst", "新闻", "闻情", "情绪"]


def test_import_and_search(tmp_path: Path) -> None:
    d = tmp_path / "Research_Papers_for_Users"
    d.mkdir()
    (d / "1.md").write_text("# Research Paper 1: Short interest and returns\n- 来源: https://x\n---\n"
                            + "Short interest predicts negative returns. " * 10)
    (d / "2.md").write_text("# Momentum in news\n---\n" + "News sentiment momentum. " * 10)
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        assert import_ideas(s, [d]) == 2
        assert import_ideas(s, [d]) == 0  # idempotent
        hits = IdeaIndex.load(s).search("short interest", 1)
    assert hits[0].title.startswith("Research Paper 1") and hits[0].url == "https://x"
