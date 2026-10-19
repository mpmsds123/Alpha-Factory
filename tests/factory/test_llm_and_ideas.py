import json
from pathlib import Path

import pytest
from sqlalchemy import select

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ideas import IdeaIndex, import_ideas, tokens
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM
from alpha_orchestrator.factory.models import IdeaORM, LLMCallORM


def scripted(*replies: str):  # type: ignore[no-untyped-def]
    queue = list(replies)
    return lambda model, system, user: (queue.pop(0), 100, 50)


def test_group_fields_keeps_known_families_and_is_ledgered() -> None:
    ledger = Ledger("sqlite://")
    reply = json.dumps(
        {"groups": {"EPS revisions": ["eps_rev", "made_up"], "Dispersion": ["eps_std"]}}
    )
    llm = FactoryLLM(FactoryConfig(), ledger, complete=scripted(reply))
    out = llm.group_fields({"id": "ds"}, [{"family": "eps_rev"}, {"family": "eps_std"}])
    assert out == {"eps_rev": "EPS revisions", "eps_std": "Dispersion"}
    with ledger.session() as s:
        (call,) = s.scalars(select(LLMCallORM))
    assert call.purpose == "group_fields" and call.model == FactoryConfig().llm.model


def test_combinations_use_the_strong_model_and_cap_the_count() -> None:
    ledger = Ledger("sqlite://")
    reply = json.dumps(
        {"combinations": [{"expression": "rank(a) - rank(b)", "why": "w"}] * 9 + [{"nope": 1}]}
    )
    llm = FactoryLLM(FactoryConfig(), ledger, complete=scripted(reply))
    out = llm.combinations({"id": "ds"}, [], [], n=3)
    assert len(out) == 3 and out[0] == {"expression": "rank(a) - rank(b)", "why": "w"}
    with ledger.session() as s:
        (call,) = s.scalars(select(LLMCallORM))
    assert call.model == FactoryConfig().llm.strong_model


def test_bad_json_is_retried_then_raises() -> None:
    ledger = Ledger("sqlite://")
    llm = FactoryLLM(FactoryConfig(), ledger, complete=scripted("not json", '{"description": "d"}'))
    assert llm.describe({}) == "d"
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
    (d / "1.md").write_text(
        "# Research Paper 1: Short interest and returns\n- 来源: https://x\n---\n"
        + "Short interest predicts negative returns. " * 10
    )
    (d / "2.md").write_text("# Momentum in news\n---\n" + "News sentiment momentum. " * 10)
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        assert import_ideas(s, [d]) == 2
        assert import_ideas(s, [d]) == 0  # idempotent
        hits = IdeaIndex.load(s).search("short interest", 1)
    assert hits[0].title.startswith("Research Paper 1") and hits[0].url == "https://x"


def test_homework_thread_becomes_one_document_per_answer(tmp_path: Path) -> None:
    f = tmp_path / "作业帖.md"
    f.write_text(
        "# 系列课程Course 1作业\n## 学员作业与回复\n\n"
        "--- #1  2026-05-01  (0赞) ---\nlong count 是做多的股票数 " + "x" * 100 + "\n\n"
        "# 系列课程Course 3作业\n## 学员作业与回复\n\n"
        "--- #1  2026-05-20  (2赞) ---\n模板一：group_rank(ts_delta(pcr_oi_30, 5), sector) "
        + "期权情绪"
        * 30
        + "\n\n"
        "--- #2  2026-05-21  (0赞) ---\n已完成，没有表达式" + "y" * 100 + "\n",
        encoding="utf-8",
    )
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        assert import_ideas(s, [f]) == 1
        (doc,) = s.scalars(select(IdeaORM))
    assert doc.source == "作业区" and doc.title == "系列课程Course 3作业 · 作业 #1"
    assert "pcr_oi_30" in doc.body
