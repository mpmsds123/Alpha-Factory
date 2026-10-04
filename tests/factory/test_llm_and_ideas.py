import json
from pathlib import Path

import pytest
from sqlalchemy import select

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ideas import IdeaIndex, import_ideas, tokens
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM, FieldInfo
from alpha_orchestrator.factory.models import IdeaORM, LLMCallORM


def scripted(*replies: str):  # type: ignore[no-untyped-def]
    queue = list(replies)
    return lambda model, system, user: (queue.pop(0), 100, 50)


def test_ideas_keep_only_real_fields_and_complete_templates() -> None:
    ledger = Ledger("sqlite://")
    reply = json.dumps(
        {
            "ideas": [
                {
                    "title": "Revision drift",
                    "mechanism": "m",
                    "template": "group_rank(ts_delta({A}, {w}), industry)",
                    "slots": {"A": ["eps_rev", "made_up"]},
                    "windows": [21, "63", "x"],
                    "decay": 99,
                    "neutralization": "industry",
                    "sources": ["P"],
                },
                {
                    "title": "Ghost",
                    "mechanism": "m",
                    "template": "rank({A})",
                    "slots": {"A": ["no"]},
                },
                {
                    "title": "No slot",
                    "mechanism": "m",
                    "template": "rank({A} - {B})",
                    "slots": {"A": ["eps_rev"]},
                },
            ]
        }
    )
    llm = FactoryLLM(FactoryConfig(), ledger, complete=scripted(reply))
    out = llm.ideas({"id": "ds"}, [FieldInfo("eps_rev", "MATRIX", "d")], [], [], [])
    assert [i["title"] for i in out] == ["Revision drift"]
    idea = out[0]
    assert idea["slots"] == {"A": ["eps_rev"]} and idea["windows"] == [21, 63]
    assert idea["decay"] == 30 and idea["neutralization"] == "INDUSTRY"
    with ledger.session() as s:
        (call,) = s.scalars(select(LLMCallORM))
    assert call.purpose == "ideas" and call.prompt_tokens == 100
    assert call.model == FactoryConfig().llm.strong_model


def test_review_keeps_known_verdicts_and_defaults_the_dataset_decision() -> None:
    ledger = Ledger("sqlite://")
    reply = json.dumps(
        {
            "ideas": [
                {
                    "id": "h1",
                    "verdict": "dig",
                    "reason": "r",
                    "expressions": [{"expression": "rank(a)", "why": "w"}] * 20,
                },
                {"id": "h2", "verdict": "celebrate"},
            ],
            "dataset": {"decision": "party"},
        }
    )
    llm = FactoryLLM(FactoryConfig(), ledger, complete=scripted(reply))
    out = llm.review({}, [], [], [], dig_per_idea=3)
    assert [v["id"] for v in out["ideas"]] == ["h1"] and len(out["ideas"][0]["expressions"]) == 3
    assert out["dataset"]["decision"] == "continue"


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
