"""第 2 问 可追溯 and 第 6 问 查重: everything here is a mechanical check, no LLM.

An idea must carry one piece of evidence, either
  {"type": "card", "id": "Q3"}                                   a method card (cards.py), or
  {"type": "paper", "id": "2523", "quote": "...verbatim text..."} an official document.
Official documents = the platform's research-paper posts and BRAIN TIPS (225 in the ledger
on 2026-10-14). Homework posts are never evidence: they are other people's formulas.
"""

import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha_orchestrator.factory.cards import IDEA_CARDS
from alpha_orchestrator.factory.ideas import tokens
from alpha_orchestrator.factory.models import IdeaORM

OFFICIAL_SOURCES = ("Research_Papers_for_Consultants", "Research_Papers_for_Users", "BRAIN_TIPS")
DOCS_SHOWN = 5  # documents offered to the LLM per idea
DOC_CHARS = 2000  # characters of each document shown
QUOTE_MAX = 400  # a quote is a passage, not a whole document
DUPLICATE_OVERLAP = 0.5  # 第 6 问: measured on the ledger, duplicates 0.56-0.62, others <= 0.38
REQUIRED = ("title", "mechanism", "long", "short", "horizon_days", "evidence")
WORD = re.compile(r"[a-z0-9]+")


def official_docs(s: Session) -> list[dict[str, str]]:
    rows = s.scalars(select(IdeaORM).where(IdeaORM.source.in_(OFFICIAL_SOURCES)))
    return [{"id": str(d.id), "title": d.title, "text": d.body or ""} for d in rows]


def retrieve(query: str, docs: list[dict[str, str]], k: int = DOCS_SHOWN) -> list[dict[str, str]]:
    """By meaning words (the dataset's description and fields), never by dataset name."""
    q = set(tokens(query))
    if not q:
        return []
    scored = []
    for i, d in enumerate(docs):
        words = tokens(f"{d['title']} {d['text']}")
        if not words:
            continue
        hits = sum(1 for w in words if w in q)
        scored.append((hits / (len(words) ** 0.5), i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [
        {"id": docs[i]["id"], "title": docs[i]["title"], "text": docs[i]["text"][:DOC_CHARS]}
        for score, i in scored[:k]
        if score > 0
    ]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def check_idea(card: dict[str, Any], docs: dict[str, dict[str, str]]) -> tuple[bool, str]:
    """Format and evidence (第 2 问 规则 1). Returns (ok, reason when not ok)."""
    missing = [k for k in REQUIRED if not str(card.get(k, "")).strip()]
    if missing:
        return False, f"missing: {', '.join(missing)}"
    ev = card.get("evidence")
    if not isinstance(ev, dict):
        return False, "evidence must be an object"
    kind = str(ev.get("type", "")).lower()
    ref = str(ev.get("id", "")).strip()
    if kind == "card":
        if ref.upper() not in IDEA_CARDS:
            return False, f"card {ref!r} is not on the card list"
        return True, ""
    if kind == "paper":
        doc = docs.get(ref)
        if doc is None:
            return False, f"document {ref!r} was not among the documents given"
        quote = _norm(str(ev.get("quote", "")))
        if len(quote) < 20 or len(quote) > QUOTE_MAX:
            return False, f"quote must be 20-{QUOTE_MAX} characters copied from the document"
        if quote not in _norm(doc["text"]):
            return False, f"the quote is not found word for word in document {ref}"
        return True, ""
    return False, "evidence type must be card or paper"


def evidence_key(card: dict[str, Any]) -> str:
    ev = card.get("evidence") or {}
    return f"{str(ev.get('type', '')).lower()}:{str(ev.get('id', '')).strip().upper()}"


def overlap(a: str, b: str) -> float:
    x, y = set(WORD.findall(a.lower())), set(WORD.findall(b.lower()))
    return len(x & y) / len(x | y) if x and y else 0.0


def duplicate_of(card: dict[str, Any], earlier: list[dict[str, str]]) -> str | None:
    """第 6 问: same evidence and search words overlapping at least half -> the title it
    repeats. `earlier` rows carry title, query and evidence_key."""
    key = evidence_key(card)
    query = str(card.get("query", ""))
    for e in earlier:
        if e["evidence_key"] == key and overlap(query, e["query"]) >= DUPLICATE_OVERLAP:
            return e["title"]
    return None
