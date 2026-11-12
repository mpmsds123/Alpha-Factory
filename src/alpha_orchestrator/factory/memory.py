"""Memory A (priors from documents) and Memory B (one record per hypothesis).

Similarity is the program's: explicit tags (dataset category, region, field type,
hypothesis type) with fixed weights, no embeddings. What was retrieved for a decision is
returned as ids so the decision log can show it.
"""

import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha_orchestrator.factory.config import Memory
from alpha_orchestrator.factory.models import (
    DecisionORM,
    HStatus,
    HypothesisORM,
    PriorORM,
    Verdict,
)
from alpha_orchestrator.factory.util import utc_now

TAG_KEYS = ("category", "region", "field_type", "hypothesis_type")
UNKNOWN = {"", "unknown", None}
POSITIVE = {Verdict.PASSED, Verdict.SUPPORTED}
NEGATIVE = {Verdict.IDEA_FAILED}
FAILED = {Verdict.IDEA_FAILED, Verdict.IMPLEMENTATION_FAILED, Verdict.BUDGET}
SCAN_LIMIT = 2000  # newest closed hypotheses considered per retrieval (bounds work)
PRIOR_ID = re.compile(r"^P(\d+)$")


def tags_of(h: HypothesisORM) -> dict[str, str]:
    """The program's tags for a hypothesis."""
    return {
        "category": (h.category or "").lower(),
        "region": h.region or "",
        "field_type": (h.field_type or "").upper(),
        "hypothesis_type": h.hypothesis_type or "unknown",
    }


def similarity(a: dict[str, Any], b: dict[str, Any], weights: dict[str, int]) -> int:
    """Sum of the weights of the tags two records share (unknown tags never match)."""
    return sum(w for k, w in weights.items() if a.get(k) not in UNKNOWN and a.get(k) == b.get(k))


def record_view(h: HypothesisORM) -> dict[str, Any]:
    """What a decision sees of a past hypothesis (train-period numbers only)."""
    return {
        "id": f"H{h.id}",
        "claim": h.claim[:400],
        "tags": tags_of(h),
        "field": h.field_id,
        "verdict": h.verdict,
        "reason": h.reason[:300],
        "lesson": h.lesson[:300],
        "best_train_sharpe": round(h.best_train_sharpe, 2),
        "simulations": h.sims_used,
    }


def retrieve_records(
    s: Session, tags: dict[str, Any], cfg: Memory, exclude: set[int] | None = None
) -> list[HypothesisORM]:
    """The k closed hypotheses most similar by tags; a failure is always kept in view
    when one qualifies, so the model sees what did not work as well as what did."""
    rows = list(
        s.scalars(
            select(HypothesisORM)
            .where(HypothesisORM.status == HStatus.CLOSED)
            .order_by(HypothesisORM.id.desc())
            .limit(SCAN_LIMIT)
        )
    )
    scored = [
        (similarity(tags, tags_of(h), cfg.weights), h)
        for h in rows
        if h.id not in (exclude or set())
    ]
    scored = [(sc, h) for sc, h in scored if sc >= cfg.min_score]
    scored.sort(key=lambda x: (-x[0], -x[1].id))
    top = [h for _, h in scored[: cfg.k_records]]
    if top and not any(h.verdict in FAILED for h in top):
        failure = next((h for _, h in scored if h.verdict in FAILED), None)
        if failure is not None:
            top[-1] = failure
    return top


def prior_view(p: PriorORM) -> dict[str, Any]:
    return {
        "id": f"P{p.id}",
        "claim": p.claim[:400],
        "source": f"{p.source_type}: {p.source_title[:120]}",
        "confidence": round(p.confidence, 2),
    }


def retrieve_priors(s: Session, tags: dict[str, Any], cfg: Memory) -> list[PriorORM]:
    """Priors that apply to these tags (a prior with no tags applies everywhere), most
    specific first, then most trusted."""
    out = []
    for p in s.scalars(select(PriorORM)):
        cats = [c.lower() for c in (p.tags or {}).get("categories") or []]
        types = list((p.tags or {}).get("hypothesis_types") or [])
        if cats and tags.get("category") not in cats:
            continue
        if (
            types
            and tags.get("hypothesis_type") not in UNKNOWN
            and tags.get("hypothesis_type") not in types
        ):
            continue
        specific = bool(cats) + bool(types and tags.get("hypothesis_type") in types)
        out.append((specific, p.confidence, p))
    out.sort(key=lambda x: (-x[0], -x[1], x[2].id))
    return [p for _, _, p in out[: cfg.k_priors]]


def update_priors(s: Session, h: HypothesisORM) -> list[int]:
    """When a hypothesis closes, every prior its decisions cited learns from the verdict:
    supported or passed -> one success; idea failed -> one failure. Implementation
    failures and budget stops say nothing about the claim and change nothing."""
    if h.verdict in POSITIVE:
        field = "a"
    elif h.verdict in NEGATIVE:
        field = "b"
    else:
        return []
    cited: set[int] = set()
    for d in s.scalars(select(DecisionORM).where(DecisionORM.hypothesis_id == h.id)):
        for c in d.cited or []:
            m = PRIOR_ID.match(str(c))
            if m:
                cited.add(int(m.group(1)))
    changed = []
    for pid in sorted(cited):
        p = s.get(PriorORM, pid)
        if p is None:
            continue
        setattr(p, field, getattr(p, field) + 1.0)
        p.history = [
            *(p.history or []),
            {
                "at": utc_now().isoformat(),
                "hypothesis": f"H{h.id}",
                "verdict": h.verdict,
                "confidence": round(p.confidence, 4),
            },
        ]
        changed.append(pid)
    return changed


# ---------- seeding Memory A ----------
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in(quote: str, body: str) -> bool:
    """The quote must appear in the document (whitespace and case aside)."""
    q = _norm(quote)
    return len(q) >= 10 and q in _norm(body)


def add_prior(
    s: Session,
    *,
    claim: str,
    source_type: str,
    source_ref: str,
    source_title: str,
    quote: str,
    tags: dict[str, Any],
    cfg: Memory,
) -> PriorORM | None:
    """Insert a prior unless the same claim from the same source exists."""
    found = s.scalar(
        select(PriorORM).where(PriorORM.claim == claim, PriorORM.source_ref == source_ref)
    )
    if found is not None:
        return None
    a, b = cfg.prior_seed.get(source_type, [1.0, 1.0])
    p = PriorORM(
        claim=claim,
        source_type=source_type,
        source_ref=source_ref,
        source_title=source_title,
        quote=quote,
        tags=tags,
        a=a,
        b=b,
        history=[{"at": utc_now().isoformat(), "seed": source_type, "confidence": a / (a + b)}],
    )
    s.add(p)
    s.flush()
    return p


def read_documents(db_path: str, sources: list[str]) -> list[dict[str, str]]:
    """Official documents stored by the earlier factory (read-only)."""
    path = Path(db_path).expanduser()
    if not path.is_file() or not sources:
        return []
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        marks = ",".join("?" * len(sources))
        rows = con.execute(
            f"select id, source, title, body from ideas where source in ({marks})", sources
        ).fetchall()
    finally:
        con.close()
    return [{"id": str(r[0]), "source": r[1], "title": r[2], "body": r[3] or ""} for r in rows]


def seed_priors(
    s_factory: Callable[[], Any],
    documents: list[dict[str, str]],
    extract: Callable[[dict[str, str]], list[dict[str, Any]]],
    cfg: Memory,
    spent: Callable[[], float],
) -> dict[str, int]:
    """Turn documents into priors. `extract` returns claims for one document (the LLM
    step); the program keeps only claims whose quote really is in the document, and
    stops at the seeding dollar cap."""
    counts = {"documents": 0, "claims": 0, "kept": 0, "quote_not_found": 0}
    start = spent()
    for doc in documents:
        if spent() - start >= cfg.seed_usd_cap:
            counts["stopped_at_cap"] = 1
            break
        with s_factory() as s:
            done = s.scalar(select(PriorORM.id).where(PriorORM.source_ref == f"doc:{doc['id']}"))
        if done is not None:
            continue
        counts["documents"] += 1
        for c in extract(doc):
            counts["claims"] += 1
            if not quote_in(c["quote"], doc["body"]):
                counts["quote_not_found"] += 1
                continue
            with s_factory() as s:
                if add_prior(
                    s,
                    claim=c["claim"],
                    source_type="official",
                    source_ref=f"doc:{doc['id']}",
                    source_title=doc["title"],
                    quote=c["quote"],
                    tags={
                        "categories": [x.lower() for x in c.get("categories") or []],
                        "hypothesis_types": list(c.get("hypothesis_types") or []),
                    },
                    cfg=cfg,
                ):
                    counts["kept"] += 1
    return counts


# ---------- importing the v7 ledger into Memory B ----------
def _v7_verdict(status: str, why: str) -> str:
    # v7 "DONE" meant the regular Sharpe line was reached; none of those cleared the
    # platform's submission check, so they count as supported, not passed.
    if status == "DONE":
        return Verdict.SUPPORTED
    if "cannot be told from noise" in why:
        return Verdict.IDEA_FAILED
    if "no formula passed" in why:
        return Verdict.SUPPORTED
    return Verdict.INVALID


def import_v7(s: Session, db_path: str) -> int:
    """One record per finished v7 idea, marked source='v7' and hypothesis type unknown.
    Its Sharpe is the whole-IS figure of that design (no train/test split then)."""
    path = Path(db_path).expanduser()
    if not path.is_file():
        return 0
    if s.scalar(select(HypothesisORM.id).where(HypothesisORM.source == "v7").limit(1)):
        return 0
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = con.execute("""
            select r.id, r.status, r.title, r.mechanism, r.best_sharpe, r.created_at,
                   coalesce(json_extract(r.extra, '$.why_dead'), ''),
                   coalesce(json_extract(r.extra, '$.group.fields[0]'), ''),
                   c.region, c.delay, c.universe, c.dataset_id, c.dataset_name, c.category,
                   (select count(*) from candidates k where json_extract(k.extra, '$.idea') = r.id)
            from research_ideas r join cells c on c.cell_id = r.cell_id
            where r.status in ('DEAD', 'DONE')
            order by r.created_at
            """).fetchall()
        types = dict(con.execute("select cell_id || '|' || field_id, type from fields").fetchall())
        cells = dict(con.execute("select dataset_id || '|' || region, cell_id from cells"))
    finally:
        con.close()
    n = 0
    for (
        _rid,
        status,
        title,
        mechanism,
        best,
        _created,
        why,
        field_id,
        region,
        delay,
        universe,
        dataset_id,
        dataset_name,
        category,
        sims,
    ) in rows:
        verdict = _v7_verdict(status, why)
        if verdict == Verdict.INVALID:
            continue
        cell = cells.get(f"{dataset_id}|{region}", "")
        s.add(
            HypothesisORM(
                source="v7",
                region=region,
                delay=int(delay),
                universe=universe,
                dataset_id=dataset_id,
                dataset_name=dataset_name or "",
                category=(category or "").lower(),
                field_id=field_id,
                field_type=types.get(f"{cell}|{field_id}", ""),
                claim=f"{title}: {mechanism}"[:2000],
                hypothesis_type="unknown",
                status=HStatus.CLOSED,
                verdict=verdict,
                reason=why[:500],
                lesson=why[:300],
                best_train_sharpe=float(best or 0.0),
                sims_used=int(sims or 0),
                closed_at=utc_now(),
            )
        )
        n += 1
    return n
