import sqlite3
from pathlib import Path

from alpha_orchestrator.factory import memory
from alpha_orchestrator.factory.config import Memory
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.models import (
    DecisionORM,
    HStatus,
    HypothesisORM,
    PriorORM,
    Verdict,
)

CFG = Memory()


def _closed(s, **kw) -> HypothesisORM:  # type: ignore[no-untyped-def]
    base = {
        "region": "R1",
        "dataset_id": "d",
        "category": "cat_a",
        "field_type": "MATRIX",
        "hypothesis_type": "reversal",
        "status": HStatus.CLOSED,
        "verdict": Verdict.SUPPORTED,
    }
    h = HypothesisORM(**{**base, **kw})
    s.add(h)
    s.flush()
    return h


def test_similarity_ignores_unknown_tags() -> None:
    a = {"category": "x", "region": "R", "field_type": "M", "hypothesis_type": "unknown"}
    assert memory.similarity(a, dict(a), CFG.weights) == 5
    assert memory.similarity({"category": ""}, {"category": ""}, CFG.weights) == 0


def test_retrieval_ranks_by_tags_and_keeps_a_failure_in_view() -> None:
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        ids = [_closed(s).id for _ in range(6)]  # all supported, full match
        fail = _closed(s, verdict=Verdict.IDEA_FAILED, region="R2").id  # score 6
        _closed(s, category="other", hypothesis_type="level")  # score 2: below min
        _closed(s, status=HStatus.OPEN)
        tags = {"category": "cat_a", "region": "R1", "field_type": "MATRIX"}
        tags["hypothesis_type"] = "reversal"
        got = [h.id for h in memory.retrieve_records(s, tags, CFG, exclude={ids[0]})]
    assert len(got) == 5 and fail in got and ids[0] not in got
    assert got[:4] == sorted(ids[1:], reverse=True)[:4]


def test_prior_retrieval_and_update() -> None:
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        general = memory.add_prior(
            s,
            claim="c1",
            source_type="official",
            source_ref="doc:1",
            source_title="t",
            quote="q" * 10,
            tags={},
            cfg=CFG,
        )
        specific = memory.add_prior(
            s,
            claim="c2",
            source_type="forum",
            source_ref="doc:2",
            source_title="t",
            quote="q" * 10,
            tags={"categories": ["cat_a"]},
            cfg=CFG,
        )
        memory.add_prior(
            s,
            claim="c3",
            source_type="own",
            source_ref="x",
            source_title="t",
            quote="",
            tags={"hypothesis_types": ["level"]},
            cfg=CFG,
        )
        assert (
            memory.add_prior(
                s,
                claim="c1",
                source_type="official",
                source_ref="doc:1",
                source_title="t",
                quote="",
                tags={},
                cfg=CFG,
            )
            is None
        )
        assert general is not None and specific is not None
        got = memory.retrieve_priors(s, {"category": "cat_a", "hypothesis_type": "reversal"}, CFG)
        assert [p.id for p in got] == [specific.id, general.id]
        h = _closed(s, verdict=Verdict.IDEA_FAILED)
        s.add(DecisionORM(hypothesis_id=h.id, step="x", cited=[f"P{general.id}", "H3", "P999"]))
        s.flush()
        assert memory.update_priors(s, h) == [general.id]
        assert (general.a, general.b) == (3.0, 3.0)
        assert general.history[-1]["verdict"] == Verdict.IDEA_FAILED
        h.verdict = Verdict.BUDGET
        assert memory.update_priors(s, h) == []
        h.verdict = Verdict.PASSED
        memory.update_priors(s, h)
        assert general.a == 4.0


def test_seed_keeps_only_quoted_claims_and_stops_at_cap() -> None:
    ledger = Ledger("sqlite://")
    docs = [
        {"id": "1", "title": "Doc one", "body": "Prices  drift after   news events."},
        {"id": "2", "title": "Doc two", "body": "other"},
    ]
    claims = [
        {"claim": "news drift", "quote": "prices drift after news", "categories": ["News"]},
        {"claim": "made up", "quote": "not in the document at all"},
    ]
    spent = iter([0.0, 0.0, 9.0])
    counts = memory.seed_priors(ledger.session, docs, lambda d: claims, CFG, lambda: next(spent))
    assert counts == {
        "documents": 1,
        "claims": 2,
        "kept": 1,
        "quote_not_found": 1,
        "stopped_at_cap": 1,
    }
    with ledger.session() as s:
        p = s.query(PriorORM).one()
        assert (p.source_ref, p.tags["categories"], p.confidence) == ("doc:1", ["news"], 0.6)


def test_import_v7_maps_verdicts(tmp_path: Path) -> None:
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
        create table research_ideas (id text, cell_id text, status text, title text,
            mechanism text, best_sharpe real, created_at text, extra text);
        create table cells (cell_id text, region text, delay int, universe text,
            dataset_id text, dataset_name text, category text);
        create table fields (cell_id text, field_id text, type text);
        create table candidates (id text, extra text);
        create table ideas (id int, source text, title text, body text);
        insert into cells values ('c1', 'R1', 1, 'U', 'd1', 'D one', 'Cat');
        insert into fields values ('c1', 'f1', 'MATRIX');
        insert into research_ideas values
          ('i1', 'c1', 'DONE', 'T', 'M', 1.7, '1', '{"group": {"fields": ["f1"]}}'),
          ('i2', 'c1', 'DEAD', 'T', 'M', 0.5, '2', '{"why_dead": "cannot be told from noise"}'),
          ('i3', 'c1', 'DEAD', 'T', 'M', 1.3, '3', '{"why_dead": "no formula passed the line"}'),
          ('i4', 'c1', 'DEAD', 'T', 'M', 0.0, '4', '{"why_dead": "no valid formula"}'),
          ('i5', 'c1', 'RETIRED', 'T', 'M', 0.0, '5', '{}');
        insert into candidates values ('k1', '{"idea": "i1"}'), ('k2', '{"idea": "i1"}');
        insert into ideas values (1, 'official', 'A', 'body'), (2, 'forum', 'B', 'x');
        """)
    con.commit()
    con.close()
    ledger = Ledger("sqlite://")
    with ledger.session() as s:
        assert memory.import_v7(s, str(db)) == 3
    with ledger.session() as s:
        assert memory.import_v7(s, str(db)) == 0  # once only
        rows = s.query(HypothesisORM).order_by(HypothesisORM.id).all()
        assert [r.verdict for r in rows] == ["supported", "idea_failed", "supported"]
        assert (rows[0].field_type, rows[0].sims_used, rows[0].category) == ("MATRIX", 2, "cat")
    assert memory.import_v7(ledger.session().__enter__(), str(tmp_path / "none.db")) == 0
    docs = memory.read_documents(str(db), ["official"])
    assert [d["title"] for d in docs] == ["A"]
    assert memory.read_documents(str(db), []) == []
