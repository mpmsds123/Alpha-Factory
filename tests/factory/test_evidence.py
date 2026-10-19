"""第 2、6 问: evidence and duplicate checks are mechanical and catch the ledger's real cases."""

from alpha_orchestrator.factory import evidence as ev
from alpha_orchestrator.factory.study import scanner_worth

DOC = {
    "id": "2523",
    "title": "Using Abnormal Analyst Coverage",
    "text": "We show that abnormal coverage contains valuable information about individual "
    "firm ex-ante crash risk (proxied by implied volatility smirk).",
}
BASE = {
    "title": "t",
    "mechanism": "m",
    "long": "l",
    "short": "s",
    "horizon_days": 21,
    "query": "analyst coverage growth",
}


def test_a_card_must_be_on_the_list() -> None:
    assert ev.check_idea({**BASE, "evidence": {"type": "card", "id": "q3"}}, {})[0]
    ok, why = ev.check_idea({**BASE, "evidence": {"type": "card", "id": "Q9"}}, {})
    assert not ok and "not on the card list" in why


def test_a_quote_must_be_found_word_for_word() -> None:
    docs = {"2523": DOC}
    good = {
        "type": "paper",
        "id": "2523",
        "quote": "abnormal coverage contains  valuable information",
    }
    assert ev.check_idea({**BASE, "evidence": good}, docs)[0]
    bad = {**good, "quote": "abnormal coverage predicts higher returns for the stock"}
    ok, why = ev.check_idea({**BASE, "evidence": bad}, docs)
    assert not ok and "word for word" in why
    ok, why = ev.check_idea({**BASE, "evidence": {**good, "id": "1"}}, docs)
    assert not ok and "not among the documents" in why


def test_missing_fields_are_named() -> None:
    ok, why = ev.check_idea({"title": "t"}, {})
    assert not ok and "mechanism" in why and "evidence" in why


def test_the_ledgers_two_repeats_are_caught_and_others_are_not() -> None:
    card = {"query": "network momentum diffusion recent", "evidence": {"type": "card", "id": "Q2"}}
    earlier = [
        {
            "title": "Persistent Network Momentum Trend",
            "query": "network momentum trend persistent recent diffusion",
            "evidence_key": "card:Q2",
        },
        {"title": "Other", "query": "cash flow yield", "evidence_key": "card:Q2"},
    ]
    assert ev.duplicate_of(card, earlier) == "Persistent Network Momentum Trend"
    other_evidence = [{**earlier[0], "evidence_key": "card:Q3"}]
    assert ev.duplicate_of(card, other_evidence) is None


def test_scanner_worth_keeps_two_percent_and_up(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "fields_USA_TOP3000_1_good.json").write_text("[]")
    (tmp_path / "fields_USA_TOP3000_1_bad.json").write_text("[]")
    head = "expression,sharpe,fitness,turnover\n"
    good = head + "a,1.7,1.2,0.1\n" + "b,0.2,0.1,0.1\n" * 9
    bad = head + "a,1.2,1.0,0.1\n" + "b,0.2,0.1,0.1\n" * 99
    (tmp_path / "USA_good_v6_1阶.csv").write_text(good, encoding="utf-8")
    (tmp_path / "USA_bad_1阶.csv").write_text(bad, encoding="utf-8")
    rows = scanner_worth(str(tmp_path))
    assert [r["dataset"] for r in rows] == ["good"]
    assert rows[0]["regular"] == 1 and rows[0]["universe"] == "TOP3000"
