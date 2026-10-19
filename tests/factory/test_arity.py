"""现行设计 第 4 步: positional inputs are checked against the platform's own definitions."""

from alpha_orchestrator.factory.validate import arity_issues, check, signature

DEFS = {
    "winsorize": ("winsorize(x, std=4)", ""),
    "ts_backfill": ("ts_backfill(x,lookback = d, k=1)", ""),
    "add": ("add(x, y, filter = false), x + y", "Adds two or more inputs"),
    "ts_rank": ("ts_rank(x, d, constant = 0)", ""),
    "rank": ("rank(x, rate=2)", ""),
}
SIG = {k: sig for k, v in DEFS.items() if (sig := signature(*v))}


def test_the_formula_that_cancelled_a_batch_is_caught() -> None:
    assert arity_issues("winsorize(a, 0.01, 0.99)", SIG)
    assert not arity_issues("winsorize(a, std=3)", SIG)


def test_common_forms_pass() -> None:
    for e in ["ts_backfill(a, 60)", "add(a, b, c)", "ts_rank(a, 20)", "rank(a)"]:
        assert not arity_issues(e, SIG), e
    assert arity_issues("ts_rank(a)", SIG)
    assert arity_issues("rank(a, 2)", SIG)


def test_check_reports_it() -> None:
    v = check(
        "winsorize(a, 0.01, 0.99)", operators={"winsorize"}, fields={"a": "MATRIX"}, signatures=SIG
    )
    assert not v.ok and "positional" in v.issues[0]


def test_group_input_must_be_a_grouping() -> None:
    from alpha_orchestrator.factory.validate import group_issues

    f = {"a": "MATRIX", "g": "GROUP"}
    assert group_issues("group_rank(a, a)", f)  # the batch-cancelling case of 10/14
    for ok in [
        "group_rank(a, industry)",
        "group_rank(a, g)",
        "group_mean(a, 1, sector)",
        'group_neutralize(a, bucket(rank(a), range="0,1,0.1"))',
    ]:
        assert not group_issues(ok, f), ok
