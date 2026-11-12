from pathlib import Path

import pytest

from alpha_orchestrator.factory import tree
from alpha_orchestrator.factory.validate import check, signature
from alpha_orchestrator.factory.vocab import Vocabulary
from tests.factory.fakes import OPERATORS, VOCAB

SIGS = {k: s for k, d in OPERATORS.items() if (s := signature(d)) is not None}


def _check(expr: str, fields: dict[str, str] | None = None) -> list[str]:
    return check(
        expr,
        operators=set(OPERATORS),
        signatures=SIGS,
        fields=fields or {"f1": "MATRIX", "f2": "MATRIX", "v1": "VECTOR"},
        vocab=VOCAB,
        windows=(5, 21, 63),
        max_operators=4,
        max_fields=2,
    ).issues


def test_parse_and_canonical_text() -> None:
    node = tree.parse("ORDER( w_avg(f1,5) - f2 * 2 )")
    assert tree.text(node) == "order((w_avg(f1, 5) - (f2 * 2)))"
    assert tree.text(tree.parse("a > 1 ? -b : c")) == "((a > 1) ? -(b) : c)"
    assert tree.text(tree.parse("bins(f1, range='0.1,1,0.1')")) == "bins(f1, range='0.1,1,0.1')"
    assert tree.parse("-3").value == "-3"
    for bad in ("order(", "f1 +", "a b", "order(f1;)", "", "f1 ? f2"):
        with pytest.raises(ValueError):
            tree.parse(bad)


def test_shape_ignores_numbers_and_cosmetic_wrapper() -> None:
    a = tree.shape(tree.parse("order(w_avg(f1, 5))"), VOCAB)
    b = tree.shape(tree.parse("w_avg(f1, 21)"), VOCAB)
    assert a == b == "w_avg(f1, #)"


def test_counts_and_windows() -> None:
    node = tree.parse("w_fill(w_avg(f1, 5), 21) + 1")
    assert tree.operator_count(node, VOCAB) == 2  # w_avg and +; w_fill is free
    assert tree.windows(node, VOCAB) == [21, 5]
    assert tree.names(node) == {"f1"}
    assert tree.size(node) == 7


@pytest.mark.parametrize(
    ("parent", "child", "kind"),
    [
        ("w_avg(f1, 5)", "w_avg(f1, 5)", "same"),
        ("w_avg(f1, 5)", "w_avg(f1, 21)", "window"),
        ("w_avg(w_chg(f1, 5), 5)", "w_avg(w_chg(f1, 21), 21)", "window"),
        ("w_avg(f1, 5)", "w_avg(f2, 5)", "field"),
        ("w_avg(f1, 5)", "w_chg(f1, 5)", "operator"),
        ("w_avg(f1, 5)", "order(w_avg(f1, 5))", "rewrite"),
        ("w_avg(f1, 5)", "w_chg(f2, 5)", "rewrite"),
        ("w_avg(f1, 5)", "w_avg(f2, 21)", "rewrite"),
    ],
)
def test_change_type(parent: str, child: str, kind: str) -> None:
    assert tree.change_type(tree.parse(parent), tree.parse(child), {"f1", "f2"}) == kind


@pytest.mark.parametrize(
    ("expr", "sign"),
    [
        ("order(f1)", 1),
        ("flipit(f1)", -1),
        ("-w_avg(f1, 5)", -1),
        ("times(-1, f1)", -1),
        ("f1 / 2", 1),
        ("2 - f1", -1),
        ("minus(f2, f1)", -1),
        ("plus(f1, f2)", 1),
        ("when(f2 > 0, f1, -1)", 1),
        ("absval(f1)", None),
        ("f1 * f2", None),
        ("f2 > f1 ? 1 : 0", None),
        ("w_avg(f2, f1)", None),
    ],
)
def test_signs(expr: str, sign: int | None) -> None:
    assert tree.signs(tree.parse(expr), VOCAB).get("f1") == sign


def test_alignment() -> None:
    node = tree.parse("flipit(f1)")
    assert tree.alignment(node, VOCAB, "f1", -1) == "match"
    assert tree.alignment(node, VOCAB, "f1", 1) == "opposite"
    assert tree.alignment(node, VOCAB, "f1", 0) == "unknown"
    assert tree.alignment(tree.parse("absval(f1)"), VOCAB, "f1", 1) == "unknown"


def test_valid_expression() -> None:
    v = check(
        "gnorm(order(w_avg(f1, 21)), zone)",
        operators=set(OPERATORS),
        signatures=SIGS,
        fields={"f1": "MATRIX"},
        vocab=VOCAB,
        windows=(21,),
        max_operators=8,
        max_fields=3,
    )
    assert v.ok, v.issues
    assert (v.fields_used, v.operators) == (["f1"], 3)
    assert v.shape == "gnorm(order(w_avg(f1, #)), zone)"


@pytest.mark.parametrize(
    ("expr", "needle"),
    [
        ("nosuch(f1)", "unknown operator"),
        ("w_avg(f1)", "positional"),
        ("gnorm(f1, f2)", "grouping input"),
        ("order(f9)", "not allowed"),
        ("order(v1)", "needs a reducer"),
        ("w_avg(f1, 17)", "window"),
        ("order(order(order(order(order(f1)))))", "operators"),
        ("plus(f1, f2, red_avg(v1))", "data fields"),
        ("order(f1", "does not parse"),
    ],
)
def test_invalid_expressions(expr: str, needle: str) -> None:
    assert any(needle in i for i in _check(expr)), _check(expr)


def test_group_maker_and_reducer_accepted() -> None:
    assert _check("gnorm(f1, bins(f2))") == []
    assert _check("order(red_avg(v1))") == []


def test_signature_reading() -> None:
    assert signature("w_avg(x, d)") == (2, 2)
    assert signature("plus(x, y, ..)") == (2, -1)
    assert signature("bins(x, range='0.1,1,0.1')") == (1, 1)
    assert signature("pick(x, lookback = d)") == (1, 2)
    assert signature("many(x)", "takes two or more inputs") == (1, -1)
    assert signature("not a call") is None


def test_vocabulary_loads_from_private_folder(tmp_path: Path) -> None:
    assert Vocabulary.load(tmp_path) == Vocabulary()
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "platform.yaml").write_text(
        "grouping_fields: [Zone]\ngroup_input: {gnorm: 1}\nlanguage_notes_files: [notes.txt]\n"
    )
    v = Vocabulary.load(tmp_path)
    assert v.grouping_fields == frozenset({"zone"}) and v.group_input == {"gnorm": 1}
    assert v.language_notes == "hello"
