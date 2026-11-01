"""Does a formula buy what its idea says? And is an idea specific enough to be proven wrong?"""

import pytest

from alpha_orchestrator.factory import engine as eng
from alpha_orchestrator.factory.direction import direction_label, expected_sign, field_signs


@pytest.mark.parametrize(
    ("expression", "x"),
    [
        ("rank(x)", 1),
        ("-rank(x)", -1),
        ("reverse(rank(x))", -1),
        ("multiply(-1, rank(x))", -1),
        ("rank(x) * -1", -1),
        ("rank(x) / -2", -1),
        ("multiply(-1, reverse(rank(x)))", 1),  # two flips
        ("rank(ts_delta(ts_backfill(x, 63), 21))", 1),
        ("-group_rank(ts_backfill(x, 63), industry)", -1),
        ("winsorize(-x, std=4)", -1),
        ("trade_when(volume > adv20, -rank(x), -1)", -1),
        ("rank(abs(x))", None),  # cannot be told
        ("rank(x / cap)", None),  # ratio of two signals
        ("rank(ts_std_dev(x, 21))", None),
    ],
)
def test_sign_of_a_field_in_a_formula(expression: str, x: int | None) -> None:
    assert field_signs(expression)["x"] == x


def test_subtract_flips_the_second_field() -> None:
    assert field_signs("rank(subtract(a, b))") == {"a": 1, "b": -1}
    assert field_signs("rank(a - b)") == {"a": 1, "b": -1}


def test_a_field_used_with_both_signs_is_unknown() -> None:
    assert field_signs("rank(x) - rank(x)")["x"] is None


def test_bad_text_gives_no_signs_instead_of_raising() -> None:
    assert field_signs("rank(x $ 3)") == {}
    assert field_signs("") == {}


def test_label_match_opposite_mixed_unknown() -> None:
    want = {"a": 1, "b": -1}
    assert direction_label("rank(a - b)", want) == "match"
    assert direction_label("-rank(a - b)", want) == "opposite"
    assert direction_label("rank(a + b)", want) == "mixed"
    assert direction_label("rank(abs(a))", want) == "unknown"
    assert direction_label("rank(c)", want) == "unknown"


def test_words_for_direction() -> None:
    assert expected_sign("high") == 1 and expected_sign("Higher is better") == 1
    assert expected_sign("low") == -1 and expected_sign("lower is better") == -1
    assert expected_sign("maybe") is None


GOOD = {
    "name": "n",
    "meaning": "m",
    "why": "w",
    "horizon_days": 63,
    "falsifier": "top-minus-bottom return is not positive over 63 days",
    "field_signs": {"a": "high", "b": "low"},
}


def test_an_idea_card_keeps_window_falsifier_and_signs() -> None:
    card = eng._idea_card(GOOD, ["a", "b"])
    assert card is not None
    assert card["horizon_days"] == 63
    assert card["field_signs"] == {"a": "high", "b": "low"}
    assert eng._group_signs(card) == {"a": 1, "b": -1}


@pytest.mark.parametrize(
    "change",
    [
        {"horizon_days": 30},  # not one of the allowed windows
        {"horizon_days": None},
        {"horizon_days": "soon"},
        {"falsifier": ""},
        {"falsifier": "wrong"},  # too short to say anything
        {"field_signs": {"a": "high"}},  # b missing
        {"field_signs": {"a": "high", "b": "sideways"}},
        {"field_signs": None},
    ],
)
def test_an_idea_that_cannot_be_proven_wrong_is_dropped(change: dict) -> None:  # type: ignore[type-arg]
    assert eng._idea_card({**GOOD, **change}, ["a", "b"]) is None


def test_first_gate_is_above_the_measured_noise_line() -> None:
    assert eng.FIRST_GATE == 1.0


def test_groups_saved_before_the_window_existed_still_start_an_idea() -> None:
    legacy = {"name": "n", "meaning": "m", "direction": "higher is better", "why": "w"}
    assert eng._group_signs(legacy) == {}
    assert direction_label("rank(a)", eng._group_signs(legacy)) == "unknown"
