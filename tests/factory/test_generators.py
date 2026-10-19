from alpha_orchestrator.factory import generators as g
from alpha_orchestrator.factory.validate import check, operator_count

OPS = {
    "rank",
    "ts_rank",
    "ts_zscore",
    "ts_delta",
    "ts_mean",
    "ts_av_diff",
    "ts_std_dev",
    "ts_backfill",
    "vec_avg",
    "group_rank",
    "group_zscore",
    "group_neutralize",
    "trade_when",
    "multiply",
    "hump",
    "ts_decay_linear",
    "sign",
    "abs",
}
FIELDS = {"eps_fy1": "MATRIX", "eps_fy2": "MATRIX", "news_vec": "VECTOR"}
S = g.base_settings("USA", "TOP3000", 1, "analyst", ["INDUSTRY", "SUBINDUSTRY", "MARKET"])


def test_settings_follow_region_rules() -> None:
    assert S["neutralization"] == "INDUSTRY" and S["maxTrade"] == "OFF"
    asi = g.base_settings("ASI", "MINVOL1M", 1, "analyst", ["COUNTRY", "INDUSTRY"])
    assert asi["neutralization"] == "COUNTRY" and asi["maxTrade"] == "ON"


def test_families_drop_horizon_tokens() -> None:
    fams = g.group_families(
        [
            {"id": "eps_fy1", "coverage": 0.9},
            {"id": "eps_fy2", "coverage": 0.95},
            {"id": "iv_call_30"},
            {"id": "iv_call_60"},
            {"id": "sales"},
        ]
    )
    assert set(fams) == {"eps", "iv_call", "sales"}
    assert fams["eps"][0]["id"] == "eps_fy2"  # best coverage represents the family


def test_template_expands_prepared_fields_windows_and_caps() -> None:
    import random

    prep = {"eps_fy1": "ts_backfill(eps_fy1, 66)", "news_vec": "vec_avg(news_vec)"}
    out = g.expand_template(
        "group_rank(ts_delta({A}, {w}), industry)",
        {"A": ["eps_fy1", "news_vec"]},
        [21, 63],
        prep,
        S,
        limit=10,
        rng=random.Random(1),
    )
    exprs = {p.expression for p in out}
    assert "group_rank(ts_delta(ts_backfill(eps_fy1, 66), 63), industry)" in exprs
    assert "group_rank(ts_delta(vec_avg(news_vec), 21), industry)" in exprs
    assert len(out) == 4 and all(p.stage == "IDEA" for p in out)
    assert all(check(p.expression, operators=OPS, fields=FIELDS).ok for p in out)
    pair = g.expand_template(
        "rank({A} - {B})",
        {"A": ["eps_fy1", "eps_fy2"], "B": ["eps_fy1", "eps_fy2"]},
        [],
        {},
        S,
        limit=10,
        rng=random.Random(1),
    )
    assert {p.expression for p in pair} == {"rank(eps_fy1 - eps_fy2)", "rank(eps_fy2 - eps_fy1)"}
    assert (
        len(g.expand_template("rank({A})", {"A": ["a", "b", "c"]}, [], {}, S, 2, random.Random(0)))
        == 2
    )


def test_validator_catches_platform_errors_locally() -> None:
    assert not check("rank(news_vec)", operators=OPS, fields=FIELDS).ok
    assert not check("rank(nope)", operators=OPS, fields=FIELDS).ok
    assert not check("frobnicate(eps_fy1)", operators=OPS, fields=FIELDS).ok
    assert not check("rank(eps_fy1", operators=OPS, fields=FIELDS).ok
    v = check("group_rank(ts_rank(eps_fy1, 63), industry)", operators=OPS, fields=FIELDS)
    assert v.ok and v.atom and v.fields_used == {"eps_fy1"}
    assert not check("rank(eps_fy1) * rank(volume)", operators=OPS, fields=FIELDS).atom


def test_operator_count_skips_backfill_and_unary_minus() -> None:
    assert operator_count("-rank(ts_backfill(eps_fy1, 63))", OPS) == 1
    assert operator_count("rank(a) + rank(b)", OPS) == 3


def test_flip_and_orders() -> None:
    assert g.flip("rank(x)", S, -1.3).expression == "multiply(-1, rank(x))"
    assert g.flip("multiply(-1, rank(x))", S, -1.3) is None
    assert len(g.second_order("rank(x)", S)) == 9
    assert all(p.expression.startswith("trade_when(") for p in g.third_order("rank(x)", S))


def test_repairs_route_by_failure() -> None:
    roles = {
        p.role
        for p in g.repair_proposals("rank(x)", S, ["HIGH_TURNOVER"], 0.8, ["INDUSTRY"], False)
    }
    assert roles == {"decay", "hump"}
    roles = {
        p.role
        for p in g.repair_proposals(
            "rank(x)", S, ["LOW_SHARPE"], 0.1, ["INDUSTRY", "SUBINDUSTRY"], False
        )
    }
    assert roles == set()  # below the fine-tune gate nothing parametric is allowed
    roles = {
        p.role
        for p in g.repair_proposals(
            "rank(x)", S, ["LOW_SHARPE"], 0.1, ["INDUSTRY", "SUBINDUSTRY"], True
        )
    }
    assert roles == {"neutralization"}


def test_robustness_variants() -> None:
    props = g.robustness_proposals(
        "ts_rank(x, 21)", S, ["INDUSTRY", "SUBINDUSTRY"], ["TOP3000", "TOP1000", "TOP500"]
    )
    roles = [p.role for p in props]
    assert roles[:2] == ["rank", "sign"]
    assert "ts_rank(x, 19)" in [p.expression for p in props]
    assert any(p.settings["universe"] == "TOP1000" for p in props if p.role == "sub_universe")


def test_decay_pairs_with_turnover() -> None:
    assert [g.decay_for_turnover(t) for t in (None, 0.1, 0.3, 0.5, 0.9)] == [0, 0, 4, 8, 16]
