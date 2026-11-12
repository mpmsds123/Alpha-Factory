"""A made-up expression language for tests, so no platform vocabulary appears in git."""

from alpha_orchestrator.factory.vocab import Vocabulary

VOCAB = Vocabulary(
    grouping_fields=frozenset({"zone"}),
    common_fields=frozenset({"px"}),
    free_operators=frozenset({"w_fill"}),
    vector_reducer_prefix="red_",
    group_input={"gnorm": 1},
    group_makers=frozenset({"bins"}),
    window_prefixes=("w_",),
    increasing=frozenset({"order", "w_avg", "w_chg", "gnorm", "red_avg", "w_fill", "bins"}),
    decreasing=frozenset({"flipit"}),
    all_increasing=frozenset({"plus"}),
    arithmetic={"times": "*", "minus": "-"},
    conditional={"when": 1},
    cosmetic_wrappers=frozenset({"order"}),
    negate_template="times(-1, {x})",
    language_notes="order(x) ranks; w_avg(x, d) averages over d days.",
)

# name -> definition, as the platform's operator list gives them
OPERATORS = {
    "order": "order(x)",
    "w_avg": "w_avg(x, d)",
    "w_chg": "w_chg(x, d)",
    "w_vol": "w_vol(x, d)",
    "w_fill": "w_fill(x, d)",
    "gnorm": "gnorm(x, group)",
    "bins": "bins(x, range='0.1,1,0.1')",
    "red_avg": "red_avg(x)",
    "flipit": "flipit(x)",
    "plus": "plus(x, y, ..)",
    "times": "times(x, y)",
    "minus": "minus(x, y)",
    "when": "when(cond, x, other)",
    "absval": "absval(x)",
}
