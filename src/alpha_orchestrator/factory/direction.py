"""Does a formula buy what its idea says it should buy?

An idea states, per field, whether a HIGHER value should predict higher returns ("high")
or lower ("low"). A formula is a positive signal (positive = long), so a "low" field must
enter it with a negative sign and a "high" field with a positive sign.

`field_signs(expression)` walks the formula and returns, for every field, +1 (formula rises
when the field rises), -1 (formula falls when the field rises) or None (cannot be told, e.g.
the field sits inside abs() or a ratio's denominator). Only a definite sign that contradicts
the idea is reported; None never causes a rejection.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from alpha_orchestrator.factory.util import tokenize_expression

# Operators that rise with their FIRST input (the other inputs are windows, groups or flags).
INCREASING = frozenset(
    {
        "rank", "zscore", "scale", "winsorize", "normalize", "ts_backfill", "group_backfill",
        "ts_mean", "ts_sum", "ts_decay_linear", "ts_rank", "ts_delta", "ts_zscore",
        "ts_av_diff", "ts_max", "ts_min", "ts_median", "ts_scale", "ts_quantile", "sign",
        "log", "sqrt", "group_rank", "group_zscore", "group_neutralize", "group_scale",
        "vec_avg", "vec_sum", "vec_max", "vec_min", "densify", "ts_delay", "hump",
        "ts_decay_exp_window", "ts_returns", "kth_element", "ts_step", "ts_target_tvr_decay",
    }
)  # fmt: skip
DECREASING = frozenset({"reverse"})
# Operators whose result keeps the sign of EACH value input (sum, mean, max, min of inputs).
ALL_INPUTS_INCREASING = frozenset({"add", "max", "min", "group_mean", "sum"})
Sign = int | None


def _merge(a: dict[str, Sign], b: dict[str, Sign]) -> dict[str, Sign]:
    out = dict(a)
    for k, v in b.items():
        out[k] = v if k not in out or out[k] == v else None
    return out


def _flip(d: dict[str, Sign]) -> dict[str, Sign]:
    return {k: (None if v is None else -v) for k, v in d.items()}


def _unknown(d: dict[str, Sign]) -> dict[str, Sign]:
    return {k: None for k in d}


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.t, self.i = tokens, 0

    def peek(self) -> str | None:
        """Return the next token without consuming it."""
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self) -> str:
        """Consume and return the next token."""
        tok = self.t[self.i]
        self.i += 1
        return tok

    # Each parse method returns (signs, number) where number is the literal value when the
    # sub-expression is a plain number (so multiply(-1, x) can be read), else None.
    def expr(self) -> tuple[dict[str, Sign], float | None]:
        """Parse a full expression, including the ternary operator."""
        left = self.additive()
        if self.peek() == "?":
            self.take()
            a, _ = self.expr()
            if self.peek() == ":":
                self.take()
            b, _ = self.expr()
            return _merge(_unknown(left[0]), _merge(a, b)), None
        if self.peek() in {"<", ">", "<=", ">=", "==", "!=", "&&", "||"}:
            self.take()
            right = self.expr()
            return _unknown(_merge(left[0], right[0])), None
        return left

    def additive(self) -> tuple[dict[str, Sign], float | None]:
        """Parse a sum or difference."""
        signs, num = self.term()
        while self.peek() in {"+", "-"}:
            op = self.take()
            r, rnum = self.term()
            if op == "-":
                r = _flip(r)
            signs = _merge(signs, r)
            num = None if num is None or rnum is None else (num + rnum if op == "+" else num - rnum)
        return signs, num

    def term(self) -> tuple[dict[str, Sign], float | None]:
        """Parse a product, quotient, power or modulo."""
        signs, num = self.unary()
        while self.peek() in {"*", "/", "^", "%"}:
            op = self.take()
            r, rnum = self.unary()
            signs = _product(op, signs, num, r, rnum)
            num = None
        return signs, num

    def unary(self) -> tuple[dict[str, Sign], float | None]:
        """Parse an optionally negated operand."""
        if self.peek() == "-":
            self.take()
            s, n = self.unary()
            return _flip(s), (None if n is None else -n)
        if self.peek() == "+":
            self.take()
            return self.unary()
        return self.atom()

    def atom(self) -> tuple[dict[str, Sign], float | None]:
        """Parse a number, field, call or parenthesised expression."""
        tok = self.take()
        if tok == "(":
            inner = self.expr()
            if self.peek() == ")":
                self.take()
            return inner
        if re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", tok):
            return {}, float(tok)
        if tok[0] in "\"'":
            return {}, None
        if re.fullmatch(r"[A-Za-z_]\w*", tok):
            if self.peek() == "(":
                return self.call(tok.lower())
            return {tok: 1}, None
        return {}, None

    def call(self, name: str) -> tuple[dict[str, Sign], float | None]:
        """Parse an operator call's arguments."""
        self.take()  # "("
        args: list[tuple[dict[str, Sign], float | None]] = []
        while self.peek() not in {")", None}:
            # a named argument (std=4, range="...") carries no field
            if (
                self.i + 1 < len(self.t)
                and re.fullmatch(r"[A-Za-z_]\w*", self.t[self.i])
                and self.t[self.i + 1] == "="
            ):
                self.i += 2
                self.expr()
            else:
                args.append(self.expr())
            if self.peek() == ",":
                self.take()
        if self.peek() == ")":
            self.take()
        return _apply(name, args)


def _product(
    op: str, a: dict[str, Sign], an: float | None, b: dict[str, Sign], bn: float | None
) -> dict[str, Sign]:
    """Sign after `a op b`; a plain number on one side sets the sign, two signals are unknown."""
    if op == "*":
        if an is not None and not a:
            return b if an > 0 else _flip(b) if an < 0 else _unknown(b)
        if bn is not None and not b:
            return a if bn > 0 else _flip(a) if bn < 0 else _unknown(a)
    if op == "/" and bn is not None and not b and bn != 0:
        return a if bn > 0 else _flip(a)
    return _unknown(_merge(a, b))


def _apply(
    name: str, args: list[tuple[dict[str, Sign], float | None]]
) -> tuple[dict[str, Sign], float | None]:
    if not args:
        return {}, None
    first = args[0][0]
    if name in INCREASING:
        # inputs after the first are windows/groups; a field there is not "the signal"
        rest: dict[str, Sign] = {}
        for a in args[1:]:
            rest = _merge(rest, _unknown(a[0]))
        return _merge(first, rest), None
    if name in DECREASING:
        return _flip(first), None
    if name in ALL_INPUTS_INCREASING:
        out: dict[str, Sign] = {}
        for a in args:
            out = _merge(out, a[0])
        return out, None
    if name == "subtract" and len(args) >= 2:
        return _merge(first, _flip(args[1][0])), None
    if name == "multiply" and len(args) >= 2:
        out = args[0][0]
        num = args[0][1]
        for nxt in args[1:]:
            out = _product("*", out, num, nxt[0], nxt[1])
            num = None
        return out, None
    if name == "divide" and len(args) == 2:
        return _product("/", args[0][0], args[0][1], args[1][0], args[1][1]), None
    if name in {"trade_when"} and len(args) >= 2:
        cond = _unknown(args[0][0])
        tail = _unknown(args[2][0]) if len(args) > 2 else {}
        return _merge(_merge(cond, tail), args[1][0]), None
    if name == "if_else" and len(args) == 3:
        return _merge(_unknown(args[0][0]), _merge(args[1][0], args[2][0])), None
    out = {}
    for a in args:  # unknown operator or non-monotone (abs, ts_std_dev, ts_corr ...)
        out = _merge(out, _unknown(a[0]))
    return out, None


def field_signs(expression: str) -> dict[str, Sign]:
    """+1 / -1 / None per identifier in the formula (operators and groups are filtered by the
    caller, which only looks up the idea's own fields)."""
    try:
        tokens = tokenize_expression(expression)
        parser = _Parser(tokens)
        signs, _ = parser.expr()
    except (ValueError, IndexError):
        return {}
    return signs


def expected_sign(word: str) -> int | None:
    """'high' -> +1 (higher value should predict higher returns), 'low' -> -1."""
    w = str(word).strip().lower()
    if w in {"high", "higher", "higher is better", "+", "+1", "positive"}:
        return 1
    if w in {"low", "lower", "lower is better", "-", "-1", "negative"}:
        return -1
    return None


def direction_label(expression: str, expected: Mapping[str, int]) -> str:
    """How a formula sits against its idea: 'match', 'opposite', 'mixed' or 'unknown'.

    Recorded, not enforced: in the 2026-10-31 ledger the 10 first formulas that bet against
    their idea's stated direction all had a positive Sharpe (mean 1.07, against 0.20 for the
    478 that matched), so rejecting them would have removed the best ones."""
    got = field_signs(expression)
    known = [(got.get(fid), want) for fid, want in expected.items() if got.get(fid) is not None]
    if not known:
        return "unknown"
    hits = [have == want for have, want in known]
    return "match" if all(hits) else "opposite" if not any(hits) else "mixed"
