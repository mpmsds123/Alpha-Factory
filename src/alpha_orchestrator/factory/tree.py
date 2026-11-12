"""Expressions as trees: parse, print canonically, compare, and read a field's sign.

Grammar (lowest precedence first):
    ternary   a ? b : c
    or/and    || &&
    compare   < > <= >= == !=
    additive  + -
    term      * / %
    power     ^
    unary     -x
    atom      number | 'string' | name | name(args) | (expr)
    arg       expr | name = expr          (named input)
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from alpha_orchestrator.factory.util import tokenize_expression
from alpha_orchestrator.factory.vocab import Vocabulary

Sign = int | None


@dataclass(frozen=True)
class Node:
    kind: str  # call, name, num, str, binop, unary, ternary, kwarg
    value: str = ""
    children: tuple[Node, ...] = field(default_factory=tuple)

    def walk(self) -> Iterator[Node]:
        """Yield this node and every descendant, depth first."""
        yield self
        for c in self.children:
            yield from c.walk()


class ParseError(ValueError):
    pass


class _Parser:
    COMPARE = ("<", ">", "<=", ">=", "==", "!=")

    def __init__(self, tokens: list[str]) -> None:
        self.t = tokens
        self.i = 0

    def peek(self, k: int = 0) -> str | None:
        return self.t[self.i + k] if self.i + k < len(self.t) else None

    def take(self, want: str | None = None) -> str:
        tok = self.peek()
        if tok is None or (want is not None and tok != want):
            raise ParseError(f"expected {want or 'more input'}, got {tok}")
        self.i += 1
        return tok

    def parse(self) -> Node:
        node = self.ternary()
        if self.peek() is not None:
            raise ParseError(f"unexpected {self.peek()}")
        return node

    def ternary(self) -> Node:
        cond = self.logic()
        if self.peek() == "?":
            self.take()
            a = self.ternary()
            self.take(":")
            b = self.ternary()
            return Node("ternary", "?", (cond, a, b))
        return cond

    def _binary(self, ops: tuple[str, ...], nxt: str) -> Node:
        left: Node = getattr(self, nxt)()
        while self.peek() in ops:
            op = self.take()
            left = Node("binop", op, (left, getattr(self, nxt)()))
        return left

    def logic(self) -> Node:
        return self._binary(("&&", "||"), "compare")

    def compare(self) -> Node:
        return self._binary(self.COMPARE, "additive")

    def additive(self) -> Node:
        return self._binary(("+", "-"), "term")

    def term(self) -> Node:
        return self._binary(("*", "/", "%"), "power")

    def power(self) -> Node:
        base = self.unary()
        if self.peek() == "^":
            self.take()
            return Node("binop", "^", (base, self.unary()))
        return base

    def unary(self) -> Node:
        if self.peek() == "-":
            self.take()
            inner = self.unary()
            if inner.kind == "num":
                return Node("num", "-" + inner.value)
            return Node("unary", "-", (inner,))
        if self.peek() == "+":
            self.take()
            return self.unary()
        return self.atom()

    def atom(self) -> Node:
        tok = self.take()
        if tok == "(":
            inner = self.ternary()
            self.take(")")
            return inner
        if tok[0] in "\"'":
            return Node("str", tok)
        if tok[0].isdigit() or tok[0] == ".":
            return Node("num", tok)
        if tok[0].isalpha() or tok[0] == "_":
            if self.peek() == "(":
                return self.call(tok)
            return Node("name", tok)
        raise ParseError(f"unexpected {tok}")

    def call(self, name: str) -> Node:
        self.take("(")
        args: list[Node] = []
        while self.peek() != ")":
            if self.peek(1) == "=" and (self.peek() or "")[:1].isalpha():
                key = self.take()
                self.take("=")
                args.append(Node("kwarg", key, (self.ternary(),)))
            else:
                args.append(self.ternary())
            if self.peek() == ",":
                self.take()
            elif self.peek() != ")":
                raise ParseError(f"expected , or ) in {name}(...)")
        self.take(")")
        return Node("call", name.lower(), tuple(args))


def parse(expression: str) -> Node:
    """Parse an expression; raises ParseError (a ValueError) when it is not well formed."""
    try:
        tokens = tokenize_expression(expression)
    except ValueError as exc:
        raise ParseError(str(exc)) from exc
    if not tokens:
        raise ParseError("empty expression")
    return _Parser(tokens).parse()


def text(node: Node) -> str:
    """Canonical text: lower-case operators, single spaces, every infix fully bracketed."""
    if node.kind in ("name", "num", "str"):
        return node.value
    if node.kind == "call":
        return f"{node.value}({', '.join(text(c) for c in node.children)})"
    if node.kind == "kwarg":
        return f"{node.value}={text(node.children[0])}"
    if node.kind == "unary":
        return f"-({text(node.children[0])})"
    if node.kind == "ternary":
        a, b, c = (text(x) for x in node.children)
        return f"({a} ? {b} : {c})"
    a, b = (text(x) for x in node.children)
    return f"({a} {node.value} {b})"


def positional(node: Node) -> list[Node]:
    """A call's inputs that are not passed by name."""
    return [c for c in node.children if c.kind != "kwarg"]


def shape(node: Node, vocab: Vocabulary) -> str:
    """Structure without numbers and without cosmetic outer wrappers: two expressions with
    the same shape differ only in windows and constants."""
    while node.kind == "call" and node.value in vocab.cosmetic_wrappers and len(node.children) == 1:
        node = node.children[0]

    def strip(n: Node) -> Node:
        if n.kind == "num":
            return Node("num", "#")
        return Node(n.kind, n.value, tuple(strip(c) for c in n.children))

    return text(strip(node))


def size(node: Node) -> int:
    """Number of nodes."""
    return sum(1 for _ in node.walk())


def names(node: Node) -> set[str]:
    """Every bare identifier (data fields and grouping identifiers)."""
    return {n.value for n in node.walk() if n.kind == "name"}


def operator_count(node: Node, vocab: Vocabulary) -> int:
    """Calls and infix operators, as the platform counts them (free operators excluded)."""
    return sum(
        1
        for n in node.walk()
        if (n.kind == "call" and n.value not in vocab.free_operators)
        or n.kind in ("binop", "ternary")
    )


def windows(node: Node, vocab: Vocabulary) -> list[int]:
    """Integer look-back inputs of window operators."""
    out = []
    for n in node.walk():
        if n.kind == "call" and n.value.startswith(vocab.window_prefixes or ("\x00",)):
            for arg in positional(n)[1:]:
                if arg.kind == "num" and arg.value.lstrip("-").isdigit():
                    out.append(int(arg.value))
    return out


# ---------- comparing a child with its parent ----------
def _diffs(
    a: Node, b: Node, path: tuple[int, ...] = ()
) -> list[tuple[tuple[int, ...], Node, Node]]:
    if a == b:
        return []
    same_head = a.kind == b.kind and len(a.children) == len(b.children) and a.children
    if same_head and a.value == b.value:
        out = []
        for i, (x, y) in enumerate(zip(a.children, b.children, strict=True)):
            out.extend(_diffs(x, y, (*path, i)))
        return out
    if (
        same_head
        and a.kind == "call"
        and all(x == y for x, y in zip(a.children, b.children, strict=True))
    ):
        return [(path, a, b)]  # only the operator name differs
    return [(path, a, b)]


def change_type(parent: Node, child: Node, fields: set[str]) -> str:
    """What kind of change turns parent into child: same, window, constant, field,
    operator or rewrite (anything bigger than one of those)."""
    found = _diffs(parent, child)
    if not found:
        return "same"
    kinds = set()
    for _, a, b in found:
        if a.kind == b.kind == "num":
            kinds.add("window")
        elif a.kind == b.kind == "name" and a.value in fields and b.value in fields:
            kinds.add("field")
        elif (
            a.kind == b.kind == "call"
            and a.value != b.value
            and a.children == b.children
            and len(found) == 1
        ):
            kinds.add("operator")
        else:
            return "rewrite"
    if kinds == {"window"}:
        return "window"  # several windows moved together still count as one change
    if len(found) == 1:
        return kinds.pop()
    return "rewrite"


# ---------- sign of a field in the expression ----------
def _merge(a: Mapping[str, Sign], b: Mapping[str, Sign]) -> dict[str, Sign]:
    out = dict(a)
    for k, v in b.items():
        out[k] = v if k not in out else (out[k] if out[k] == v else None)
    return out


def _flip(d: Mapping[str, Sign]) -> dict[str, Sign]:
    return {k: (-v if v is not None else None) for k, v in d.items()}


def _unknown(d: Mapping[str, Sign]) -> dict[str, Sign]:
    return {k: None for k in d}


def _const(n: Node) -> float | None:
    if n.kind == "num":
        try:
            return float(n.value)
        except ValueError:
            return None
    return None


def _arith(op: str, a: Node, b: Node, vocab: Vocabulary) -> dict[str, Sign]:
    sa, sb = signs(a, vocab), signs(b, vocab)
    if op == "+":
        return _merge(sa, sb)
    if op == "-":
        return _merge(sa, _flip(sb))
    ca, cb = _const(a), _const(b)
    if op == "*":
        if ca is not None:
            return sb if ca > 0 else _flip(sb) if ca < 0 else _unknown(sb)
        if cb is not None:
            return sa if cb > 0 else _flip(sa) if cb < 0 else _unknown(sa)
    if op == "/" and cb:
        return sa if cb > 0 else _flip(sa)
    return _unknown(_merge(sa, sb))


def signs(node: Node, vocab: Vocabulary) -> dict[str, Sign]:
    """For every identifier: +1 when the expression rises with it, -1 when it falls,
    None when that cannot be told (inside abs, a denominator, a condition, ...)."""
    if node.kind == "name":
        return {node.value: 1}
    if node.kind in ("num", "str"):
        return {}
    if node.kind == "unary":
        return _flip(signs(node.children[0], vocab))
    if node.kind == "ternary":
        cond, a, b = node.children
        return _merge(_unknown(signs(cond, vocab)), _merge(signs(a, vocab), signs(b, vocab)))
    if node.kind == "binop":
        if node.value in ("+", "-", "*", "/"):
            return _arith(node.value, node.children[0], node.children[1], vocab)
        out: dict[str, Sign] = {}
        for c in node.children:
            out = _merge(out, _unknown(signs(c, vocab)))
        return out
    if node.kind == "kwarg":
        return _unknown(signs(node.children[0], vocab))
    args = positional(node)
    rest: dict[str, Sign] = {}
    for c in node.children:
        rest = _merge(rest, _unknown(signs(c, vocab)))
    if not args:
        return rest
    name = node.value
    if name in vocab.arithmetic and len(args) == 2:
        return _arith(vocab.arithmetic[name], args[0], args[1], vocab)
    if name in vocab.increasing:
        return {**rest, **signs(args[0], vocab)}
    if name in vocab.decreasing:
        return {**rest, **_flip(signs(args[0], vocab))}
    if name in vocab.all_increasing:
        out = {}
        for a in args:
            out = _merge(out, signs(a, vocab))
        return out
    if name in vocab.conditional and vocab.conditional[name] < len(args):
        return {**rest, **signs(args[vocab.conditional[name]], vocab)}
    return rest


def alignment(node: Node, vocab: Vocabulary, field_id: str, direction: int) -> str:
    """How the expression bets on the hypothesis field: match, opposite or unknown."""
    got = signs(node, vocab).get(field_id)
    if got is None or direction == 0:
        return "unknown"
    return "match" if got == direction else "opposite"
