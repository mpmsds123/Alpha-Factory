"""Local pre-checks. One bad expression fails a whole multi-simulation (官方 API 课),
so nothing reaches the platform before passing here."""

import re
from collections.abc import Mapping, Set
from dataclasses import dataclass, field

from alpha_orchestrator.domain.identity import tokenize_expression
from alpha_orchestrator.gates.structural import GROUP_CLASSIFICATION_IDENTIFIERS

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
FREE_OPERATORS = frozenset({"ts_backfill", "group_backfill"})  # 规则: not counted
MAX_OPERATORS = 8
KEYWORDS = frozenset({"true", "false", "nan", "inf"})
# Price-volume fields exist in every region (dataset pv1). Using them breaks ATOM purity.
COMMON_FIELDS = frozenset(
    {"close", "open", "high", "low", "volume", "vwap", "returns", "adv20", "cap", "sharesout"}
)
VECTOR_REDUCERS_PREFIX = "vec_"


@dataclass
class Verdict:
    ok: bool
    issues: list[str] = field(default_factory=list)
    operator_count: int = 0
    fields_used: set[str] = field(default_factory=set)
    atom: bool = True


def operator_count(expression: str, operators: Set[str]) -> int:
    tokens = tokenize_expression(expression)
    count = 0
    for i, tok in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if nxt == "(" and tok.lower() in operators and tok.lower() not in FREE_OPERATORS:
            count += 1
        elif tok in {"+", "-", "*", "/", "^", "<", ">", "<=", ">=", "==", "!=", "?", "&&", "||"}:
            prev = tokens[i - 1] if i else None
            if tok == "-" and (prev is None or prev in {"(", ",", "?", ":"}):
                continue  # unary minus on a literal or expression start
            count += 1
    return count


def check(
    expression: str,
    *,
    operators: Set[str],
    fields: Mapping[str, str],
    extra_fields: Set[str] = COMMON_FIELDS,
) -> Verdict:
    """`fields` maps this dataset's field ids to MATRIX/VECTOR/GROUP."""
    verdict = Verdict(ok=True)
    try:
        tokens = tokenize_expression(expression)
    except ValueError as exc:
        return Verdict(ok=False, issues=[str(exc)])
    depth = 0
    for tok in tokens:
        depth += tok == "("
        depth -= tok == ")"
        if depth < 0:
            break
    if depth != 0:
        verdict.issues.append("unbalanced parentheses")

    ops = {o.lower() for o in operators}
    stack: list[str] = []
    for i, tok in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if tok == "(":
            prev = tokens[i - 1] if i else ""
            stack.append(prev.lower() if IDENT.fullmatch(prev) else "")
            continue
        if tok == ")":
            if stack:
                stack.pop()
            continue
        if not IDENT.fullmatch(tok):
            continue
        low = tok.lower()
        if nxt == "(":
            if low not in ops:
                verdict.issues.append(f"unknown operator {tok}")
            continue
        if nxt == "=":  # named argument, e.g. range="0.1,1,0.1"
            continue
        if low in KEYWORDS or low in GROUP_CLASSIFICATION_IDENTIFIERS:
            continue
        if tok in fields:
            verdict.fields_used.add(tok)
            if fields[tok] == "VECTOR" and not any(
                f.startswith(VECTOR_REDUCERS_PREFIX) for f in stack
            ):
                verdict.issues.append(f"vector field {tok} needs a vec_* reducer")
            continue
        if low in extra_fields:
            verdict.fields_used.add(low)
            verdict.atom = False
            continue
        verdict.issues.append(f"unknown field {tok}")

    verdict.operator_count = operator_count(expression, ops)
    if verdict.operator_count > MAX_OPERATORS:
        verdict.issues.append(f"{verdict.operator_count} operators > {MAX_OPERATORS}")
    verdict.ok = not verdict.issues
    return verdict
