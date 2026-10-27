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


def signature(definition: str, description: str = "") -> tuple[int, int] | None:
    """(fewest, most positional inputs; -1 = no limit) from the platform's operator definition,
    e.g. "winsorize(x, std=4)" -> (1, 1): optional inputs must be passed by name. None
    when the definition is not of the form name(...)."""
    head = definition.strip().splitlines()[0] if definition.strip() else ""
    m = re.match(r"\s*[A-Za-z_]\w*\s*\(", head)
    if not m:
        return None
    body = _inside(head, m.end() - 1)
    if body is None:
        return None
    required, variadic, optional_positional = 0, False, 0
    for part in _split_args(body):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            variadic = True
            part = part.replace(".", "").strip()
            if part:
                required += 1
            continue
        named = re.match(r"^[A-Za-z_]\w*\s*=(?!=)\s*(.*)$", part)
        if named:
            if re.fullmatch(r"[a-z]", named.group(1).strip()):
                optional_positional += 1  # "lookback = d": a value slot, may be positional
            continue  # optional, by name
        required += 1
    low = description.lower()
    if "two or more" in low or "at least 2" in low:
        variadic = True
    return required, -1 if variadic else required + optional_positional


def _inside(text: str, open_at: int) -> str | None:
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_at + 1 : i]
    return None


def _split_args(body: str) -> list[str]:
    parts, depth, quote, cur = [], 0, "", []
    for ch in body:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'“”":
            quote = "”" if ch == "“" else ch
            cur.append(ch)
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur))
    return parts


def arity_issues(expression: str, signatures: Mapping[str, tuple[int, int]]) -> list[str]:
    """Wrong number of positional inputs. 2026-10-14: winsorize(x, 0.01, 0.99) cancelled a
    whole batch of 10 ("Invalid number of inputs : 3, should be exactly 1")."""
    issues = []
    for m in re.finditer(r"([A-Za-z_]\w*)\s*\(", expression):
        sig = signatures.get(m.group(1).lower())
        if sig is None:
            continue
        body = _inside(expression, m.end() - 1)
        if body is None:
            continue
        args = [a for a in _split_args(body) if a.strip()]
        positional = [a for a in args if not re.match(r"^\s*[A-Za-z_]\w*\s*=(?!=)", a)]
        low, high = sig
        if len(positional) < low or (high >= 0 and len(positional) > high):
            want = f"{low}" if high == low else f"{low}-{high}" if high > 0 else f"{low}+"
            issues.append(
                f"{m.group(1)} takes {want} positional input(s), got {len(positional)}; "
                "pass optional inputs by name (e.g. winsorize(x, std=4))"
            )
    return issues


# 2026-10-14: group_rank(x, <an ordinary field>) cancelled a batch ("Incompatible unit
# ... expected Unit[Group:1]"). The group input sits at this positional index.
GROUP_INPUT = {
    "group_rank": 1,
    "group_zscore": 1,
    "group_neutralize": 1,
    "group_scale": 1,
    "group_max": 1,
    "group_min": 1,
    "group_backfill": 1,
    "group_mean": 2,
}
GROUP_MAKERS = ("bucket(", "densify(", "group_cartesian_product(")


def group_issues(expression: str, fields: Mapping[str, str]) -> list[str]:
    issues = []
    for m in re.finditer(r"([A-Za-z_]\w*)\s*\(", expression):
        at = GROUP_INPUT.get(m.group(1).lower())
        if at is None:
            continue
        body = _inside(expression, m.end() - 1)
        if body is None:
            continue
        args = [a.strip() for a in _split_args(body) if a.strip()]
        positional = [a for a in args if not re.match(r"^[A-Za-z_]\w*\s*=(?!=)", a)]
        if len(positional) <= at:
            continue
        g = positional[at]
        if (
            g.lower() in GROUP_CLASSIFICATION_IDENTIFIERS
            or fields.get(g) == "GROUP"
            or g.lower().startswith(GROUP_MAKERS)
        ):
            continue
        issues.append(
            f"{m.group(1)} needs a grouping input (industry, sector, subindustry, market, "
            f"a GROUP field or bucket(...)), got {g[:60]}"
        )
    return issues


def check(
    expression: str,
    *,
    operators: Set[str],
    fields: Mapping[str, str],
    extra_fields: Set[str] = COMMON_FIELDS,
    signatures: Mapping[str, tuple[int, int]] | None = None,
    max_operators: int | None = MAX_OPERATORS,
) -> Verdict:
    """`max_operators` is the Power Pool limit; None for a regular-only formula.
    `fields` maps this dataset's field ids to MATRIX/VECTOR/GROUP."""
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

    if signatures:
        verdict.issues.extend(arity_issues(expression, signatures))
    verdict.issues.extend(group_issues(expression, fields))
    verdict.operator_count = operator_count(expression, ops)
    if max_operators is not None and verdict.operator_count > max_operators:
        verdict.issues.append(f"{verdict.operator_count} operators > {max_operators}")
    verdict.ok = not verdict.issues
    return verdict
