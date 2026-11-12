"""Local checks before anything is simulated. One bad expression cancels a whole
multi-simulation, so nothing reaches the platform without passing here.

Operators and their signatures come from the platform's live operator list; grouping
identifiers and the like come from the private vocabulary.
"""

import re
from collections.abc import Mapping, Set
from dataclasses import dataclass, field

from alpha_orchestrator.factory import tree
from alpha_orchestrator.factory.vocab import Vocabulary


@dataclass
class Verdict:
    ok: bool
    issues: list[str] = field(default_factory=list)
    node: tree.Node | None = None
    canonical: str = ""
    shape: str = ""
    operators: int = 0
    nodes: int = 0
    fields_used: list[str] = field(default_factory=list)


def signature(definition: str, description: str = "") -> tuple[int, int] | None:
    """(fewest, most positional inputs; -1 = no limit) read from an operator definition
    such as "name(x, d, flag=false)": inputs with a default must be passed by name, except
    a one-letter default (a value slot). None when the definition is not name(...)."""
    head = definition.strip().splitlines()[0] if definition.strip() else ""
    m = re.match(r"\s*[A-Za-z_]\w*\s*\((.*)\)", head)
    if not m:
        return None
    required, variadic, optional_positional = 0, False, 0
    depth, cur, parts, quote = 0, "", [], ""
    for ch in m.group(1):
        if quote:
            quote = "" if ch == quote else quote
            cur += ch
            continue
        if ch in "'\"":
            quote = ch
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    for part in (p.strip() for p in parts):
        if not part:
            continue
        if ".." in part:
            variadic = True
            if part.replace(".", "").strip():
                required += 1
            continue
        named = re.match(r"^[A-Za-z_]\w*\s*=(?!=)\s*(.*)$", part)
        if named:
            if re.fullmatch(r"[a-z]", named.group(1).strip()):
                optional_positional += 1
            continue
        required += 1
    low = description.lower()
    if "two or more" in low or "at least 2" in low:
        variadic = True
    return required, -1 if variadic else required + optional_positional


def check(
    expression: str,
    *,
    operators: Set[str],
    signatures: Mapping[str, tuple[int, int]],
    fields: Mapping[str, str],
    vocab: Vocabulary,
    windows: tuple[int, ...],
    max_operators: int,
    max_fields: int,
) -> Verdict:
    """`fields` maps every data field this expression may use to its type
    (MATRIX / VECTOR / GROUP)."""
    try:
        node = tree.parse(expression)
    except ValueError as exc:
        return Verdict(ok=False, issues=[f"does not parse: {exc}"])
    v = Verdict(ok=True, node=node, canonical=tree.text(node), shape=tree.shape(node, vocab))
    ops = {o.lower() for o in operators}
    allowed = {k: t for k, t in fields.items()}
    used: set[str] = set()

    def visit(n: tree.Node, reducers: int) -> None:
        if n.kind == "call":
            if n.value not in ops:
                v.issues.append(f"unknown operator {n.value}")
            sig = signatures.get(n.value)
            args = tree.positional(n)
            if sig is not None:
                low, high = sig
                if len(args) < low or (high >= 0 and len(args) > high):
                    want = f"{low}" if high == low else f"{low}-{high}" if high > 0 else f"{low}+"
                    v.issues.append(
                        f"{n.value} takes {want} positional input(s), got {len(args)}; "
                        "pass optional inputs by name"
                    )
            at = vocab.group_input.get(n.value)
            if at is not None and at < len(args):
                g = args[at]
                is_group = (
                    (g.kind == "name" and g.value.lower() in vocab.grouping_fields)
                    or (g.kind == "name" and allowed.get(g.value) == "GROUP")
                    or (g.kind == "call" and g.value in vocab.group_makers)
                )
                if not is_group:
                    v.issues.append(f"{n.value} needs a grouping input, got {tree.text(g)[:60]}")
            inner = reducers + (
                1
                if vocab.vector_reducer_prefix and n.value.startswith(vocab.vector_reducer_prefix)
                else 0
            )
            for c in n.children:
                visit(c, inner)
            return
        if n.kind == "kwarg":
            visit(n.children[0], reducers)
            return
        if n.kind == "name":
            name = n.value
            if name.lower() in vocab.grouping_fields or name.lower() in ("true", "false", "nan"):
                return
            if name not in allowed:
                v.issues.append(f"field {name} is not allowed for this hypothesis")
                return
            used.add(name)
            if allowed[name] == "VECTOR" and vocab.vector_reducer_prefix and reducers == 0:
                v.issues.append(f"vector field {name} needs a reducer")
            return
        for c in n.children:
            visit(c, reducers)

    visit(node, 0)
    v.fields_used = sorted(used)
    v.operators = tree.operator_count(node, vocab)
    v.nodes = tree.size(node)
    if v.operators > max_operators:
        v.issues.append(f"{v.operators} operators > {max_operators}")
    if len(used) > max_fields:
        v.issues.append(f"{len(used)} data fields > {max_fields}")
    bad = sorted({w for w in tree.windows(node, vocab) if w not in windows})
    if bad:
        v.issues.append(f"window(s) {bad} not in the allowed set {list(windows)}")
    v.ok = not v.issues
    return v
