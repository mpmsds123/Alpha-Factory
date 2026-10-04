"""Deterministic expression generators: idea templates, orders 2-3, flips, repairs, stress
tests.

Each returns `Proposal`s; the pipeline validates, de-duplicates and queues them.
Window choices are restricted to economically meaningful lengths (one week, month,
quarter, half-year, year), per the official guidance against fitted values like 17 or 37.
"""

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from alpha_orchestrator.gates.sign_flip import negate

WINDOWS = (5, 21, 63, 126, 252)
GROUP_OPERATORS = ("group_rank", "group_zscore", "group_neutralize")
GROUPS = ("industry", "subindustry", "sector")
NEUTRALIZATION_BY_CATEGORY = {  # 官方 Neutralization 文档按数据类别的推荐
    "fundamental": "INDUSTRY",
    "analyst": "INDUSTRY",
    "earnings": "INDUSTRY",
    "news": "SUBINDUSTRY",
    "sentiment": "INDUSTRY",
    "socialmedia": "INDUSTRY",
    "option": "MARKET",
    "pv": "MARKET",
    "institutions": "SECTOR",
    "macro": "SECTOR",
}
MAX_TRADE_REGIONS = frozenset({"ASI", "JPN", "HKG", "KOR", "TWN"})  # 规则: must be ON
COUNTRY_FIRST = frozenset({"EUR", "ASI", "GLB"})  # 官方新人指南
GROUP_WRAP = re.compile(r"\bgroup_(?:rank|zscore|neutralize)\(")
HORIZON = re.compile(r"^(?:\d+|\d+[dwmqy]|fy\d+|fq\d+|[qhy]\d|ltm|ntm|ttm)$")  # horizons only


@dataclass
class Proposal:
    expression: str
    settings: dict[str, Any]
    stage: str
    role: str = ""
    change: str = ""
    priority: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------- settings ----------
def base_settings(
    region: str, universe: str, delay: int, category: str, neutralizations: Sequence[str] = ()
) -> dict[str, Any]:
    neut = (
        "COUNTRY"
        if region in COUNTRY_FIRST
        else NEUTRALIZATION_BY_CATEGORY.get(category, "INDUSTRY")
    )
    if neutralizations and neut not in neutralizations:
        neut = next(
            (n for n in ("INDUSTRY", "SUBINDUSTRY", "MARKET") if n in neutralizations),
            neutralizations[0],
        )
    return {
        "region": region,
        "universe": universe,
        "delay": delay,
        "decay": 0,
        "neutralization": neut,
        "truncation": 0.08,
        "maxTrade": "ON" if region in MAX_TRADE_REGIONS else "OFF",
    }


def decay_for_turnover(turnover: float | None) -> int:
    """官方 machine_lib: pair higher-turnover signals with a larger decay."""
    if turnover is None:
        return 0
    for limit, decay in ((0.6, 16), (0.4, 8), (0.25, 4)):
        if turnover > limit:
            return decay
    return 0


# ---------- fields ----------
def family_key(field_id: str) -> str:
    """Near-duplicate fields share a family: horizon/version tokens are dropped."""
    parts = [p for p in field_id.lower().split("_") if p and not HORIZON.match(p)]
    return "_".join(parts) or field_id.lower()


def group_families(fields: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    families: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for f in fields:
        families[family_key(str(f["id"]))].append(f)
    for members in families.values():
        members.sort(key=lambda f: (-(float(f.get("coverage") or 0)), str(f["id"])))
    return dict(families)


# ---------- an idea's template ----------
def _filler(values: Mapping[str, str]) -> Callable[[re.Match[str]], str]:
    return lambda m: values.get(m.group(1), m.group(0))


PLACEHOLDER = re.compile(r"\{([A-Za-z]\w*)\}")


def expand_template(
    template: str,
    slots: Mapping[str, Sequence[str]],
    windows: Sequence[int],
    prep: Mapping[str, str],
    settings: dict[str, Any],
    limit: int,
    rng: Any,
) -> list[Proposal]:
    """`{A}`, `{B}` ... take field ids from `slots` (each entering through its `prep`),
    `{w}` takes each window. Every combination is built, shuffled (官方: random sampling
    beats the first N in field order), and capped at `limit`. A combination that puts the
    same field in two slots is skipped."""
    names = [n for n in dict.fromkeys(PLACEHOLDER.findall(template)) if n != "w"]
    combos: list[dict[str, str]] = [{}]
    for name in names:
        combos = [{**c, name: f} for c in combos for f in slots.get(name, [])]
    combos = [c for c in combos if len(set(c.values())) == len(c)]
    wins = list(windows) if "{w}" in template else [0]
    out = []
    for combo in combos:
        for w in wins:
            values = {n: prep.get(f, f) for n, f in combo.items()}
            values["w"] = str(w)
            expr = PLACEHOLDER.sub(_filler(values), template)
            label = ",".join(f"{n}={f}" for n, f in combo.items())
            out.append(
                Proposal(expr, settings, "IDEA", role="idea", change=f"{label} w={w}".strip())
            )
    rng.shuffle(out)
    return out[:limit]


# ---------- order 2 / 3 ----------
def second_order(
    expression: str, settings: dict[str, Any], extra_groups: Sequence[str] = ()
) -> list[Proposal]:
    out = []
    for op in GROUP_OPERATORS:
        for group in (*GROUPS, *extra_groups):
            out.append(
                Proposal(
                    f"{op}({expression}, {group})",
                    settings,
                    "SECOND",
                    role=op,
                    change=f"wrap {op} by {group}",
                )
            )
    return out


THIRD_ORDER_EVENTS = (
    ("volume > adv20", "-1", "trade on above-average volume"),
    ("ts_std_dev(returns, 21) < ts_std_dev(returns, 126)", "-1", "trade in calm regimes"),
    ("ts_std_dev(returns, 21) > ts_std_dev(returns, 126)", "-1", "trade in volatile regimes"),
    ("1", "abs(returns) > 0.1", "stop out after a 10% move"),
)


def third_order(expression: str, settings: dict[str, Any]) -> list[Proposal]:
    return [
        Proposal(
            f"trade_when({entry}, {expression}, {exit_})",
            settings,
            "THIRD",
            role="trade_when",
            change=why,
        )
        for entry, exit_, why in THIRD_ORDER_EVENTS
    ]


# ---------- flips and repairs ----------
def flip(expression: str, settings: dict[str, Any], sharpe: float) -> Proposal | None:
    negated = negate(expression)
    if negated is None:
        return None
    return Proposal(
        negated,
        settings,
        "FLIP",
        role="flip",
        change=f"parent sharpe {sharpe:.2f}; negation mirrors the signal",
    )


def repair_proposals(
    expression: str,
    settings: dict[str, Any],
    fails: Iterable[str],
    turnover: float | None,
    neutralizations: Sequence[str],
    may_finetune: bool,
) -> list[Proposal]:
    """One allowed fix per failing check, one variable at a time (设计 第 5 步)."""
    out: list[Proposal] = []
    fails = set(fails)
    if "HIGH_TURNOVER" in fails or (turnover or 0) > 0.4:
        new_decay = min(
            512, max(decay_for_turnover(turnover), int(settings.get("decay", 0)) * 2 or 4)
        )
        out.append(
            Proposal(
                expression,
                {**settings, "decay": new_decay},
                "REPAIR",
                role="decay",
                change=f"decay {settings.get('decay', 0)}->{new_decay}",
            )
        )
        out.append(
            Proposal(
                f"hump({expression}, hump=0.01)",
                settings,
                "REPAIR",
                role="hump",
                change="hump to cut small rebalances",
            )
        )
    if "LOW_SUB_UNIVERSE_SHARPE" in fails:
        split = (
            f"ts_decay_linear({expression}, 5) * rank(volume * close) + "
            f"ts_decay_linear({expression}, 10) * (1 - rank(volume * close))"
        )
        out.append(
            Proposal(
                split,
                settings,
                "REPAIR",
                role="liquidity_split",
                change="decay liquid and illiquid names separately",
            )
        )
    if "CONCENTRATED_WEIGHT" in fails:
        out.append(
            Proposal(
                f"rank({expression})",
                {**settings, "truncation": 0.05},
                "REPAIR",
                role="rank_truncate",
                change="rank and truncation 0.05",
            )
        )
    if may_finetune and fails & {"LOW_SHARPE", "LOW_FITNESS"}:
        current = settings.get("neutralization")
        for neut in ("SUBINDUSTRY", "INDUSTRY", "SECTOR", "STATISTICAL", "SLOW_AND_FAST"):
            if neut != current and neut in neutralizations:
                out.append(
                    Proposal(
                        expression,
                        {**settings, "neutralization": neut},
                        "REPAIR",
                        role="neutralization",
                        change=f"neutralization {current}->{neut}",
                    )
                )
    return out


# ---------- robustness ----------
INT = re.compile(r"(?<![\w.])(\d+)(?![\w.])")


def perturb_windows(expression: str) -> list[str]:
    """Each meaningful window nudged ~10% both ways (21 -> 19, 23), one at a time."""
    out = []
    for m in INT.finditer(expression):
        w = int(m.group(1))
        if w not in WINDOWS or w == 5:
            continue
        for nw in (round(w * 0.9), round(w * 1.1)):
            out.append(expression[: m.start()] + str(nw) + expression[m.end() :])
    return out


def robustness_proposals(
    expression: str,
    settings: dict[str, Any],
    neutralizations: Sequence[str],
    universes: Sequence[str],
) -> list[Proposal]:
    out = [
        Proposal(f"rank({expression})", settings, "ROBUST", role="rank"),
        Proposal(f"sign({expression})", settings, "ROBUST", role="sign"),
    ]
    out += [
        Proposal(e, settings, "ROBUST", role="perturb") for e in perturb_windows(expression)[:2]
    ]
    alt = next(
        (
            n
            for n in ("SUBINDUSTRY", "INDUSTRY", "MARKET", "SECTOR")
            if n in neutralizations and n != settings.get("neutralization")
        ),
        None,
    )
    if alt:
        out.append(
            Proposal(
                expression, {**settings, "neutralization": alt}, "ROBUST", role="neutralization"
            )
        )
    smaller = _smaller_universe(str(settings.get("universe")), universes)
    if smaller:
        out.append(
            Proposal(expression, {**settings, "universe": smaller}, "ROBUST", role="sub_universe")
        )
    return out


def _smaller_universe(current: str, universes: Sequence[str]) -> str | None:
    if not current.startswith("TOP") or not current[3:].isdigit():
        return None
    size = int(current[3:])
    tops = sorted(
        (
            int(u[3:])
            for u in universes
            if u.startswith("TOP") and u[3:].isdigit() and int(u[3:]) < size
        ),
        reverse=True,
    )
    return f"TOP{tops[0]}" if tops else None


# ---------- refinement structures (dig deeper) ----------
# Library structures put around a family's best survey formula E. Which ones get tried is
# decided by their recorded success on this data category (Thompson sampling), not here.
STRUCTURES: dict[str, str] = {
    "subindustry": "group_rank({E}, subindustry)",
    "sector": "group_rank({E}, sector)",
    "cap_bucket": 'group_rank({E}, bucket(rank(cap), range="0.1,1,0.1"))',
    "smooth": "ts_decay_linear({E}, 10)",
    "volume_event": "trade_when(volume > adv20, {E}, -1)",
    "peer_zscore": "group_zscore({E}, industry)",
}


def structure(name: str, expression: str) -> str:
    return STRUCTURES[name].replace("{E}", expression)


# ---------- combinations (dig across fields) ----------
# The random arm: forms valid for any two fields. The LLM arm writes its own; both are
# simulated side by side so the LLM has to beat random to keep its budget.
COMBO_FORMS: tuple[str, ...] = (
    "rank({A}) * rank({B})",
    "rank({A}) - rank({B})",
    "trade_when(rank({B}) < 0.3, rank({A}), -1)",
    'group_rank({A}, bucket(rank({B}), range="0.2,1,0.2"))',
)


def random_combos(
    preps: Sequence[str], n: int, rng: Any, settings: dict[str, Any]
) -> list[Proposal]:
    out: list[Proposal] = []
    if len(preps) < 2:
        return out
    for _ in range(n * 4):
        if len(out) >= n:
            break
        a, b = rng.sample(list(preps), 2)
        form = rng.choice(COMBO_FORMS)
        expr = form.replace("{A}", a).replace("{B}", b)
        if any(p.expression == expr for p in out):
            continue
        out.append(Proposal(expr, settings, "COMBO", role="random", change=form))
    return out
