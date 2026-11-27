"""Weekly performance: the README table and the changelog lines, recomputed from the database.

Counts and rates only (no expressions, fields or dataset names), so the output is safe for
the public README. A week is the 7 quota days ending on the report day.
"""

from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.report import _day, stage_rows

START = "<!-- performance:start -->"
END = "<!-- performance:end -->"
ANCHOR = "## Quick start"  # the section goes in front of this heading the first time

COUNTS = (
    ("Hypotheses opened", "hypotheses_opened"),
    ("Expressions proposed", "expressions_proposed"),
    ("Expressions passing local checks", "expressions_valid"),
    ("Simulations completed", "simulations_done"),
    ("Round-1 gate cleared (hypotheses)", "round1_cleared"),
    ("Train gate passed", "train_pass"),
    ("Test-period gate passed", "oos_pass"),
    ("Platform checks passed", "platform_pass"),
    ("Stored for review", "stored_for_review"),
)


def totals(rows: list[dict[str, Any]], first: str = "", last: str = "9999") -> dict[str, float]:
    """Sum the daily stage rows whose day is in [first, last]."""
    out: dict[str, float] = defaultdict(float)
    for r in rows:
        if first <= r["day"] <= last:
            for k, v in r.items():
                if k != "day" and isinstance(v, int | float) and not isinstance(v, bool):
                    out[k] += v
    return out


def _rate(num: float, den: float, pct: bool = True) -> str:
    if not den:
        return "–"
    return f"{100 * num / den:.1f}%" if pct else f"{num / den:.3f}"


def _column(c: dict[str, float]) -> list[str]:
    cells = [f"{int(c[k]):,}" for _, k in COUNTS]
    cells.append(_rate(c["expressions_valid"], c["expressions_proposed"]))
    cells.append(f"{c['llm_usd']:.2f}")
    cells.append(_rate(c["llm_usd"], c["simulations_done"], pct=False))
    return cells


def windows(today: date) -> tuple[tuple[str, str], tuple[str, str]]:
    """(this week, prior week) as inclusive day ranges."""
    end = today
    start = end - timedelta(days=6)
    prior_end = start - timedelta(days=1)
    prior_start = prior_end - timedelta(days=6)
    return (start.isoformat(), end.isoformat()), (prior_start.isoformat(), prior_end.isoformat())


def table(rows: list[dict[str, Any]], today: date) -> str:
    """The README section body: one table, this week against the prior week and all time."""
    (w0, w1), (p0, p1) = windows(today)
    week, prior, total = totals(rows, w0, w1), totals(rows, p0, p1), totals(rows)
    first = rows[0]["day"] if rows else today.isoformat()
    labels = [label for label, _ in COUNTS] + [
        "Share of expressions passing local checks",
        "LLM cost (USD)",
        "LLM cost per simulation (USD)",
    ]
    lines = [
        f"Updated {today.isoformat()} from the run database. This week: {w0} to {w1} "
        f"(quota days, Pacific). All time: since {first}.",
        "",
        "| Stage | This week | Prior week | All time |",
        "|---|---:|---:|---:|",
    ]
    for label, a, b, c in zip(labels, _column(week), _column(prior), _column(total), strict=True):
        lines.append(f"| {label} | {a} | {b} | {c} |")
    return "\n".join(lines)


def changelog_lines(rows: list[dict[str, Any]], today: date) -> str:
    """The **Ran** and **Found** lines of a weekly CHANGELOG entry."""
    (w0, w1), _ = windows(today)
    c = totals(rows, w0, w1)
    return (
        f"- Ran. {w0} to {w1}: {int(c['hypotheses_opened'])} hypotheses, "
        f"{int(c['expressions_proposed']):,} expressions proposed "
        f"({int(c['expressions_valid']):,} passed local checks), "
        f"{int(c['simulations_done']):,} simulations, LLM cost ${c['llm_usd']:.2f}.\n"
        f"- Found. Round-1 gate cleared by {int(c['round1_cleared'])} hypotheses; "
        f"train gate {int(c['train_pass'])}, test-period gate {int(c['oos_pass'])}, "
        f"platform checks {int(c['platform_pass'])}; "
        f"{int(c['stored_for_review'])} stored for review."
    )


def put_section(readme: str, body: str) -> str:
    """Replace the performance block, or add it (with its heading) before ANCHOR."""
    block = f"{START}\n{body}\n{END}"
    if START in readme and END in readme:
        head, rest = readme.split(START, 1)
        _, tail = rest.split(END, 1)
        return head + block + tail
    section = f"## Recent performance\n\n{block}\n\n"
    if ANCHOR in readme:
        return readme.replace(ANCHOR, section + ANCHOR, 1)
    return readme.rstrip("\n") + "\n\n" + section


def report(ledger: Ledger, config: FactoryConfig, now: datetime | None = None) -> tuple[str, str]:
    """(README section body, changelog lines) for the quota day containing `now`."""
    rows, _ = stage_rows(ledger, config)
    today = date.fromisoformat(_day(now or datetime.now().astimezone(), config))
    return table(rows, today), changelog_lines(rows, today)


def write_readme(path: Path, body: str) -> bool:
    """Update the README in place; True when it changed."""
    old = path.read_text(encoding="utf-8")
    new = put_section(old, body)
    if new != old:
        path.write_text(new, encoding="utf-8")
    return new != old
