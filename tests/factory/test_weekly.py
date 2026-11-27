from datetime import date

from alpha_orchestrator.factory import weekly


def _row(day: str, **kw: float) -> dict[str, object]:
    return {"day": day, **kw}


ROWS = [
    _row(
        "2026-11-10", simulations_done=5, expressions_proposed=10, expressions_valid=4, llm_usd=0.5
    ),
    _row(
        "2026-11-20",
        simulations_done=100,
        expressions_proposed=200,
        expressions_valid=150,
        llm_usd=1.0,
    ),
    _row("2026-11-26", simulations_done=20, stored_for_review=1, llm_usd=0.25),
]


def test_windows_are_two_runs_of_seven_days() -> None:
    assert weekly.windows(date(2026, 11, 26)) == (
        ("2026-11-20", "2026-11-26"),
        ("2026-11-13", "2026-11-19"),
    )


def test_table_splits_week_prior_and_all_time() -> None:
    body = weekly.table(ROWS, date(2026, 11, 26))
    assert "| Simulations completed | 120 | 0 | 125 |" in body
    assert "| Stored for review | 1 | 0 | 1 |" in body
    assert "| Share of expressions passing local checks | 75.0% | – | 73.3% |" in body
    assert "| LLM cost (USD) | 1.25 | 0.00 | 1.75 |" in body
    assert "All time: since 2026-11-10" in body


def test_changelog_lines() -> None:
    text = weekly.changelog_lines(ROWS, date(2026, 11, 26))
    assert "120 simulations, LLM cost $1.25" in text and "1 stored for review" in text


def test_section_is_added_once_then_replaced() -> None:
    readme = "# T\n\nintro\n\n## Quick start\n\nrun it\n"
    first = weekly.put_section(readme, "v1")
    assert first.index("## Recent performance") < first.index("## Quick start")
    second = weekly.put_section(first, "v2")
    assert "v2" in second and "v1" not in second
    assert second.count("## Recent performance") == 1
    assert weekly.put_section("# T\n", "x").endswith(weekly.END + "\n\n")
