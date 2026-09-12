import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs"


def test_portfolio_documents_exist_and_are_linked() -> None:
    expected = (
        "ARCHITECTURE.md",
        "TEST_EVIDENCE.md",
        "DESIGN_TRADEOFFS.md",
        "KNOWN_LIMITS.md",
        "60_SECOND_EXPLANATION.md",
        "case-studies/sanitized-replay.md",
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for relative_path in expected:
        path = DOCS / relative_path
        assert path.is_file()
        assert path.read_text(encoding="utf-8").strip()
        assert f"docs/{relative_path}" in readme


def test_case_study_matches_sanitized_replay_fixture() -> None:
    fixture = json.loads(
        (ROOT / "tests" / "fixtures" / "basic_replay.json").read_text(encoding="utf-8")
    )
    case_study = (DOCS / "case-studies" / "sanitized-replay.md").read_text(encoding="utf-8")
    normalized_case_study = " ".join(case_study.split())

    assert len(fixture["candidates"]) == 2
    assert len(fixture["scenarios"]) == 2
    for scenario in fixture["scenarios"]:
        expression = scenario["expression"]
        metrics = scenario["result"]["metrics"]
        assert f"`{expression}`" in case_study
        for name in ("sharpe", "fitness", "turnover"):
            assert f"`{metrics[name]:.2f}`" in case_study
    assert "2 distinct candidates" in normalized_case_study
    assert "2 completed simulation results" in normalized_case_study
    assert "No live platform check was performed" in normalized_case_study


def test_architecture_documents_preserve_authority_boundaries() -> None:
    architecture = (DOCS / "ARCHITECTURE.md").read_text(encoding="utf-8")
    limits = (DOCS / "KNOWN_LIMITS.md").read_text(encoding="utf-8")
    normalized_architecture = " ".join(architecture.split())
    normalized_limits = " ".join(limits.split())

    assert "# Current Package Architecture" in architecture
    assert "SQLite is the source of truth" in architecture
    assert "typed proposals only" in architecture
    assert "there is no submission path" in normalized_architecture
    assert "remain unverified" in normalized_limits
    assert "There is no dashboard and no automatic submission" in normalized_limits


def test_technical_explanation_is_concise_and_states_the_safety_boundary() -> None:
    explanation = (DOCS / "60_SECOND_EXPLANATION.md").read_text(encoding="utf-8")
    words = re.findall(r"\b[\w-]+\b", explanation.removeprefix("# 60-Second Technical Explanation"))
    normalized_explanation = " ".join(explanation.split())

    assert 110 <= len(words) <= 170
    assert "The LLM has narrow authority" in normalized_explanation
    assert "no automatic submission path" in normalized_explanation
    assert "human review" in normalized_explanation
