"""The repository's public documentation is one README plus a CHANGELOG, kept generic."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
PUBLIC_DOCS = (ROOT / "README.md", ROOT / "CHANGELOG.md")


def test_only_readme_and_changelog_are_top_level_docs() -> None:
    markdown = sorted(p.name for p in ROOT.glob("*.md"))
    assert markdown == ["CHANGELOG.md", "README.md"]
    assert not (ROOT / "docs").exists()


def test_readme_covers_problem_flow_and_how_to_run() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for heading in ("## The problem", "## How it works", "## Quick start", "## Configuration"):
        assert heading in readme
    assert "review approve" in readme
    assert "Nothing is ever submitted automatically" in readme


def test_public_docs_stay_platform_generic() -> None:
    platform_name = re.compile(r"WorldQuant|(?<![A-Z_])BRAIN(?![A-Z_])")
    # An operator call applied to a field, e.g. name(arg — formulas are never published.
    expression = re.compile(r"\b[a-z]+_[a-z_]*\(\s*[a-z]")
    for path in PUBLIC_DOCS:
        text = path.read_text(encoding="utf-8")
        assert not platform_name.search(text), path.name
        assert not expression.search(text), path.name
