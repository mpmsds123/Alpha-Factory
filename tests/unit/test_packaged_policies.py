import tomllib
from pathlib import Path

from alpha_orchestrator import __version__


def test_package_versions_match() -> None:
    root = Path(__file__).parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == __version__
