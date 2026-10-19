import tomllib
from pathlib import Path

from alpha_orchestrator import __version__


def test_package_versions_match() -> None:
    root = Path(__file__).parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == __version__


def test_packaged_default_policies_match_repository_versions() -> None:
    root = Path(__file__).parents[2]
    packaged = root / "src" / "alpha_orchestrator" / "default_policies"
    for name in (
        "routing.v1.yaml",
        "routing.v2.yaml",
        "limits.v1.yaml",
        "limits.v2.yaml",
        "limits.v3.yaml",
        "error-taxonomy.v1.yaml",
        "correlation.v1.yaml",
    ):
        assert (root / "policies" / name).read_bytes() == (packaged / name).read_bytes()


def test_packaged_default_prompts_match_repository_versions() -> None:
    root = Path(__file__).parents[2]
    packaged = root / "src" / "alpha_orchestrator" / "default_prompts"
    for name in ("generator.v1.md", "repair-planner.v1.md"):
        assert (root / "prompts" / name).read_bytes() == (packaged / name).read_bytes()
