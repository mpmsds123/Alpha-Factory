from pathlib import Path


def test_packaged_default_policies_match_repository_versions() -> None:
    root = Path(__file__).parents[2]
    packaged = root / "src" / "alpha_orchestrator" / "default_policies"
    for name in ("routing.v1.yaml", "limits.v1.yaml", "error-taxonomy.v1.yaml"):
        assert (root / "policies" / name).read_bytes() == (packaged / name).read_bytes()
