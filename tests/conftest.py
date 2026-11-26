from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _platform_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Tests never read the private vocabulary or contact the real platform, and never
    write to the real memory folder, crowded list or PnL cache (home is a temp folder)."""
    monkeypatch.setenv("PLATFORM_API_URL", "https://api.example.test")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
