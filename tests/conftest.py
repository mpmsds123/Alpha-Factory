import pytest


@pytest.fixture(autouse=True)
def _platform_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never read the private vocabulary or contact the real platform."""
    monkeypatch.setenv("PLATFORM_API_URL", "https://api.example.test")
