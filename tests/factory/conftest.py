import pytest


@pytest.fixture(autouse=True)
def _reports_in_tmp(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    """FactoryConfig() writes reports to ./reports. Run every factory test from a temporary
    folder so a test never overwrites the real 工厂_待挑选.csv (it did on 2026-11-01)."""
    monkeypatch.chdir(tmp_path)
