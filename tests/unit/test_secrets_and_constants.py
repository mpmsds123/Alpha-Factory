"""Secrets come from the environment first, then from private files; pricing hours are stable."""

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

from alpha_orchestrator.brain.session import Credentials
from alpha_orchestrator.factory.config import LLM_KEY_ENV, llm_key_available, load_llm_key
from alpha_orchestrator.factory.llm import is_peak

ROOT = Path(__file__).parents[2]


def _refresh_session_module() -> ModuleType:
    """Import scripts/refresh_session.py, which is not part of the package."""
    spec = importlib.util.spec_from_file_location(
        "refresh_session", ROOT / "scripts" / "refresh_session.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (LLM_KEY_ENV, "ALPHA_LAB_BRAIN_USERNAME", "ALPHA_LAB_BRAIN_PASSWORD"):
        monkeypatch.delenv(name, raising=False)


def test_llm_key_prefers_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key_file = tmp_path / "key.json"
    key_file.write_text(json.dumps({"api_key": "from-file"}), encoding="utf-8")
    monkeypatch.setenv(LLM_KEY_ENV, "from-env")
    assert load_llm_key(str(key_file)) == "from-env"
    assert llm_key_available(str(tmp_path / "missing.json"))


def test_llm_key_falls_back_to_file(tmp_path: Path) -> None:
    key_file = tmp_path / "key.json"
    key_file.write_text(json.dumps({"api_key": "from-file"}), encoding="utf-8")
    assert load_llm_key(str(key_file)) == "from-file"
    assert llm_key_available(str(key_file))
    assert not llm_key_available(str(tmp_path / "missing.json"))


def test_llm_key_file_without_key_is_rejected(tmp_path: Path) -> None:
    key_file = tmp_path / "key.json"
    key_file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="api_key"):
        load_llm_key(str(key_file))


def test_credentials_prefer_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHA_LAB_BRAIN_USERNAME", "user@example.com")
    monkeypatch.setenv("ALPHA_LAB_BRAIN_PASSWORD", "secret")
    assert Credentials.load() == Credentials(email="user@example.com", password="secret")


def test_explicit_credentials_file_wins_over_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ALPHA_LAB_BRAIN_USERNAME", "env@example.com")
    monkeypatch.setenv("ALPHA_LAB_BRAIN_PASSWORD", "env-secret")
    path = tmp_path / "creds.json"
    path.write_text(json.dumps({"email": "file@example.com", "password": "pw"}), encoding="utf-8")
    assert Credentials.load(path).email == "file@example.com"


def test_credentials_need_both_environment_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ALPHA_LAB_BRAIN_USERNAME", "only-user@example.com")
    path = tmp_path / "creds.json"
    path.write_text(json.dumps(["file@example.com", "pw"]), encoding="utf-8")
    monkeypatch.setenv("BRAIN_CREDENTIALS_FILE", str(path))
    assert Credentials.load().email == "file@example.com"


def test_refresh_script_reads_environment_then_private_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = _refresh_session_module()
    path = tmp_path / "brain_credentials.txt"
    path.write_text(json.dumps(["file-user", "file-pw"]), encoding="utf-8")
    path.chmod(0o600)
    assert script.load_credentials(path) == ("file-user", "file-pw")
    monkeypatch.setenv("ALPHA_LAB_BRAIN_USERNAME", "env-user")
    monkeypatch.setenv("ALPHA_LAB_BRAIN_PASSWORD", "env-pw")
    assert script.load_credentials(tmp_path / "missing.txt") == ("env-user", "env-pw")


def test_refresh_script_rejects_shared_credentials_file(tmp_path: Path) -> None:
    script = _refresh_session_module()
    path = tmp_path / "brain_credentials.txt"
    path.write_text(json.dumps(["user", "pw"]), encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(SystemExit, match="chmod 600"):
        script.load_credentials(path)


@pytest.mark.parametrize(
    ("utc", "peak"),
    [
        (datetime(2026, 11, 9, 2, 0, tzinfo=UTC), True),  # Mon 10:00 Beijing
        (datetime(2026, 11, 9, 4, 30, tzinfo=UTC), False),  # Mon 12:30 Beijing (lunch)
        (datetime(2026, 11, 9, 7, 0, tzinfo=UTC), True),  # Mon 15:00 Beijing
        (datetime(2026, 11, 9, 10, 0, tzinfo=UTC), False),  # Mon 18:00 Beijing
        (datetime(2026, 11, 14, 2, 0, tzinfo=UTC), False),  # Sat 10:00 Beijing
    ],
)
def test_peak_price_hours(utc: datetime, peak: bool) -> None:
    assert is_peak(utc) is peak
