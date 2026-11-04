"""login --then-run must hand the run an open session (bug of 2026-11-03: the session was
closed in a finally block before the run used it)."""

from typing import Any

from alpha_orchestrator.factory import cli


class _Session:
    def __init__(self) -> None:
        self.closed = False

    def ensure(self, interactive: bool = False) -> float:
        return 3600.0

    def close(self) -> None:
        self.closed = True


def test_then_run_gets_an_open_session_and_closes_it_afterwards(monkeypatch: Any) -> None:
    made: list[_Session] = []
    seen: list[bool] = []

    def new_session() -> _Session:
        made.append(_Session())
        return made[-1]

    def fake_run(**_kw: Any) -> None:
        assert cli._live_session is made[0]
        seen.append(made[0].closed)

    monkeypatch.setattr(cli, "BrainSession", new_session)
    monkeypatch.setattr(cli, "run", fake_run)
    cli.login(then_run=True)
    assert seen == [False]  # open while the run uses it
    assert made[0].closed and cli._live_session is None


def test_login_alone_closes_the_session(monkeypatch: Any) -> None:
    made: list[_Session] = []
    monkeypatch.setattr(cli, "BrainSession", lambda: made.append(_Session()) or made[-1])
    monkeypatch.setattr(cli, "run", lambda **_kw: (_ for _ in ()).throw(AssertionError))
    cli.login(then_run=False)
    assert made[0].closed
