import json
from pathlib import Path

import httpx
import pytest

from alpha_orchestrator.brain.api import BrainAPI, parse_simulation_options, simulation_payload
from alpha_orchestrator.brain.client import BrainClient, RateLimited
from alpha_orchestrator.brain.session import (
    AuthenticationFailed,
    BrainSession,
    Credentials,
    NeedBiometric,
    SessionStore,
)

CREDS = Credentials(email="me@example.com", password="pw")


def make(handler, tmp_path: Path, clock=lambda: 10_000.0):  # type: ignore[no-untyped-def]
    session = BrainSession(
        CREDS, SessionStore(tmp_path), transport=httpx.MockTransport(handler), clock=clock
    )
    return session, BrainClient(session, sleep=lambda _s: None)


def test_credentials_accepts_both_file_shapes(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"email": "x@y", "password": "p"}))
    b = tmp_path / "b.json"
    b.write_text(json.dumps(["x@y", "p"]))
    assert Credentials.load(a) == Credentials.load(b) == Credentials("x@y", "p")


def test_login_saves_cookies_privately(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"].startswith("Basic ")
        return httpx.Response(201, headers={"set-cookie": "t=abc; Path=/"})

    session, _ = make(handler, tmp_path)
    session.login()
    saved = json.loads(session.store.cookies_path.read_text())
    assert [(c["name"], c["value"]) for c in saved] == [("t", "abc")]
    assert oct(session.store.cookies_path.stat().st_mode)[-3:] == "600"


def test_same_cookie_name_on_two_domains_round_trips(tmp_path: Path) -> None:
    """The platform sets `t` for more than one domain; saving must not crash."""
    from alpha_orchestrator.brain.session import SessionStore

    store = SessionStore(tmp_path)
    jar = httpx.Cookies()
    jar.set("t", "api", domain="api.worldquantbrain.com", path="/")
    jar.set("t", "web", domain=".worldquantbrain.com", path="/")
    store.save_cookies(jar)
    again = store.load_cookies()
    assert sorted(c.value for c in again.jar) == ["api", "web"]
    # files written by the earlier version still load
    store.cookies_path.write_text(json.dumps({"t": "old"}))
    assert [c.value for c in store.load_cookies().jar] == ["old"]
    # a stale domain-less copy next to the server's cookie is dropped, on load and on save
    store.cookies_path.write_text(
        json.dumps(
            [
                {"name": "t", "value": "stale", "domain": "", "path": "/"},
                {"name": "t", "value": "fresh", "domain": ".api.worldquantbrain.com", "path": "/"},
            ]
        )
    )
    loaded = store.load_cookies()
    assert [c.value for c in loaded.jar] == ["fresh"]
    mixed = httpx.Cookies()
    mixed.set("t", "stale")
    mixed.set("t", "fresh", domain=".api.worldquantbrain.com")
    store.save_cookies(mixed)
    assert [c["value"] for c in json.loads(store.cookies_path.read_text())] == ["fresh"]


def test_persona_challenge_raises_with_url(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            headers={
                "WWW-Authenticate": "persona",
                "Location": "/authentication/persona?inquiry=1",
            },
        )

    session, _ = make(handler, tmp_path)
    with pytest.raises(NeedBiometric) as info:
        session.login()
    assert info.value.url.endswith("/authentication/persona?inquiry=1")
    assert session.store.meta()["pending_biometric"] == info.value.url


def test_relogin_is_rationed(tmp_path: Path) -> None:
    session, _ = make(lambda r: httpx.Response(201), tmp_path)
    session.login()
    with pytest.raises(AuthenticationFailed):
        session.login()
    # a person running `alpha-factory login` is not held back by the cooldown
    session.login(interactive=True)


def test_client_retries_429_then_succeeds(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json={"ok": True}, headers={"X-Ratelimit-Remaining": "4321"})

    _, client = make(handler, tmp_path)
    assert client.request("GET", "/x").json() == {"ok": True}
    assert client.quota.remaining == 4321


def test_client_gives_up_on_persistent_429(tmp_path: Path) -> None:
    _, client = make(lambda r: httpx.Response(429), tmp_path)
    with pytest.raises(RateLimited):
        client.request("GET", "/x")


def test_wait_json_follows_retry_after(tmp_path: Path) -> None:
    seq = [httpx.Response(200, headers={"Retry-After": "2"}), httpx.Response(200, json={"v": 1})]
    _, client = make(lambda r: seq.pop(0), tmp_path)
    assert client.wait_json("/alphas/a") == {"v": 1}


def test_multi_simulation_round_trip(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/simulations":
            body = json.loads(request.content)
            assert isinstance(body, list) and len(body) == 2
            return httpx.Response(201, headers={"Location": "/simulations/M1"})
        if path == "/simulations/M1":
            return httpx.Response(200, json={"status": "COMPLETE", "children": ["C1", "C2"]})
        if path == "/simulations/C1":
            return httpx.Response(200, json={"status": "COMPLETE", "alpha": "A1"})
        if path == "/simulations/C2":
            return httpx.Response(200, json={"status": "ERROR", "message": "bad field"})
        raise AssertionError(path)

    _, client = make(handler, tmp_path)
    api = BrainAPI(client)
    payloads = [
        simulation_payload("rank(close)", {"region": "USA"}),
        simulation_payload("rank(open)", {"region": "USA"}),
    ]
    progress = api.poll(api.start_multi_simulation(payloads))
    assert progress.done and progress.children == ["C1", "C2"]
    assert api.child("C1").alpha_id == "A1"
    assert api.child("C2").message == "bad field"


def test_paged_datafields(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        rows = [{"id": f"f{offset + i}"} for i in range(50 if offset == 0 else 10)]
        return httpx.Response(200, json={"count": 60, "results": rows})

    _, client = make(handler, tmp_path)
    fields = BrainAPI(client).datafields("USA", 1, "TOP3000", "fundamental6")
    assert len(fields) == 60 and fields[-1]["id"] == "f59"


def test_simulation_options_flatten() -> None:
    tree = {
        "region": {"choices": {"instrumentType": {"EQUITY": [{"value": "USA"}]}}},
        "universe": {
            "choices": {
                "instrumentType": {
                    "EQUITY": {"region": {"USA": [{"value": "TOP3000"}, {"value": "TOP1000"}]}}
                }
            }
        },
        "neutralization": {
            "choices": {"instrumentType": {"EQUITY": {"region": {"USA": [{"value": "INDUSTRY"}]}}}}
        },
        "delay": {
            "choices": {
                "instrumentType": {"EQUITY": {"region": {"USA": [{"value": 1}, {"value": 0}]}}}
            }
        },
    }
    assert parse_simulation_options(tree) == {
        "USA": {
            "universes": ["TOP3000", "TOP1000"],
            "neutralizations": ["INDUSTRY"],
            "delays": [1, 0],
        }
    }


def test_yearly_stats_and_pnl_tables(tmp_path: Path) -> None:
    table = {
        "schema": {"properties": [{"name": "date"}, {"name": "pnl"}]},
        "records": [["2020-01-02", 10.0], ["2020-01-03", -2.5]],
    }
    _, client = make(lambda r: httpx.Response(200, json=table), tmp_path)
    assert BrainAPI(client).daily_pnl("A") == {"2020-01-02": 10.0, "2020-01-03": -2.5}
