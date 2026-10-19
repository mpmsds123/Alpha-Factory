"""BRAIN authentication: password login, biometric hand-off, cookie cache.

The platform asks for a face check (Persona) every few hours. Software cannot and must
not complete it, so an unattended run that hits one stops with `NeedBiometric`, carrying
the URL a human opens. `alpha-factory login` is the interactive path that finishes it.

Re-authentication is rationed: at most one password login per `RELOGIN_COOLDOWN_SECONDS`
to avoid repeated authentication attempts.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

API_URL = "https://api.worldquantbrain.com"
DEFAULT_CREDENTIALS = Path(".runtime/platform-session.json")
DEFAULT_STATE_DIR = Path("~/.alpha_factory")
RELOGIN_COOLDOWN_SECONDS = 1800.0


class NeedBiometric(Exception):
    """The platform requires a human face check before this session can continue."""

    def __init__(self, url: str) -> None:
        super().__init__(f"Biometric verification required: {url}")
        self.url = url


class AuthenticationFailed(Exception):
    """Credentials were rejected, or re-login was refused to protect the account."""


@dataclass(frozen=True)
class Credentials:
    email: str
    password: str

    @classmethod
    def load(cls, path: Path | None = None) -> "Credentials":
        source = path or Path(os.getenv("BRAIN_CREDENTIALS_FILE", str(DEFAULT_CREDENTIALS)))
        data = json.loads(source.expanduser().read_text(encoding="utf-8"))
        if isinstance(data, list) and len(data) == 2:
            return cls(email=str(data[0]), password=str(data[1]))
        if isinstance(data, dict) and data.get("email") and data.get("password"):
            return cls(email=str(data["email"]), password=str(data["password"]))
        raise AuthenticationFailed(f"Credentials file has no email/password: {source}")


def _current(cookies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop a domain-less cookie when the server has set the same name with a domain: the
    domain-less one is a stale copy from an older cookie file, and sending both makes the
    platform read the stale one (seen 2026-10-04: 204 with it, 200 without)."""
    with_domain = {c["name"] for c in cookies if c.get("domain")}
    return [c for c in cookies if c.get("domain") or c["name"] not in with_domain]


class SessionStore:
    """Cookies and login bookkeeping kept outside the repository, readable only by you."""

    def __init__(self, state_dir: Path | None = None) -> None:
        self.dir = (state_dir or DEFAULT_STATE_DIR).expanduser()
        self.cookies_path = self.dir / "brain_cookies.json"
        self.meta_path = self.dir / "brain_session.json"

    def load_cookies(self) -> httpx.Cookies:
        jar = httpx.Cookies()
        if not self.cookies_path.exists():
            return jar
        data = json.loads(self.cookies_path.read_text())
        if isinstance(data, dict):  # older files: {name: value}
            for name, value in data.items():
                jar.set(str(name), str(value))
            return jar
        for c in _current(data):
            jar.set(
                str(c["name"]),
                str(c["value"]),
                domain=str(c.get("domain") or ""),
                path=str(c.get("path") or "/"),
            )
        return jar

    def save_cookies(self, cookies: httpx.Cookies) -> None:
        """Every cookie with its domain and path: the platform sets the same name (`t`)
        for more than one domain, so name alone is not a key."""
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = _current(
            [
                {"name": c.name, "value": c.value, "domain": c.domain, "path": c.path}
                for c in cookies.jar
            ]
        )
        self.cookies_path.write_text(json.dumps(payload))
        self.cookies_path.chmod(0o600)

    def meta(self) -> dict[str, Any]:
        if not self.meta_path.exists():
            return {}
        return dict(json.loads(self.meta_path.read_text()))

    def update_meta(self, **values: Any) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        merged = {**self.meta(), **values}
        self.meta_path.write_text(json.dumps(merged))


class BrainSession:
    """Owns the HTTP client and keeps it authenticated."""

    def __init__(
        self,
        credentials: Credentials | None = None,
        store: SessionStore | None = None,
        *,
        base_url: str = API_URL,
        transport: httpx.BaseTransport | None = None,
        clock: Any = time.time,
    ) -> None:
        self._credentials = credentials
        self.store = store or SessionStore()
        self.base_url = base_url
        self._clock = clock
        self.http = httpx.Client(
            base_url=base_url,
            cookies=self.store.load_cookies(),
            timeout=60.0,
            follow_redirects=False,
            transport=transport,
        )

    @property
    def credentials(self) -> Credentials:
        if self._credentials is None:
            self._credentials = Credentials.load()
        return self._credentials

    def close(self) -> None:
        self.http.close()

    def seconds_remaining(self) -> float:
        """Server-reported session lifetime; 0 when the cookie is missing or expired."""
        response = self.http.get("/authentication")
        if response.status_code != 200:
            return 0.0
        try:
            expiry = float(response.json()["token"]["expiry"])
        except (KeyError, TypeError, ValueError):
            return 0.0
        self.store.update_meta(expires_at=self._clock() + expiry)
        return expiry

    def ensure(self, minimum_seconds: float = 600.0, *, interactive: bool = False) -> float:
        """Return remaining seconds, logging in with the password when needed."""
        remaining = self.seconds_remaining()
        if remaining >= minimum_seconds:
            return remaining
        self.login(interactive=interactive)
        return self.seconds_remaining()

    def login(self, *, interactive: bool = False) -> None:
        """The cooldown guards unattended runs from hammering the password; a person at
        the keyboard running `alpha-factory login` is not held back by it."""
        last = float(self.store.meta().get("last_password_login", 0.0))
        if not interactive and self._clock() - last < RELOGIN_COOLDOWN_SECONDS:
            raise AuthenticationFailed(
                "A password login happened less than 30 minutes ago; refusing another "
                "to protect the account. Run `alpha-factory login` interactively."
            )
        self.store.update_meta(last_password_login=self._clock())
        creds = self.credentials
        response = self.http.post("/authentication", auth=(creds.email, creds.password))
        if response.status_code == 401 and response.headers.get("WWW-Authenticate") == "persona":
            url = urljoin(str(response.url), response.headers["Location"])
            self.store.update_meta(pending_biometric=url)
            raise NeedBiometric(url)
        if response.status_code not in (200, 201):
            raise AuthenticationFailed(f"Login rejected: HTTP {response.status_code}")
        self.store.save_cookies(self.http.cookies)
        self.store.update_meta(pending_biometric=None)

    def complete_biometric(self, url: str) -> bool:
        """POST back to the Persona URL after the human finished the face check."""
        response = self.http.post(url)
        if response.status_code == 201:
            self.store.save_cookies(self.http.cookies)
            self.store.update_meta(pending_biometric=None)
            return True
        return False
