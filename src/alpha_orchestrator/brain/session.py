"""Platform authentication: password login, biometric hand-off, cookie cache.

The platform asks for a face check (Persona) every few hours. Software cannot and must
not complete it, so an unattended run that hits one stops with `NeedBiometric`, carrying
the URL a human opens. `alpha-factory login` is the interactive path that finishes it.

Re-authentication is rationed: at most one password login per `RELOGIN_COOLDOWN_SECONDS`.
Repeated forced logins have locked this account before.

Shared login: other programs on the same account (the template scanner) can share one
session. `BRAIN_SHARED_COOKIES` lists their cookie files (os.pathsep-separated, same
JSON shape). Before a password login or a face-check stop, a still-valid shared cookie is
adopted; every cookie this session saves is copied to those files. One face check then
unlocks every program.
"""

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

URL_ENV = "PLATFORM_API_URL"
PRIVATE_DIR_ENV = "ALPHA_FACTORY_PRIVATE_DIR"
DEFAULT_PRIVATE_DIR = Path("~/.alpha_factory/private")
DEFAULT_CREDENTIALS = Path("~/secrets/platform-brain.json")
USERNAME_ENV = "ALPHA_LAB_BRAIN_USERNAME"
PASSWORD_ENV = "ALPHA_LAB_BRAIN_PASSWORD"
DEFAULT_STATE_DIR = Path("~/.alpha_factory")
RELOGIN_COOLDOWN_SECONDS = 1800.0
SHARED_COOKIES_ENV = "BRAIN_SHARED_COOKIES"


def platform_url() -> str:
    """The platform's API address: PLATFORM_API_URL, else `api_url` in the private
    platform.yaml. It is not written into the repository."""
    url = os.getenv(URL_ENV, "")
    if not url:
        folder = Path(os.getenv(PRIVATE_DIR_ENV, str(DEFAULT_PRIVATE_DIR))).expanduser()
        path = folder / "platform.yaml"
        if path.is_file():
            import yaml

            url = str((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("api_url") or "")
    if not url:
        raise AuthenticationFailed(f"Set {URL_ENV} or api_url in the private platform.yaml")
    return url.rstrip("/")


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
        """Load credentials from the environment, else from a private JSON file.

        ALPHA_LAB_BRAIN_USERNAME and ALPHA_LAB_BRAIN_PASSWORD win when both are set and
        no explicit path is given; otherwise the file at ``path``, BRAIN_CREDENTIALS_FILE
        or ~/secrets/platform-brain.json is read.
        """
        user, password = os.getenv(USERNAME_ENV, ""), os.getenv(PASSWORD_ENV, "")
        if path is None and user and password:
            return cls(email=user, password=password)
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


def _write_private_json(path: Path, payload: Any) -> None:
    """Sync staged private JSON before atomically replacing the destination."""
    encoded = json.dumps(payload).encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


class SessionStore:
    """Cookies and login bookkeeping kept outside the repository, readable only by you."""

    def __init__(self, state_dir: Path | None = None, shared: list[Path] | None = None) -> None:
        self.dir = (
            state_dir or Path(os.getenv("BRAIN_STATE_DIR", str(DEFAULT_STATE_DIR)))
        ).expanduser()
        self.cookies_path = self.dir / "brain_cookies.json"
        self.meta_path = self.dir / "brain_session.json"
        if shared is None:
            raw = os.getenv(SHARED_COOKIES_ENV, "")
            shared = [Path(p) for p in raw.split(os.pathsep) if p.strip()]
        self.shared = [p.expanduser() for p in shared]

    def load_cookies(self, path: Path | None = None) -> httpx.Cookies:
        """Return the saved cookies (or those in `path`) as a cookie jar (empty if none)."""
        jar = httpx.Cookies()
        path = path or self.cookies_path
        if not path.exists():
            return jar
        data = json.loads(path.read_text())
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
        _write_private_json(self.cookies_path, payload)
        for other in self.shared:  # best effort: the other program keeps working either way
            try:
                if other.parent.is_dir():
                    _write_private_json(other, payload)
            except OSError:
                pass

    def meta(self) -> dict[str, Any]:
        """Return the saved session metadata (empty if none saved)."""
        if not self.meta_path.exists():
            return {}
        return dict(json.loads(self.meta_path.read_text()))

    def update_meta(self, **values: Any) -> None:
        """Merge values into the saved session metadata."""
        self.dir.mkdir(parents=True, exist_ok=True)
        merged = {**self.meta(), **values}
        _write_private_json(self.meta_path, merged)


class BrainSession:
    """Owns the HTTP client and keeps it authenticated."""

    def __init__(
        self,
        credentials: Credentials | None = None,
        store: SessionStore | None = None,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        clock: Any = time.time,
    ) -> None:
        self._credentials = credentials
        self.store = store or SessionStore()
        base_url = base_url or platform_url()
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
        """Return the login credentials, loading them on first use."""
        if self._credentials is None:
            self._credentials = Credentials.load()
        return self._credentials

    def close(self) -> None:
        """Close the underlying HTTP client."""
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

    def adopt_shared(self, minimum_seconds: float = 600.0) -> bool:
        """Use another program's still-valid session (see the module note). On success the
        cookies are saved here and any pending face check is cleared."""
        own = httpx.Cookies(self.http.cookies)
        for path in self.store.shared:
            try:
                jar = self.store.load_cookies(path)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if not len(jar):
                continue
            self.http.cookies = jar
            if self.seconds_remaining() >= minimum_seconds:
                self.store.save_cookies(self.http.cookies)
                self.store.update_meta(pending_biometric=None)
                return True
        self.http.cookies = own
        return False

    def ensure(self, minimum_seconds: float = 600.0, *, interactive: bool = False) -> float:
        """Return remaining seconds, logging in with the password when needed."""
        if not (self.store.meta().get("pending_biometric") and self.adopt_shared(minimum_seconds)):
            self._require_human_if_pending(interactive)
        remaining = self.seconds_remaining()
        if remaining >= minimum_seconds:
            return remaining
        if self.adopt_shared(minimum_seconds):
            return self.seconds_remaining()
        self.login(interactive=interactive)
        return self.seconds_remaining()

    def login(self, *, interactive: bool = False) -> None:
        """The cooldown guards unattended runs from hammering the password; a person at
        the keyboard running `alpha-factory login` is not held back by it. A valid shared
        session (another program already logged in) is used instead of a password login."""
        if not interactive and self.adopt_shared():
            return
        self._require_human_if_pending(interactive)
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

    def _require_human_if_pending(self, interactive: bool) -> None:
        pending = self.store.meta().get("pending_biometric")
        if pending and not interactive:
            raise NeedBiometric(str(pending))

    def complete_biometric(self, url: str) -> bool:
        """POST back to the Persona URL after the human finished the face check."""
        response = self.http.post(url)
        if response.status_code == 201:
            self.store.save_cookies(self.http.cookies)
            self.store.update_meta(pending_biometric=None)
            return True
        return False
