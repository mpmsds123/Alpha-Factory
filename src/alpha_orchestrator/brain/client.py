"""One request path for every BRAIN call: waits, back-off, quota bookkeeping.

BRAIN answers "still computing" with 200 + `Retry-After`; it throttles with 429; and it
sometimes drops connections outright. Each is handled here once, so callers only ever see
a finished response, `RateLimited` after patience runs out, or a session problem.
"""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from alpha_orchestrator.brain.session import BrainSession


class RateLimited(Exception):
    """The platform kept answering 429 past the back-off budget."""


class SessionExpired(Exception):
    """A request came back 401 after the session's one allowed re-login."""


@dataclass
class Quota:
    limit: int | None = None
    remaining: int | None = None
    reset_seconds: int | None = None

    def update(self, headers: httpx.Headers) -> None:
        for attr, name in (
            ("limit", "X-Ratelimit-Limit"),
            ("remaining", "X-Ratelimit-Remaining"),
            ("reset_seconds", "X-Ratelimit-Reset"),
        ):
            value = headers.get(name)
            if value is not None and value.strip().lstrip("-").isdigit():
                setattr(self, attr, int(value))


class BrainClient:
    def __init__(
        self,
        session: BrainSession,
        *,
        sleep: Callable[[float], None] = time.sleep,
        max_attempts: int = 8,
    ) -> None:
        self.session = session
        self.sleep = sleep
        self.max_attempts = max_attempts
        self.quota = Quota()

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        relogged = False
        for attempt in range(self.max_attempts):
            try:
                response = self.session.http.request(method, path, **kwargs)
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.RemoteProtocolError,
                httpx.TimeoutException,
            ):
                self.sleep(min(5.0 * 2**attempt, 60.0))
                continue
            self.quota.update(response.headers)
            if response.status_code == 401:
                if relogged:
                    raise SessionExpired(f"{method} {path} rejected after re-login")
                self.session.login()
                relogged = True
                continue
            if response.status_code == 429:
                wait = _retry_after(response) or min(30.0 * (attempt + 1), 300.0)
                self.sleep(wait + random.uniform(0, 3))
                continue
            return response
        if method.upper() == "GET":
            raise RateLimited(f"{method} {path} gave up after {self.max_attempts} attempts")
        raise RateLimited(f"{method} {path} kept failing")

    def wait_json(self, path: str, *, max_wait_seconds: float = 1800.0) -> Any:
        """GET until the platform stops asking us to come back; None for empty/4xx."""
        waited = 0.0
        while True:
            response = self.request("GET", path)
            retry = _retry_after(response)
            if retry and waited < max_wait_seconds:
                self.sleep(retry)
                waited += retry
                continue
            if response.status_code >= 400 or not response.content.strip():
                return None
            return response.json()


def _retry_after(response: httpx.Response) -> float:
    try:
        return float(response.headers.get("Retry-After", 0) or 0)
    except ValueError:
        return 0.0
