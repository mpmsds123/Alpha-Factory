"""Refresh the platform session cookie file used by ALPHA_LAB_BRAIN_SESSION_FILE.

The core package never handles account passwords; it only reads a cookie file that
someone has already produced. This script is that someone: it exchanges your
credentials for session cookies, handling the platform's biometric (persona)
challenge by opening the verification link in a browser and waiting for you.

Flow: POST /authentication -> 401 + ``WWW-Authenticate: persona`` -> complete the
challenge in a browser -> POST the challenge Location -> 201 -> write cookies.

Usage:
    python scripts/refresh_session.py                    # writes ~/brain-session.json
    python scripts/refresh_session.py --out /path/to/file.json

Credentials come from ALPHA_LAB_BRAIN_USERNAME / ALPHA_LAB_BRAIN_PASSWORD when both are
set, otherwise from scripts/brain_credentials.txt: a JSON array ``["user", "password"]``
readable only by you (chmod 600). That file is git-ignored.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import stat
import time
import webbrowser
from pathlib import Path
from urllib.parse import urljoin

import httpx2 as httpx

logger = logging.getLogger("refresh_session")

API_BASE_URL = "https://api.worldquantbrain.com"
AUTHENTICATION_PATH = "/authentication"
SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_CREDENTIALS_PATH = SCRIPTS_DIR / "brain_credentials.txt"
DEFAULT_OUTPUT_PATH = Path.home() / "brain-session.json"
USERNAME_ENV = "ALPHA_LAB_BRAIN_USERNAME"
PASSWORD_ENV = "ALPHA_LAB_BRAIN_PASSWORD"

# A 429 from the platform means one of two very different things:
# - ordinary rate limiting (e.g. after many metadata calls): back off and retry;
# - BIOMETRICS_THROTTLED: the biometric challenge itself is rate limited. Retrying
#   does not help; wait out the 1-2 hour window or verify once in a browser first.
BIOMETRICS_THROTTLED_MARKER = "BIOMETRICS_THROTTLED"
# Four attempts with a linearly growing pause (60 s, 120 s, 180 s) covers the
# platform's usual short rate-limit window without hammering it.
AUTHENTICATION_RETRY_ATTEMPTS = 4
RETRY_PAUSE_STEP_SECONDS = 60.0
# Generous enough for a slow authentication round trip; the call is made once.
HTTP_TIMEOUT_SECONDS = 60.0
# Only the start of an error body is shown; it is enough to identify the problem.
ERROR_BODY_PREVIEW_CHARS = 300


def load_credentials(path: Path) -> tuple[str, str]:
    """Return (user, password) from the environment, else from a private JSON file."""
    user, password = os.environ.get(USERNAME_ENV, ""), os.environ.get(PASSWORD_ENV, "")
    if user and password:
        return user, password
    if not path.is_file():
        raise SystemExit(
            f"Credentials file not found: {path}\n"
            f"Set {USERNAME_ENV}/{PASSWORD_ENV}, or create the file containing a JSON array, "
            'e.g. ["your_email@example.com", "your_password"]'
        )
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise SystemExit(f"Credentials file must not be group/world readable: chmod 600 {path}")
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, list) or len(payload) != 2:
        raise SystemExit(f"Credentials file must be a JSON array [user, password]: {path}")
    user, password = payload
    if not isinstance(user, str) or not isinstance(password, str) or not user or not password:
        raise SystemExit(f"User and password in the credentials file must be non-empty: {path}")
    return user, password


def authenticate(client: httpx.Client, user: str, password: str) -> None:
    """Authenticate the client, walking the user through a biometric challenge if needed."""
    response: httpx.Response | None = None
    for attempt in range(AUTHENTICATION_RETRY_ATTEMPTS):
        response = client.post(AUTHENTICATION_PATH, auth=(user, password))
        if response.status_code != 429:
            break
        pause = RETRY_PAUSE_STEP_SECONDS * (attempt + 1)
        if BIOMETRICS_THROTTLED_MARKER in response.text:
            raise SystemExit(
                "Biometric verification is rate limited (BIOMETRICS_THROTTLED); the "
                "credentials are fine.\nLog in once in a browser and complete the face "
                "check, or wait 1-2 hours before running this script again."
            )
        logger.warning("[429] rate limited; retry %d in %.0f s", attempt + 2, pause)
        time.sleep(pause)

    assert response is not None
    if response.status_code in (200, 201):
        return

    challenge_scheme = response.headers.get("WWW-Authenticate", "").casefold()
    if response.status_code == 401 and challenge_scheme.startswith("persona"):
        location = response.headers.get("Location")
        if not location:
            raise SystemExit("Persona verification requested but no Location header was sent.")
        challenge_url = urljoin(str(response.url), location)
        logger.info("Biometric verification required; opening the browser:\n  %s", challenge_url)
        with contextlib.suppress(Exception):
            webbrowser.open(challenge_url)
        input("Finish the verification in the browser, then press Enter...")
        challenge_response = client.post(challenge_url)
        while challenge_response.status_code != 201:
            input("Not verified yet, or the link expired. Verify again, then press Enter...")
            challenge_response = client.post(challenge_url)
        return

    raise SystemExit(
        f"Authentication failed (HTTP {response.status_code}): "
        f"{response.text[:ERROR_BODY_PREVIEW_CHARS]}"
    )


def write_session_file(client: httpx.Client, output_path: Path) -> None:
    """Write the client's cookies as the JSON list the session loader expects (mode 600)."""
    cookies = [
        {"name": name, "value": value} for name, value in client.cookies.items() if name and value
    ]
    if not cookies:
        raise SystemExit("Authentication appeared to succeed but returned no cookies.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cookies, ensure_ascii=False, indent=2), encoding="utf-8")
    output_path.chmod(0o600)
    logger.info("Wrote new session file: %s", output_path)


def main() -> None:
    """Parse arguments, authenticate and write the session file."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--credentials",
        type=Path,
        default=DEFAULT_CREDENTIALS_PATH,
        help=f"credentials file (default: {DEFAULT_CREDENTIALS_PATH})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"session cookie file to write (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    user, password = load_credentials(args.credentials)
    with httpx.Client(
        base_url=API_BASE_URL, timeout=HTTP_TIMEOUT_SECONDS, follow_redirects=False
    ) as client:
        authenticate(client, user, password)
        write_session_file(client, args.out)


if __name__ == "__main__":
    main()
