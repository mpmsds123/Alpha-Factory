"""Desktop notifications so an unattended run can call a human back."""

import contextlib
import logging
import subprocess
import sys

# The notification is always echoed to stderr, even when the caller has not configured
# logging, so this logger carries its own handler and does not propagate.
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# osascript normally returns in well under a second; 10 s stops a hung notification
# centre from blocking the run.
OSASCRIPT_TIMEOUT_SECONDS = 10


def notify(message: str, title: str = "Alpha 工厂") -> None:
    """Log the message to stderr and, on macOS, show it as a desktop notification."""
    logger.info("[%s] %s", title, message)
    if sys.platform != "darwin":
        return
    script = f"display notification {_quote(message)} with title {_quote(title)}"
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        subprocess.run(["osascript", "-e", script], timeout=OSASCRIPT_TIMEOUT_SECONDS, check=False)


def _quote(text: str) -> str:
    """Quote text as an AppleScript string literal."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
