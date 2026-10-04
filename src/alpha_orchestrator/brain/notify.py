"""Desktop notifications so an unattended run can call a human back."""

import subprocess
import sys


def notify(message: str, title: str = "Alpha 工厂") -> None:
    print(f"[{title}] {message}", file=sys.stderr, flush=True)
    if sys.platform != "darwin":
        return
    script = f"display notification {_quote(message)} with title {_quote(title)}"
    try:
        subprocess.run(["osascript", "-e", script], timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
