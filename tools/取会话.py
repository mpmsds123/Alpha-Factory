"""刷新 BRAIN 会话 cookie 文件，供 ALPHA_LAB_BRAIN_SESSION_FILE 使用。

这个脚本本身进 git（.gitignore 里只单独排除了同目录的 tools/brain_credentials.txt
和生成出来的会话文件本身，不是整个 tools/）。核心包
（src/alpha_orchestrator）故意不碰账号密码，只吃一个别人已经登录好的 cookie
文件——这个脚本就是"别人"：用你的账号密码换 cookie，同时正确处理 BRAIN 的
persona 生物识别挑战，而不是让你每次手动从浏览器里抠 cookie。

逻辑抄的是本地 IQC 项目 wq/auth.py 里跑得通的那一套（POST /authentication ->
401 + WWW-Authenticate: persona -> 打开 Location 链接人工过一次验证 -> 再 POST
Location 拿到 201），只是换成本项目已有的 httpx 依赖，并且把结果写成
load_session_cookies() 认的格式。

用法:
    python tools/取会话.py                       # 输出到 ~/brain-session.json
    python tools/取会话.py --out /path/to/file.json

凭证文件: tools/brain_credentials.txt，JSON 数组 ["user", "password"]（同
IQC 的格式，可以直接从那边复制一份过来）。这个文件同样不会进 git。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import stat
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import urljoin

import httpx2 as httpx

API_BASE_URL = "https://api.worldquantbrain.com"
AUTHENTICATION_PATH = "/authentication"
TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_CREDENTIALS_PATH = TOOLS_DIR / "brain_credentials.txt"
DEFAULT_OUTPUT_PATH = Path.home() / "brain-session.json"

# 429 时 BRAIN 有两种截然不同的含义，必须分开处理：
# - 普通限流（比如刚抓了很多字段）: 退避重试就行。
# - BIOMETRICS_THROTTLED: 是生物识别发起频率被限，不是账号密码问题，重试没用，
#   要么等 1-2 小时窗口过去，要么先去浏览器手动过一次人脸验证。
BIOMETRICS_THROTTLED_MARKER = "BIOMETRICS_THROTTLED"
AUTHENTICATION_RETRY_ATTEMPTS = 4


def load_credentials(path: Path) -> tuple[str, str]:
    if not path.is_file():
        raise SystemExit(
            f"没找到凭证文件: {path}\n"
            '请创建它，内容是 JSON 数组，例如: ["your_email@example.com", "your_password"]'
        )
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise SystemExit(f"凭证文件不能让同组或其他用户读取: chmod 600 {path}")
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, list) or len(payload) != 2:
        raise SystemExit(f"凭证文件格式不对，应为 [用户名, 密码] 的 JSON 数组: {path}")
    user, password = payload
    if not isinstance(user, str) or not isinstance(password, str) or not user or not password:
        raise SystemExit(f"凭证文件里的用户名/密码必须是非空字符串: {path}")
    return user, password


def authenticate(client: httpx.Client, user: str, password: str) -> None:
    response: httpx.Response | None = None
    for attempt in range(AUTHENTICATION_RETRY_ATTEMPTS):
        response = client.post(AUTHENTICATION_PATH, auth=(user, password))
        if response.status_code != 429:
            break
        pause = 60.0 * (attempt + 1)
        if BIOMETRICS_THROTTLED_MARKER in response.text:
            raise SystemExit(
                "生物识别被限流 (BIOMETRICS_THROTTLED)，不是账号密码错误。\n"
                "去 https://platform.worldquantbrain.com 用浏览器登录并手动完成一次人脸验证，\n"
                "或者等 1-2 小时限流窗口过去再跑这个脚本。"
            )
        print(f"  [429] 限流，{pause:.0f} 秒后重试第 {attempt + 2} 次…", file=sys.stderr)
        time.sleep(pause)

    assert response is not None
    if response.status_code in (200, 201):
        return

    challenge_scheme = response.headers.get("WWW-Authenticate", "").casefold()
    if response.status_code == 401 and challenge_scheme.startswith("persona"):
        location = response.headers.get("Location")
        if not location:
            raise SystemExit("BRAIN 要求 persona 验证但响应里没有 Location，无法继续。")
        challenge_url = urljoin(str(response.url), location)
        print("需要生物识别验证，正在打开浏览器完成一次人脸验证：")
        print(f"  {challenge_url}")
        with contextlib.suppress(Exception):
            webbrowser.open(challenge_url)
        input("在浏览器里完成验证后，回来按 Enter 继续…")
        challenge_response = client.post(challenge_url)
        while challenge_response.status_code != 201:
            input("还没通过，或者验证链接已过期。重新验证后按 Enter 再试一次…")
            challenge_response = client.post(challenge_url)
        return

    raise SystemExit(f"认证失败 (HTTP {response.status_code}): {response.text[:300]}")


def write_session_file(client: httpx.Client, output_path: Path) -> None:
    cookies = [
        {"name": name, "value": value} for name, value in client.cookies.items() if name and value
    ]
    if not cookies:
        raise SystemExit("认证看起来成功了，但没有拿到任何 cookie，无法写出会话文件。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cookies, ensure_ascii=False, indent=2), encoding="utf-8")
    output_path.chmod(0o600)
    print(f"已写入新的会话文件: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--credentials",
        type=Path,
        default=DEFAULT_CREDENTIALS_PATH,
        help=f"凭证文件路径 (默认: {DEFAULT_CREDENTIALS_PATH})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"输出的会话 cookie 文件路径 (默认: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    user, password = load_credentials(args.credentials)
    with httpx.Client(base_url=API_BASE_URL, timeout=60.0, follow_redirects=False) as client:
        authenticate(client, user, password)
        write_session_file(client, args.out)


if __name__ == "__main__":
    main()
