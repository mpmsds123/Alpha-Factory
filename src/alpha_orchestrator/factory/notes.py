"""The factory's three memory files, in the private memory folder.

AlphaForge keeps what its agents must know in three small files instead of one growing
document (its rule file silently stopped being read past ~24 KB):
  RULES.md     short, standing rules every LLM call sees; capped in size
  STATUS.md    rewritten at the end of every run: today's slots, counts, why it stopped
  LESSONS.md   numbered lessons, appended only when a trigger fires (below), never
               twice for the same key; the latest few are shown to every LLM call

Lesson triggers (AlphaForge's rule L14, adapted):
  rate_jump     the round-1 pass rate moved by more than 30% against the past 7 days
  wasted        2 or more batches in a run came back with every simulation failed
  first_pass    the first stored alpha of a slot or a region
  quota         a run stopped on the platform's quota or refusals
  crowded       a field was added to the crowded list after a correlation failure
The Memory B records (one per hypothesis) stay in the database as before.
"""

import re
from datetime import date
from pathlib import Path

RULES = "RULES.md"
STATUS = "STATUS.md"
LESSONS = "LESSONS.md"
RULES_MAX_CHARS = 4000  # every LLM call carries it; keep it short
LESSONS_SHOWN = 8
RATE_JUMP = 0.30

DEFAULT_RULES = """# Rules (shown to every LLM call; keep under 4,000 characters)

1. One hypothesis, one mechanism. Every expression must follow from the claim.
2. Respect the slot: its operator and field limits are hard limits.
3. A slot outside the pool must use more than the pool's operators or fields.
4. Never use a field on the crowded list, except the hypothesis field itself.
5. When a fix is given, make exactly that change and nothing else.
6. After a correlation failure, keep the claim and change the structure; do not
   re-scale or re-window the failed expression.
7. Prefer fewer, clearer expressions over many near-copies.
"""


def folder(memory_dir: str) -> Path:
    path = Path(memory_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def rules(memory_dir: str) -> str:
    """The rules text (created from the default on first use), capped in size."""
    path = folder(memory_dir) / RULES
    if not path.is_file():
        path.write_text(DEFAULT_RULES, encoding="utf-8")
    return path.read_text(encoding="utf-8")[:RULES_MAX_CHARS]


def write_status(memory_dir: str, lines: list[str]) -> Path:
    path = folder(memory_dir) / STATUS
    body = f"# Status, {date.today().isoformat()}\n\n" + "\n".join(f"- {x}" for x in lines)
    path.write_text(body + "\n", encoding="utf-8")
    return path


_ENTRY = re.compile(r"^## L(\d+) \[([^\]]+)\]", re.M)


def _entries(text: str) -> list[tuple[int, str]]:
    return [(int(m.group(1)), m.group(2)) for m in _ENTRY.finditer(text)]


def add_lesson(memory_dir: str, trigger: str, key: str, text: str) -> int | None:
    """Append a numbered lesson unless one with this trigger and key exists. Returns its
    number, or None when it was a repeat."""
    path = folder(memory_dir) / LESSONS
    old = path.read_text(encoding="utf-8") if path.is_file() else "# Lessons\n"
    tag = f"{trigger}:{key}"
    entries = _entries(old)
    if any(t == tag for _, t in entries):
        return None
    n = max((i for i, _ in entries), default=0) + 1
    entry = f"\n## L{n} [{tag}] {date.today().isoformat()}\n{text.strip()}\n"
    path.write_text(old.rstrip("\n") + "\n" + entry, encoding="utf-8")
    return n


def recent_lessons(memory_dir: str, k: int = LESSONS_SHOWN) -> list[str]:
    """The last k lessons, one line each, for the LLM."""
    path = Path(memory_dir).expanduser() / LESSONS
    if not path.is_file():
        return []
    blocks = re.split(r"\n(?=## L\d+ )", path.read_text(encoding="utf-8"))
    out = []
    for b in blocks:
        if not b.startswith("## L"):
            continue
        head, _, body = b.partition("\n")
        number = head.split()[1]
        out.append(f"{number}: {' '.join(body.split())}"[:300])
    return out[-k:]


def rate_jumped(today: float, past: float) -> bool:
    """True when today's rate differs from the past rate by more than 30% of it."""
    if past <= 0:
        return False
    return abs(today - past) / past > RATE_JUMP
