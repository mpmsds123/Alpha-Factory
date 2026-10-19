"""The idea library: papers and forum posts the LLM must ground every hypothesis in.

Imported once into the ledger (so it outlives the source folders), retrieved with BM25.
Tokens are English words plus Chinese character bigrams, which needs no segmenter and
works for mixed-language posts. Official research-paper posts are boosted: they often
name the exact BRAIN datasets and fields that implement the paper.
"""

import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha_orchestrator.factory.models import IdeaORM

WORD = re.compile(r"[a-z][a-z0-9_]{1,}")
CJK = re.compile(r"[一-鿿]+")
STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "are",
        "from",
        "alpha",
        "alphas",
        "data",
        "brain",
        "worldquant",
        "https",
        "com",
        "www",
        "support",
    }
)
BOOST = {
    "Research_Papers_for_Consultants": 1.6,
    "Research_Papers_for_Users": 1.5,
    "BRAIN_TIPS": 1.2,
}
MAX_BODY = 6000


def tokens(text: str) -> list[str]:
    low = text.lower()
    out = [w for w in WORD.findall(low) if w not in STOP]
    for run in CJK.findall(text):
        out += [run[i : i + 2] for i in range(len(run) - 1)] or [run]
    return out


@dataclass
class Doc:
    source: str
    title: str
    url: str
    body: str


def parse_post_file(path: Path, source: str) -> Doc | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    title = next((ln[2:].strip() for ln in lines if ln.startswith("# ")), path.stem)
    url = next((ln.split(":", 1)[1].strip() for ln in lines if ln.strip().startswith("- 来源")), "")
    body = text.split("---", 1)[1] if "---" in text else text
    body = body.strip()
    return Doc(source, title, url, body[:MAX_BODY]) if len(body) > 80 else None


def parse_compilation(path: Path, source: str) -> Iterator[Doc]:
    """One big markdown file of '# title' sections (the forum scrape in Downloads)."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    for chunk in re.split(r"\n(?=# )", text):
        head, _, rest = chunk.partition("\n")
        title = head.lstrip("# ").strip()
        url_match = re.search(r"https://support\.worldquantbrain\.com\S+", rest)
        body = re.sub(r"=+", "", rest).strip()
        if title and len(body) > 120 and not title.startswith("WQ "):
            yield Doc(source, title[:500], url_match.group(0) if url_match else "", body[:MAX_BODY])


HOMEWORK_POST = re.compile(r"^--- #(\d+)\s.*---$", re.M)
COURSE = re.compile(r"^# (.*Course\s*\d+.*)$", re.M)
HAS_EXPRESSION = re.compile(r"\b(?:ts_\w+|group_\w+|rank|trade_when|vec_\w+)\s*\(")


def parse_homework(path: Path, source: str) -> Iterator[Doc]:
    """The official homework thread: every student answer is its own document, titled by
    its course, because most answers are dataset-specific templates with a rationale."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    courses = [(m.start(), m.group(1).strip()) for m in COURSE.finditer(text)]
    posts = list(HOMEWORK_POST.finditer(text))
    for i, m in enumerate(posts):
        end = posts[i + 1].start() if i + 1 < len(posts) else len(text)
        body = text[m.end() : end]
        body = body.split("\n# ", 1)[0].strip()  # stop at the next course heading
        course = next((name for pos, name in reversed(courses) if pos < m.start()), "作业")
        # Course 1 answers are platform-basics quizzes; templates start with course 2.
        if re.search(r"Course\s*1\D", course + " ") or not HAS_EXPRESSION.search(body):
            continue
        if len(body) > 80:
            yield Doc(source, f"{course[:200]} · 作业 #{m.group(1)}", "", body[:MAX_BODY])


def collect(paths: Iterable[Path]) -> Iterator[Doc]:
    for path in paths:
        path = path.expanduser()
        if path.is_file():
            head = path.read_text(encoding="utf-8", errors="ignore")[:200_000]
            if HOMEWORK_POST.search(head):
                yield from parse_homework(path, "作业区")
            else:
                yield from parse_compilation(path, path.stem)
        elif path.is_dir():
            for f in sorted(path.rglob("*.md")):
                doc = parse_post_file(f, f.parent.name)
                if doc:
                    yield doc


def import_ideas(s: Session, paths: Iterable[Path]) -> int:
    existing = {(src, t) for src, t in s.execute(select(IdeaORM.source, IdeaORM.title))}
    added = 0
    for doc in collect(paths):
        if (doc.source, doc.title) in existing:
            continue
        s.add(IdeaORM(source=doc.source, title=doc.title, url=doc.url, body=doc.body))
        existing.add((doc.source, doc.title))
        added += 1
    return added


class IdeaIndex:
    """BM25 over title (weighted x3) + body."""

    def __init__(self, docs: list[IdeaORM], k1: float = 1.4, b: float = 0.75) -> None:
        self.docs = docs
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(d.title) * 3 + tokens(d.body)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / len(self.len) if self.len else 1.0
        df: Counter[str] = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def load(cls, s: Session) -> "IdeaIndex":
        return cls(list(s.scalars(select(IdeaORM))))

    def search(self, query: str, k: int = 6) -> list[IdeaORM]:
        q = Counter(tokens(query))
        scores = []
        for i, tf in enumerate(self.tf):
            score = 0.0
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                norm = (
                    f
                    * (self.k1 + 1)
                    / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
                )
                score += self.idf.get(term, 0.0) * norm
            if score:
                scores.append((score * BOOST.get(self.docs[i].source, 1.0), i))
        scores.sort(reverse=True)
        return [self.docs[i] for _, i in scores[:k]]
