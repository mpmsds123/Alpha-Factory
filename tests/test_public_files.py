"""Nothing platform-specific may be committed: no operator names, dataset ids, field
names or expressions in any tracked file, and only README and CHANGELOG as Markdown.

On a machine with the private vocabulary and the factory databases, every tracked file
is scanned for their operator names and dataset/field ids; elsewhere (CI) only the
generic checks run.
"""

import os
import re
import sqlite3
import subprocess
from pathlib import Path

import pytest

from alpha_orchestrator.factory.vocab import Vocabulary

ROOT = Path(__file__).parents[1]
PRIVATE = Path(os.getenv("ALPHA_FACTORY_PRIVATE_DIR", "~/.alpha_factory/private")).expanduser()
DATABASES = [
    Path(p).expanduser() for p in ("~/.alpha_factory/factory.db", "~/.alpha_factory/factory_v8.db")
]
PLATFORM_NAME = re.compile(r"WorldQuant|worldquant|(?<![A-Z_])BRAIN(?![A-Z_])")
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
THIS = Path(__file__).resolve()


def tracked() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return [ROOT / p for p in out if (ROOT / p).is_file() and (ROOT / p).resolve() != THIS]


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""


def test_only_readme_and_changelog_are_markdown() -> None:
    assert sorted(p.name for p in tracked() if p.suffix == ".md") == ["CHANGELOG.md", "README.md"]


def test_no_platform_name_in_tracked_files() -> None:
    bad = [str(p.relative_to(ROOT)) for p in tracked() if PLATFORM_NAME.search(_text(p))]
    assert not bad, bad


def _private_names() -> set[str]:
    v = Vocabulary.load(PRIVATE)
    names: set[str] = set()
    for group in (
        v.increasing,
        v.decreasing,
        v.all_increasing,
        v.free_operators,
        v.group_makers,
        v.cosmetic_wrappers,
    ):
        names |= set(group)
    names |= set(v.group_input) | set(v.conditional)
    # common English words (rank, sign, log, ...) are not evidence; compound names are
    return {n for n in names if "_" in n}


def _private_ids() -> set[str]:
    ids: set[str] = set()
    for db in DATABASES:
        if not db.is_file():
            continue
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            for table, cols in (
                ("cells", ("dataset_id",)),
                ("fields", ("field_id",)),
                ("catalog", ("dataset_id", "field_id")),
            ):
                try:
                    for col in cols:
                        ids |= {r[0] for r in con.execute(f"select distinct {col} from {table}")}
                except sqlite3.Error:
                    continue
        finally:
            con.close()
    return {i for i in ids if i and len(i) >= 5 and re.search(r"[_\d]", i)}


def test_no_private_operator_dataset_or_field_names() -> None:
    names = _private_names() | _private_ids()
    if not names:
        pytest.skip("no private vocabulary or databases on this machine")
    hits = {}
    for p in tracked():
        found = set(WORD.findall(_text(p))) & names
        if found:
            hits[str(p.relative_to(ROOT))] = sorted(found)[:5]
    assert not hits, hits
