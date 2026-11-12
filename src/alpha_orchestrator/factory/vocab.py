"""Platform vocabulary, loaded from the private folder at run time.

The repository stays platform-generic: operator names, grouping identifiers and language
notes live in `<private_dir>/platform.yaml` (and the files it names), never in git.
The live operator list and signatures come from the platform's API.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VOCAB_FILE = "platform.yaml"


@dataclass(frozen=True)
class Vocabulary:
    # identifiers that name a grouping (valid group inputs; not counted as data fields)
    grouping_fields: frozenset[str] = frozenset()
    # data fields every dataset may borrow (using one breaks single-dataset purity)
    common_fields: frozenset[str] = frozenset()
    # operators the platform does not count toward the operator limit
    free_operators: frozenset[str] = frozenset()
    # prefix of operators that reduce a vector field to a number
    vector_reducer_prefix: str = ""
    # operator -> positional index of its group input
    group_input: dict[str, int] = field(default_factory=dict)
    # operators whose result is itself a grouping
    group_makers: frozenset[str] = frozenset()
    # operator-name prefixes whose integer inputs are look-back windows
    window_prefixes: tuple[str, ...] = ()
    # sign rules: result rises with the first input / falls with it / rises with every input
    increasing: frozenset[str] = frozenset()
    decreasing: frozenset[str] = frozenset()
    all_increasing: frozenset[str] = frozenset()
    # operator names that mean + - * / written as calls
    arithmetic: dict[str, str] = field(default_factory=dict)
    # conditional operators -> positional index of the value they pass through
    conditional: dict[str, int] = field(default_factory=dict)
    # whole-expression wrappers ignored when comparing shapes (cosmetic rescaling)
    cosmetic_wrappers: frozenset[str] = frozenset()
    # how a negation is written
    negate_template: str = "-({x})"
    # free text shown to the LLM about the expression language
    language_notes: str = ""

    @classmethod
    def load(cls, private_dir: str | Path) -> "Vocabulary":
        """Read the private vocabulary; an empty one when the file is missing."""
        folder = Path(private_dir).expanduser()
        path = folder / VOCAB_FILE
        if not path.is_file():
            return cls()
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        notes = ""
        for name in raw.get("language_notes_files") or []:
            note = folder / name
            if note.is_file():
                notes += note.read_text(encoding="utf-8") + "\n"

        def low(key: str) -> frozenset[str]:
            return frozenset(str(x).lower() for x in raw.get(key) or [])

        return cls(
            grouping_fields=low("grouping_fields"),
            common_fields=low("common_fields"),
            free_operators=low("free_operators"),
            vector_reducer_prefix=str(raw.get("vector_reducer_prefix") or ""),
            group_input={str(k).lower(): int(v) for k, v in (raw.get("group_input") or {}).items()},
            group_makers=low("group_makers"),
            window_prefixes=tuple(str(x).lower() for x in raw.get("window_prefixes") or []),
            increasing=low("increasing"),
            decreasing=low("decreasing"),
            all_increasing=low("all_increasing"),
            arithmetic={str(k).lower(): str(v) for k, v in (raw.get("arithmetic") or {}).items()},
            conditional={str(k).lower(): int(v) for k, v in (raw.get("conditional") or {}).items()},
            cosmetic_wrappers=low("cosmetic_wrappers"),
            negate_template=str(raw.get("negate_template") or "-({x})"),
            language_notes=notes.strip(),
        )
