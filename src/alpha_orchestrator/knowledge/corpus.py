"""Traceable corpus inventory and explicit draft-card extraction."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from alpha_orchestrator.domain.enums import EvidenceLevel, MethodCardStatus
from alpha_orchestrator.domain.identity import sha256_text
from alpha_orchestrator.domain.knowledge import (
    MethodCard,
    SourceDocument,
    SourceReference,
    SourceSection,
)
from alpha_orchestrator.storage.repository import Repository

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
CARD_BLOCK_RE = re.compile(r"```method-card\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
SENSITIVE_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|authorization|password|session[_-]?token|access[_-]?token|"
    r"account[_-]?id)\b\s*[:=]\s*\S+"
)
COMPLETE_ALPHA_RE = re.compile(
    r"(?ims)(?:```(?:alpha|fast[-_ ]?expr|worldquant[-_ ]?alpha)\b.*?```)"
    r"|(?:^\s*(?:alpha[_-]?)?expression\s*[:=]\s*\S.+$)"
)


@dataclass(frozen=True)
class InventoryResult:
    inventoried: int
    excluded: int
    source_ids: tuple[str, ...]


class CorpusInventory:
    def __init__(self, repository: Repository) -> None:
        self.repository = repository

    def inventory(self, root: Path, source_type: EvidenceLevel) -> InventoryResult:
        resolved_root = root.expanduser().resolve()
        if not resolved_root.is_dir():
            raise ValueError(f"Corpus path is not a directory: {root}")
        files = sorted(
            {
                *resolved_root.rglob("*.md"),
                *resolved_root.rglob("*.markdown"),
            }
        )
        source_ids: list[str] = []
        excluded_count = 0
        for path in files:
            resolved_path = path.resolve()
            if not resolved_path.is_relative_to(resolved_root) or not resolved_path.is_file():
                continue
            relative = resolved_path.relative_to(resolved_root).as_posix()
            text = resolved_path.read_text(encoding="utf-8")
            source_id = (
                "SRC-" + sha256_text(f"{source_type}:{resolved_root.as_posix()}:{relative}")[:20]
            )
            reason = None
            if SENSITIVE_RE.search(text):
                reason = "CREDENTIAL_OR_ACCOUNT_DATA_DETECTED"
            elif COMPLETE_ALPHA_RE.search(text):
                reason = "COMPLETE_ALPHA_EXPRESSION_DETECTED"
            document = SourceDocument(
                source_id=source_id,
                corpus_root_locator=str(resolved_root),
                relative_locator=relative,
                title=_document_title(text, resolved_path.stem),
                source_type=source_type,
                content_hash=sha256_text(text),
                excluded=reason is not None,
                exclusion_reason=reason,
                sections=_section_inventory(source_id, text),
            )
            self.repository.upsert_source_document(document)
            source_ids.append(source_id)
            excluded_count += int(reason is not None)
        return InventoryResult(
            inventoried=len(source_ids),
            excluded=excluded_count,
            source_ids=tuple(source_ids),
        )

    def extract_draft_cards(self, source_id: str) -> list[MethodCard]:
        document = self.repository.get_source_document(source_id)
        if document.excluded:
            raise ValueError(f"Source is excluded: {document.exclusion_reason}")
        path = Path(document.corpus_root_locator) / document.relative_locator
        text = path.read_text(encoding="utf-8")
        if sha256_text(text) != document.content_hash:
            raise ValueError("Source content changed after inventory; inventory it again")
        cards: list[MethodCard] = []
        for block_index, match in enumerate(CARD_BLOCK_RE.finditer(text), start=1):
            if SENSITIVE_RE.search(match.group(1)):
                raise ValueError("Method-card block contains credential-like content")
            payload = yaml.safe_load(match.group(1))
            if not isinstance(payload, dict):
                raise ValueError("Method-card block must contain a YAML mapping")
            card = self._draft_from_payload(document, payload, match.start(), text, block_index)
            cards.append(self.repository.upsert_method_card(card))
        if not cards:
            raise ValueError("Source contains no explicit ```method-card block")
        return cards

    @staticmethod
    def _draft_from_payload(
        document: SourceDocument,
        payload: dict[str, Any],
        block_start: int,
        text: str,
        block_index: int,
    ) -> MethodCard:
        line_number = text.count("\n", 0, block_start) + 1
        values = dict(payload)
        values.setdefault(
            "card_id",
            "MC-" + sha256_text(f"{document.source_id}:{line_number}:{block_index}")[:16],
        )
        values["status"] = MethodCardStatus.DRAFT
        values["reviewer"] = None
        values["approval_timestamp"] = None
        values["source_references"] = [
            SourceReference(
                source_id=document.source_id,
                section_locator=f"line:{line_number}",
            ).model_dump(mode="json")
        ]
        card = MethodCard.model_validate(values)
        if card.evidence_level is not document.source_type:
            raise ValueError("Method-card evidence level must match its inventoried source type")
        return card


def _document_title(text: str, fallback: str) -> str:
    match = HEADING_RE.search(text)
    return match.group(2).strip() if match else fallback


def _section_inventory(source_id: str, text: str) -> list[SourceSection]:
    lines = text.splitlines()
    headings = [
        (index, match.group(1).strip())
        for index, line in enumerate(lines, start=1)
        if (match := re.match(r"^#{1,6}\s+(.+?)\s*$", line))
    ]
    if not headings:
        locator = f"lines:1-{max(1, len(lines))}"
        return [
            SourceSection(
                section_id="SEC-" + sha256_text(f"{source_id}:{locator}")[:20],
                heading="Document",
                section_locator=locator,
                content_hash=sha256_text(text),
            )
        ]
    sections: list[SourceSection] = []
    for position, (start, heading) in enumerate(headings):
        end = headings[position + 1][0] - 1 if position + 1 < len(headings) else len(lines)
        locator = f"lines:{start}-{end}"
        section_text = "\n".join(lines[start - 1 : end])
        sections.append(
            SourceSection(
                section_id="SEC-" + sha256_text(f"{source_id}:{locator}")[:20],
                heading=heading,
                section_locator=locator,
                content_hash=sha256_text(section_text),
            )
        )
    return sections
