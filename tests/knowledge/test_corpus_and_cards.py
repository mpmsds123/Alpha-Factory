from pathlib import Path

import pytest

from alpha_orchestrator.domain.enums import DataType, EvidenceLevel, MethodCardStatus
from alpha_orchestrator.domain.knowledge import MethodCard, SourceReference
from alpha_orchestrator.knowledge.corpus import CorpusInventory
from alpha_orchestrator.storage.repository import MissingRecord, Repository


def card_block(evidence_level: str = "training", card_id: str = "MC-COVERAGE-001") -> str:
    return f"""# Coverage diagnostics

This private body is never copied into SQLite.

```method-card
card_id: {card_id}
title: Diagnose low coverage before changing the expression
version: v1
applicable_failure_codes: [LOW_COVERAGE]
applicable_data_types: [MATRIX]
prerequisites:
  - Confirm the frozen field snapshot.
diagnostic_signals:
  - Coverage falls below the configured diagnostic floor.
allowed_repair_actions:
  - Replace one sparse input field with a documented compatible field.
prohibited_actions:
  - Do not relax a platform rule.
stop_condition: Stop when no compatible higher-coverage field exists.
evidence_level: {evidence_level}
```

## Notes

Additional private notes remain outside the database.
"""


def test_inventory_records_only_hashes_and_sections_and_extracts_draft(
    tmp_path: Path, repository: Repository
) -> None:
    corpus = tmp_path / "private-corpus"
    corpus.mkdir()
    source = corpus / "coverage.md"
    source.write_text(card_block(), encoding="utf-8")
    (corpus / "credentials.md").write_text(
        "# Excluded\napi_key=EXAMPLE_NOT_A_REAL_CREDENTIAL", encoding="utf-8"
    )
    (corpus / "alpha.md").write_text(
        "# Excluded Alpha\n```alpha\nrank(close)\n```", encoding="utf-8"
    )
    inventory = CorpusInventory(repository)
    result = inventory.inventory(corpus, EvidenceLevel.TRAINING)
    assert result.inventoried == 3
    assert result.excluded == 2
    first_ids = result.source_ids
    assert inventory.inventory(corpus, EvidenceLevel.TRAINING).source_ids == first_ids

    documents = repository.list_source_documents()
    safe = next(document for document in documents if document.relative_locator == "coverage.md")
    excluded = next(
        document for document in documents if document.relative_locator == "credentials.md"
    )
    excluded_alpha = next(
        document for document in documents if document.relative_locator == "alpha.md"
    )
    assert len(safe.sections) == 2
    assert safe.sections[0].section_locator.startswith("lines:")
    assert excluded.excluded
    assert excluded.exclusion_reason == "CREDENTIAL_OR_ACCOUNT_DATA_DETECTED"
    assert excluded_alpha.exclusion_reason == "COMPLETE_ALPHA_EXPRESSION_DETECTED"

    cards = inventory.extract_draft_cards(safe.source_id)
    assert len(cards) == 1
    assert cards[0].status is MethodCardStatus.DRAFT
    assert cards[0].source_references[0].source_id == safe.source_id
    assert cards[0].reviewer is None
    with pytest.raises(ValueError, match="excluded"):
        inventory.extract_draft_cards(excluded.source_id)
    with pytest.raises(ValueError, match="excluded"):
        inventory.extract_draft_cards(excluded_alpha.source_id)


def test_approval_is_explicit_and_approved_versions_are_immutable(
    tmp_path: Path, repository: Repository
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "card.md").write_text(card_block(), encoding="utf-8")
    inventory = CorpusInventory(repository)
    source_id = inventory.inventory(corpus, EvidenceLevel.TRAINING).source_ids[0]
    draft = inventory.extract_draft_cards(source_id)[0]
    assert repository.list_method_cards(MethodCardStatus.APPROVED) == []
    with pytest.raises(ValueError, match="Reviewer"):
        repository.approve_method_card(draft.card_id, " ")
    approved = repository.approve_method_card(draft.card_id, "Research Reviewer")
    assert approved.status is MethodCardStatus.APPROVED
    assert approved.reviewer == "Research Reviewer"
    assert approved.approval_timestamp is not None
    changed = approved.model_copy(update={"title": "Mutated title"})
    with pytest.raises(ValueError, match="immutable"):
        repository.upsert_method_card(changed)
    repository.upsert_method_card(approved)
    reextracted = inventory.extract_draft_cards(source_id)[0]
    assert reextracted.status is MethodCardStatus.APPROVED
    persisted = repository.list_method_cards(MethodCardStatus.APPROVED)[0]
    assert persisted.reviewer == "Research Reviewer"


def test_same_relative_locator_in_distinct_roots_has_distinct_stable_source_ids(
    tmp_path: Path, repository: Repository
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "card.md").write_text(card_block(), encoding="utf-8")
    (second_root / "card.md").write_text(card_block(), encoding="utf-8")
    inventory = CorpusInventory(repository)
    first_id = inventory.inventory(first_root, EvidenceLevel.TRAINING).source_ids[0]
    second_id = inventory.inventory(second_root, EvidenceLevel.TRAINING).source_ids[0]
    assert first_id != second_id
    assert inventory.inventory(first_root, EvidenceLevel.TRAINING).source_ids[0] == first_id


def test_extraction_rejects_stale_missing_and_mislabeled_sources(
    tmp_path: Path, repository: Repository
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "card.md"
    source.write_text(card_block(), encoding="utf-8")
    inventory = CorpusInventory(repository)
    source_id = inventory.inventory(corpus, EvidenceLevel.TRAINING).source_ids[0]
    source.write_text(card_block() + "\nchanged", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after inventory"):
        inventory.extract_draft_cards(source_id)

    source.write_text("# No explicit block", encoding="utf-8")
    source_id = inventory.inventory(corpus, EvidenceLevel.TRAINING).source_ids[0]
    with pytest.raises(ValueError, match="no explicit"):
        inventory.extract_draft_cards(source_id)

    source.write_text(card_block(evidence_level="official"), encoding="utf-8")
    source_id = inventory.inventory(corpus, EvidenceLevel.TRAINING).source_ids[0]
    with pytest.raises(ValueError, match="evidence level"):
        inventory.extract_draft_cards(source_id)


def test_inventory_validation_and_missing_source(tmp_path: Path, repository: Repository) -> None:
    inventory = CorpusInventory(repository)
    with pytest.raises(ValueError, match="not a directory"):
        inventory.inventory(tmp_path / "missing", EvidenceLevel.UNVERIFIED)
    with pytest.raises(MissingRecord):
        repository.get_source_document("SRC-MISSING")
    with pytest.raises(MissingRecord):
        repository.approve_method_card("MC-MISSING", "Reviewer")


def test_method_card_contract_requires_complete_approval() -> None:
    values = {
        "card_id": "MC-TEST-001",
        "title": "Test card",
        "version": "v1",
        "applicable_data_types": [DataType.MATRIX],
        "prerequisites": ["One prerequisite"],
        "diagnostic_signals": ["One signal"],
        "allowed_repair_actions": ["One action"],
        "prohibited_actions": ["One prohibition"],
        "stop_condition": "Stop deterministically.",
        "evidence_level": EvidenceLevel.UNVERIFIED,
        "source_references": [SourceReference(source_id="SRC-ONE", section_locator="lines:1-2")],
    }
    with pytest.raises(ValueError, match="Approved cards"):
        MethodCard(
            **values,
            status=MethodCardStatus.APPROVED,
            reviewer="Reviewer",
        )
