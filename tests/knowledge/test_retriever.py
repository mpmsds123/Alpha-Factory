from datetime import UTC, datetime

from alpha_orchestrator.domain.enums import DataType, EvidenceLevel, MethodCardStatus
from alpha_orchestrator.domain.knowledge import MethodCard, SourceReference
from alpha_orchestrator.knowledge.retriever import MethodCardRetriever
from alpha_orchestrator.storage.repository import Repository


def card(
    card_id: str,
    *,
    status: MethodCardStatus,
    failure_codes: list[str],
    data_type: DataType = DataType.MATRIX,
) -> MethodCard:
    approved = status is MethodCardStatus.APPROVED
    return MethodCard(
        card_id=card_id,
        title=f"Targeted {failure_codes[0] if failure_codes else 'general'} diagnosis",
        version="v1",
        status=status,
        applicable_failure_codes=failure_codes,
        applicable_data_types=[data_type],
        prerequisites=["Inspect coverage and the frozen field snapshot"],
        diagnostic_signals=["Low coverage is persistent"],
        allowed_repair_actions=["Change one field"],
        prohibited_actions=["Do not change limits"],
        stop_condition="Stop after one failed targeted repair",
        evidence_level=EvidenceLevel.TRAINING,
        source_references=[SourceReference(source_id="SRC-ONE", section_locator="lines:1-10")],
        reviewer="Reviewer" if approved else None,
        approval_timestamp=datetime.now(UTC) if approved else None,
    )


def test_retriever_uses_only_approved_cards_and_deterministic_tags(
    repository: Repository,
) -> None:
    repository.upsert_method_card(
        card(
            "MC-LOW-COVERAGE",
            status=MethodCardStatus.APPROVED,
            failure_codes=["LOW_COVERAGE"],
        )
    )
    repository.upsert_method_card(
        card(
            "MC-CONCENTRATION",
            status=MethodCardStatus.APPROVED,
            failure_codes=["CONCENTRATED_WEIGHT"],
        )
    )
    repository.upsert_method_card(
        card(
            "MC-DRAFT",
            status=MethodCardStatus.DRAFT,
            failure_codes=["LOW_COVERAGE"],
        )
    )
    repository.upsert_method_card(
        card(
            "MC-VECTOR",
            status=MethodCardStatus.APPROVED,
            failure_codes=["LOW_COVERAGE"],
            data_type=DataType.VECTOR,
        )
    )
    repository.upsert_method_card(
        card(
            "MC-GENERAL",
            status=MethodCardStatus.APPROVED,
            failure_codes=[],
        )
    )
    retriever = MethodCardRetriever(repository)
    first = retriever.retrieve(
        query="diagnose low coverage",
        data_type=DataType.MATRIX,
        failure_codes={"LOW_COVERAGE"},
    )
    second = retriever.retrieve(
        query="diagnose low coverage",
        data_type=DataType.MATRIX,
        failure_codes={"LOW_COVERAGE"},
    )
    assert [item.card.card_id for item in first] == ["MC-LOW-COVERAGE"]
    assert first == second
    assert first[0].score > 0
    assert retriever.retrieve(query="x", data_type=DataType.MATRIX, limit=0) == []
    assert retriever.retrieve(query="x", data_type=DataType.VECTOR, failure_codes={"OTHER"}) == []
