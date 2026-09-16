"""Deterministic tag filtering with a compact BM25-style lexical ranker."""

import math
import re
from collections import Counter
from dataclasses import dataclass

from alpha_orchestrator.domain.enums import DataType, MethodCardStatus
from alpha_orchestrator.domain.knowledge import MethodCard
from alpha_orchestrator.storage.repository import Repository

TERM_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class RetrievedCard:
    card: MethodCard
    score: float


class MethodCardRetriever:
    def __init__(self, repository: Repository) -> None:
        self.repository = repository

    def retrieve(
        self,
        *,
        query: str,
        data_type: DataType,
        failure_codes: set[str] | None = None,
        limit: int = 5,
    ) -> list[RetrievedCard]:
        if limit < 1:
            return []
        requested_failures = {code.casefold() for code in failure_codes or set()}
        cards = [
            card
            for card in self.repository.list_method_cards(MethodCardStatus.APPROVED)
            if data_type in card.applicable_data_types
            and (
                not requested_failures
                or requested_failures & {code.casefold() for code in card.applicable_failure_codes}
            )
        ]
        if not cards:
            return []
        documents = [_card_terms(card) for card in cards]
        query_terms = _terms(query) | requested_failures | {data_type.casefold()}
        document_frequency = Counter(term for document in documents for term in set(document))
        average_length = sum(len(document) for document in documents) / len(documents)
        ranked: list[RetrievedCard] = []
        for card, document in zip(cards, documents, strict=True):
            score = _bm25_score(
                query_terms,
                document,
                document_frequency,
                len(documents),
                average_length,
            )
            card_failures = {code.casefold() for code in card.applicable_failure_codes}
            score += 5.0 * len(requested_failures & card_failures)
            score += 1.0
            ranked.append(RetrievedCard(card=card, score=round(score, 8)))
        return sorted(ranked, key=lambda item: (-item.score, item.card.card_id))[:limit]


def _terms(value: str) -> set[str]:
    return {match.group(0).casefold() for match in TERM_RE.finditer(value)}


def _card_terms(card: MethodCard) -> list[str]:
    values = [
        card.title,
        *card.applicable_failure_codes,
        *(data_type.value for data_type in card.applicable_data_types),
        *card.prerequisites,
        *card.diagnostic_signals,
        *card.allowed_repair_actions,
        *card.prohibited_actions,
        card.stop_condition,
    ]
    return [term for value in values for term in _terms(value)]


def _bm25_score(
    query_terms: set[str],
    document: list[str],
    document_frequency: Counter[str],
    document_count: int,
    average_length: float,
) -> float:
    if not query_terms or not document:
        return 0.0
    frequencies = Counter(document)
    score = 0.0
    k1 = 1.5
    b = 0.75
    for term in query_terms:
        frequency = frequencies[term]
        if not frequency:
            continue
        inverse_frequency = math.log(
            1 + (document_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
        )
        denominator = frequency + k1 * (1 - b + b * len(document) / max(1.0, average_length))
        score += inverse_frequency * frequency * (k1 + 1) / denominator
    return score
