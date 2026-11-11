import json
from datetime import date, timedelta

import pytest

from alpha_orchestrator.factory.config import FactoryConfig
from alpha_orchestrator.factory.experiment import (
    Proposal,
    Protocol,
    proposal_digest,
    propose,
    replay,
)
from alpha_orchestrator.factory.ledger import Ledger
from alpha_orchestrator.factory.llm import FactoryLLM

CARD = {
    "hypothesis": "candidate has better incremental PnL",
    "competing_explanation": "candidate only tracks baseline",
    "evidence_ids": ["field-definition"],
    "candidate": "candidate",
    "benchmark": "benchmark",
    "direction": "higher",
    "falsifier": "relative daily pnl is non-positive in both blocks",
}


def test_propose_validates_model_evidence() -> None:
    def fake(model: str, system: str, user: str) -> tuple[str, int, int]:
        assert json.loads(user)["allowed_series"] == ["candidate", "benchmark"]
        return json.dumps(CARD), 1, 1

    llm = FactoryLLM(FactoryConfig(), Ledger("sqlite://"), complete=fake)
    allowed = ["candidate", "benchmark"]
    assert propose(llm, {"field-definition": "mock"}, allowed).direction == "higher"
    with pytest.raises(ValueError, match="unknown evidence"):
        propose(llm, {}, allowed)


def test_signed_pnl_replay_has_three_outcomes_and_binds_proposal() -> None:
    proposal = Proposal.model_validate(CARD)
    start = date(2026, 1, 2)
    days = [start + timedelta(days=i) for i in range(-1, 4)]
    protocol = Protocol(
        proposal_digest=proposal_digest(proposal),
        approved_on=days[0],
        start=start,
        end=days[-1],
        min_observations=4,
        min_signed_mean=1,
    )
    series = {
        "candidate": dict(zip(days, [0.0, 2.0, 4.0, 6.0, 8.0], strict=True)),
        "benchmark": dict.fromkeys(days, 0.0),
    }
    assert replay(proposal, protocol, series)["verdict"] == "supports"
    series["candidate"] = {day: -pnl for day, pnl in series["candidate"].items()}
    assert replay(proposal, protocol, series)["verdict"] == "conflicts"
    del series["candidate"][days[1]]
    assert replay(proposal, protocol, series)["verdict"] == "insufficient_evidence"
    with pytest.raises(ValueError, match="frozen"):
        replay(proposal.model_copy(update={"direction": "lower"}), protocol, series)
