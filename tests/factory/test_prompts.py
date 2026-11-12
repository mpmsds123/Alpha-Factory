import pytest
from pydantic import ValidationError

from alpha_orchestrator.factory import prompts


def test_versions_carry_a_text_hash() -> None:
    assert prompts.HYPOTHESIS.version.startswith("hypothesis.v1-")
    assert len({p.version for p in prompts.ALL.values()}) == len(prompts.ALL)


def test_decision_requires_consistent_payload() -> None:
    base = {"reason": "nothing left here", "lesson": "a lesson worth keeping"}
    with pytest.raises(ValidationError):
        prompts.DecisionOut.model_validate({**base, "action": "mutate"})
    with pytest.raises(ValidationError):
        prompts.DecisionOut.model_validate({**base, "action": "revise"})
    assert prompts.DecisionOut.model_validate({**base, "action": "drop"}).action == "drop"


def test_hypothesis_schema_is_strict() -> None:
    good = {
        "claim": "x" * 40,
        "direction": "higher_value_lower_return",
        "hypothesis_type": "reversal",
        "holding_days": 21,
        "turnover_band": "mid",
        "falsifier": "no reversal at all",
    }
    assert prompts.HypothesisOut.model_validate(good).holding_days == 21
    for bad in ({**good, "holding_days": 17}, {**good, "surprise": 1}, {**good, "claim": "x"}):
        with pytest.raises(ValidationError):
            prompts.HypothesisOut.model_validate(bad)


def test_prompts_name_no_platform() -> None:
    text = " ".join(p.system + p.task for p in prompts.ALL.values()).lower()
    assert "fastexpr" not in text and "brain" not in text
