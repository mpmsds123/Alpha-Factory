"""Private evidence may contain identifiers and keys; only approved aggregates leave it."""

import json
import runpy
from pathlib import Path

import pytest

EXPORTER = runpy.run_path(str(Path(__file__).parents[2] / "scripts/build_public_progress.py"))


def inputs():  # type: ignore[no-untyped-def]
    evidence = {
        "live": {"hundred_sim_stage": "passed"},
        "local_verification": {"passed": 2, "skipped": 0, "coverage_percent": 86.0},
        "deployment": {"regular_scheduled_invocation_verified": True},
        "local_allocation_rollover_with_real_requests_verified": True,
    }
    research = {
        "observed_at_utc": "2026-10-26T21:16:05-07:00",
        "current_run": {
            "candidate_rows": 3,
            "statuses": {"DONE": 2, "RUNNING": 1},
            "metric_shortfalls_using_current_thresholds": {"fitness_below_floor": 2},
            "locally_marked_ready": 0,
            "locally_marked_robust_pass": 0,
        },
    }
    return evidence, research


def test_export_cannot_carry_unknown_private_fields() -> None:
    evidence, research = inputs()
    poison = "PRIVATE-CREDENTIAL-FIELD-EXPRESSION-ID-PATH"
    evidence["source"] = poison
    evidence["local_verification"]["error"] = poison
    evidence["deployment"]["source"] = poison
    research["all_stored_candidates"] = {poison: {"formula": poison}}
    research["current_run"]["by_dataset"] = {poison: {"id": poison}}
    result = EXPORTER["build_snapshot"](evidence, research)
    assert poison not in json.dumps(result)
    assert poison not in EXPORTER["render"](result)
    assert result["research_snapshot_at_utc"] == "2026-10-27T04:16:05+00:00"


@pytest.mark.parametrize("unsafe", ["PRIVATE-IDENTIFIER", True, -1])
def test_non_count_values_cannot_hide_in_an_allowed_field(unsafe: object) -> None:
    evidence, research = inputs()
    research["current_run"]["candidate_rows"] = unsafe
    with pytest.raises(ValueError):
        EXPORTER["build_snapshot"](evidence, research)


def test_timestamp_is_parsed_instead_of_copied_as_free_text() -> None:
    evidence, research = inputs()
    research["observed_at_utc"] = "2026-10-27T04:16:05Z\nPRIVATE-IDENTIFIER"
    with pytest.raises(ValueError):
        EXPORTER["build_snapshot"](evidence, research)
