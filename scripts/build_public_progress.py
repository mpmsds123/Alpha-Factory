"""Export aggregate progress from local evidence without copying private payloads.

This command has no network or publishing capability. Review generated changes before
publishing. Source evidence files stay outside the public repository.
"""

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def count(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Expected a nonnegative aggregate count")
    return value


def timestamp(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected a timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp needs a timezone")
    return parsed.astimezone(UTC).isoformat()


def build_snapshot(evidence: dict[str, Any], research: dict[str, Any]) -> dict[str, Any]:
    """Construct a new object from explicit numeric/boolean fields only."""
    current = research["current_run"]
    statuses = current["statuses"]
    metrics = current["metric_shortfalls_using_current_thresholds"]
    checks = evidence["local_verification"]
    coverage = checks["coverage_percent"]
    if (
        type(coverage) not in (int, float)
        or not math.isfinite(coverage)
        or not 0 <= coverage <= 100
    ):
        raise ValueError("Invalid coverage percentage")
    return {
        "schema_version": 1,
        "research_snapshot_at_utc": timestamp(research["observed_at_utc"]),
        "scope": "Private working system; separate from the README research cohort",
        "research": {
            "candidate_rows": count(current["candidate_rows"]),
            "completed": count(statuses.get("DONE", 0)),
            "inflight": count(statuses.get("RUNNING", 0)),
            "failed": count(statuses.get("FAILED", 0)),
            "locally_marked_ready": count(current["locally_marked_ready"]),
            "locally_marked_robust_pass": count(current["locally_marked_robust_pass"]),
            "shortfalls": {
                k: count(metrics.get(k, 0))
                for k in (
                    "fitness_below_floor",
                    "sharpe_below_floor",
                    "margin_below_floor",
                    "turnover_outside_range",
                )
            },
        },
        "engineering": {
            "hundred_case_stage_passed": evidence["live"].get("hundred_sim_stage") == "passed",
            "regular_scheduled_start_verified": evidence["deployment"].get(
                "regular_scheduled_invocation_verified"
            )
            is True,
            "local_rollover_with_real_requests_verified": evidence.get(
                "local_allocation_rollover_with_real_requests_verified"
            )
            is True,
            "private_working_tree_checks": {
                "passed": count(checks["passed"]),
                "skipped": count(checks["skipped"]),
                "coverage_percent": coverage,
            },
        },
        "limits": [
            "Private runtime changes are not all in the public source snapshot.",
            "Private working-tree checks are not the public repository CI results.",
            "The 100-case stage is a subset of the campaign, not an additional 100 results.",
            "Candidates include related variants, not independent discoveries.",
            "Failure categories overlap; their counts must not be summed.",
            "Local rollover does not verify the platform global quota counter.",
            "A controlled yield improvement and out-of-sample value remain unproven.",
        ],
    }


def render(snapshot: dict[str, Any]) -> str:
    r = snapshot["research"]
    e = snapshot["engineering"]
    local = e["private_working_tree_checks"]

    def outcome(value: bool) -> str:
        return "Observed" if value else "Not yet verified"

    return f"""# Development progress

Research snapshot: **{snapshot["research_snapshot_at_utc"]}**.
This is a reviewed snapshot, not a continuously streaming dashboard.

[Development journal](progress/README.md) · [Machine-readable snapshot](progress/latest.json)

## Current position

The working system is under live acceptance, and research quality is still being evaluated.
No controlled improvement in qualified-alpha yield has been established.

The operational notes below concern a **private working system**. Its newest changes have
not all been ported into the public source. The README's previously documented research
cohort is separate: do not add its figures to this campaign or compare them as an A/B test.

## Current campaign

| Measure | Snapshot |
|---|---:|
| Candidate simulations recorded | {r["candidate_rows"]} |
| Completed | {r["completed"]} |
| In flight | {r["inflight"]} |
| Execution failures | {r["failed"]} |
| Locally marked ready for further checks | {r["locally_marked_ready"]} |
| Locally marked robustness passes | {r["locally_marked_robust_pass"]} |

Candidates include multiple versions of the same idea. Completion means a result was
returned and recorded; it does not mean the alpha passed the full acceptance criteria.
The 100-case execution stage is included in this campaign, not added to it.

## What is verified

| Working-system check | Evidence status |
|---|---|
| 100-case execution and recovery stage | {outcome(e["hundred_case_stage_passed"])} |
| Regular scheduled startup | {outcome(e["regular_scheduled_start_verified"])} |
| Local rollover + accepted work | {outcome(e["local_rollover_with_real_requests_verified"])} |
| Human biometric pause/resume on the latest change | Still under observation |
| Controlled research-yield improvement | Not established |
| Out-of-sample predictive value | Not established |

Private working-tree verification: **{local["passed"]} passed, {local["skipped"]} skipped;
{local["coverage_percent"]:.2f}% coverage**. These are not the public repository CI results.
For the public source's recorded checks, see [test evidence](TEST_EVIDENCE.md) and its CI runs.
The platform's global remaining-quota counter was not exposed by the read-only probes.

## What the results suggest testing next

Of {r["completed"]} completed candidates, {r["shortfalls"]["fitness_below_floor"]} fell below
the current fitness floor. A high headline Sharpe can coexist with excessive trading and
poor fitness. The next question is whether refinement spends budget improving a headline
metric while leaving the checks that prevent qualification unresolved.

This is a hypothesis, not an established cause. A candidate change needs a fixed baseline,
the same available history and simulation budget, unchanged gates, and repeated comparison.
Historical cohorts used different versions and datasets and are not a fair control group.

## Follow or contribute

Updates record meaningful changes: **observation → change or hypothesis → evidence → next step**.
Routine polling is not a new milestone. Dated entries retain the earlier interpretation and
its limits; corrections should be explicit. Publishing follows review of the aggregate export.

Readers can [open an issue](https://github.com/mpmsds123/Alpha-Factory/issues/new) with a
reproducible failure, a competing explanation, or a proposed test, and can reproduce the
[offline demo](case-studies/sanitized-replay.md).

Only aggregate counts, engineering findings and reviewed limitations are exported. Credentials,
platform field identifiers, formulas, alpha IDs, raw logs and local filesystem paths stay private.

The exporter in `scripts/build_public_progress.py` reads explicitly allowed fields only. It
has no network access or publishing step; generated changes still require a publication review.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docs"
    )
    args = parser.parse_args()
    snapshot = build_snapshot(
        json.loads(args.evidence.read_text()), json.loads(args.research.read_text())
    )
    progress = args.output_dir / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    (progress / "latest.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    (args.output_dir / "PROGRESS.md").write_text(render(snapshot))
    print("Generated aggregate progress snapshot; nothing was published.")


if __name__ == "__main__":
    main()
