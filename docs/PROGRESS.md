# Development progress

Research snapshot: **2026-10-27T04:16:05.640463+00:00**.
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
| Candidate simulations recorded | 180 |
| Completed | 179 |
| In flight | 1 |
| Execution failures | 0 |
| Locally marked ready for further checks | 0 |
| Locally marked robustness passes | 0 |

Candidates include multiple versions of the same idea. Completion means a result was
returned and recorded; it does not mean the alpha passed the full acceptance criteria.
The 100-case execution stage is included in this campaign, not added to it.

## What is verified

| Working-system check | Evidence status |
|---|---|
| 100-case execution and recovery stage | Observed |
| Regular scheduled startup | Observed |
| Local rollover + accepted work | Observed |
| Human biometric pause/resume on the latest change | Still under observation |
| Controlled research-yield improvement | Not established |
| Out-of-sample predictive value | Not established |

Private working-tree verification: **296 passed, 5 skipped;
86.02% coverage**. These are not the public repository CI results.
For the public source's recorded checks, see [test evidence](TEST_EVIDENCE.md) and its CI runs.
The platform's global remaining-quota counter was not exposed by the read-only probes.

## What the results suggest testing next

Of 179 completed candidates, 177 fell below
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
