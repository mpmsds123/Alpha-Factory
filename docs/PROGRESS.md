# Development progress

Campaign accounting checked locally: **2026-10-30T05:01:55.331650+00:00**.
Research-metric audit retained from **2026-10-28T03:15:11.004085+00:00**; private software
checks were recorded on **Oct 27**. No new platform observation is implied.
This is a reviewed snapshot, not a continuously streaming dashboard.

[Development journal](progress/README.md) · [Machine-readable snapshot](progress/latest.json)

## Current position

Research execution is **paused for result reconciliation**. Production acceptance is incomplete.
The factory has a ceiling of **1,000 simulation tasks per allocation day**; the independent
scanner has 4,000. Scanner results are excluded from every factory count below.
No controlled improvement in qualified-alpha yield has been established.

The operational notes below concern a **private working system**. Its newest changes have
not all been ported into the public source. The README's previously documented research
cohort is separate: do not add its figures to this campaign or compare them as an A/B test.

## Current campaign

| Measure | Snapshot |
|---|---:|
| Accepted simulation tasks in this campaign | 344 |
| Completed | 338 |
| Actively in flight | 0 |
| Quarantined simulation tasks (UNKNOWN) | 6 |
| Quarantined batches containing those tasks | 1 |
| New factory simulation tasks on Oct 29, as of ledger check | 0 |
| New batch submissions on Oct 29, as of ledger check | 0 |
| New research LLM calls on Oct 29, as of ledger check | 0 |
| Submitted alphas | 0 |
| Execution failures | 0 |
| Locally marked ready for further checks | 0 |
| Locally marked robustness passes | 0 |

Candidates include multiple versions of the same idea. Completion means a result was
returned and recorded; it does not mean the alpha passed the full acceptance criteria.
The 100-simulation execution stage is included in this campaign, not added to it.
A simulation task evaluates one formula/settings combination on the platform; a batch groups
several tasks. A program launch, read-only query, LLM call, or local test is not a simulation.
The six quarantined tasks are **not execution failures or confirmed completions**. Four matching
result identities were independently verified, but two terminal outcomes are still missing;
the entire batch and all six reservations remain unchanged. No replacement tasks were sent.

## What is verified

| Working-system check | Evidence status |
|---|---|
| 100-case execution and recovery stage | Observed |
| Earlier schedule startup | Observed historically |
| New daytime schedule invocation | Observed; correctly blocked by UNKNOWN guard |
| New daytime research execution | Not yet verified |
| Local rollover + accepted work | Observed |
| Human verification handoff | Previously observed; current block is reconciliation |
| Latest startup and session-persistence changes in live use | Not yet verified |
| Overall production acceptance | Incomplete |
| Controlled research-yield improvement | Not established |
| Out-of-sample predictive value | Not established |

Private working-tree verification: **305 passed, 5 skipped;
86.22% coverage**. These are not the public repository CI results.
For the public source's recorded checks, see [test evidence](TEST_EVIDENCE.md) and its CI runs.
The platform's global remaining-quota counter was not exposed by the read-only probes.

## Today: research review, execution remains paused

A read-only literature and code comparison covered **10 full-text papers and five
repositories**, producing **30 proposed improvement directions**. It compared the private
working implementation with the reviewed methods. This was an analysis deliverable: no
factory code or prompt was changed, no research LLM was called and no simulation was created
for the review. The private report and source material are not part of this public update.

The prioritized hypotheses concern task-level model evaluation, appropriately scoped reuse
of earlier research experience, hypothesis-to-formula semantic checks, and duplicate/family
diagnostics. These are proposals, not evidence of improved qualified-alpha yield. None was
implemented or tested in a controlled platform experiment today.

Two local authentication-slot checks are recorded for this date. They did not request
verification while result reconciliation remained the blocking condition. The scheduled
worker remains disabled and unloaded; zero factory workers were present at the ledger check.
These checks are not simulations. See the [Oct 29 journal](progress/2026-10-29.md).

## Current blocker and next evidence

A returned result failed the submitted-identity check. The factory retained the original
progress reference and stopped. That progress endpoint subsequently returned server errors.
An exhaustive read-only search of the fixed account-history range did not locate the two
missing factory identities. Account-history records are not additional factory simulations,
and absence from that list is not proof of failure or non-acceptance.

The required next evidence is the original child-task terminal status with its matching
result, or an explicit terminal error. Operator review is required. Do not recreate the tasks,
release reservations, or infer a failure merely to restart research.

Offline engineering completed on Oct 27: preserve result evidence before quarantine; write
private session state atomically with restrictive permissions; acquire worker ownership before
any authentication. Local fault tests passed. These changes preserve research gates and
human submission control; they have not resolved the historical missing outcomes or proved
better alpha yield. See the [Oct 27 journal](progress/2026-10-27.md).

Of 338 completed campaign candidates, 331 fall below the current fitness floor when evaluated
retrospectively. Related variants and overlapping failed checks prevent treating this as an
independent discovery sample. A fixed-budget research comparison remains a plan, not an
executed experiment. No controlled yield improvement or out-of-sample value is established.

## Follow or contribute

Updates record meaningful changes: **observation → change or hypothesis → evidence → next step**.
Routine polling is not a new milestone. Dated entries retain the earlier interpretation and
its limits; corrections should be explicit. Daily publication is scheduled for 22:00 Pacific.
This Oct 29 update combines a dated local ledger check, a completed research review, and
earlier research-metric and engineering evidence. Publishing the journal does not refresh platform results or authentication.

Readers can [open an issue](https://github.com/mpmsds123/Alpha-Factory/issues/new) with a
reproducible failure, a competing explanation, or a proposed test, and can reproduce the
[offline demo](case-studies/sanitized-replay.md).

Only aggregate counts, engineering findings and reviewed limitations are exported. Credentials,
platform field identifiers, formulas, alpha IDs, raw logs and local filesystem paths stay private.

The exporter in `scripts/build_public_progress.py` reads explicitly allowed fields only. It
has no network access or publishing step; generated changes still require a publication review.
