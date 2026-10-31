# Development progress

Campaign accounting checked locally: **2026-10-31T05:02:59.565084+00:00**.
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
| New factory simulation tasks on Oct 30, as of ledger check | 0 |
| New batch submissions on Oct 30, as of ledger check | 0 |
| New research LLM calls on Oct 30, as of ledger check | 0 |
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

## Today: a tested exception proposal, not a running factory

At the owner's request, a candidate implementation was prepared for a short, bounded
operational exception. It would keep the original uncertain batch immutable, isolate its
dataset and permit unrelated new work only within an explicit expiry. Any new uncertainty
would still stop execution. Default research gates and human submission control are preserved.

The candidate passed **317 local tests, with 5 skipped and 86.38% coverage**, plus lint,
format and type checks. These checks concern a working copy: the candidate has **not been
deployed or started**. The last deployed private-source checks remain 305 passed/5 skipped.
Deployment is awaiting an additional explicit write approval; no code from this candidate
was uploaded here. Offline checks do not prove live recovery or improved research yield.

The owner completed a new human-verification handoff, and the platform confirmed it. A bounded
read of the original progress reference returned not-found; this does not establish either
missing task's final outcome. The existing quarantine and all six reservations remain.
Three reserved authentication-slot checks are recorded for today. No new factory simulation
or research LLM call is recorded. See the [Oct 30 journal](progress/2026-10-30.md).

## Current blocker and next evidence

A returned result failed the submitted-identity check. The factory retained the original
progress reference and stopped. That progress endpoint subsequently returned server errors.
Today a bounded recheck returned not-found. This is not a terminal outcome.
An earlier exhaustive read-only search of the fixed account-history range did not locate the two
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
This Oct 30 update combines a dated local ledger check, candidate-only engineering checks,
and earlier research-metric and deployed-source evidence. Publishing the journal does not refresh platform results or authentication.

Readers can [open an issue](https://github.com/mpmsds123/Alpha-Factory/issues/new) with a
reproducible failure, a competing explanation, or a proposed test, and can reproduce the
[offline demo](case-studies/sanitized-replay.md).

Only aggregate counts, engineering findings and reviewed limitations are exported. Credentials,
platform field identifiers, formulas, alpha IDs, raw logs and local filesystem paths stay private.

The exporter in `scripts/build_public_progress.py` reads explicitly allowed fields only. It
has no network access or publishing step; generated changes still require a publication review.
