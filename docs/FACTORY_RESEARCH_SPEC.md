# Alpha Factory research specification

Revision 1. The release target is a single-worker, resumable BRAIN research pipeline
with auditable LLM decisions and manual submission. The release entrypoint is
`alpha-factory`. The production deployment procedure is in the
[runbook](PRODUCTION_RUNBOOK.md).

## Operating contract

| Actor | Authority |
| --- | --- |
| LLM | Propose an economic hypothesis, competing explanation and one available intervention. Predict an observable result, state a falsifier or abstain. |
| Factory | Select the fixed-rule comparison, validate expressions, reserve simulation budget, identify results, apply the recorded scoring rule and persist the lineage. |
| BRAIN | Execute simulations and return candidate metrics and platform checks. |
| Operator | Set the research scope and budget, complete platform verification, resolve ambiguous batches and approve submission. |

The default v7 configuration uses its existing research policy. Set `refinement_trial: true`
in a pilot configuration to activate the paired comparison. For a refinement pilot,
select the dataset combinations explicitly and leave `direction_trial` disabled.
Study dataset selection in a separate pilot. Run both against the same authoritative
ledger and use the same daily allocation limits.

## Paired refinement protocol, revision 1

**Eligibility.** Use the first eligible refinement round of an idea, with at most one
comparison per idea and per parent fingerprint. Require at least one concrete failed
check and two distinct, validated, uncached expression-only fixes from the existing
diagnosis menu. Both choices inherit the parent's complete simulation settings.
If the comparison is ineligible, use the usual v7 refinement path.

**Assignment.** The fixed rule ranks eligible fixes by `(wins + 1)/(tries + 2)` in the
available repair history for the diagnosed problem, breaking ties by menu order. A
historical win means fewer failed checks than the parent. The model receives the same
eligible menu and historical counts, together with the parent metrics and idea. It
returns a bounded JSON proposal with `action`, `fix`, `hypothesis`, `alternative`,
`prediction`, `falsifier`, and `why`. It can test one listed fix or abstain. A malformed
proposal follows the existing bounded LLM retry and pause behavior.

**Execution.** A single SQLite transaction stores the plan, candidate IDs and queued
work. Budget one child per distinct choice, up to two new children. Agreement shares one
child and yields one paired tie. Abstention runs the fixed rule only. The queue shuffles
distinct choices before insertion and uses the normal budget, batching and result-identity
controls. The normal v7 gates determine promotion. The paired score remains a separate
research measurement.

**Recovery.** An accepted simulation is polled using its saved BRAIN location. A create
with ambiguous acceptance enters the existing reconciliation path. An accepted pair can
straddle a quota window and stays pending until all assigned children are terminal.
Disable the trial to stop new assignments. Existing assignments settle from their saved
candidate IDs and policy snapshot.

**Scoring.** All recorded target checks must explicitly return `PASS`, with no new failed
checks relative to the parent. An absent, pending or nonbinary target check, a failed
child, or a nonfinite or suspicious Sharpe makes that arm unusable. Save signed Sharpe,
cleared targets and newly failed checks for observed arms. The record page displays
proposals and outcomes, including abstentions and incomplete evidence.

**Research review.** Before a live comparison, freeze the cohort, observation budget and
stopping rule. Count by parent. Show agreement, abstention, unusable arms and unresolved
assignments alongside wins. The paired endpoint measures target-check clearance on
in-sample BRAIN results. For candidate yield, join recorded arm candidates to platform
review outcomes, count unique candidates, and report consumed simulations and deferred
checks. For an economic claim, freeze the selected formula and use a prospective
evaluation period with a specified benchmark. Related parents can share fields, names
and dates. Treat pilot totals as descriptive.

## Release acceptance

| Check | Required evidence |
| --- | --- |
| Build and compatibility | Repository formatting, lint, type checks and full offline tests pass. A preexisting ledger opens without loss of stored results. |
| Paired protocol | Same-parent, same-settings choices are persisted before dispatch. Agreement spends one child. Abstention records no LLM child. Missing target checks leave the arm unusable. |
| Restart and budget | In the fake platform test, reopen SQLite after acceptance and recover the original batch with zero additional creates. On the live canary, reconcile BRAIN IDs, quota reservations and restart behavior. |
| Check and submit | `review recheck` updates existing candidates under the writer lock using the platform check endpoint. Manual approval remains the submission entrypoint. |
| Release readiness | Record the commit, configuration, ledger backup, acceptance evidence and rollback decision in the deployment record. Advance the live canary stages using the runbook. |

Offline engineering verification, live operating acceptance and the research benefit
each have their own evidence. Report their status separately when releasing.

## Acceptance evidence, November 10, 2026

- Offline checks passed. Formatting and lint passed, mypy checked 82 source files,
  and the test suite reported 388 passed, 5 skipped and 87.16% branch-aware coverage.
- The SQLite restart test interrupts polling after the fake platform accepts the
  batch. It verifies the original assignment, the recovered results and zero
  additional simulation creates on restart.
- The production preflight read the authoritative ledger. It found one in-flight
  batch and zero unknown batches. BRAIN returned HTTP 204 on the authentication
  probe and the local session records a pending biometric challenge. Live recovery
  and canary acceptance remain pending. After verification, recover the in-flight
  batch and reconcile it before allocating new live canary work.
