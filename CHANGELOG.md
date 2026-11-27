# Changelog

Newest first. Every entry before 2026-10-31 is taken from the git history; the short hash
points to the commit. Real expressions and results are not published (see README).
Operational detail that used to live in `docs/` is summarized in the entries below.

Entry format for new days: **Ran** (what was run, in counts), **Found** (aggregate results
only), **Changed** (code or design), **Open** (unresolved).

## 2026-11-26 (weekly)

- Ran. 2026-11-21 to 2026-11-27: 143 hypotheses, 683 expressions proposed (262 passed local checks), 264 simulations, LLM cost $2.57.
- Found. Round-1 gate cleared by 8 hypotheses; train gate 9, test-period gate 8, platform checks 0; 0 stored for review.
- Changed. Earlier GitHub history joined to the cleaned local version (a7a7c86).
- Changed. Daily slots, fix table, local checks, platform retry, memory files, submit switch, daily schedule (#2).
- Changed. Daily run stops 15 minutes before the quota day resets (#3).
- Changed. One login shared with the scanner; daily run waits for the face check (#4).
- Changed. Pool slot first until it has an alpha in the quota day (#5).
- Changed. Weekly performance table in the README (#6).
- Open. First train-gate and test-period passes this week (9 and 8), but none cleared the
  platform checks: 4 retries for production correlation, 1 for concentrated weight.
- Open. No daily run between 2026-11-21 and 2026-11-25 (the schedule started 2026-11-25).
  On 2026-11-26 the run stopped at a face check, then twice on the LLM provider (402
  insufficient balance, then an empty reply); the evening rerun did 150 simulations and
  stopped at the deadline.
- Open. Share of expressions passing local checks fell from 70.6% to 38.4%.

## 2026-11-27 (pool route for pool-slot alphas)

- Found. After the pool-first fix the evening run produced the first v8 alphas through
  the train and test-period gates (9 in its first hour, train Sharpe up to 2.9, test
  Sharpe 2.4 to 2.9). All were rejected at the platform step, mainly on the regular
  production correlation (0.89 to 0.96), and every WARNING was counted as a failure.
- Changed. Alphas of the pool slot are judged by the pool route, ported from the
  template scanner: pool correlation PASS, nothing outside the regular-only tests fails,
  and a pure pool alpha must match a current pool theme. Before the check the factory
  writes the Idea / Rationale description the pool tests need, from the hypothesis.
- Changed. On the regular route a WARNING no longer blocks; FAIL and PENDING still do.
  A pool-route failure no longer puts the expression's fields on the crowded list.
- Changed. `review approve` uses the same routes and the same description.

## 2026-11-26 (one login with the scanner, face-check wait, empty-reply retry)

- Ran. The 15:00 daily run stopped at a face check before any simulation; after the
  login the LLM provider refused calls (account balance), then a 15-minute call came back
  with no answer and ended the run. 3 simulations in all.
- Changed. One platform login shared with the template scanner: a still-valid cookie from
  either program is used before a password login or a face-check stop, and every saved
  cookie is copied to the other's file, so one face check unlocks both.
- Changed. `daily` no longer ends the day at a face check: it opens the check page and
  resumes once the check is done (here or through the scanner's login), up to the day's
  deadline.
- Changed. `scanner_runs` is set, which turns on the originality check against the
  scanner's formulas.
- Changed. An LLM reply with no choices counts as a bad answer and is retried, instead of
  being treated as a provider outage that stops the run.
- Changed. The pool slot comes first until it has an alpha in the quota day. The
  evening run gave two of every three hypotheses to the slots that must exceed the pool
  limits; 240 of the 249 expressions written for them stayed inside those limits and were
  rejected, so the run spent its LLM budget on 9 simulations.
- Changed. `alpha-factory weekly` writes a Recent performance table (this week, prior
  week, all time; counts and rates only) into the README and prints the changelog lines.
- Open. No v8 hypothesis has passed the train gate yet (5 cleared round 1 in 1,261
  simulations); this is the main problem to work on.

## 2026-11-25 (daily slots, fix table, local checks, scheduled runs)

- Changed. Copied the working parts of a public pipeline that delivers 4 alphas a day
  into the existing program-driven loop (the loop itself is unchanged).
- Changed. Daily slots: pool, single dataset, multi dataset, plus a spare place; each
  hypothesis gets a slot that sets its operator and field limits; the slots outside the
  pool must exceed the pool limits; the multi-dataset slot adds fields of a second dataset.
  Nothing new opens once 4 alphas are stored in the quota day.
- Changed. Fix table: each failure maps to one change; decay fixes are applied by the
  program without an LLM call; LLM children that make another change are rejected.
- Changed. Local checks: daily-PnL self-correlation against submitted alphas (0.60), a
  crowded-field list, the 64-operator cap.
- Changed. A platform-check failure reopens the hypothesis once with the failure and its
  fix; a production-correlation failure crowds the expression's fields.
- Changed. Three memory files (rules, daily status, lessons with fixed triggers) next to
  the database records; rules and recent lessons go to every LLM call.
- Changed. Submit switch (approve by default, or auto up to 4 a day, with a pause file).
- Changed. `daily` command (restarts after pauses, stops on a face check) and a launchd
  job writer, replacing the "runs by hand only" rule.
- Open. The combined-alpha slot is not built; its place is the spare slot. The local
  self-correlation reads the account's submitted alphas from the API, not yet run live.

## 2026-11-12 (v8: hypothesis factory)

- Changed. Rebuilt around hypotheses instead of field x template. The program picks a
  dataset (any delay-1 dataset, by Thompson draw on past round-1 results) and a field; the
  LLM states a hypothesis and writes 1-3 expressions; after each round it diagnoses (idea
  vs implementation) and decides mutate / revise / drop inside program budgets (12
  simulations per hypothesis, 6 for one revision, round-1 gate 1.0, 3 improvement
  rounds, 2 dry rounds). Children must make exactly the change they declare.
- Changed. Every simulation holds out the last 2 years as a test period. The LLM sees
  training numbers only; winners must keep a test Sharpe of 0.5 and then pass every
  platform check. Submission gates are the official delay-1 numbers.
- Changed. One new database with hypotheses, expressions, simulations, decisions and
  priors. Every LLM answer is schema-checked, cites the retrieved records and priors it
  used (uncited answers flagged) and is logged with prompt version, model and cost.
  Priors carry a Beta confidence updated from the factory's own verdicts.
- Changed. Daily LLM dollar cap and a dollars-per-stored-alpha guard; per-stage CSVs
  written during runs.
- Changed. Removed the earlier orchestrator, the v7 engine, the launchd scheduler and the
  shell launcher. Platform vocabulary moved to a private folder; a test scans tracked
  files for it.
- Open. Not yet run against the platform. Thresholds measured on v7 (round-1 gate,
  minimum gain) need re-measuring on train-period numbers.

## 2026-11-12

- Found. Funnel of the v7 design (2026-11-03 to 11-11, about 2,930 simulations): 805 ideas,
  2,406 first formulas, 156 ideas past the first gate (19%), 13 formulas past the regular
  line and the robustness test, 0 past the platform's submission check. The first gate lost
  the most: 648 ideas, about two thirds of all simulations. None of the 805 ideas used a
  field the template scanner had found signal in: v7 skipped that filter, and the scanner's
  output had moved to a single master table the factory did not read. In the scanner's
  own results, 64% of simulations on fields with signal reach the first gate's Sharpe
  level, against 10% elsewhere.
- Changed. Per-stage funnel CSVs in the reports folder, refreshed during a run and by the
  read-only `alpha-factory funnel` command. The scanner's master table is read; v7 mines only
  datasets and fields with scanner signal; first formulas that repeat a scanner formula's
  structure are rejected (originality check after AlphaAgent). Thresholds are unchanged.
- Open. Whether the first-gate pass rate rises in a live run; signal fields are a limited
  supply (about one day of simulations) that grows as the scanner runs.

## 2026-11-11

- Changed: repository cleanup to professional standards, with no change in behaviour.
  Stale notes, specs and duplicate policy/prompt copies removed; `tools/` became `scripts/`;
  secrets can come from environment variables (`.env.example`); dependencies pinned in
  `requirements.txt`; black adopted as formatter; English comments, docstrings and named
  constants throughout the factory; tests for secret loading; one README replaces the
  `docs/` folder.

## 2026-11-10

- Changed. Documented the research and release contracts. Added a versioned paired
  refinement protocol with durable assignment, settlement and a fixed-rule comparison.
- Changed. Added `review recheck` for existing candidates and their deferred platform checks.
- Ran. Offline: 388 passed, 5 skipped, 87.16% branch coverage. A restart test interrupts
  polling after the fake platform accepts a batch and confirms zero additional creates.
- Ran. Live recovery with both daily caps at zero: an expired progress URL recorded its nine
  children as lost and created nothing new. Ten saved candidates were re-checked; none
  reached the review list.
- Ran. First live canary (10 children): the worker was stopped before its first poll; a new
  worker reconciled ten distinct results to the ten original candidates with no duplicates.
- Ran. Expanded canary (cumulative cap 100, paired refinement trial on): 40 distinct results
  matched 40 candidates across six batches; no duplicates or unresolved batches.
- Found. Three paired-refinement trials settled; the model and the fixed rule chose the same
  fix each time, so comparative benefit is still unmeasured. No new review candidates.
- Open. Full-volume and scheduled acceptance need observation across a real quota reset.

## 2026-10-31

- Changed: README now leads with `alpha-factory`; broken link to a removed file fixed; this
  changelog added.

## 2026-10-30

- Changed: explicitly authorized, expiring quarantine for an unresolved batch (`16823fe`).
- Changed: an authorized run continues through the current allocation instead of stopping at a
  fixed time (`c3af9ab`).

## 2026-10-27

- Changed: the factory owns its ledger before authentication; a competing startup is rejected
  (`081a2d4`).
- Changed: private session state is written atomically, so an interruption keeps the previous
  complete state (`75c2aef`).
- Changed: reconciliation evidence is stored before a result is quarantined (`8aa96ae`).

## 2026-10-26

- Changed: budgets, reconciliation, recovery and promotion gates hardened (`d9cc5ac`).
- Changed: dataset selection restored to the scanner's results after the three-dataset pilot
  (`5e7bd17`).
- Changed: finished simulations are drained before new LLM planning; duplicate refills stopped
  (`41d526c`).
- Changed: confirmed authentication and throttle refusals stay resumable without phantom
  requests (`658782c`); completion counts labelled per invocation (`59fd598`).
- Changed: allocation day aligned with the scanner's 21:00 Pacific window (`0fd7e84`).
- Changed: incomplete LLM output is rejected and research decisions are preserved (`2a6da2c`).
- Changed: a biometric-verification pause survives worker restarts (`6af1abb`).
- Changed: acceptance gates and margin evidence are passed to the diagnosis step (`6d1d547`).

## 2026-10-16

- Changed: enhancement stage. A formula with Sharpe >= 1.2 that missed the pass line is first
  compared under 10-bucket volume or market-cap grouping (4 tries), then the best one gets a
  conditional-trading wrapper (2 tries) (`0842777`).

## 2026-10-15

- Changed: second official run restricted to the 3 datasets that showed signal on 10/15; design
  version v5, counting restarted (`097d91f`).
- Changed: scout 8 field groups first; if none passes 0.8 the dataset is dropped, otherwise more
  groups are added until the dataset budget is used (`9294c79`).
- Changed: extra groups when budget remains (at most 2, de-duplicated); near-miss formulas keep
  improving up to 3 rounds; variants that only change a window or number are rejected
  (`d59fced`).
- Changed: first official run capped at 300 simulations; a cheaper model writes formulas
  (`06ad51f`).
- Fixed: an LLM outage stops the day without mislabelling datasets (`9717213`).

## 2026-10-14

- Changed: current design. The LLM reads all field descriptions and forms groups, writes formulas
  per group, then diagnoses the full history; operator argument counts are checked against the
  platform definition (`fcd42b3`).
- Changed: statistical redesign following ten written questions on noise, thresholds and budget
  (`a21f6ad`).

## 2026-10-09

- Changed: no control arm; datasets already judged by the template scanner go first so results
  compare like for like (`e4c2f5a`).

## 2026-10-06

- Changed: high turnover is a failing check to fix by raising decay; only |Sharpe| > 3 is treated
  as suspect (`4898d9d`).

## 2026-10-05

- Changed: design v1, steps 1 to 9: dataset choice by coverage (>= 60%), idea tables, first tests,
  improvement, luck and robustness checks, daily run (`bacec6e`, `f6d7fa3`, `460e106`).
- Changed: region evidence compares first formulas only, never improved ones (`817613c`).
- Changed: no carrying forward for news and price data (`6fa546c`).

## 2026-10-04

- Added: statistical pipeline: sample, test against noise, allocate by evidence (`3ef2d28`).
- Fixed: login cookie handling (`b172bdd`, `b71ab08`, `e1d0006`); invalid-settings rejections
  fail only that formula (`d25a238`).

## 2026-10-03

- Added: first build of the factory: ledger and cell planner, simulator with pacing and resume,
  expression generators with local pre-checks, idea library and LLM layer, research loop,
  selection, dashboard, `alpha-factory` CLI, config file and scheduled run.
- Added: mining without an LLM and recovery of strongly negative signals (`e1024e3`).

## 2026-09-09 to 2026-09-19

- Added: `alpha-lab` in six milestones: offline auditable engine, knowledge and LLM layer,
  authorized live adapter, MCP interface, portfolio documents, daily production workflow.
- Fixed: concurrency default and a group-classification false positive (`643be4c`); a crash when
  resuming a candidate left SIMULATING after an interrupted retry (`bb69fd4`).
