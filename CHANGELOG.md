# Changelog

Newest first. Every entry before 2026-10-31 is taken from the git history; the short hash
points to the commit. Real expressions and results are not published (see README).
Operational detail that used to live in `docs/` is summarized in the entries below.

Entry format for new days: **Ran** (what was run, in counts), **Found** (aggregate results
only), **Changed** (code or design), **Open** (unresolved).

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
