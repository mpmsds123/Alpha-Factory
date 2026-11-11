# Changelog

Newest first. Every entry before 2026-10-31 is taken from the git history; the short hash
points to the commit. Real expressions and results are not published (see README).

Entry format for new days: **Ran** (what was run, in counts), **Found** (aggregate results
only), **Changed** (code or design), **Open** (unresolved).

## 2026-11-10

- Changed. Documented the research and release contracts. Added a versioned paired
  refinement protocol with durable assignment, settlement and a fixed-rule comparison.
- Changed. Added `review recheck` for existing candidates and their deferred platform checks.
- Open. BRAIN authentication must be restored for live recovery and canary acceptance.

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
  `trade_when` wrapper (2 tries) (`0842777`).

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
