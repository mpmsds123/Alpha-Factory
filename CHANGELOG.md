# Changelog

Each entry records what was run, what the data showed, what changed, and what is still open. Newest first. Figures cover the current design (2026-10-16 onward) unless stated. Daily development notes before this page are in [docs/progress](docs/progress/README.md); this file supersedes the research figures in earlier README versions.

## 2026-11-06 · weekly report (design v7)

Window: allocation days 2026-10-31 to 2026-11-06 (Pacific). The 10-30 evening run is in the previous entry. Design v6 ran until the evening of 11-03 (222 backtests), v7 after it (1,992 backtests).

**1) Summary**

2,214 backtests completed and 0 submittable ideas entered the pick-list; 7 of the 9 ideas that reached the platform's submission check were stopped only by the account's daily submission-count limit, which the factory records as a permanent failure.

**2) Performance**

| | This week | Prior week |
|---|---|---|
| Backtests completed | 2,214 | 375 |
| Quota use (1,000/day × 7) | 31.6% | 5.4% |
| Submittable ideas (in pick-list) | 0 | 0 |
| Submittable ideas per 1,000 backtests | 0.00 | 0.00 |
| DeepSeek spend | ¥23.31 | ¥8.90 |
| Spend per submittable idea | n/a (none) | n/a (none) |

Prior-week figures use the same report function over 10-24 to 10-30. Backtests by Pacific date: 11-01 222, 11-03 964, 11-04 343, 11-05 588, 11-06 97; none on 10-31 or 11-02.

Coverage: 55 datasets in 8 regions. Backtests / datasets by region: USA 500 / 15, EUR 497 / 12, ASI 473 / 6, DEU 319 / 3, JPN 169 / 6, AMR 147 / 7, GLB 85 / 5, GBR 24 / 1.

Idea funnel (ideas finished this week):

| Step | Ideas | Pass rate |
|---|---|---|
| Ideas with ≥ 1 valid backtest | 528 | |
| First batch best \|Sharpe\| ≥ 1.0 | 115 | 21.8% |
| Any backtest with Sharpe ≥ 1.2 (incl. improvements) | 97 | 84.3% |
| Sharpe, fitness, turnover and margin lines all met | 28 | 28.9% |
| All in-backtest platform checks pass | 9 | 32.1% |
| Submission check passed → pick-list | 0 | **0%** |

A funnel comparison with the prior week is not possible: all 98 ideas created that week were later retired, and the report excludes retired ideas (51 more were retired this week).

**3) What the bandit learned**

Success = the variant fails fewer platform checks than its parent. 408 menu fixes this week, 48 successes (11.8%); the model chose every fix, the bandit fallback was not used. Pairs with ≥ 3 attempts:

| Problem | Fix | Success / attempts |
|---|---|---|
| Excess turnover | raise decay setting | 9 / 11 |
| Excess turnover | 5-day smoothing | 9 / 12 |
| Weak Sharpe | switch neutralisation level | 5 / 9 |
| Weak Sharpe | peer-group normalisation | 5 / 19 |
| Weak Sharpe | trade only in volatile regimes | 4 / 17 |
| Weak Sharpe | neutralise within liquidity buckets | 3 / 15 |
| Weak Sharpe | standardise within size buckets | 1 / 5 |
| Weak fitness | neutralise within size buckets | 1 / 6 |
| Weak fitness | neutralise within liquidity buckets | 1 / 5 |
| Unstable over time | neutralise within liquidity buckets | 2 / 11 |
| Unstable over time | 21-day smoothing | 2 / 14 |
| Unstable over time | switch neutralisation level | 1 / 6 |
| Unstable over time | trade only in volatile regimes | 1 / 7 |
| Unstable over time | trade only on activity spikes | 0 / 6 |
| Concentrated in few stocks | peer-group normalisation | 1 / 4 |
| Concentrated in few stocks | standardise within size buckets | 0 / 4 |
| Concentrated in few stocks | clip outliers | 0 / 4 |
| Fails in part of the universe | switch neutralisation level | 3 / 49 |
| Fails in part of the universe | standardise within size buckets | 0 / 37 |
| Fails in part of the universe | neutralise within size buckets | 0 / 21 |
| Fails in part of the universe | neutralise within liquidity buckets | 0 / 19 |
| Fails in part of the universe | standardise within liquidity buckets | 0 / 16 |
| Under another line (margin or unlisted check) | neutralise within size buckets | 0 / 34 |
| Under another line (margin or unlisted check) | neutralise within liquidity buckets | 0 / 32 |
| Under another line (margin or unlisted check) | switch neutralisation level | 0 / 27 |
| Under another line (margin or unlisted check) | trade only on activity spikes | 0 / 14 |

Turnover is the one problem the menu solves (18 / 23). Sub-universe and "other" problems took 249 attempts for 3 successes.

**4) Changes this week**

- 11-01: first-formula gate raised from 0.8 to 1.0; ideas must state a horizon, a falsifier and a sign per field. Basis: measured no-signal Sharpe SD 0.36 to 0.58, not 0.316; no idea that later reached 1.2 had a first formula under 1.0.
- 11-01: no new idea while 20 simulations are queued (up to 48 were waiting on 10-31).
- 11-01: pool review: the model proposes stop or enhance per idea from program-computed hit rates; the program validates. Off by default.
- 11-01: a batch whose progress link expired is shut instead of crashing the run.
- 11-03: daily progress file; the run reuses the biometric-checked session.
- 11-03, design v6: the model works only on fields where the scanner found signal (such fields pass again in another template 23% of the time vs about 0% for the rest; 1,871 fields).
- 11-03, design v7: free-form refinement replaced by a fixed menu of fixes with Thompson sampling (206 free-form variants: mean ΔSharpe −0.215, 95% CI −0.257 to −0.174, none gained 0.35); idea memory across datasets; datasets drawn at random; 6 ideas in flight (v6 used 17% of quota).
- 11-04: fixed the login hand-off that shut the session before the run used it (cause of the 11-02 stop).
- 11-04: four first-day v7 bugs: unlisted checks routed to "other" (79 variants, 0 helped) now diagnosed by pattern; a fix could wrap itself; an unsupported region setting (12 refused); the weekly report counted artefacts as submittable.

**5) Risks / known issues**

- Submission check misreads an account limit as an alpha failure. 10 alphas from 9 ideas reached the submission check. 8 alphas (7 ideas) failed only the daily submission-count check (4 of 4 used that day). The factory flagged them as permanently failed and never re-checked them. Of the other two, one had already been submitted and one failed the in-sample ladder and minimum-duration checks. Caveats for the 8: all carry a warning on recent two-year Sharpe (0.52 to 1.23 against 1.58), correlation checks were still pending, and all 8 come from one dataset (DEU model), so they are likely correlated with each other.
- Runs stop on empty field grouping. 3 of the 4 run ends this week (11-05, and 11-06 twice) were "no field groups". The progress file reports this as an LLM failure from balance or network, but all 1,010 DeepSeek calls this week returned successfully. The cause is in how the grouping answer is accepted. 11-05 and 11-06 together used 685 of 2,000 backtests.
- Fix budget spent where the menu does not work. After the 11-04 rerouting, "under another line" went 0 / 29 and "fails in part of the universe" 3 / 141. Thompson sampling should move away from these fixes, but every fix on the menu for these problems is failing.
- Lost backtests: 71 of 2,285 completed records have no result. 20 were lost when progress links expired, 15 cancelled for a unit mismatch in the formula, 12 refused for the unsupported region, 12 were platform-side errors, 8 hit the resource limit and 4 had no child result. One batch was refused pending biometric login (11-05).
- In-backtest check failures, by check (2,214 backtests; 115 or 5.2% passed all): Sharpe 1,790; fitness 1,754; recent two-year Sharpe 1,665; sub-universe Sharpe 869; Japan sub-region Sharpe 472; robust-universe return 368 and Sharpe 361; concentrated weight 349; in-sample ladder 179; excess turnover 169.

**6) Next week**

- **Change:** in the pick-list step, a failure of the daily submission-count check means "retry later", not "alpha failed". Such alphas are not flagged as permanently failed. Alphas whose only recorded failure is that check (this week's 8) become eligible again and are re-checked on the next run.
- **Basis:** the submission step has the smallest pass rate in the funnel, 0 of 9 ideas. 7 of those 9 were blocked only by this account-level check.
- **Expected effect:** the 8 held alphas get a fresh check on the first run. At most 7 submittable ideas would result (3.2 per 1,000 backtests at this week's backtest count). The likely number is smaller because of the two-year Sharpe warnings, the pending correlation checks and the single source dataset.
- **Test next week:**
  - No alpha is flagged as permanently failed on the submission-count check alone.
  - Each of the 8 has a check result dated after the change.
  - The pick-list has at least one row, or each of the 8 has a recorded reason that is not the account limit.
  - Report the submission-step pass rate next Friday.

## 2026-10-31 (evening): offline audit of the LLM touchpoints

**Ran**
- No simulations and no model calls. Read-only analysis of the ledger copy for the current design: 817 completed backtests (12 in flight), 190 ideas, 331 DeepSeek calls.

**Found**
- Model spend is ¥19.61: improvement diagnosis 51%, field grouping 32%, first-order formulas 17%. Each task uses one model, so the ledger cannot compare models.
- 16 of 138 pro-model calls failed and cost ¥3.12 (16% of spend). 15 were empty outputs that had been counted as successes until corrected on 10-27; all 8 failed grouping calls ended at exactly 4,096 output tokens.
- All 559 first-order formulas use only fields from their idea's declared group. The declared horizon is 21 days for all 190 ideas, so it carries no information.
- The idea's stated direction predicts the sign of Sharpe only for "higher is better": 66% of 383 formulas are positive, against 50% (n = 117) and 57% (n = 40) for "lower is better", which is no better than a coin flip. Sign is read from a leading negation only; formulas that encode direction by subtraction order are not classified.
- No exact repeat among 829 records. The 559 first-order formulas reduce to 218 structural skeletons, the five most common covering 28%; most ideas apply three wrappers (level, change, within-industry rank) to one field.
- Improvement diagnoses match the parent's measured failure in every case checked (121 of 121 low fitness, 23 of 23 high turnover). Turnover fixes lowered turnover in 83% of 23 variants; fitness fixes raised fitness in 26% of 121. 38 of 168 variants (23%) beat their parent; none reached the +0.35 replacement threshold. Parents are the best of three first-order formulas, so some regression is expected; separating it from the model's contribution needs a random-variant control.

**Changed**
- No code or configuration.

**Open**
- Eight candidate changes, each to be tested one at a time against the current design on identical datasets and budget: a control protocol; templates with program-side field-family expansion, testing one representative first; a bounded work queue; cross-idea memory (an offline predictability check first); a noise-calibrated entry gate (observed σ 0.37 to 0.52 against 0.316 configured); a random-variant control for the improvement step on the 50 existing parents (about 168 simulations); program-screened, model-proposed keep/drop decisions; falsifiable idea specifications.
- Reconciling the six quarantined tasks and the six in-flight tasks is the precondition for further simulations.

## 2026-10-31

**Ran**
- Read the full backtest ledger (3,862 records, 2026-10-03 to 2026-10-31) and split it by design version: probe/survey (2,429), four ideas per dataset (180), first LLM loop (424), current design (829).
- Re-derived every statistic on the README from the ledger: 817 completed backtests, 186 ideas, 7 datasets in the current design.
- Ran the factory on the evening of 10-30 Pacific on a dataset unrelated to the quarantined batch: 92 new records, 24 ideas, one more dataset closed.

**Found**
- The first-formula gate assumes a no-information Sharpe SD of 0.316. Observed SD is 0.52 overall and 0.37 to 0.39 in the two lowest-yield datasets. The gate's false-positive rate per idea is 11% to 33%, not 3.4% (README E1).
- The improvement step has not helped: 0 of 168 variants gained 0.35; mean change −0.21 (95% CI −0.26 to −0.17); no control arm exists (E3).
- Enhancement has a negative mean (−0.10) but produced the highest Sharpe in the ledger, 1.83 (E4).
- Datasets differ strongly in the share of first formulas above 0.8: 4.9% to 42.6% (χ² = 53.6, p = 2.5×10⁻¹⁰) (E5).
- All 22 formulas with Sharpe ≥ 1.58 fail at least one platform check; the most common are the sub-period (16) and sub-universe (11) checks (E7).
- Quota is not the binding constraint: 17% of capacity used over 5 active days (E9).
- Every closed dataset carries the label "noise", including one with 22.5% gate-level formulas (E6).

**Changed**
- README rewritten as a problem-first statistical design: formulas, worked example, parameter and assumption tables, per-choice evidence, results and limitations.
- Earlier README figures (102 ideas, 523 backtests, 59 early stops) are replaced; they came from a cohort that mixes design versions.
- No code or threshold was changed.

**Open**
- One batch of six tasks remains quarantined with unknown outcome; six more were in flight at the last record.
- Public code lags the private working system; the progress log of 10-30 predates the evening's 92 records.
- No control arm, no out-of-sample test, no non-LLM comparison.
- Decisions pending with the owner: re-estimate σ per dataset; raise the gate from 0.8 to 1.0; relabel dataset closure; fund a random-variant control for the improvement step.
