# Changelog

Each entry records what was run, what the data showed, what changed, and what is still open. Newest first. Figures cover the current design (2026-10-16 onward) unless stated. Daily development notes before this page are in [docs/progress](docs/progress/README.md); this file supersedes the research figures in earlier README versions.

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
