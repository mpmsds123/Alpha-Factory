# Changelog

Each entry records what was run, what the data showed, what changed, and what is still open. Newest first. Figures cover the current design (2026-10-16 onward) unless stated. Daily development notes before this page are in [docs/progress](docs/progress/README.md); this file supersedes the research figures in earlier README versions.

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
