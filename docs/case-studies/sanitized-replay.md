# Sanitized Replay Case Study

## Question and evidence boundary

Can the offline engine reconstruct a bounded two-candidate research run and distinguish a locally
strong result from a weak result without making a platform or tradability claim?

This case was executed on 2026-09-11 with package version 0.4.0 immediately before the portfolio
release. It used `tests/fixtures/basic_replay.json`, a sanitized fixture containing demo fields and
expressions only. No LLM, network request, live WorldQuant simulation, platform check, or submission
was involved.

## Frozen run contract

- Dataset: `demo-dataset`
- Cell: `USA x TOP1000 x Delay 1`
- Data type: `MATRIX`
- Adapter: `replay`
- Routing and limit policies: `routing.v1` and `limits.v1`
- Run ID: `20281e45-251b-4008-bb9e-d9ec69640d4f` in a disposable SQLite ledger

## Reproduction

```bash
alpha-lab init --db case.db
alpha-lab run create \
  --dataset demo-dataset --region USA --universe TOP1000 --delay 1 \
  --adapter replay --db case.db
alpha-lab run execute RUN_ID --adapter replay \
  --fixture tests/fixtures/basic_replay.json --db case.db
alpha-lab report RUN_ID --db case.db
```

## Recorded result

- 2 proposal events became 2 distinct candidates; 0 duplicates were reused.
- Both candidates passed structural validation.
- 2 simulation attempt rows produced 2 completed simulation results.
- There were 0 infrastructure failures and 0 unresolved attempts.
- `rank(ts_mean(close, 5))` recorded Sharpe `0.70`, fitness `0.40`, and turnover `0.20`.
  Deterministic routing labeled it `STRONG`, leaving 1 candidate in
  `READY_FOR_PLATFORM_CHECK`.
- `rank(volume)` recorded Sharpe `0.10`, fitness `0.10`, and turnover `0.15`.
  Deterministic routing labeled it `DISCARD`, leaving 1 candidate `ARCHIVED`.
- The run stopped as `COMPLETED` with no stop reason. Automatic submission was unavailable.

## Verdict

The replay demonstrates candidate accounting, pre-simulation gates, persisted attempts,
deterministic routing, and report reconstruction on sanitized inputs. `READY_FOR_PLATFORM_CHECK`
allocates the next unit of research effort; it is not evidence that the expression is robust,
profitable out of sample, approved by WorldQuant, or suitable for capital. No live platform check
was performed.
