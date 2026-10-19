# Offline Replay Report

- Adapter: `replay`
- Dataset: `demo-dataset`
- Candidates evaluated: `2`
- Completed results: `2`
- Candidate states: `1 READY_FOR_PLATFORM_CHECK`, `1 ARCHIVED`
- Gate outcomes: `1 STRONG`, `6 PASS`, `1 DISCARD`, `6 PENDING`
- Infrastructure failures: `0`
- External calls: `0`

| Candidate | State | Sharpe* | Fitness* | Turnover* |
|---|---|---:|---:|---:|
| candidate-a | completed | 0.70 | 0.40 | 0.20 |
| candidate-b | completed | 0.10 | 0.10 | 0.15 |

The stronger synthetic result is routed more favorably; the weaker result remains recorded as
negative research evidence. Every value above comes from `tests/fixtures/basic_replay.json`.

\* Synthetic fixture values, not live or out-of-sample performance.
