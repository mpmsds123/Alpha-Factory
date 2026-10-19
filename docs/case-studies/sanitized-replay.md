# Sanitized replay

The bundled fixture contains 2 distinct candidates and 2 completed simulation results. It is synthetic and exercises orchestration behavior only.

| Expression | Sharpe | Fitness | Turnover |
|---|---:|---:|---:|
| `rank(ts_mean(close, 5))` | `0.70` | `0.40` | `0.20` |
| `rank(volume)` | `0.10` | `0.10` | `0.15` |

The first candidate is routed more favorably than the second, demonstrating deterministic result handling rather than investment merit. No live platform check was performed.
