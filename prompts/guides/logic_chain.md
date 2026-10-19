# Single-signal decision guide

This short guide is independently written for the public repository.

1. State the economic mechanism and expected direction before choosing an operator.
2. Check missingness, update frequency, coverage, and extreme values.
3. Choose one primary representation: level, recent change, historical deviation, or peer-relative value.
4. Use a group transform only when the mechanism requires comparison among defensible peers.
5. Add smoothing or conditional activation only to address an observed failure such as turnover or stale data.
6. Change one structural decision at a time so parent-child results remain interpretable.
7. Stop when the fixed repair count or backtest budget is exhausted.

