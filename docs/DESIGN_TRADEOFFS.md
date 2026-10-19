# Design tradeoffs

- SQLite and one local worker favor auditability and restart safety over distributed throughput.
- Typed LLM output and deterministic gates reduce flexibility but make failures inspectable.
- Fixed budgets and bounded repairs may stop a promising branch early; they also prevent unlimited search.
- Excluding private pilot artifacts limits public reproducibility while protecting platform data and expressions.
- Keeping the existing internal package name avoids a risky cosmetic refactor.
