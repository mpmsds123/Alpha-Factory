# Operator selection guide

This public guide describes decision categories rather than reusable alpha formulas.

- Cleaning operators address missing observations and extreme values.
- Cross-sectional rank or standardization is appropriate when relative position matters more than raw scale.
- Time-series change or normalization is appropriate when the signal is a change from the asset's own history.
- Group transforms require an economic reason that values are comparable only within peers.
- Smoothing and decay are implementation choices used to address observed turnover or instability.
- Conditional activation requires a separately motivated state variable and must not be added only because a backtest improved.
- Combination operators require a mechanism for interaction; do not combine fields merely to search more variants.
- Prefer the smallest expression that tests the stated hypothesis.

