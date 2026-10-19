# Sanitized Replay Example

This example is a deterministic, offline execution trace. It uses synthetic identifiers and
metrics; no credential, private field, proprietary expression, or live platform result is present.

Run it with:

```bash
make demo
```

Artifacts:

- [`input.json`](input.json): frozen synthetic metadata, candidates, and simulator responses;
- [`decision_trace.json`](decision_trace.json): compact state transitions and budget accounting;
- [`report.md`](report.md): representative report produced from ledger rows.

What the example proves: two proposals can be validated, persisted, simulated through the replay
adapter, and reported deterministically without a network call. It does not prove predictive value.
