# Alpha Research Orchestrator

An offline-first, evidence-gated research engine for bounded and reconstructable Alpha
experiments. SQLite is the source of truth: candidate identities, state transitions, gate
decisions, simulation attempts, retries, lineage, policy snapshots, and stop reasons are
persisted.

This repository is an independent clean-room implementation. CNHKMCP and BrainMachine were
read-only architectural evidence; this project does not import, vendor, modify, or depend on
either system. The frozen contracts and safety boundaries are user-authored design decisions,
and all implementation code here is new.

## Milestone 1 boundary

Implemented:

- installable Python 3.11+ package and English Typer CLI;
- Pydantic v2 boundary contracts;
- SQLAlchemy 2.x SQLite schema and Alembic migration;
- conservative canonical identity and pre-simulation deduplication;
- transactional state transitions, decisions, events, and parent-child lineage;
- versioned routing, limit, and error-taxonomy policies;
- deterministic structural validation, routing, backpressure, and stopping;
- bounded retries with stable idempotency keys;
- `FakeBrainAdapter` and sanitized `ReplayBrainAdapter`;
- deterministic Markdown run reports and structured secret-redacting logs.

Explicitly deferred: LLMs, corpus ingestion, live WorldQuant access, MCP, a Web dashboard, and
automatic submission. A local routing result is never described as platform approval.

## Install and verify

```bash
python -m pip install -e '.[dev]'
ruff format --check .
ruff check .
mypy src
pytest
```

Automated tests use only in-memory fakes or local replay JSON. They never contact WorldQuant.

## Offline example

```bash
alpha-lab init --db alpha_lab.db
RUN_ID=$(alpha-lab run create \
  --dataset demo-dataset --region USA --universe TOP1000 --delay 1 \
  --adapter replay --db alpha_lab.db)
alpha-lab run execute "$RUN_ID" --adapter replay \
  --fixture tests/fixtures/basic_replay.json --db alpha_lab.db
alpha-lab run status "$RUN_ID" --db alpha_lab.db
alpha-lab report "$RUN_ID" --db alpha_lab.db
```

`run stop` records an exact human stop reason. `run resume` reconstructs work from SQLite and
does not repeat completed simulations. There is intentionally no `submit` command.

## Trust boundary

- Policy thresholds allocate research effort; they are not evidence of tradability.
- `READY_FOR_PLATFORM_CHECK` means only that offline routing selected a candidate.
- `PASS`, `FAIL`, `WARNING`, `PENDING`, and `ERROR` remain distinct platform-check values.
- Credentials belong only in environment variables or user-owned session files. They are
  redacted from persisted errors and logs and are never included in fixtures.
