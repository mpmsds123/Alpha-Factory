# Alpha Research Orchestrator

An offline-first, evidence-gated research engine for bounded and reconstructable Alpha
experiments. SQLite is the source of truth: candidate identities, state transitions, gate
decisions, simulation attempts, retries, lineage, policy snapshots, and stop reasons are
persisted.

This repository is an independent clean-room implementation. CNHKMCP and BrainMachine were
read-only architectural evidence; this project does not import, vendor, modify, or depend on
either system. The frozen contracts and safety boundaries are user-authored design decisions,
and all implementation code here is new.

## Implemented milestones

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
- deterministic Markdown run reports and structured secret-redacting logs;
- private Markdown inventory containing hashes and traceable section locators, not raw text;
- explicit paraphrased `method-card` block extraction, human approval, and immutable approved
  versions;
- approved-card-only tag filtering with deterministic BM25-style lexical ranking;
- versioned generator and repair prompts persisted with their content hashes;
- proposal-only CandidateGenerator with explicit uncertainty and single-parent RepairPlanner
  contracts;
- OpenAI-compatible client configuration, strict Pydantic output validation, and an LLM request
  ledger that stores hashes, frozen snapshot/MethodCard version manifests, and usage metadata
  rather than raw inputs or outputs;
- explicitly configured live BRAIN access using a protected external session-cookie file;
- content-addressed field, operator, and setting snapshots frozen to one research cell;
- idempotent simulation creation, one-step polling, normalized metrics and error classes, and
  bounded `Retry-After` handling;
- live platform-check recording that preserves `PASS`, `FAIL`, `WARNING`, `PENDING`, and `ERROR`
  and promotes only a confirmed all-pass response to human review;
- a local stdio MCP 2.x server exposing five narrow ledger and replay tools with typed structured
  output and explicit read/write annotations.

Explicitly deferred: a Web dashboard and automatic submission. A local routing result is never
described as platform approval. LLM and BRAIN calls occur only when the user explicitly supplies
their separate configurations; replay remains fully offline.

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

## Private corpus and method cards

```bash
alpha-lab corpus inventory /path/to/private/markdown \
  --source-type training --db alpha_lab.db
alpha-lab corpus extract SRC-... --db alpha_lab.db
alpha-lab cards list --status draft --db alpha_lab.db
alpha-lab cards approve MC-... --reviewer "Reviewer Name" --db alpha_lab.db
```

Extraction recognizes only explicit `method-card` YAML fences whose content is already
paraphrased. Drafts cannot influence retrieval until a human approval record exists. Forum
experience can guide a proposal after approval, but deterministic gates never convert it into a
platform rule.

## Optional proposal LLM

The library exposes `CandidateGenerator` and `RepairPlanner`; both validate complete output before
inserting any candidate. Configure the client only through the `ALPHA_LAB_LLM_*` environment
variables shown in `.env.example`. API keys are represented as secret values in memory and are not
persisted. Automated tests intercept the HTTP boundary and never contact an external endpoint.

## Authorized live boundary

Live mode is disabled unless `ALPHA_LAB_BRAIN_SESSION_FILE` points to an external JSON cookie
mapping or browser-exported cookie list. On Unix, the file must have no group or other-user access
(for example, `chmod 600 /outside/repository/brain-session.json`). The adapter probes
`/authentication` before any protected request and classifies an inquiry challenge as
`HUMAN_VERIFICATION_REQUIRED`; browser login alone is not treated as proof that the HTTP client is
authenticated.

```bash
alpha-lab run create \
  --dataset DATASET_ID --region USA --universe TOP1000 --delay 1 \
  --adapter live --db alpha_lab.db
alpha-lab run execute RUN_ID --adapter live --db alpha_lab.db
alpha-lab platform-check CANDIDATE_ID --db alpha_lab.db
```

Each execution performs at most one poll of a pending simulation and persists the external ID, so
`run resume` continues without creating the simulation again. `platform-check` requires a
completed live simulation and never submits it. Before execution, proposals are inserted through
the library's `CandidateGenerator`; the live adapter deliberately does not generate expressions.
Endpoint paths can be overridden with the `ALPHA_LAB_BRAIN_*_PATH` variables in `.env.example`.

Milestone 3 verification used intercepted HTTP responses only: no real WorldQuant login,
simulation, or platform check was performed. The default endpoint contract therefore remains to
be confirmed against an authorized account before first live use.

## MCP interface

`alpha-lab-mcp` runs the local server over stdio using the official MCP Python SDK 2.x. Configure
the ledger and optional replay fixture in the host process; tool callers cannot replace either
path:

```bash
export ALPHA_LAB_MCP_DATABASE_PATH=/absolute/path/alpha_lab.db
export ALPHA_LAB_MCP_REPLAY_FIXTURE=/absolute/path/sanitized-replay.json
alpha-lab-mcp
```

The exposed tools are `alpha_run_status`, `alpha_run_report`, `alpha_method_cards_list`,
`alpha_replay_execute`, and `alpha_run_stop`. The first three are read-only. The two write tools
can only advance/resume an existing replay run or record the fixed `MCP_REQUESTED_STOP` reason.
There are no MCP tools for live BRAIN access, LLM invocation, corpus reads, card approval, arbitrary
filesystem paths, or Alpha submission.

## Trust boundary

- Policy thresholds allocate research effort; they are not evidence of tradability.
- `READY_FOR_PLATFORM_CHECK` means only that offline routing selected a candidate.
- `PASS`, `FAIL`, `WARNING`, `PENDING`, and `ERROR` remain distinct platform-check values.
- Credentials belong only in environment variables or user-owned session files. They are
  redacted from persisted errors and logs and are never included in fixtures.
- The package has no submit method or CLI command; every surviving Alpha stops at human review.
