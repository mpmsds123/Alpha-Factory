# Codex build prompt

Build a new repository named `alpha-research-orchestrator` from the adjacent `IMPLEMENTATION_SPEC.md`.

The new repository must be independent from CNHKMCP and BrainMachine. Treat `/Users/meilinpan/BrainMachine`, the installed `cnhkmcp` package, and all private corpus directories as read-only evidence. Do not modify, import, vendor, or depend on them. Do not run a live WorldQuant simulation or platform check without a separate explicit user request.

Follow these delivery rules:

1. Implement the milestones in order. Complete Milestone 1 before adding any LLM or live-network behavior.
2. Use Python 3.11+, Pydantic v2, SQLAlchemy 2.x with Alembic, Typer, HTTPX, pytest, respx, Ruff, and an installable `pyproject.toml` package.
3. Make SQLite the source of truth. Every candidate transition, gate decision, prompt version, simulation attempt, retry, lineage edge, and stop reason must be reconstructable.
4. Enforce candidate deduplication, queue backpressure, idempotency, error classification, lineage, and all configured ceilings in deterministic code.
5. The LLM may propose candidates and single-parent repairs only. It may not validate syntax, classify transport errors, change limits, declare a platform pass, or submit an Alpha.
6. Implement `FakeBrainAdapter` and `ReplayBrainAdapter` first. Automated tests must never contact WorldQuant.
7. Include sanitized regression fixtures for the historical CNHKMCP failure modes described in the specification. Do not copy private Alpha expressions or raw forum/training text.
8. Do not implement automatic submission or a Web dashboard. Defer MCP until the offline CLI and core engine pass all acceptance criteria.
9. Never write credentials to the database, logs, reports, fixtures, prompts, or Git.
10. Keep all code, schemas, prompts, CLI text, comments, tests, and documentation in English.

For each milestone:

- begin by restating its file boundary and acceptance criteria;
- implement production code and tests together;
- run formatting, linting, type checks and the complete test suite;
- report exact verification results and any unimplemented boundary;
- commit the milestone only after its acceptance criteria pass.

Start with Milestone 1 only. Do not begin Milestone 2 in the same change.
