# Current Package Architecture

Alpha Factory is a single-operator, local Python application. SQLite is the source of truth for dataset cells, ideas, candidates, batches, LLM calls, decisions, lineage, results, and stop reasons.

The LLM is deliberately outside the control plane and sends typed proposals only. It can group fields, propose hypotheses/formulas, and explain a repair. Deterministic code verifies identifiers and operator contracts, rejects duplicates, enforces budgets, dispatches simulations, classifies results, and decides whether another bounded step is allowed.

The active workflow lives in `src/alpha_orchestrator/factory/`. Platform access lives in `brain/`, canonical identity and timestamps in `domain/`, and structural gates in `gates/`. Replay adapters, storage contracts, and tests provide the local safety harness.

Every external batch is recorded before polling. A restart reads unresolved ledger rows and resumes instead of silently recreating work. Final candidates stop at a human review boundary; there is no submission path in the orchestration layer.

