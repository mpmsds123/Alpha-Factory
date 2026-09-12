# 60-Second Technical Explanation

Alpha Research Orchestrator is an offline-first Python system for running bounded, auditable Alpha
experiments. A researcher starts the CLI, which freezes one dataset and market cell, then sends
typed proposals through deterministic identity, syntax, budget, and routing gates. SQLite is the
source of truth for every candidate, decision, lineage edge, simulation attempt, retry, policy
snapshot, and stop reason, so a report can reconstruct the run without console history. Replay and
fake adapters make the complete workflow testable without WorldQuant. An optional live adapter is
separately configured, persists external IDs, and performs only one resumable poll at a time. The
LLM has narrow authority: it may propose candidates or a single-parent repair, but deterministic
code validates, classifies errors, enforces limits, and interprets platform states. A local strong
metric means only “consider a platform check.” The package has no automatic submission path; every
survivor stops at human review. A narrow MCP server exposes ledger reads, replay execution, and a
fixed stop action without expanding that authority boundary.
