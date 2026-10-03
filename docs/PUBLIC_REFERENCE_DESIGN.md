# Public reference design

This project remains an independent implementation. Public repositories were reviewed for
operational patterns only; their code, prompts, credentials, Alpha expressions, and research
archives are not copied or imported.

## Patterns retained

- **Persistent queue and crash resume:** SQLite records work before dispatch, and stable identities
  prevent a restart from creating duplicate simulations.
- **Active, bounded dispatch:** the account ceiling and the currently learned concurrency are
  separate. The effective value grows after clean batches and shrinks after a platform 429.
- **Long-lived worker:** pending simulations are polled by one resumable worker instead of asking a
  human to run a sequence of `resume` commands.
- **Survivor-first routing:** completed candidates enter platform checks immediately; a slow tail
  cannot strand the best result.
- **Human-only submission:** automation stops at a reconstructable review report.

## References reviewed

- CNHKMCP 4.3.6 installed-package audit: generation, inspection, simulation, decision, enhancement,
  and implementation stages; treated as read-only architectural evidence.
- `aaditya-v-more/worldquant-orchestrator`: persisted adaptive concurrency, request pacing, and
  separate read/write retry behavior.
- `angel4angelov-glitch/wq-alpha-pipeline`: SQLite crash recovery, active-futures dispatch, process
  locking, and explicit survivor/correlation stages.
- `QZYWQ/Worldquantbrain`: durable queues, run artifacts, and resumable long-task harnesses.

These references do not establish current platform rules. Live BRAIN responses and the frozen run
contract remain authoritative for this repository.
