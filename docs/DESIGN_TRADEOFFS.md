# Design Tradeoffs

| Decision | Benefit | Cost accepted |
| --- | --- | --- |
| SQLite as the source of truth | Transactions, inspectability, and simple replay on one machine | No distributed workers or high write concurrency |
| Conservative canonical identity | Duplicate proposals do not consume simulation budget | Semantically equivalent but syntactically different formulas can remain distinct |
| Deterministic BM25-style retrieval | Offline, reproducible, and easy to audit | Less semantic recall than an embedding service |
| Replay-first adapters | Full workflow testing without credentials or network drift | Live endpoint compatibility still requires authorized verification |
| Explicit live configuration | Accidental live execution is difficult | More operator setup and no seamless browser-session reuse |
| One poll per execute | Every wait is resumable and externally controllable | No background job continuously waits for completion |
| Single-parent repair only | Lineage and causal attribution remain understandable | No crossover or open-ended autonomous search |
| LLM proposal-only authority | Creative suggestions cannot override rules or evidence | Deterministic validators may reject ideas the model could have explained |
| CLI first, narrow MCP second | One stable service layer and a small automation surface | MCP cannot operate live mode or manage the private corpus |
| Human-only submission | Prevents research routing from becoming an execution decision | A reviewer must move results into any external submission workflow |
| No dashboard | Engineering effort remains on evidence and state correctness | Operators use CLI reports or build a separate read-only view |

The central choice is deliberate conservatism: an attractive metric changes research routing, not
the truth status of an Alpha. This keeps a local threshold, a platform response, and a human capital
decision as three separate layers.
