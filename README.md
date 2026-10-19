# Alpha Factory

Alpha Factory is a local research system for **LLM-guided alpha research under a fixed backtest budget**. It turns a dataset snapshot into traceable hypotheses, validates every proposed expression with deterministic code, allocates a bounded simulation budget, learns from recorded failures, and stops at human review.

The authority split is deliberate: the LLM proposes and diagnoses; code owns validation, deduplication, budgets, scheduling, platform-state handling, persistence, and stopping.

## Why this project exists

Alpha research can waste most of its budget on invalid expressions, repeated ideas, weak datasets, and unstructured retries. Alpha Factory makes those decisions explicit and reconstructable:

1. freeze a dataset and market cell;
2. group economically related fields from descriptions;
3. ask the LLM for hypotheses and a small set of distinct formulas;
4. reject unknown fields, invalid operator arity, invalid group inputs, duplicates, and policy violations locally;
5. dispatch accepted candidates in resumable batches;
6. diagnose completed results and permit only bounded, evidence-linked repairs;
7. apply robustness and platform checks;
8. store the decision trail in SQLite and stop before submission.

## What I implemented

- a stateful orchestration loop with restart-safe SQLite persistence;
- typed LLM requests and structured response validation;
- local checks for field existence, vector reduction, operator arity, group inputs, and complexity;
- canonical fingerprints and pre-simulation deduplication;
- budget ceilings, batch pacing, adaptive concurrency, and explicit stop reasons;
- result-driven repair with bounded rounds and parent-child lineage;
- dataset/candidate filtering, robustness checks, and human-only final selection;
- replay/fake adapters and automated tests that run without external calls;
- deterministic HTML/CSV reporting generated from ledger rows.

These are engineering claims, not claims of predictive performance. The repository does **not** claim that the LLM improves alpha quality, that any candidate is production-ready, or that local tests establish out-of-sample value.

## Architecture

```text
dataset metadata -> LLM hypotheses/formulas -> deterministic validation
        -> deduplicated queue -> budgeted simulator -> diagnosis/repair
        -> robustness/platform checks -> SQLite evidence -> human review
```

The recruiter-facing entry point is `alpha-factory`. Supporting modules provide identity, validation, adapter, persistence, and replay primitives used by the main workflow.

## Safe local verification

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
ruff check .
```

Tests use fakes, intercepted HTTP responses, or sanitized replay fixtures. They do not log in to or spend quota on an external platform.

## Configuration and live boundary

`factory.yaml` contains public example values only. Credentials and session material must stay outside the repository. Live research requires separately authorized LLM and platform sessions; no credential, cookie, real field identifier, proprietary expression, external alpha identifier, or production database is included here.

The code can prepare candidates for review, but submission is never automatic. `policy.auto_submit` remains `false`, and a person must approve any final action.

## Evidence boundary

- Automated tests demonstrate local code behavior and failure handling.
- The included configuration is illustrative, not a record of a production run.
- Private pilot reports were excluded because they contain platform field names, expressions, and identifiers.
- No public result here should be read as proof of tradability or platform approval.

See [Architecture](docs/ARCHITECTURE.md), [test evidence](docs/TEST_EVIDENCE.md), [design tradeoffs](docs/DESIGN_TRADEOFFS.md),
[known limits](docs/KNOWN_LIMITS.md), the [60-second explanation](docs/60_SECOND_EXPLANATION.md),
and the [sanitized replay case study](docs/case-studies/sanitized-replay.md).

## License and third-party boundary

Project source is released under the MIT License. Runtime dependencies remain under their own licenses. No third-party project source code, prompts, private research notes, or proprietary expressions are vendored in this public snapshot; see [NOTICE](NOTICE.md).

