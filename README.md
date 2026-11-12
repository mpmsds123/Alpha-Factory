# Alpha Factory

A research factory for daily cross-sectional equity signals on a simulation platform. An
LLM proposes economic hypotheses and the expressions that test them; the program owns
every budget, gate and stop. Nothing is ever submitted automatically.

## The problem

Fixed-template scanners cover what their templates can express and nothing else. Free
LLM generation covers more, but it overfits: thousands of loosely motivated variants,
selected on in-sample results, look good and then fail out of sample. This factory tests
**hypotheses**, not field × template combinations, and searches each one with a small,
fixed budget, a stopping rule and a held-out period.

## How it works

```
program: pick a dataset and a field        (Thompson draw; coverage, family, usage rules)
LLM (a): hypothesis                         mechanism, direction, type, holding period,
                                            expected turnover, falsifier
LLM (b): 1-3 expressions                    checked locally: operators, arity, fields,
                                            windows, direction, originality, repeats
program: simulate                           with the last 2 years held out as a test period
LLM (c): diagnosis                          idea supported / idea failed / implementation failed
program: stopping rule                      round-1 gate, rounds, dry rounds, budget
LLM (d): decision                           mutate (one declared change per child), revise, drop
program: gates                              train numbers -> test period -> platform checks
you:     review approve                     the only path that submits
```

* **Program layer** (never touched by the LLM): submission gates copied from the official
  documentation, simulation budget per hypothesis (12; 6 for a revision), the stopping
  rule, the out-of-sample gate, a daily LLM dollar cap and a dollars-per-passing-alpha
  guard. Every number and its source is in `src/alpha_orchestrator/factory/config.py`.
* **LLM layer**: four decision points, strict JSON validated against a schema. Each answer
  must cite the memory records and priors it used; citing an id that was not retrieved
  is an error, citing nothing is flagged. The LLM only ever sees training-period numbers.
* **Memory A (priors)**: claims from documents, each with its source, a verbatim quote
  and a Beta(a, b) confidence that the factory's own results update.
* **Memory B (experience)**: one record per hypothesis. Records are retrieved by
  program-defined tags (dataset category, region, field type, hypothesis type), never by
  embeddings, and every decision logs which ids it was shown.
* **Expressions are trees**: canonical text, structure shape, operator and node counts,
  and the exact change between a child and its parent.

Prior art this borrows from: R&D-Agent(Q) (hypothesis → experiment → feedback that
separates the idea from its implementation), AlphaAgent (originality against existing
formulas, hypothesis-expression alignment, complexity limits), FunSearch/AlphaEvolve
(a deterministic evaluator outside the LLM, best-so-far shown in the mutation prompt,
each hypothesis its own island).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
make install
```

### Private folder

Platform vocabulary never enters the repository. It lives in `~/.alpha_factory/private/`:

* `platform.yaml`: the API address, the expression-language name, grouping identifiers,
  operator sign rules, preferred universes, default neutralizations, document sources
  that count as official, and the names of the language-notes files shown to the LLM
* `factory.local.yaml`: machine-specific overrides of `factory.yaml` (paths, regions)

### Secrets

* LLM key: `ALPHA_FACTORY_LLM_API_KEY`, or `{"api_key": "..."}` in the `llm.key_file`
* Platform login: `ALPHA_LAB_BRAIN_USERNAME` / `ALPHA_LAB_BRAIN_PASSWORD`, or the
  credentials file named by `BRAIN_CREDENTIALS_FILE`

### Run

```bash
alpha-factory memory import-v7        # once: earlier results become experience records
alpha-factory priors seed             # once: claims from official documents (capped cost)
alpha-factory login --then-run        # face check if asked, then a run of up to 12 hours
alpha-factory review list             # alphas that passed every gate
alpha-factory review approve <id>     # submit one (cannot be undone)
```

`alpha-factory run`, `status`, `stats`, `priors add`, `review reject` and `recover-batch`
are also available; each has `--help`. In VS Code, run
`python -m alpha_orchestrator.factory.cli run` from the integrated terminal. Runs are
started by hand; there is no scheduler.

## Configuration

`factory.yaml` overrides the defaults in `config.py` (budgets, cost caps, models per
step, memory weights); the submission gates are not overridable. Progress is written
during a run to `reports/` (git-ignored, it contains real expressions):

| File | What |
|---|---|
| `stages.csv` | per quota day: counts entering and surviving each stage, LLM dollars |
| `rejections.csv` | why expressions were rejected before simulation |
| `by_prompt_model.csv` | calls, dollars and passing alphas per dollar by step, prompt version and model |
| `hypotheses.csv` | Memory B, one row per hypothesis |
| `priors.csv` | Memory A with current confidence |
| `review_queue.csv` | alphas stored for review |

The database (`~/.alpha_factory/factory_v8.db`) holds the tables `hypotheses`,
`expressions` (with `parent_id` lineage), `simulations` (train, test and whole-period
metrics apart), `decisions` (step, inputs, retrieved ids, output, citations, flags,
prompt version, model, cost), `priors`, plus a catalog cache and in-flight batches.

## Safety boundaries and known limits

* Only `review approve` submits; it re-runs the platform checks first.
* One writer per database (OS file lock); interrupted batches are re-polled, never resent.
* The out-of-sample gate is only as honest as its isolation: no prompt may contain a
  test-period value (enforced in the LLM client), and nothing selects on it before the gate.
* Thresholds marked "ledger" in `config.py` were measured on earlier runs and should be
  re-measured once this design has its own data.

## Development

```bash
make check   # black, ruff, mypy, pytest (coverage >= 85%)
```

Tests use a made-up expression language and an in-memory platform; a guard test scans
every tracked file for platform vocabulary.

## License

MIT
