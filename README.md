# Alpha Factory

An LLM-assisted research pipeline that searches for quantitative trading signals ("alphas")
on a cloud backtesting platform, under a fixed daily simulation budget, and stops every
result at human review.

## The problem

Mining alphas by brute force (expanding templates over every data field) burns simulation
quota on ideas with no economic reason behind them, and the few that pass are often noise
that happened to clear a threshold. Doing it by hand does not scale. This project lets a
language model propose *reasoned* research ideas while deterministic code owns everything
that has to be trustworthy: budgets, validation, statistics, identity, crash recovery and
the audit trail. Nothing is ever submitted automatically.

## How it works

One run is one allocation day. The loop repeats until the daily budget, the time limit or
the available ideas run out:

```
 pick a dataset ─► LLM groups its fields into economic ideas ─► LLM writes first formulas
 (scanner signal only)
        ▲                                                          │
        │                                        local validator rejects invalid ones
        │                                                          ▼
 next idea / dataset ◄── idea dies (below the luck line) ◄── simulate on the platform
                                                                   │ promising
                                                                   ▼
                      diagnose failed checks ─► pick fixes (menu + Thompson sampling)
                                                                   │ passes the line
                                                                   ▼
                      robustness test ─► platform submission checks ─► review list
                                                                   │
                                                  you approve ─► submit (manual only)
```

- **Scanner evidence first.** Only datasets and fields in which the separate template scanner
  found signal are mined, and a first formula that repeats the structure of a formula the
  scanner already ran is sent back as a rewrite rather than a new alpha.
- **Ideas, not templates.** The LLM reads field descriptions and past outcomes and proposes
  field groups with an economic rationale; every LLM call is budgeted and stored with hashes.
- **Statistics before effort.** An idea whose best first formula does not clear the level
  that noise alone would reach for that many tries is dropped instead of tuned.
- **Bounded refinement.** Failed checks are mapped to a fixed menu of fixes; a Thompson-sampling
  bandit chooses among them using the record of past repairs.
- **Budget and recovery.** Simulations are reserved in a SQLite ledger *before* they are sent,
  so a crash or restart never overspends or duplicates work; one worker owns one ledger, and an
  ambiguous platform response blocks further runs until it is reconciled.
- **Human control.** Survivors are re-checked by the platform and land on a review list.
  `alpha-factory review approve` is the only code path that submits.

The repository also contains **`alpha-lab`**, the earlier offline-first engine (replay
adapter, method-card retrieval, narrow MCP server). It is kept working and is what the
offline demo below exercises; `alpha-factory` is the current runtime.

## Quick start

Requires Python 3.11+ (verified on 3.12, macOS).

```bash
python3 -m venv .venv
source .venv/bin/activate
make install          # pinned versions from requirements.txt, then the package itself
make check            # black, ruff, mypy and the full offline test suite
make demo             # offline end-to-end run against a sanitized replay fixture
```

Tests and the demo never contact the platform or an LLM.

### Configure secrets

Secrets come only from environment variables or private files outside the repository.
Copy `.env.example` to `.env` (git-ignored), fill in what you use, and load it:

```bash
set -a; source .env; set +a
```

| Purpose | Environment variable | Fallback |
| --- | --- | --- |
| LLM key for `alpha-factory` | `ALPHA_FACTORY_LLM_API_KEY` | `llm.key_file` in `factory.yaml` |
| Platform login | `ALPHA_LAB_BRAIN_USERNAME`, `ALPHA_LAB_BRAIN_PASSWORD` | `BRAIN_CREDENTIALS_FILE` |
| `alpha-lab` LLM and live session | `ALPHA_LAB_LLM_*`, `ALPHA_LAB_BRAIN_*` | none (offline) |

Credential files must be readable only by you (`chmod 600`).

### Run the factory

```bash
alpha-factory doctor                  # read-only: budget, in-flight and blocked batches
alpha-factory login --then-run        # complete the platform's verification, then run a day
alpha-factory run                     # run one allocation day (resumes where it stopped)
alpha-factory dashboard               # open the local dashboard
alpha-factory funnel                  # read-only: rewrite the per-stage funnel CSVs in reports/
alpha-factory review list             # candidates waiting for you
alpha-factory review approve ID       # the only command that submits
alpha-factory schedule install        # optional: daily run via macOS launchd
```

If a run stops on an ambiguous batch, inspect it on the platform and reconnect it with
`alpha-factory recover-batch BATCH_ID --location /simulations/VERIFIED_ID`; never delete it.

## Configuration

- **`factory.yaml`** overrides the defaults in `src/alpha_orchestrator/factory/config.py`,
  where every tunable number carries a comment saying where it came from. Key settings:
  daily simulation caps (`budget.v1_daily_cap`, `budget.daily_sim_cap`; the lower wins), the
  quota reset time, pass thresholds, LLM models and the regions to search.
- **Trial switches** (`direction_trial`, `refinement_trial`) turn on paired experiments that
  compare an LLM choice with a random or fixed-rule control on the same inputs; both are off
  by default and only descriptive until a stopping rule is frozen in advance.
- **`alpha-lab` policies** (routing, limits, error taxonomy, correlation) are versioned YAML
  files in `src/alpha_orchestrator/default_policies/`.

## Safety boundaries and known limits

- No automatic submission exists in any CLI command, MCP tool or adapter.
- Local thresholds allocate research effort; they are not evidence of tradability. Offline
  tests prove engineering controls (budget, recovery, identity), not alpha quality.
- Missing, pending or erroneous platform checks never promote a candidate.
- Live acceptance is staged (10, then 100, then the full daily cap on one ledger) and is
  recorded in `CHANGELOG.md`; no economic edge is claimed.
- Single worker, single SQLite ledger, one machine. Not built for distributed workloads.
- Real expressions, results, reports and the ledger stay on the owner's machine
  (see `.gitignore`).

## Project layout

```
src/alpha_orchestrator/
  factory/        alpha-factory: engine, scheduler, LLM layer, selection, reports, CLI
  brain/          platform HTTP client, session handling, notifications
  adapters/       alpha-lab adapters: fake, replay, live
  orchestration/  alpha-lab run engine and platform checks
  storage/        alpha-lab SQLAlchemy models and repository (migrations/ for Alembic)
  gates/ domain/ knowledge/ llm/ mcp/ reporting/   shared rules, models and services
  default_policies/ default_prompts/              packaged policies and prompts
tests/            pytest suite (offline: fakes and replay fixtures only)
scripts/          refresh_session.py (session cookie helper), daily_run.sh (alpha-lab day)
seeds/            template file for the deterministic alpha-lab generator
factory.yaml      factory configuration
```

## Development

```bash
make check                 # black --check, ruff, mypy (strict), pytest with coverage >= 85%
black . && ruff check .    # format and lint
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org/). History and
live-run evidence are in [CHANGELOG.md](CHANGELOG.md).

## License

MIT, see [LICENSE](LICENSE).
