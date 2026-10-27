# Private Alpha Factory: production rollout

The supported private research entrypoint is `alpha-factory`. The older `alpha-lab`
workflow remains in this repository, but its replay/demo does not validate the factory.
This release is a production candidate. Live acceptance and rollout evidence must be
recorded separately from offline tests. It does not establish an economic edge.

## Install and preflight

Use Python 3.12, a dedicated virtual environment, and this checkout. The exact packages
used for local verification are recorded in `requirements-verified.txt` (including dev
checks; this is a record of the macOS environment, not a cross-platform lock).

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-verified.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/alpha-factory doctor --config /absolute/path/factory.yaml
.venv/bin/alpha-factory doctor --live --config /absolute/path/factory.yaml
```

`doctor` never creates or migrates the ledger and never performs password login or
simulation. It reports local reservations, uncertain batches, in-flight work and the
last completion. `--live` adds a read-only authentication probe. A successful preflight
is not an end-to-end acceptance result.

Use `BRAIN_STATE_DIR` for a dedicated private session directory. Credentials remain in
the existing external credential file. Cookies and keys must never enter the source
bundle. When BRAIN requires Persona verification, the owner completes it; do not retry
password login in a loop. Login and run must use the same session directory.

## Budget and recovery

Both `v1_daily_cap` and `daily_sim_cap` are 1,000 in the release configuration. The lower
value wins. Reservations count child simulations at request creation, including pending
and unknown outcomes. Confirmed refusals release their reservations. The local day is UTC;
verify the account's actual simulation reset and header semantics during rollout. This
ledger cannot coordinate spending by another independent scanner or another ledger.

`quota_reserve` is configurable and defaults to zero for this factory's allocated quota.
The final batch is trimmed to the remaining allocation and reported headroom. A quota is
a ceiling: exhausted research ideas may stop earlier. Do not invent weak tests just to
consume the remainder. Research thresholds are preserved. Production clears the old three-dataset pilot
whitelist and uses the existing scanner eligibility/ranking, with one active dataset.
The 100-child pilot dataset allowance is raised to 1,000. Explicitly selected budget-stopped
cells with unfinished work can resume; prior candidates and failure evidence are retained.
Ready simulation work is drained before new LLM planning. A refill that supplies no new
field groups stops further refills for that dataset.

Only one worker can own a ledger. Its OS lock releases after a crash. Existing in-flight
URLs are polled before new LLM proposals. The deadline stops further work between HTTP
operations; an individual network request/backoff may still take its configured timeout.

An ambiguous creation response or a crash before its URL commits leaves an `UNKNOWN`
batch. The allocation remains reserved and further runs block. Inspect the request on the
platform and reconnect its verified progress URL:

```sh
alpha-factory recover-batch BATCH_ID --location /simulations/VERIFIED_ID --config CONFIG
alpha-factory run --config CONFIG
```

Never delete an unknown batch or requeue it merely because its URL is missing. If the
platform provides no reconciliation mechanism, leave it blocked for operator review.
Returned successful results must match the submitted expression and settings. Failed
children without identities retain their errors at batch scope rather than guessing
from response position.

LLM transport retries are bounded; a daily call cap and output-token cap are configurable.
Unavailable or malformed LLM responses pause research without declaring the idea noise.
Calls retain request/system/response hashes; older ledger rows receive empty provenance
fields because their original inputs cannot be reconstructed. Token cost fields remain
estimates from the existing configured price table, not verified invoices or a spending cap.

## Promotion and acceptance

Empty, warning, pending, erroneous or missing required platform checks cannot promote a
candidate. Self/prod correlation must be present, finite and within the configured bound.
A robustness flag with no actual ratios is insufficient. Submission remains manual and
checks are revalidated immediately before a requested submission.

Roll out against one authoritative ledger: at most 10 live simulations first, then 100,
then the 1,000 daily ceiling. Reuse that same ledger as caps increase. Before moving a
snapshot back to the original location, stop the old worker, take a verified backup and
prove the original has not changed since the snapshot. Never run both ledgers against the
same allocation. Do not activate a schedule until the live canary and restart test pass.

Acceptance requires:

- Every accepted child resolves to the correct ledger candidate or an explicitly blocked
  reconciliation case; no silent retries of ambiguous creates.
- Zero daily overspend and duplicate accepted creates across worker restarts.
- Controlled behavior for invalid formulas, throttling, auth expiry, LLM failure and timeout.
- A live restart resumes existing URLs without recreating requests, and the platform and
  ledger counts reconcile across an actual quota reset.
- A ready candidate retains economic rationale, lineage, settings, results and completed
  checks; an empty shortlist is reported honestly.

For scheduled operation, `alpha-factory schedule install --config CONFIG` pins the code
checkout, exact configuration and current `BRAIN_STATE_DIR`. It uses macOS local time.
Persona challenges still require a person and generate an explicit pause. A working Mac,
network access and notification delivery must be verified under real scheduled execution.

## Verification

```sh
ruff format --check .
ruff check .
PYTHONPATH=src mypy src
PYTHONPATH=src pytest
```

Tests include lost creation responses, restart reservations, quota-tail trimming,
unknown-result quarantine, worker ownership, deadlines, result identity, incomplete
checks, LLM outage recovery, provenance migration and a synthetic 1,000-child day.
These establish engineering controls; they do not demonstrate live Alpha quality,
research independence, out-of-sample returns or the benefit of LLM proposals.
