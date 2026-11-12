#!/usr/bin/env bash
# Run one full day of the legacy `alpha-lab` pipeline with a single command:
# create a run -> generate candidates -> simulate -> resume until nothing is pending
# -> platform checks / correlation de-dup / single-parent repair -> report.
#
# The generator is chosen from the environment:
#   ALPHA_LAB_LLM_API_KEY and ALPHA_LAB_LLM_MODEL set -> --research-query: the LLM reads
#     approved method cards to generate candidates and plan repairs.
#   otherwise -> --templates: expand the first-order templates in $SEED_FILE (default
#     seeds/first_order.txt) over every matrix field of the dataset; no LLM needed.
#   Strongly negative results are sign-flipped and re-run on both paths.
#
# The one step this cannot automate is biometric login: when the session has expired
# or needs face verification, the script exits and tells you to run
# scripts/refresh_session.py.
#
# Usage:
#   scripts/daily_run.sh [dataset] [region] [universe] [delay] [research question]
# Defaults: dataset=(built-in default below) region=USA universe=TOP3000 delay=1,
#   research question=$RESEARCH_QUERY or the built-in default below.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

echo "[0/4] Local configuration check (offline)"
alpha-lab doctor --db "$ROOT/alpha_lab.db"

DATASET="${1:-analyst_base_ref}"
REGION="${2:-USA}"
UNIVERSE="${3:-TOP3000}"
DELAY="${4:-1}"
DEFAULT_QUERY="Generate cross-sectional alpha ideas from the ${DATASET} dataset for ${REGION} ${UNIVERSE} delay ${DELAY}. Follow the approved generation-stage method cards for field selection, transformation order, preprocessing, and promotion thresholds."
RESEARCH_QUERY="${5:-${RESEARCH_QUERY:-$DEFAULT_QUERY}}"
CORRELATION_POLICY="${CORRELATION_POLICY:-$ROOT/src/alpha_orchestrator/default_policies/correlation.v1.yaml}"
LIMIT_POLICY="${LIMIT_POLICY:-$ROOT/src/alpha_orchestrator/default_policies/limits.v3.yaml}"
DB="$ROOT/alpha_lab.db"
ACTIVE_RUN_FILE="$ROOT/.runtime/active_run_id"
export ALPHA_LAB_BRAIN_SESSION_FILE="${ALPHA_LAB_BRAIN_SESSION_FILE:-$HOME/brain-session.json}"

SEED_FILE="${SEED_FILE:-$ROOT/seeds/first_order.txt}"
if [ -n "${ALPHA_LAB_LLM_API_KEY:-}" ] && [ -n "${ALPHA_LAB_LLM_MODEL:-}" ]; then
  GEN_LABEL="research-query, reads method cards"
  GEN_ARGS=(--research-query "$RESEARCH_QUERY")
  RESUME_ARGS=(--research-query "$RESEARCH_QUERY")
else
  GEN_LABEL="templates: $SEED_FILE (no LLM configured; deterministic templates)"
  GEN_ARGS=(--templates "$SEED_FILE")
  RESUME_ARGS=()
fi

RESUME_WAIT_SECONDS=60

mkdir -p "$(dirname "$ACTIVE_RUN_FILE")"
if [ -s "$ACTIVE_RUN_FILE" ]; then
  RUN_ID=$(tr -d '[:space:]' < "$ACTIVE_RUN_FILE")
  echo "[1/4] Resuming saved run: $RUN_ID"
else
  echo "[1/4] Creating run: $DATASET / $REGION / $UNIVERSE / delay=$DELAY"
  if ! RUN_ID=$(alpha-lab run create \
  --dataset "$DATASET" --region "$REGION" --universe "$UNIVERSE" --delay "$DELAY" \
  --data-type MATRIX --adapter live --limit-policy "$LIMIT_POLICY" --db "$DB" 2>&1); then
    echo "$RUN_ID" | grep -q "HUMAN_VERIFICATION_REQUIRED\|AUTHENTICATION" && {
      echo "Session expired or face verification required; run: python scripts/refresh_session.py" >&2
    }
    echo "$RUN_ID" >&2
    exit 1
  fi
  printf '%s\n' "$RUN_ID" > "$ACTIVE_RUN_FILE"
  echo "  run_id=$RUN_ID"
fi

echo "[2/4] Generating candidates and running the first simulations ($GEN_LABEL)"
alpha-lab run execute "$RUN_ID" --adapter live \
  "${GEN_ARGS[@]}" \
  --limit-policy "$LIMIT_POLICY" \
  --correlation-policy "$CORRELATION_POLICY" \
  --db "$DB"

echo "[3/4] Collecting pending simulations until done or a manual login is needed"
attempt=0
while true; do
  attempt=$((attempt + 1))
  SUMMARY=$(alpha-lab run resume "$RUN_ID" --adapter live \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
    --limit-policy "$LIMIT_POLICY" \
    --correlation-policy "$CORRELATION_POLICY" \
    --db "$DB")
  echo "  resume #${attempt}: $SUMMARY"
  if echo "$SUMMARY" | grep -q '"status": "COMPLETED"'; then
    break
  fi
  if echo "$SUMMARY" | grep -q "AUTHENTICATION\|HUMAN_VERIFICATION_REQUIRED\|SESSION_EXPIRING_SOON"; then
    echo "Login required; state is saved. Refresh the session and rerun this command to resume." >&2
    exit 2
  fi
  sleep "$RESUME_WAIT_SECONDS"
done

echo "[4/4] Report"
alpha-lab report "$RUN_ID" --db "$DB"
rm -f "$ACTIVE_RUN_FILE"
