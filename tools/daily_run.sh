#!/usr/bin/env bash
# 一条命令跑完一天的流程：建run -> LLM生成候选(读方法论卡) -> 跑模拟 ->
# 反复resume直到没有pending -> 平台检查/相关性去重/单亲修复自动结算 -> 出报告。
#
# 这条脚本走的是 --research-query 路径，不是 --templates：只有这条路径会实例化
# CandidateGenerator/RepairPlanner，只有实例化了它们，approved 的方法卡才会真正
# 被检索、被用来生成候选和指导修复。--templates 路径是纯静态种子回放，从不读
# 方法论卡，跑得再多也不会用上卡库里的东西 (src/alpha_orchestrator/cli/app.py
# 的 `if research_query is not None` 分支)。
#
# 拦不住的唯一一步：session过期/被要求人脸验证时，这个脚本会直接报错退出并
# 告诉你去跑 tools/取会话.py，不会假装能自动搞定生物识别。
#
# 用法:
#   tools/daily_run.sh [数据集] [region] [universe] [delay] [研究问题]
# 默认:
#   数据集=analyst_base_ref region=USA universe=TOP3000 delay=1
#   研究问题=$RESEARCH_QUERY 环境变量，否则用下面内置的默认问题
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

DATASET="${1:-analyst_base_ref}"
REGION="${2:-USA}"
UNIVERSE="${3:-TOP3000}"
DELAY="${4:-1}"
DEFAULT_QUERY="Generate cross-sectional alpha ideas from the ${DATASET} dataset for ${REGION} ${UNIVERSE} delay ${DELAY}. Follow the approved generation-stage method cards for field selection, transformation order, preprocessing, and promotion thresholds."
RESEARCH_QUERY="${5:-${RESEARCH_QUERY:-$DEFAULT_QUERY}}"
CORRELATION_POLICY="${CORRELATION_POLICY:-$ROOT/policies/correlation.v1.yaml}"
DB="$ROOT/alpha_lab.db"
export ALPHA_LAB_BRAIN_SESSION_FILE="${ALPHA_LAB_BRAIN_SESSION_FILE:-$HOME/brain-session.json}"

if [ -z "${ALPHA_LAB_LLM_API_KEY:-}" ] || [ -z "${ALPHA_LAB_LLM_MODEL:-}" ]; then
  echo "缺 ALPHA_LAB_LLM_API_KEY / ALPHA_LAB_LLM_MODEL：不设这两个环境变量，" >&2
  echo "--research-query 路径会在生成候选那一步直接报错，CandidateGenerator/" >&2
  echo "RepairPlanner 起不来，方法卡也就用不上。先 export 再跑这个脚本。" >&2
  exit 1
fi

MAX_RESUME_ATTEMPTS=8
RESUME_WAIT_SECONDS=60

echo "[1/4] 建run: $DATASET / $REGION / $UNIVERSE / delay=$DELAY"
if ! RUN_ID=$(alpha-lab run create \
  --dataset "$DATASET" --region "$REGION" --universe "$UNIVERSE" --delay "$DELAY" \
  --data-type MATRIX --adapter live --db "$DB" 2>&1); then
  echo "$RUN_ID" | grep -q "HUMAN_VERIFICATION_REQUIRED\|AUTHENTICATION" && {
    echo "会话过期或需要人脸验证，先跑: python tools/取会话.py" >&2
  }
  echo "$RUN_ID" >&2
  exit 1
fi
echo "  run_id=$RUN_ID"

echo "[2/4] 生成候选并跑第一轮模拟 (research-query, 读方法论卡)"
echo "  query: $RESEARCH_QUERY"
alpha-lab run execute "$RUN_ID" --adapter live \
  --research-query "$RESEARCH_QUERY" \
  --correlation-policy "$CORRELATION_POLICY" \
  --db "$DB"

echo "[3/4] 反复resume直到没有pending模拟 (最多 $MAX_RESUME_ATTEMPTS 次，每次间隔 ${RESUME_WAIT_SECONDS}s)"
for attempt in $(seq 1 "$MAX_RESUME_ATTEMPTS"); do
  SUMMARY=$(alpha-lab run resume "$RUN_ID" --adapter live \
    --research-query "$RESEARCH_QUERY" \
    --correlation-policy "$CORRELATION_POLICY" \
    --db "$DB")
  echo "  第${attempt}次resume: $SUMMARY"
  if ! echo "$SUMMARY" | grep -q "UNRESOLVED_PENDING_SIMULATION"; then
    break
  fi
  if [ "$attempt" -lt "$MAX_RESUME_ATTEMPTS" ]; then
    sleep "$RESUME_WAIT_SECONDS"
  fi
done

echo "[4/4] 报告"
alpha-lab report "$RUN_ID" --db "$DB"
