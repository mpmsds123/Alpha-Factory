#!/usr/bin/env bash
# 一条命令跑完一天的流程：建run -> LLM生成候选(读方法论卡) -> 跑模拟 ->
# 反复resume直到没有pending -> 平台检查/相关性去重/单亲修复自动结算 -> 出报告。
#
# 两种生成方式，按环境变量自动选：
#   设了 ALPHA_LAB_LLM_API_KEY + ALPHA_LAB_LLM_MODEL -> --research-query 路径：
#     LLM 读已批准的方法卡生成候选，RepairPlanner 按卡修复。
#   没设 -> --templates 路径：用 $SEED_FILE（默认 seeds/first_order.txt）把一阶模板
#     展开到数据集的全部 MATRIX 字段，不需要任何 LLM。强负 sharpe 的结果由引擎
#     自动取反再跑一次（gates/sign_flip.py），两条路径都有这一步。
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

echo "[0/4] 本地配置检查（不联网）"
alpha-lab doctor --db "$ROOT/alpha_lab.db"

DATASET="${1:-analyst_base_ref}"
REGION="${2:-USA}"
UNIVERSE="${3:-TOP3000}"
DELAY="${4:-1}"
DEFAULT_QUERY="Generate cross-sectional alpha ideas from the ${DATASET} dataset for ${REGION} ${UNIVERSE} delay ${DELAY}. Follow the approved generation-stage method cards for field selection, transformation order, preprocessing, and promotion thresholds."
RESEARCH_QUERY="${5:-${RESEARCH_QUERY:-$DEFAULT_QUERY}}"
CORRELATION_POLICY="${CORRELATION_POLICY:-$ROOT/policies/correlation.v1.yaml}"
LIMIT_POLICY="${LIMIT_POLICY:-$ROOT/policies/limits.v3.yaml}"
DB="$ROOT/alpha_lab.db"
ACTIVE_RUN_FILE="$ROOT/.runtime/active_run_id"
export ALPHA_LAB_BRAIN_SESSION_FILE="${ALPHA_LAB_BRAIN_SESSION_FILE:-$HOME/brain-session.json}"

SEED_FILE="${SEED_FILE:-$ROOT/seeds/first_order.txt}"
if [ -n "${ALPHA_LAB_LLM_API_KEY:-}" ] && [ -n "${ALPHA_LAB_LLM_MODEL:-}" ]; then
  GEN_LABEL="research-query, 读方法论卡"
  GEN_ARGS=(--research-query "$RESEARCH_QUERY")
  RESUME_ARGS=(--research-query "$RESEARCH_QUERY")
else
  GEN_LABEL="templates: $SEED_FILE（没配 LLM，走确定性模板）"
  GEN_ARGS=(--templates "$SEED_FILE")
  RESUME_ARGS=()
fi

RESUME_WAIT_SECONDS=60

mkdir -p "$(dirname "$ACTIVE_RUN_FILE")"
if [ -s "$ACTIVE_RUN_FILE" ]; then
  RUN_ID=$(tr -d '[:space:]' < "$ACTIVE_RUN_FILE")
  echo "[1/4] 续跑已保存的run: $RUN_ID"
else
  echo "[1/4] 建run: $DATASET / $REGION / $UNIVERSE / delay=$DELAY"
  if ! RUN_ID=$(alpha-lab run create \
  --dataset "$DATASET" --region "$REGION" --universe "$UNIVERSE" --delay "$DELAY" \
  --data-type MATRIX --adapter live --limit-policy "$LIMIT_POLICY" --db "$DB" 2>&1); then
    echo "$RUN_ID" | grep -q "HUMAN_VERIFICATION_REQUIRED\|AUTHENTICATION" && {
      echo "会话过期或需要人脸验证，先跑: python tools/取会话.py" >&2
    }
    echo "$RUN_ID" >&2
    exit 1
  fi
  printf '%s\n' "$RUN_ID" > "$ACTIVE_RUN_FILE"
  echo "  run_id=$RUN_ID"
fi

echo "[2/4] 生成候选并跑第一轮模拟 ($GEN_LABEL)"
alpha-lab run execute "$RUN_ID" --adapter live \
  "${GEN_ARGS[@]}" \
  --limit-policy "$LIMIT_POLICY" \
  --correlation-policy "$CORRELATION_POLICY" \
  --db "$DB"

echo "[3/4] worker持续回收pending，直到完成或需要人工登录"
attempt=0
while true; do
  attempt=$((attempt + 1))
  SUMMARY=$(alpha-lab run resume "$RUN_ID" --adapter live \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
    --limit-policy "$LIMIT_POLICY" \
    --correlation-policy "$CORRELATION_POLICY" \
    --db "$DB")
  echo "  第${attempt}次resume: $SUMMARY"
  if echo "$SUMMARY" | grep -q '"status": "COMPLETED"'; then
    break
  fi
  if echo "$SUMMARY" | grep -q "AUTHENTICATION\|HUMAN_VERIFICATION_REQUIRED\|SESSION_EXPIRING_SOON"; then
    echo "需要你重新登录；状态已保存。刷新会话后重新运行本命令即可续跑。" >&2
    exit 2
  fi
  sleep "$RESUME_WAIT_SECONDS"
done

echo "[4/4] 报告"
alpha-lab report "$RUN_ID" --db "$DB"
rm -f "$ACTIVE_RUN_FILE"
