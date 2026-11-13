#!/usr/bin/env bash
# This file is part of the NumPilot project at Harbin Institute of Technology (HIT).
#
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
#
# Authors (accounts):
# - Su Tonghua <@sutonghua>
# - Li Xing <@li-xingHIT>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NumPilot example workflow:
# ordinary NumPy -> static five-state gate -> configured model -> numerical gate
# -> strict-NPU resident benchmark. The configured-model artifact is never AST-made.
set -Eeuo pipefail

FILE="examples/demo/large_numpy_program.py"
FILE_EXPLICIT=0
OUT_OVERRIDE="${NUMPILOT_DEMO_OUT:-}"
OUT_EXPLICIT=0
LLM_CONFIG_OVERRIDE="${NUMPILOT_LLM_CONFIG_FILE:-}"
WARMUP=3
SAMPLES=20
CORE=""

usage() {
  cat <<'EOF'
Usage: ./run_demo.sh [SOURCE.py] [options]

Options:
  --out DIR       use this new run directory
  --llm-config FILE
                  use this private LLM JSON config for every NumPilot stage
  --core N        pin both CPU and NPU workers to logical CPU N
  --quick         2 warmups + 7 samples (short validation run)
  --full          3 warmups + 30 samples (stronger evidence)
  --llm           accepted for compatibility; the configured model is always used
  -h, --help      show this help

SOURCE must be an ordinary NumPy script, assign global `result`, and contain
one `# NUMPILOT_DEVICE_INPUTS_END` marker for the resident performance scope.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --out)
      [ "$#" -ge 2 ] || { echo "错误: --out 缺少目录" >&2; exit 2; }
      OUT_OVERRIDE="$2"
      [ -n "$OUT_OVERRIDE" ] || { echo "错误: --out 目录不能为空" >&2; exit 2; }
      OUT_EXPLICIT=1
      shift 2
      ;;
    --out=*)
      OUT_OVERRIDE="${1#*=}"
      [ -n "$OUT_OVERRIDE" ] || { echo "错误: --out 目录不能为空" >&2; exit 2; }
      OUT_EXPLICIT=1
      shift
      ;;
    --llm-config)
      [ "$#" -ge 2 ] || { echo "错误: --llm-config 缺少文件路径" >&2; exit 2; }
      [ -n "$2" ] || { echo "错误: --llm-config 文件路径不能为空" >&2; exit 2; }
      LLM_CONFIG_OVERRIDE="$2"
      shift 2
      ;;
    --llm-config=*)
      LLM_CONFIG_OVERRIDE="${1#*=}"
      [ -n "$LLM_CONFIG_OVERRIDE" ] || { echo "错误: --llm-config 文件路径不能为空" >&2; exit 2; }
      shift
      ;;
    --core)
      [ "$#" -ge 2 ] || { echo "错误: --core 缺少逻辑核编号" >&2; exit 2; }
      CORE="$2"
      shift 2
      ;;
    --quick)
      WARMUP=2
      SAMPLES=7
      shift
      ;;
    --full)
      WARMUP=3
      SAMPLES=30
      shift
      ;;
    --llm)
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *.py)
      FILE="$1"
      FILE_EXPLICIT=1
      shift
      ;;
    *)
      echo "错误: 未知参数 $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$LLM_CONFIG_OVERRIDE" in
  *$'\n'*|*$'\r'*)
    echo "错误: --llm-config 路径不能包含换行符" >&2
    exit 2
    ;;
esac

PY="${NUMPILOT_PY:-python3}"
CALLER_DIR="$(pwd -P)"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit 1

# Expand only the current-user ``~`` forms.  Variables do not undergo shell
# tilde expansion after parsing, and eval would turn a user path into code.
expand_leading_tilde() {
  case "$1" in
    '~') REPLY="${HOME:?}" ;;
    '~/'*) REPLY="${HOME:?}${1#\~}" ;;
    *) REPLY="$1" ;;
  esac
}

reject_symlink_components() {
  local value="$1"
  local label="$2"
  local current="$value"
  case "$current" in
    /*) ;;
    *) current="$PWD/$current" ;;
  esac
  while :; do
    if [ -L "$current" ]; then
      echo "错误: $label 路径不能穿过符号链接: $current" >&2
      return 1
    fi
    [ "$current" = "/" ] && break
    current="${current%/*}"
    [ -n "$current" ] || current="/"
  done
}

if [ "$FILE_EXPLICIT" -eq 1 ]; then
  expand_leading_tilde "$FILE"
  FILE="$REPLY"
  case "$FILE" in
    /*) ;;
    *) FILE="$CALLER_DIR/$FILE" ;;
  esac
else
  # The built-in example is intentionally resolved from the checkout after
  # ``cd``; this keeps ``run_demo.sh`` convenient from any invocation cwd.
  :
fi
case "$FILE" in
  *$'\n'*|*$'\r'*)
    echo "错误: SOURCE 路径不能包含换行符" >&2
    exit 2
    ;;
esac
reject_symlink_components "$FILE" "SOURCE" || exit 2

# Keep credentials in the existing private JSON file; this option only passes
# its path to NumPilot.  No token/endpoint is copied into command arguments or
# the repository.  An explicit flag takes precedence over the inherited env.
if [ -n "$LLM_CONFIG_OVERRIDE" ]; then
  # Shell does not expand a leading ``~`` after it has been captured in a
  # variable (and evaluating it would be unsafe).  Expand only the two
  # unambiguous forms explicitly, then resolve relative paths from the
  # invocation directory.
  expand_leading_tilde "$LLM_CONFIG_OVERRIDE"
  LLM_CONFIG_OVERRIDE="$REPLY"
  case "$LLM_CONFIG_OVERRIDE" in
    /*) ;;
    *) LLM_CONFIG_OVERRIDE="$CALLER_DIR/$LLM_CONFIG_OVERRIDE" ;;
  esac
  reject_symlink_components "$LLM_CONFIG_OVERRIDE" "LLM 配置" || exit 2
  export NUMPILOT_LLM_CONFIG_FILE="$LLM_CONFIG_OVERRIDE"
fi

if [ "$OUT_EXPLICIT" -eq 1 ] || [ -n "$OUT_OVERRIDE" ]; then
  # Resolve a caller-provided relative path before changing into the checkout;
  # this makes ``--out runs/demo`` mean the same thing from every cwd.
  expand_leading_tilde "$OUT_OVERRIDE"
  OUT_OVERRIDE="$REPLY"
  case "$OUT_OVERRIDE" in
    /*) RUN_ROOT="$OUT_OVERRIDE" ;;
    *) RUN_ROOT="$CALLER_DIR/$OUT_OVERRIDE" ;;
  esac
else
  DEMO_HOME="$("$PY" -m numpilot paths --get demo)"
  RUN_ROOT="$DEMO_HOME/run-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
reject_symlink_components "$RUN_ROOT" "run directory" || exit 2
MIGRATION_OUT="$RUN_ROOT/migration"
ANALYSIS_OUT="$RUN_ROOT/qualified-analysis.json"
VERIFICATION_OUT="$RUN_ROOT/script-verification-report.json"
BENCHMARK_OUT="$RUN_ROOT/script-benchmark-report.json"

if [ -L "$RUN_ROOT" ] || [ -e "$RUN_ROOT" ]; then
  echo "错误: run directory 必须是全新的路径（不会覆盖既有产物）: $RUN_ROOT" >&2
  exit 2
fi
(umask 077 && mkdir -p -- "$RUN_ROOT")

# A private custom OPP may provide qualified public AsNumpy operators.  This is
# process-local and never edits the system environment.
CUSTOM_OP_API_LIB="${ASCEND_OPP_PATH:-/usr/local/Ascend/cann/opp}/vendors/customize/op_api/lib"
if [ -d "$CUSTOM_OP_API_LIB" ]; then
  export LD_LIBRARY_PATH="$CUSTOM_OP_API_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

DEMO_STARTED_AT=$(date +%s)
CURRENT_STAGE="configured model 本地检查"
if [ -t 1 ] && [ -z "${NO_COLOR+x}" ] && [ "${TERM:-}" != "dumb" ]; then
  COLOR_CYAN=$'\033[1;36m'
  COLOR_YELLOW=$'\033[0;33m'
  COLOR_RESET=$'\033[0m'
else
  COLOR_CYAN=""
  COLOR_YELLOW=""
  COLOR_RESET=""
fi

report_failure() {
  local code=$1
  local elapsed=$(( $(date +%s) - DEMO_STARTED_AT ))
  trap - ERR INT TERM
  printf '\n[NumPilot demo] FAIL · stage=%s · exit=%d · elapsed=%ss\n' \
    "$CURRENT_STAGE" "$code" "$elapsed" >&2
  printf '[NumPilot demo] 已完成的回执与诊断保留在: %s\n' "$RUN_ROOT" >&2
  exit "$code"
}

on_error() {
  local code=$?
  report_failure "$code"
}

on_interrupt() {
  local elapsed=$(( $(date +%s) - DEMO_STARTED_AT ))
  trap - ERR INT TERM
  printf '\n[NumPilot demo] INTERRUPTED · stage=%s · elapsed=%ss\n' \
    "$CURRENT_STAGE" "$elapsed" >&2
  printf '[NumPilot demo] 已完成的回执与诊断保留在: %s\n' "$RUN_ROOT" >&2
  exit 130
}

trap on_error ERR
trap on_interrupt INT TERM

banner() {
  CURRENT_STAGE="$1"
  printf '\n%s========== %s ==========%s\n' \
    "$COLOR_CYAN" "$CURRENT_STAGE" "$COLOR_RESET"
}

run() {
  printf '%s$' "$COLOR_YELLOW"
  printf ' %q' "$@"
  printf '%s\n' "$COLOR_RESET"
  "$@"
}

CONFIG_RC=0
"$PY" -m numpilot configure-llm --check || CONFIG_RC=$?
if [ "$CONFIG_RC" -ne 0 ]; then
  echo "错误: configured model 不可用；请先在仓库根目录执行 install -m 600 llm.example.json llm.json 后填写，或配置 NUMPILOT_LLM_CONFIG_FILE、使用 --llm-config，或运行 configure-llm" >&2
  report_failure "$CONFIG_RC"
fi

echo "NumPilot ordinary 分阶段流程演示"
echo "  Python : $PY"
echo "  ordinary source : $FILE"
echo "  generator : configured model（禁止 AST 生成/回退）"
echo "  benchmark : single-core, warmup=$WARMUP, samples=$SAMPLES"
echo "  run directory : $RUN_ROOT"

banner "1/5 环境与真实 NPU runtime 自检"
run "$PY" -m numpilot doctor --backend npu

banner "2/5 ordinary 五态静态分析（不执行源码、不调用模型）"
run "$PY" -m numpilot analyze "$FILE" --require-direct --out "$ANALYSIS_OUT"

banner "3/5 configured model 迁移（真实远程回执 + AST 事后审计）"
run "$PY" -m numpilot migrate "$FILE" \
  --mode script --engine llm --backend npu --fallback-policy reject \
  --out "$MIGRATION_OUT"

banner "4/5 显式执行与命名输出数值对拍"
run "$PY" -m numpilot verify-script "$FILE" \
  --migration-report "$MIGRATION_OUT/script-migration-report.json" \
  --output result --out "$VERIFICATION_OUT"

banner "5/5 strict-NPU 设备驻留性能验证"
BENCH_ARGS=(
  benchmark-script "$FILE"
  --migration-report "$MIGRATION_OUT/script-migration-report.json"
  --output result
  --out "$BENCHMARK_OUT"
  --warmup "$WARMUP"
  --samples "$SAMPLES"
)
if [ -n "$CORE" ]; then
  BENCH_ARGS+=(--core "$CORE")
fi
run "$PY" -m numpilot "${BENCH_ARGS[@]}"

banner "一屏回执摘要（只读 JSON，不重新执行用户代码）"
if ! "$PY" tools/render_demo_summary.py \
  --analysis "$ANALYSIS_OUT" \
  --migration "$MIGRATION_OUT/script-migration-report.json" \
  --verification "$VERIFICATION_OUT" \
  --benchmark "$BENCHMARK_OUT"; then
  echo "警告: 摘要渲染失败；原始回执仍保留在 $RUN_ROOT" >&2
fi

echo
echo "✅ ordinary 分阶段流程完成: $RUN_ROOT"
echo "   elapsed: $(( $(date +%s) - DEMO_STARTED_AT ))s"
echo "   headline 只引用 $BENCHMARK_OUT 中 performance_valid=true 的设备驻留结果。"
echo "   上述组件总耗时仅作诊断口径；不是直接采样 E2E，不冒充端到端单次延迟。"
