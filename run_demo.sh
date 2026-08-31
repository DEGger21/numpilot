#!/usr/bin/env bash
# NumPilot competition demo:
# ordinary NumPy -> static five-state gate -> configured model -> numerical gate
# -> strict-NPU resident benchmark.  The submitted artifact is never AST-made.
set -euo pipefail

FILE="examples/poisson_source_smooth_l1_map.py"
OUT_OVERRIDE="${NUMPILOT_DEMO_OUT:-}"
WARMUP=3
SAMPLES=20
CORE=""

usage() {
  cat <<'EOF'
Usage: ./run_demo.sh [SOURCE.py] [options]

Options:
  --out DIR       use this new run directory
  --core N        pin both CPU and NPU workers to logical CPU N
  --quick         2 warmups + 7 samples (live presentation)
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
      shift 2
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
      shift
      ;;
    *)
      echo "错误: 未知参数 $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PY="${NUMPILOT_PY:-python3}"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit 1

if [ -n "$OUT_OVERRIDE" ]; then
  RUN_ROOT="$OUT_OVERRIDE"
else
  DEMO_HOME="$("$PY" -m numpilot paths --get demo)"
  RUN_ROOT="$DEMO_HOME/competition-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
MIGRATION_OUT="$RUN_ROOT/migration"
ANALYSIS_OUT="$RUN_ROOT/qualified-analysis.json"
VERIFICATION_OUT="$RUN_ROOT/script-verification-report.json"
BENCHMARK_OUT="$RUN_ROOT/script-benchmark-report.json"

# A private custom OPP may provide qualified public AsNumpy operators.  This is
# process-local and never edits the system environment.
CUSTOM_OP_API_LIB="${ASCEND_OPP_PATH:-/usr/local/Ascend/cann/opp}/vendors/customize/op_api/lib"
if [ -d "$CUSTOM_OP_API_LIB" ]; then
  export LD_LIBRARY_PATH="$CUSTOM_OP_API_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

banner() { printf '\n\033[1;36m========== %s ==========\033[0m\n' "$1"; }
run()    { printf '\033[0;33m$ %s\033[0m\n' "$*"; "$@"; }

if ! "$PY" -m numpilot configure-llm --check; then
  echo "错误: configured model 不可用；请先配置 NUMPILOT_LLM_CONFIG_FILE 或运行 configure-llm" >&2
  exit 1
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
echo "   headline 只引用 $BENCHMARK_OUT 中 performance_valid=true 的设备驻留结果。"
echo "   一次组件和仅为诊断账目；不是直接采样 E2E，不冒充端到端单次延迟。"
