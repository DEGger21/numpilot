#!/usr/bin/env bash
# NumPilot Poisson Ritz 客户体验：证据校验 → Skill/qualification → 真机迁移 → 驻留评测。
set -euo pipefail

MODE="live"
FULL=0
LARGE_SINGLE_CORE=0
CPU_CORE=0
CPU_CORE_SET=0
VERBOSE_RUNTIME=0
OUT_OVERRIDE="${NUMPILOT_POISSON_DEMO_OUT:-}"

usage() {
  printf '%s\n' \
    "用法: ./run_poisson_demo.sh [--evidence-only] [--full | --large-single-core] [--cpu-core N] [--verbose-runtime] [--out DIR]" \
    "  默认              实时 910B4 迁移 + 小规模驻留执行（快速体验）" \
    "  --evidence-only   只校验/展示仓库内公开证据，不访问 NPU" \
    "  --full            重跑 48×48 / 49 RBF / 3000 轮完整门控评测" \
    "  --large-single-core 运行 128×128 / 225 RBF 单核 CPU 对比诊断" \
    "  --cpu-core N      为单核诊断选择逻辑核（默认 0）" \
    "  --verbose-runtime 直接显示 AsNumpy/CANN 的逐算子日志" \
    "  --out DIR         指定本次实时产物目录"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --evidence-only) MODE="evidence" ;;
    --full) FULL=1 ;;
    --large-single-core) LARGE_SINGLE_CORE=1 ;;
    --cpu-core)
      shift
      if [ "$#" -eq 0 ]; then
        printf '%s\n' "错误: --cpu-core 需要非负逻辑核编号" >&2
        exit 2
      fi
      CPU_CORE="$1"
      CPU_CORE_SET=1
      ;;
    --verbose-runtime) VERBOSE_RUNTIME=1 ;;
    --out)
      shift
      if [ "$#" -eq 0 ]; then
        printf '%s\n' "错误: --out 需要目录参数" >&2
        exit 2
      fi
      OUT_OVERRIDE="$1"
      ;;
    -h|--help) usage; exit 0 ;;
    *) printf '%s\n' "错误: 未知参数 $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [ "$FULL" -eq 1 ] && [ "$LARGE_SINGLE_CORE" -eq 1 ]; then
  printf '%s\n' "错误: --full 与 --large-single-core 不能同时使用" >&2
  exit 2
fi
if [ "$CPU_CORE_SET" -eq 1 ] && [ "$LARGE_SINGLE_CORE" -ne 1 ]; then
  printf '%s\n' "错误: --cpu-core 仅与 --large-single-core 一起使用" >&2
  exit 2
fi
if ! [[ "$CPU_CORE" =~ ^[0-9]+$ ]]; then
  printf '%s\n' "错误: --cpu-core 需要非负逻辑核编号" >&2
  exit 2
fi

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$PROJECT_DIR/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

banner() { printf '\n\033[1;36m========== %s ==========\033[0m\n' "$1"; }

banner "NumPilot Poisson Ritz 客户体验"
printf '%s\n' \
  "路线: NumPy 科学程序 → 缺口算子补全 → AsNumpy/Ascend NPU 迁移 → 数值与性能证据" \
  "原则: 任何一道门失败即停止；不把 API 命中当成 NPU 证据，不把负加速包装成优势。"

banner "1/5 公开证据包完整性与已发布结论"
python3 tools/poisson_demo_summary.py published

if [ "$MODE" = "evidence" ]; then
  printf '\n%s\n' "✅ 证据体验完成；本次未访问 NPU，也未生成新性能数字。"
  exit 0
fi

P0_ROOT="${NUMPILOT_POISSON_P0_ROOT:-$WORKSPACE_ROOT/numpilot-gap-runs/poisson-ritz-p0-20260821/direct-local-v5}"
QUAL_ROOT="${NUMPILOT_POISSON_QUALIFICATION_ROOT:-$P0_ROOT/qualification-evidence-v6-move}"
ASNUMPY_SITE="${NUMPILOT_POISSON_ASNUMPY_SITE:-$P0_ROOT/asnumpy-p0-site-v10-move}"
PY="${NUMPILOT_POISSON_PY:-$P0_ROOT/asnumpy-p0-venv-v10-move/bin/python}"
SUPPORT_SITE="${NUMPILOT_POISSON_SUPPORT_SITE:-$WORKSPACE_ROOT/numpilot-gap-runs/asnumpy-runtime-20260813/venv/lib/python3.10/site-packages}"
SKILL_ROOT="${NUMPILOT_CANNBOT_ROOT:-$WORKSPACE_ROOT/cannbot-skills}"
OVERLAY="$QUAL_ROOT/overlay/capability-overlay.json"
APPLY_OPP="$QUAL_ROOT/runtime/private-opp/vendors/apply_gradient_descent_custom"
SMOOTH_OPP="$QUAL_ROOT/runtime/private-opp/vendors/smooth_l1_loss_custom"

for required in \
  "$PY" "$ASNUMPY_SITE/asnumpy/__init__.py" "$SUPPORT_SITE/loguru/__init__.py" \
  "$OVERLAY" "$APPLY_OPP/op_api/lib/libcust_opapi.so" \
  "$SMOOTH_OPP/op_api/lib/libcust_opapi.so"; do
  if [ ! -e "$required" ]; then
    printf '%s\n' "错误: 实时体验制品缺失: $required" >&2
    printf '%s\n' "可先运行 ./run_poisson_demo.sh --evidence-only，或通过 NUMPILOT_POISSON_* 指定制品。" >&2
    exit 2
  fi
done
if [ ! -d "$SKILL_ROOT" ]; then
  printf '%s\n' "错误: CANNBot Skill checkout 缺失: $SKILL_ROOT" >&2
  exit 2
fi
if { [ "$FULL" -eq 1 ] || [ "$LARGE_SINGLE_CORE" -eq 1 ]; } \
    && ! "$PY" -c 'import threadpoolctl' >/dev/null 2>&1; then
  printf '%s\n' \
    "错误: 性能模式要求可观测 CPU BLAS 线程数，当前解释器缺少 threadpoolctl。" \
    "请先在该隔离环境安装 NumPilot 已声明的 npu 可选依赖: $PY -m pip install 'threadpoolctl>=3,<4'" >&2
  exit 2
fi
if [ "$LARGE_SINGLE_CORE" -eq 1 ] && ! command -v taskset >/dev/null 2>&1; then
  printf '%s\n' "错误: --large-single-core 需要 taskset" >&2
  exit 2
fi

if [ -n "$OUT_OVERRIDE" ]; then
  DEMO_ROOT="$OUT_OVERRIDE"
else
  STATE_DEMO="$(python3 -m numpilot paths --get demo)"
  DEMO_ROOT="$STATE_DEMO/poisson-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [ -e "$DEMO_ROOT" ]; then
  printf '%s\n' "错误: 输出目录已存在，为避免覆盖已停止: $DEMO_ROOT" >&2
  exit 2
fi
mkdir -p "$DEMO_ROOT/logs"

export ASCEND_HOME_PATH="/usr/local/Ascend/cann-9.0.0"
export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH"
export ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp"
export ASCEND_SOC_VERSION="Ascend910B"
export ASCEND_CUSTOM_OPP_PATH="$APPLY_OPP:$SMOOTH_OPP"
export LD_LIBRARY_PATH="$APPLY_OPP/op_api/lib:$SMOOTH_OPP/op_api/lib:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/lib64/plugin/opskernel:$ASCEND_HOME_PATH/lib64/plugin/nnengine"
export PYTHONPATH="$ASNUMPY_SITE:$PROJECT_DIR:$SUPPORT_SITE"
export ASNUMPY_DEBUG=0
export ASNUMPY_LOG_DIR="$DEMO_ROOT/logs"
export PYTHONUNBUFFERED=1

banner "2/5 固定 CANNBot Skill 与已资格化算子"
python3 -m numpilot setup-skills --root "$SKILL_ROOT" --check
NUMPILOT_DEMO_QUAL_ROOT="$QUAL_ROOT" "$PY" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["NUMPILOT_DEMO_QUAL_ROOT"])
summary = json.loads((root / "qualification-summary.json").read_text(encoding="utf-8"))
if summary.get("status") != "qualified_and_overlay_published":
    raise SystemExit("qualification status is not publishable")
for name, record in sorted(summary["operators"].items()):
    print(
        f"[qualification] {name}: {record['tests_passed']} tests, "
        f"{record['numerical_cases']} numerical cases, "
        f"{record['kernel_launches']} kernel launches, "
        f"max_abs={record['max_abs_error']:.3g}"
    )
print(f"[overlay] sha256={summary['overlay_sha256']}")
PY

banner "3/5 裸金属 Ascend 环境自检"
"$PY" -m numpilot doctor --backend npu

SOURCE="examples/poisson_ritz_iteration_broadcast.py"
MIGRATION_OUT="$DEMO_ROOT/migration"
MIGRATION_LOG="$DEMO_ROOT/logs/migration.log"
banner "4/5 实时分析、迁移、strict-NPU 执行与三路数值校验"
if [ "$VERBOSE_RUNTIME" -eq 1 ]; then
  "$PY" -m numpilot migrate "$SOURCE" \
    --backend npu --engine ast --out "$MIGRATION_OUT" \
    --capability-overlay "$OVERLAY" \
    --qualification-evidence-root "$QUAL_ROOT" 2>&1 | tee "$MIGRATION_LOG"
else
  if ! "$PY" -m numpilot migrate "$SOURCE" \
      --backend npu --engine ast --out "$MIGRATION_OUT" \
      --capability-overlay "$OVERLAY" \
      --qualification-evidence-root "$QUAL_ROOT" >"$MIGRATION_LOG" 2>&1; then
    tail -n 80 "$MIGRATION_LOG" >&2
    exit 2
  fi
fi
python3 tools/poisson_demo_summary.py migration "$MIGRATION_OUT/report.json"
printf '%s\n' "[日志] 逐算子 CANN 日志已保存: $MIGRATION_LOG"

PERFORMANCE_OUT="$DEMO_ROOT/live-performance"
PERFORMANCE_LOG="$DEMO_ROOT/logs/performance.log"
PERFORMANCE_ARGS=(
  --source "$SOURCE"
  --migrated "$MIGRATION_OUT/migrated.py"
  --migration-report "$MIGRATION_OUT/report.json"
  --capability-overlay "$OVERLAY"
  --qualification-evidence-root "$QUAL_ROOT"
  --cpu-threads 1
  --out "$PERFORMANCE_OUT"
)
if [ "$FULL" -eq 1 ]; then
  PERFORMANCE_ARGS+=(
    --interior-size 48 --boundary-size 48 --centre-size 7
    --sample-iterations 10 --warmup 2 --repeats 7
    --convergence-iterations 3000
    --profiler-audit evidence/poisson-ritz-p0-20260821/performance-broadcast-optimized/profiler/audit.json
  )
  PERFORMANCE_LABEL="完整 48×48 / 49 RBF / 3000 轮门控重跑"
  PERFORMANCE_RUNNER="tools/poisson_broadcast_performance_evidence.py"
  PERFORMANCE_SUMMARY="performance"
elif [ "$LARGE_SINGLE_CORE" -eq 1 ]; then
  PERFORMANCE_ARGS+=(
    --interior-size 128 --boundary-size 128 --centre-size 15
    --sample-iterations 3 --warmup 1 --repeats 5
    --convergence-iterations 20 --cpu-core "$CPU_CORE"
  )
  PERFORMANCE_LABEL="128×128 / 225 RBF / CPU 逻辑核 $CPU_CORE 放大诊断"
  PERFORMANCE_RUNNER="tools/poisson_affinity_performance_evidence.py"
  PERFORMANCE_SUMMARY="affinity-performance"
else
  PERFORMANCE_ARGS+=(
    --interior-size 8 --boundary-size 8 --centre-size 3
    --sample-iterations 2 --warmup 1 --repeats 3
    --convergence-iterations 20
  )
  PERFORMANCE_LABEL="快速驻留执行体验（性能不发布）"
  PERFORMANCE_RUNNER="tools/poisson_broadcast_performance_evidence.py"
  PERFORMANCE_SUMMARY="performance"
fi

banner "5/5 $PERFORMANCE_LABEL"
if [ "$VERBOSE_RUNTIME" -eq 1 ]; then
  "$PY" "$PERFORMANCE_RUNNER" \
    "${PERFORMANCE_ARGS[@]}" 2>&1 | tee "$PERFORMANCE_LOG"
else
  if ! "$PY" "$PERFORMANCE_RUNNER" \
      "${PERFORMANCE_ARGS[@]}" >"$PERFORMANCE_LOG" 2>&1; then
    tail -n 120 "$PERFORMANCE_LOG" >&2
    exit 2
  fi
fi
python3 tools/poisson_demo_summary.py \
  "$PERFORMANCE_SUMMARY" "$PERFORMANCE_OUT/evidence.json"

printf '\n%s\n' \
  "✅ Poisson 客户体验完成。" \
  "   本次实时产物: $DEMO_ROOT" \
  "   已发布 3000 轮证据: evidence/poisson-ritz-p0-20260821/performance-broadcast-optimized/final/evidence.json"
