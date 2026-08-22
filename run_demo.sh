#!/usr/bin/env bash
# NumPilot 裸金属演示：环境自检 → 迁移证明 → CPU 基线 → strict-NPU 评测。
#
# 用法:
#   ./run_demo.sh                      # 默认演示无已知 fallback 的 elementwise_forward
#   ./run_demo.sh examples/mlp_forward.py   # 换一个网络
#   ./run_demo.sh --full               # 额外保存一个不限制 BLAS 的 CPU 对照口径
#   ./run_demo.sh --llm                # 迁移改用已配置的大模型智能体；首次运行由框架引导填写
#
# 环境:默认使用 python3；可用 NUMPILOT_PY 指向装有 asnumpy 的解释器。
set -euo pipefail

# ── 参数解析 ──────────────────────────────────────────────
FILE="examples/elementwise_forward.py"
OUT="out_demo"
FULL=0
ENGINE_ARGS=()
ENGINE_LABEL="ast(确定性改写)"
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    --llm)  ENGINE_ARGS=(--engine llm); ENGINE_LABEL="llm(已配置大模型智能体)" ;;
    *.py)   FILE="$arg" ;;
    *)      echo "忽略未知参数: $arg" ;;
  esac
done

PY="${NUMPILOT_PY:-python3}"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit 1

# 自定义 OPP 的 op_api 动态库只注入本进程，不修改系统配置。
CUSTOM_OP_API_LIB="${ASCEND_OPP_PATH:-/usr/local/Ascend/cann/opp}/vendors/customize/op_api/lib"
if [ -d "$CUSTOM_OP_API_LIB" ]; then
  export LD_LIBRARY_PATH="$CUSTOM_OP_API_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

banner() { printf '\n\033[1;36m========== %s ==========\033[0m\n' "$1"; }
run()    { printf '\033[0;33m$ %s\033[0m\n' "$*"; "$@"; }

# 所有入口统一由 NumPilot Python 配置层处理；不 source 任意 shell 文件。
if [ "${#ENGINE_ARGS[@]}" -gt 0 ] && ! "$PY" -m numpilot configure-llm --check; then
  if [ -t 0 ] && [ -t 1 ]; then
    "$PY" -m numpilot configure-llm
  else
    echo "错误:LLM 未配置；请运行交互配置，或用 '$PY -m numpilot configure-llm --init' 生成文件后手填" >&2
    exit 1
  fi
fi

echo "NumPilot 一键演示"
echo "  python : $PY"
echo "  网络   : $FILE"
echo "  引擎   : $ENGINE_LABEL"
echo "  输出   : $OUT/"

# ── 1) 环境自检:证明真机 asnumpy / CANN / NPU 就位 ──────────
banner "1/4 环境自检 (doctor)"
run "$PY" -m numpilot doctor --backend npu

# ── 2) 迁移证明:np.* → ap.*, 真机跑通, 数值一致, asnumpy 证据 ─
banner "2/4 迁移证明 (migrate, 引擎=$ENGINE_LABEL)"
run "$PY" -m numpilot migrate "$FILE" --backend npu --out out_npu "${ENGINE_ARGS[@]}"

# ── 3) CPU 基线：固定 1 线程，并由报告记录真实线程池。────────
banner "3/4 CPU 基线 (bench-cpu, BLAS 线程=1)"
run "$PY" -m numpilot bench-cpu "$FILE" --out "$OUT"

# ── 4) strict-NPU：fallback/unknown 直接失败，不发布加速比。──
banner "4/4 strict-NPU 对比 (引擎=$ENGINE_LABEL)"
run "$PY" -m numpilot bench-npu "$FILE" --out "$OUT" "${ENGINE_ARGS[@]}"

# ── 可选：保存不限制 BLAS 的第二种 CPU 口径。────────────────
if [ "$FULL" -eq 1 ]; then
  banner "附加 numpy 不主动限制 BLAS 的对照 (--cpu-threads 0)"
  run "$PY" -m numpilot bench-cpu "$FILE" --out "${OUT}_full" --cpu-threads 0
  run "$PY" -m numpilot bench-npu "$FILE" --out "${OUT}_full"
  echo
  echo "对照说明:报告会记录实际 BLAS 线程；不要用逻辑 CPU 数推测线程池大小。"
fi

echo
echo "✅ 演示完成。请以 $OUT/npu_benchmark.json 与 benchmark_report.md 为准；"
echo "   只有 performance_valid=true 时才能引用其中的 NPU 加速比。"
