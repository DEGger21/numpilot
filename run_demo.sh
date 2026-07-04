#!/usr/bin/env bash
# NumPilot 一键演示:环境自检 → 迁移证明 → 公平性能对比(numpy 单核 vs NPU 单卡)。
#
# 用法:
#   ./run_demo.sh                      # 默认演示 big_gemm(4 层 2048 宽 GEMM 网络, NPU 发力场景)
#   ./run_demo.sh examples/mlp_forward.py   # 换一个网络
#   ./run_demo.sh --full               # 额外跑"numpy 吃满全核"对照, 直观展示 640 核作弊效果
#   ./run_demo.sh --llm                # 迁移改用真·大模型智能体(glm-5.1, 读 .env.llm);失败即终止
#
# 环境:默认用 venv /home/lixing/np310 里的 python(装了真机 asnumpy);
#       可用 NUMPILOT_PY 覆盖,例:NUMPILOT_PY=python3 ./run_demo.sh
set -u

# ── 参数解析 ──────────────────────────────────────────────
FILE="examples/big_gemm.py"
OUT="out_demo"
FULL=0
ENGINE_ARGS=""
ENGINE_LABEL="ast(确定性改写)"
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    --llm)  ENGINE_ARGS="--engine llm"; ENGINE_LABEL="llm(glm-5.1 智能体)" ;;
    *.py)   FILE="$arg" ;;
    *)      echo "忽略未知参数: $arg" ;;
  esac
done

PY="${NUMPILOT_PY:-/home/lixing/np310/bin/python}"
cd "$(dirname "$0")" || exit 1

# --llm:加载大模型凭据(.env.llm 已被 .gitignore 忽略)。strict 语义下模型失败即终止。
if [ -n "$ENGINE_ARGS" ]; then
  if [ -f ./.env.llm ]; then set -a; . ./.env.llm; set +a
  else echo "错误:--llm 需要 ./.env.llm(填 base_url/token/model)" >&2; exit 1; fi
fi

banner() { printf '\n\033[1;36m========== %s ==========\033[0m\n' "$1"; }
run()    { printf '\033[0;33m$ %s\033[0m\n' "$*"; "$@"; }

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
run "$PY" -m numpilot migrate "$FILE" --backend npu --out out_npu $ENGINE_ARGS

# ── 3) CPU 基线:numpy 钉死 1 核(与单卡 NPU 对等)────────────
banner "3/4 CPU 基线 (bench-cpu, numpy 限 1 核)"
run "$PY" -m numpilot bench-cpu "$FILE" --out "$OUT"

# ── 4) NPU 对比:放开所有限制, 读基线算加速比 ────────────────
banner "4/4 NPU 对比 (bench-npu, 满线程 + 加速比, 引擎=$ENGINE_LABEL)"
run "$PY" -m numpilot bench-npu "$FILE" --out "$OUT" $ENGINE_ARGS

# ── 可选:全核对照, 展示 640 核如何让 numpy 虚快 ─────────────
if [ "$FULL" -eq 1 ]; then
  banner "附加 numpy 吃满全核对照 (--cpu-threads 0)"
  run "$PY" -m numpilot bench-cpu "$FILE" --out "${OUT}_full" --cpu-threads 0
  run "$PY" -m numpilot bench-npu "$FILE" --out "${OUT}_full"
  echo
  echo "对照说明:同一 NPU 时延不变, 仅 numpy 从 1 核换成 640 核, 加速比即翻转——"
  echo "         印证'CPU 用 640 核 vs NPU 单卡'的不对等, 公平口径应取 3/4 步。"
fi

echo
echo "✅ 演示完成。头条:第 4 步 [对比] 行的加速比即 numpy@CPU(1核) / asnumpy@npu(单卡)。"
