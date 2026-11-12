#!/usr/bin/env bash
# Write-once runner for the complete Vitamin C grid-density chain evidence.
# Override NUMPILOT_REPO / VC_EVIDENCE_DIR when the checkout or evidence root moves.
set -euo pipefail

REPO=${NUMPILOT_REPO:-$HOME/work/numpilot}
BASE=${VC_EVIDENCE_DIR:-$HOME/work/documents/gpu4pyscf-vitamin-c-rdm-grid}
CPU_OUT="$BASE/real-numpy-fullchain-20261111-v1"
NPU_OUT="$BASE/real-asnumpy-fullchain-20261111-v1"
CPU_PY="$BASE/pyscf-venv/bin/python"
NPU_PY=${VC_RUNTIME_PYTHON:-$HOME/work/documents/gpu4pyscf-sandwich-demo/runtime-venv/bin/python}
BOOTSTRAP=${NUMPILOT_BOOTSTRAP_RECEIPT:-$HOME/work/documents/gpu4pyscf-sandwich-demo/bootstrap-receipt.json}

test ! -e "$CPU_OUT"
test ! -e "$NPU_OUT"
test ! -e "$CPU_OUT.stdout"
test ! -e "$NPU_OUT.stdout"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
cd "$REPO"

taskset -c 0 "$CPU_PY" "$REPO/examples/scientific/gpu4pyscf_vitamin_c/numpy_fullchain.py" \
  --baseline "$BASE/real-numpy-baseline-20261111-v2" \
  --out "$CPU_OUT" \
  > "$CPU_OUT.stdout" 2> "$CPU_OUT.stderr"

source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
export ASCEND_VISIBLE_DEVICES=6
export PYTHONPATH="$REPO"
export NUMPILOT_BOOTSTRAP_RECEIPT="$BOOTSTRAP"

taskset -c 0 "$NPU_PY" "$REPO/examples/scientific/gpu4pyscf_vitamin_c/asnumpy_fullchain.py" \
  --baseline "$BASE/real-numpy-baseline-20261111-v2" \
  --cpu-fullchain "$CPU_OUT" \
  --out "$NPU_OUT" \
  > "$NPU_OUT.stdout" 2> "$NPU_OUT.stderr"

python3 "$REPO/examples/scientific/gpu4pyscf_vitamin_c/aclnn_evidence.py" \
  --log "$NPU_OUT.stdout" \
  --receipt "$NPU_OUT/asnumpy_fullchain_receipt.json" \
  --out "$NPU_OUT/aclnn_evidence.json"
