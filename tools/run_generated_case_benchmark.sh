#!/usr/bin/env bash
# Route-check and CPU/NPU benchmark one migrated generated-operator case.
# Usage: run_generated_case_benchmark.sh <source.py> <migrated.py> <op> <stage> <vendor> <bench-dir>
set -uo pipefail
source_py=$1
migrated=$2
op=$3
stage=$4
vendor=$5
bench=$6
py=/home/ma-user/work/.venvs/asnumpy-runtime/bin/python
mkdir -p "$bench"
export PYTHONPATH="$stage"
export LD_LIBRARY_PATH="$vendor/op_api/lib:/usr/local/Ascend/cann-9.2.0-beta.2/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$vendor"
start=$(date +%s)
"$py" /home/ma-user/work/numpilot/tools/check_case_route.py \
  --original "$source_py" --migrated "$migrated" --op "$op" \
  --out "$bench/route-check.json" > "$bench/route-check.log" 2>&1
route_rc=$?
"$py" /home/ma-user/work/numpilot/tools/benchmark_case_perf.py \
  --original "$source_py" --migrated "$migrated" \
  --out "$bench/benchmark.json" --warmup 3 --samples 20 > "$bench/benchmark.log" 2>&1
bench_rc=$?
end=$(date +%s)
{
  echo "status=$([ $route_rc -eq 0 ] && [ $bench_rc -eq 0 ] && echo passed || echo failed)"
  echo "route_rc=$route_rc"
  echo "benchmark_rc=$bench_rc"
  echo "elapsed_seconds=$((end-start))"
} > "$bench/status.txt"
exit $([ $route_rc -eq 0 ] && [ $bench_rc -eq 0 ] && echo 0 || echo 1)
