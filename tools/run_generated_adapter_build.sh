#!/usr/bin/env bash
# Build one case-local AsNumpy wheel carrying exactly one generated adapter.
# Usage: run_generated_adapter_build.sh <build-root> <vendor-dir>
set -uo pipefail
root=$1
vendor=$2
py=/home/ma-user/work/.venvs/asnumpy-runtime/bin/python
rm -rf "$root/wheelhouse" "$root/stage"
mkdir -p "$root/wheelhouse" "$root/stage"
cd "$root/asnumpy"
start=$(date +%s)
{
  echo "status=building"
  echo "vendor=$vendor"
  echo "start=$(date -Is)"
} > "$root/status.txt"
SETUPTOOLS_SCM_PRETEND_VERSION=0.3.1 \
CMAKE_BUILD_PARALLEL_LEVEL=8 \
ASNUMPY_CUSTOM_OPP_VENDOR="$vendor" \
"$py" -m pip wheel . --no-deps --no-build-isolation -w "$root/wheelhouse" -v > "$root/wheel.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  "$py" -m pip install --no-deps --target "$root/stage" "$root"/wheelhouse/asnumpy-*.whl > "$root/stage.log" 2>&1
  rc=$?
fi
end=$(date +%s)
{
  echo "status=$([ $rc -eq 0 ] && echo built || echo build_failed)"
  echo "vendor=$vendor"
  echo "start=$(date -Is)"
  echo "end=$(date -Is)"
  echo "elapsed_seconds=$((end-start))"
  echo "rc=$rc"
  echo "wheel=$(ls "$root"/wheelhouse/asnumpy-*.whl 2>/dev/null | head -1)"
} > "$root/status.txt"
exit $rc
