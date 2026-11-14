#!/usr/bin/env bash
# Build one model-generated Ascend C operator draft in an isolated build dir.
# Usage: run_generated_asc_build.sh <case> <draft-dir> <build-dir>
set -uo pipefail
case_name=$1
draft=$2
build=$3
mkdir -p "$build"
rm -rf "$build/project"
cp -a "$draft" "$build/project"
cd "$build/project"
start_iso=$(date -Is)
start=$(date +%s)
{
  echo "status=building"
  echo "case=$case_name"
  echo "draft=$draft"
  echo "start=$start_iso"
} > "$build/status.txt"
bash build.sh --soc=ascend910b -j8 > "$build/build.log" 2>&1
rc=$?
end=$(date +%s)
{
  echo "status=$([ $rc -eq 0 ] && echo built || echo build_failed)"
  echo "case=$case_name"
  echo "draft=$draft"
  echo "start=$start_iso"
  echo "end=$(date -Is)"
  echo "elapsed_seconds=$((end-start))"
  echo "rc=$rc"
  echo "log=$build/build.log"
} > "$build/status.txt"
exit $rc
