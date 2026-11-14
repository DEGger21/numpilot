#!/usr/bin/env bash
# Install one built generated custom-OPP package into a case-local deploy dir.
# Usage: run_generated_asc_deploy.sh <build-dir> <deploy-dir>
set -uo pipefail
build=$1
deploy=$2
pkg=$(find "$build/project/build" "$build/project/build_out" -maxdepth 3 -name 'custom_opp_*.run' -print -quit 2>/dev/null)
[ -n "$pkg" ] || { echo "[deploy] no installer package found under $build"; exit 2; }
mkdir -p "$deploy"
start_iso=$(date -Is); start=$(date +%s)
echo "status=deploying" > "$deploy/status.txt"
bash "$pkg" --quiet --install-path="$deploy/install" > "$deploy/install.log" 2>&1
rc=$?
end=$(date +%s)
{
  echo "status=$([ $rc -eq 0 ] && echo deployed || echo deploy_failed)"
  echo "package=$pkg"
  echo "start=$start_iso"
  echo "end=$(date -Is)"
  echo "elapsed_seconds=$((end-start))"
  echo "rc=$rc"
} > "$deploy/status.txt"
[ $rc -eq 0 ] || { tail -n 30 "$deploy/install.log"; exit $rc; }
echo "[deploy] installed $pkg -> $deploy/install"
find "$deploy/install" -maxdepth 3 -type d -name vendors -exec ls -la {} \;
