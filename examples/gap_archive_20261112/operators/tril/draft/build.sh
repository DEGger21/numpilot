#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_PATH="${BASE_PATH}/build"
BUILD_OUT_PATH="${BASE_PATH}/build_out"

# The generated candidate is bounded to the recorded target SoC.
SUPPORT_COMPUTE_UNITS=("ascend910b")
DEFAULT_COMPUTE_UNIT="ascend910b"

CORE_NUMS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
case "${CORE_NUMS}" in
  ''|*[!0-9]*) CORE_NUMS=8 ;;
esac
if (( CORE_NUMS > 8 )); then
  CORE_NUMS=8
fi
if (( CORE_NUMS < 1 )); then
  CORE_NUMS=1
fi

die() {
  echo "[tril][ERROR] $*" >&2
  exit 1
}

info() {
  echo "[tril][INFO] $*"
}

warn() {
  echo "[tril][WARN] $*" >&2
}

usage() {
  cat <<'USAGE'
Build script for the Tril custom operator candidate.

Usage:
  bash build.sh [OPTIONS]

Options:
  -h, --help                 Print this help message
  --list-socs                List supported SoC versions
  -j[n], --jobs[=n]          Compile thread count, default is the detected CPU count capped at 8
  --soc=<soc_version>        Target SoC; only ascend910b (Ascend910B3) is admitted
  --make_clean               Remove build/ and build_out/, then exit
  -u, --ut                   Run op_host/op_api unit tests from tests/ut
      --cov                  Request UT coverage support
  -s, --st                   Run system tests from tests/st against the active package
      --mock                 Pass --mock to tests/st/run.sh
  -e, --example              Run examples from examples against the active package
      --eager                Run aclnn eager examples (default for -e)
      --graph                Run graph-mode examples
  -a, --all                  Run UT and ST tests
  --install                  Explicitly install the built .run package into the active CANN
  --skip-build               Reuse existing build artifacts and package
  --check-public-api         Check that the current Python environment exposes asnumpy.tril

Notes:
  - The script never installs a package unless --install is explicitly supplied.
  - The script never edits shell startup files or system CANN files.
  - ST and example execution require the candidate package to be active in an isolated environment.
  - Build success is not a numerical or runtime qualification claim.

Examples:
  bash build.sh --soc=ascend910b -j8
  bash build.sh --make_clean
  bash build.sh -u --soc=ascend910b -j8
  bash build.sh --install --soc=ascend910b -j8
  bash build.sh -s --soc=ascend910b -j8
  bash build.sh -e --graph --soc=ascend910b -j8
  bash build.sh -a --soc=ascend910b -j8
USAGE
}

normalize_soc() {
  local soc
  soc="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${soc}" in
    ascend910b|ascend910b3|910b3)
      printf '%s\n' "ascend910b"
      ;;
    *)
      printf '%s\n' "${soc}"
      ;;
  esac
}

setup_cann_env() {
  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    if [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
      set +u
      # shellcheck disable=SC1090
      source "${ASCEND_HOME_PATH}/set_env.sh"
      set -u
    fi
    return 0
  fi

  local candidates=()
  if [[ -n "${ASCEND_TOOLKIT_HOME:-}" ]]; then
    candidates+=("${ASCEND_TOOLKIT_HOME}")
  fi
  candidates+=(
    "/usr/local/Ascend/ascend-toolkit/latest"
    "/usr/local/Ascend/ascend-toolkit/9.2.0-beta.2"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -n "${candidate}" && -f "${candidate}/set_env.sh" ]]; then
      set +u
      # shellcheck disable=SC1090
      source "${candidate}/set_env.sh"
      set -u
      export ASCEND_HOME_PATH="${candidate}"
      return 0
    fi
  done

  die "CANN environment not found. Set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME to the pinned CANN 9.2.0-beta.2 installation."
}

clean_all() {
  info "Removing ${BUILD_PATH}"
  rm -rf "${BUILD_PATH}"
  info "Removing ${BUILD_OUT_PATH}"
  rm -rf "${BUILD_OUT_PATH}"
}

find_kernel_object() {
  local kernel_dir="${BUILD_PATH}/op_kernel/ascendc_kernels/binary/${COMPUTE_UNIT}"
  if [[ ! -d "${kernel_dir}" ]]; then
    return 1
  fi
  find "${kernel_dir}" -type f -name '*.o' -print -quit 2>/dev/null
}

resolve_package() {
  local standard_package="${BUILD_PATH}/custom_opp_ubuntu_aarch64.run"
  if [[ -s "${standard_package}" ]]; then
    printf '%s\n' "${standard_package}"
    return 0
  fi

  local found=""
  found="$(find "${BUILD_PATH}" "${BUILD_OUT_PATH}" -maxdepth 3 -type f -name '*.run' -print -quit 2>/dev/null || true)"
  if [[ -z "${found}" ]]; then
    return 1
  fi
  printf '%s\n' "${found}"
}

install_package() {
  local package_path="$1"
  warn "Explicit --install requested. This writes the built custom operator package into the active CANN installation."
  info "Installing candidate package: ${package_path}"
  "${package_path}"
}

run_ut() {
  local ut_runner="${BASE_PATH}/tests/ut/run.sh"
  if [[ ! -f "${ut_runner}" ]]; then
    die "UT runner not found: ${ut_runner}"
  fi
  info "Running UT from ${BASE_PATH}/tests/ut"
  (
    cd "${BASE_PATH}/tests/ut"
    ENABLE_COV="${ENABLE_COV}" bash ./run.sh
  )
}

run_st() {
  local st_runner="${BASE_PATH}/tests/st/run.sh"
  if [[ ! -f "${st_runner}" ]]; then
    die "ST runner not found: ${st_runner}"
  fi
  info "Running ST from ${BASE_PATH}/tests/st"
  if [[ "${ST_MOCK}" == "true" ]]; then
    (
      cd "${BASE_PATH}/tests/st"
      bash ./run.sh --mock
    )
  else
    (
      cd "${BASE_PATH}/tests/st"
      bash ./run.sh
    )
  fi
}

run_examples() {
  local example_runner="${BASE_PATH}/examples/run.sh"
  if [[ ! -f "${example_runner}" ]]; then
    die "Example runner not found: ${example_runner}"
  fi
  info "Running examples in ${EXAMPLE_MODE} mode from ${BASE_PATH}/examples"
  (
    cd "${BASE_PATH}/examples"
    bash ./run.sh "--${EXAMPLE_MODE}"
  )
}

check_public_api() {
  command -v python3 >/dev/null 2>&1 || die "python3 is required for --check-public-api"
  info "Checking public API binding: asnumpy.tril"
  python3 - <<'PY'
import sys

try:
    import asnumpy
except Exception as exc:
    raise SystemExit(
        "[tril][ERROR] Failed to import asnumpy in the current environment: "
        f"{type(exc).__name__}: {exc}"
    )

tril = getattr(asnumpy, "tril", None)
if not callable(tril):
    raise SystemExit(
        "[tril][ERROR] asnumpy.tril is not callable in the current environment"
    )

print("[tril][INFO] asnumpy.tril is importable and callable")
PY
}

THREAD_NUM="${CORE_NUMS}"
COMPUTE_UNIT="${DEFAULT_COMPUTE_UNIT}"
CLEAN_BUILD=false
ENABLE_COV="FALSE"
RUN_UT=false
RUN_ST=false
RUN_EXAMPLE=false
EXAMPLE_MODE="eager"
ST_MOCK=false
DO_INSTALL=false
SKIP_BUILD=false
CHECK_PUBLIC_API=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --list-socs)
      echo "Supported SoC versions for this candidate:"
      for soc in "${SUPPORT_COMPUTE_UNITS[@]}"; do
        echo "  - ${soc} (Ascend910B3)"
      done
      exit 0
      ;;
    --make_clean|--make-clean|--clean)
      CLEAN_BUILD=true
      shift
      ;;
    -j|--jobs)
      if [[ $# -lt 2 ]]; then
        die "$1 requires an explicit thread count"
      fi
      THREAD_NUM="$2"
      shift 2
      ;;
    --jobs=*)
      THREAD_NUM="${1#*=}"
      shift
      ;;
    -j*)
      THREAD_NUM="${1#-j}"
      THREAD_NUM="${THREAD_NUM#=}"
      if [[ -z "${THREAD_NUM}" ]]; then
        THREAD_NUM="${CORE_NUMS}"
      fi
      shift
      ;;
    -u|--ut)
      RUN_UT=true
      shift
      ;;
    --cov)
      ENABLE_COV="TRUE"
      shift
      ;;
    -s|--st)
      RUN_ST=true
      shift
      ;;
    --mock)
      ST_MOCK=true
      shift
      ;;
    -e|--example)
      RUN_EXAMPLE=true
      shift
      ;;
    --eager)
      EXAMPLE_MODE="eager"
      shift
      ;;
    --graph)
      EXAMPLE_MODE="graph"
      shift
      ;;
    -a|--all)
      RUN_UT=true
      RUN_ST=true
      shift
      ;;
    --install)
      DO_INSTALL=true
      shift
      ;;
    --skip-build)
      SKIP_BUILD=true
      shift
      ;;
    --check-public-api)
      CHECK_PUBLIC_API=true
      shift
      ;;
    --soc)
      if [[ $# -lt 2 ]]; then
        die "--soc requires a value"
      fi
      COMPUTE_UNIT="$2"
      shift 2
      ;;
    --soc=*)
      COMPUTE_UNIT="${1#*=}"
      shift
      ;;
    -*)
      usage >&2
      die "Invalid option: $1"
      ;;
    *)
      usage >&2
      die "Unexpected argument: $1"
      ;;
  esac
done

if [[ "${CLEAN_BUILD}" == "true" ]]; then
  clean_all
  exit 0
fi

COMPUTE_UNIT="$(normalize_soc "${COMPUTE_UNIT}")"
if [[ -z "${COMPUTE_UNIT}" ]]; then
  die "--soc must not be empty"
fi
if [[ "${COMPUTE_UNIT}" != "ascend910b" ]]; then
  die "Unsupported SoC '${COMPUTE_UNIT}'. This candidate targets only Ascend910B3 (ascend910b)."
fi

if ! [[ "${THREAD_NUM}" =~ ^[1-9][0-9]*$ ]]; then
  die "Invalid thread count '${THREAD_NUM}'. Expected a positive integer."
fi

setup_cann_env

command -v cmake >/dev/null 2>&1 || die "cmake is required but was not found in PATH"
if [[ ! -f "${BASE_PATH}/CMakeLists.txt" ]]; then
  die "CMakeLists.txt not found in ${BASE_PATH}"
fi

CMAKE_ARGS=(
  "-DCMAKE_BUILD_TYPE=Release"
  "-DASCEND_COMPUTE_UNIT=${COMPUTE_UNIT}"
)
if [[ "${ENABLE_COV}" == "TRUE" ]]; then
  CMAKE_ARGS+=("-DENABLE_COV=ON")
fi

if [[ "${SKIP_BUILD}" != "true" ]]; then
  info "Configuring Tril operator for ${COMPUTE_UNIT}"
  mkdir -p "${BUILD_PATH}"
  rm -f "${BUILD_PATH}/CMakeCache.txt"
  cmake -S "${BASE_PATH}" -B "${BUILD_PATH}" "${CMAKE_ARGS[@]}"

  info "Building host libraries, kernel binaries, and package with ${THREAD_NUM} threads"
  cmake --build "${BUILD_PATH}" --target all binary package -- -j "${THREAD_NUM}"

  KERNEL_OBJ="$(find_kernel_object || true)"
  if [[ -z "${KERNEL_OBJ}" ]]; then
    die "Kernel binary not found under ${BUILD_PATH}/op_kernel/ascendc_kernels/binary/${COMPUTE_UNIT}"
  fi
  info "Kernel binary: ${KERNEL_OBJ}"
else
  info "Skipping compilation because --skip-build was requested"
fi

PKG_PATH="$(resolve_package || true)"
if [[ -z "${PKG_PATH}" || ! -s "${PKG_PATH}" ]]; then
  die "Custom operator package not found under ${BUILD_PATH} or ${BUILD_OUT_PATH}"
fi
info "Package: ${PKG_PATH}"

if [[ "${DO_INSTALL}" == "true" ]]; then
  install_package "${PKG_PATH}"
  setup_cann_env
fi

if [[ "${DO_INSTALL}" != "true" ]]; then
  if [[ "${RUN_ST}" == "true" || "${RUN_EXAMPLE}" == "true" ]]; then
    warn "ST/examples will run against the currently active package. Use --install explicitly to install this build."
  fi
fi

if [[ "${CHECK_PUBLIC_API}" == "true" ]]; then
  check_public_api
fi

if [[ "${RUN_UT}" == "true" ]]; then
  run_ut
fi

if [[ "${RUN_ST}" == "true" ]]; then
  run_st
fi

if [[ "${RUN_EXAMPLE}" == "true" ]]; then
  run_examples
fi

if [[ "${RUN_UT}" != "true" && "${RUN_ST}" != "true" && "${RUN_EXAMPLE}" != "true" && "${DO_INSTALL}" != "true" ]]; then
  info "Build artifacts prepared. No package installation or device execution was performed."
fi

info "Build stage complete. Runtime qualification still requires explicit installation, numerical checks, and activation gates."