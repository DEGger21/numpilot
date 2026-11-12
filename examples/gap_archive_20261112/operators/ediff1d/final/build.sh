#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# Isolated build driver for the ediff1d registry-invoke operator draft.
#
# Scope for this draft:
#   - C++ OpType: Ediff1d
#   - API/file stem: ediff1d
#   - target SoC: Ascend910B3, expressed as CANN compute unit "ascend910b"
#   - target CANN line: 9.2.0-beta.2
#   - admitted tensor profile: float32, 2-D C-contiguous input -> 1-D float32 output
#
# This script performs an out-of-source build under ./build and does not edit CANN, ~/.bashrc, or any
# shared AsNumpy checkout. Optional test/example paths install the produced operator package only when
# the caller explicitly asks for ST or example execution.
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_OUT_DIR="${SCRIPT_DIR}/build_out"
PACKAGE_PATH="${BUILD_DIR}/custom_opp_ubuntu_aarch64.run"

SUPPORTED_SOCS=("ascend910b")
DEFAULT_SOC="ascend910b"

CORE_NUMS="$(nproc 2>/dev/null || true)"
if ! [[ "${CORE_NUMS}" =~ ^[0-9]+$ ]] || [ "${CORE_NUMS}" -lt 1 ]; then
  CORE_NUMS=8
fi
if [ "${CORE_NUMS}" -gt 8 ]; then
  CORE_NUMS=8
fi

THREAD_NUM="${CORE_NUMS}"
COMPUTE_UNIT=""
DO_CLEAN=false
RUN_UT=false
RUN_ST=false
RUN_EXAMPLE=false
ENABLE_COV=false
EXAMPLE_MODE="eager"

usage() {
  cat <<EOF
Build script for the ediff1d Ascend C registry-invoke operator draft

Usage:
  bash build.sh --soc=ascend910b [OPTIONS]

Required for build:
  --soc=<soc>             Target compute unit. Supported: ascend910b (Ascend910B3)

Options:
  -h, --help              Print this help message
  --list-socs             List supported SoC compute units
  -j[n], --jobs=[n]       Compile thread count, default ${CORE_NUMS}
  --make_clean, --clean   Remove build/ and build_out/, then exit
  -u, --ut                Run UT after a successful build
  --cov                   Request UT coverage mode (passed to tests/ut/run.sh)
  -s, --st                Install the built package and run ST after a successful build
  -e, --example           Install the built package and run the ACLNN example
  --eager                 Example mode: eager ACLNN call (default for -e)
  --graph                 Example mode: GE IR graph call
  -a, --all               Run UT and ST after a successful build

Examples:
  bash build.sh --soc=ascend910b -j8
  bash build.sh --soc=ascend910b -u
  bash build.sh --soc=ascend910b -s
  bash build.sh --soc=ascend910b -e --graph
  bash build.sh --make_clean
EOF
}

log() {
  local level="$1"
  shift
  printf '[ediff1d][%s] %s\n' "${level}" "$*"
}

die() {
  printf '[ediff1d][ERROR] %s\n' "$*" >&2
  exit 1
}

check_supported_soc() {
  local candidate="$1"
  local supported
  for supported in "${SUPPORTED_SOCS[@]}"; do
    if [ "${candidate}" = "${supported}" ]; then
      return 0
    fi
  done
  return 1
}

run_package_installer() {
  local pkg="$1"
  [ -f "${pkg}" ] || die "Operator package not found: ${pkg}"
  log INFO "Installing operator package explicitly requested by test/example flag: ${pkg}"
  if [ -x "${pkg}" ]; then
    "${pkg}"
  else
    bash "${pkg}"
  fi
}

parse_jobs_value() {
  local value="$1"
  case "${value}" in
    ''|*[!0-9]*)
      die "Invalid thread count: ${value}"
      ;;
  esac
  if [ "${value}" -lt 1 ]; then
    die "Thread count must be at least 1: ${value}"
  fi
  THREAD_NUM="${value}"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --list-socs)
      echo "Supported SoC compute units:"
      for soc in "${SUPPORTED_SOCS[@]}"; do
        if [ "${soc}" = "ascend910b" ]; then
          echo "  - ${soc} (Ascend910B3)"
        else
          echo "  - ${soc}"
        fi
      done
      exit 0
      ;;
    --make_clean|--clean)
      DO_CLEAN=true
      shift
      ;;
    -u|--ut)
      RUN_UT=true
      shift
      ;;
    --cov)
      ENABLE_COV=true
      shift
      ;;
    -s|--st)
      RUN_ST=true
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
    --soc)
      [ "$#" -ge 2 ] || die "--soc requires a value"
      COMPUTE_UNIT="$2"
      shift 2
      ;;
    --soc=*)
      COMPUTE_UNIT="${1#*=}"
      shift
      ;;
    -j)
      if [ "$#" -ge 2 ] && [[ "$2" =~ ^[0-9]+$ ]]; then
        parse_jobs_value "$2"
        shift 2
      else
        THREAD_NUM="${CORE_NUMS}"
        shift
      fi
      ;;
    -j*)
      parse_jobs_value "${1#-j}"
      shift
      ;;
    --jobs=*)
      parse_jobs_value "${1#*=}"
      shift
      ;;
    --jobs)
      [ "$#" -ge 2 ] || die "--jobs requires a value"
      parse_jobs_value "$2"
      shift 2
      ;;
    -*)
      die "Invalid option: $1"
      ;;
    *)
      die "Unexpected positional argument: $1"
      ;;
  esac
done

if [ "${DO_CLEAN}" = true ]; then
  log INFO "Removing isolated build directories"
  rm -rf "${BUILD_DIR}" "${BUILD_OUT_DIR}"
  log INFO "Clean complete"
  exit 0
fi

if [ -z "${COMPUTE_UNIT}" ]; then
  COMPUTE_UNIT="${DEFAULT_SOC}"
  log INFO "No --soc supplied; using project default ${DEFAULT_SOC}"
fi

COMPUTE_UNIT="$(printf '%s' "${COMPUTE_UNIT}" | tr '[:upper:]' '[:lower:]')"
if ! check_supported_soc "${COMPUTE_UNIT}"; then
  die "Unsupported SoC compute unit '${COMPUTE_UNIT}'. Supported: ${SUPPORTED_SOCS[*]}"
fi

case "${THREAD_NUM}" in
  ''|*[!0-9]*)
    die "Invalid thread count: ${THREAD_NUM}"
    ;;
esac
if [ "${THREAD_NUM}" -lt 1 ]; then
  die "Thread count must be at least 1"
fi

[ -f "${SCRIPT_DIR}/CMakeLists.txt" ] || die "Missing CMakeLists.txt at ${SCRIPT_DIR}/CMakeLists.txt"
command -v cmake >/dev/null 2>&1 || die "cmake was not found in PATH"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  log WARN "ASCEND_HOME_PATH is not set; source the pinned CANN 9.2.0-beta.2 environment if configuration fails"
fi

log INFO "Target SoC: ${COMPUTE_UNIT}"
log INFO "Thread count: ${THREAD_NUM}"
log INFO "Build directory: ${BUILD_DIR}"

mkdir -p "${BUILD_DIR}"
rm -f "${BUILD_DIR}/CMakeCache.txt"

CMAKE_ARGS=(
  "-DASCEND_COMPUTE_UNIT=${COMPUTE_UNIT}"
  "-DCMAKE_BUILD_TYPE=Release"
)

log INFO "Configuring project"
(
  cd "${BUILD_DIR}"
  cmake "${CMAKE_ARGS[@]}" "${SCRIPT_DIR}"
)

log INFO "Building host libraries, kernel binary, and package"
(
  cd "${BUILD_DIR}"
  cmake --build . --target all binary package -- -j "${THREAD_NUM}"
)

KERNEL_DIR="${BUILD_DIR}/op_kernel/ascendc_kernels/binary/${COMPUTE_UNIT}"
KERNEL_OBJ="$(find "${KERNEL_DIR}" -type f -name '*.o' -print -quit 2>/dev/null || true)"
[ -n "${KERNEL_OBJ}" ] || die "Kernel binary not found under ${KERNEL_DIR}"

if [ ! -f "${PACKAGE_PATH}" ] || [ ! -s "${PACKAGE_PATH}" ]; then
  FOUND_PACKAGE="$(find "${BUILD_DIR}" -maxdepth 2 -type f -name '*.run' -print -quit 2>/dev/null || true)"
  [ -n "${FOUND_PACKAGE}" ] || die "Operator package not found under ${BUILD_DIR}"
  PACKAGE_PATH="${FOUND_PACKAGE}"
fi

log INFO "Build artifacts produced"
echo "  kernel:  ${KERNEL_OBJ}"
echo "  package: ${PACKAGE_PATH}"
echo "  note:    plain build does not install or activate the package"

if [ "${RUN_UT}" = true ]; then
  UT_RUNNER="${SCRIPT_DIR}/tests/ut/run.sh"
  [ -f "${UT_RUNNER}" ] || die "UT runner not found: ${UT_RUNNER}"
  if [ "${ENABLE_COV}" = true ]; then
    COV_ENV="TRUE"
  else
    COV_ENV="FALSE"
  fi
  log INFO "Running UT"
  (
    cd "${SCRIPT_DIR}/tests/ut"
    ENABLE_COV="${COV_ENV}" bash ./run.sh
  )
fi

if [ "${RUN_ST}" = true ]; then
  ST_RUNNER="${SCRIPT_DIR}/tests/st/run.sh"
  [ -f "${ST_RUNNER}" ] || die "ST runner not found: ${ST_RUNNER}"
  run_package_installer "${PACKAGE_PATH}"
  log INFO "Running ST (Real mode)"
  (
    cd "${SCRIPT_DIR}/tests/st"
    bash ./run.sh
  )
fi

if [ "${RUN_EXAMPLE}" = true ]; then
  EXAMPLE_RUNNER="${SCRIPT_DIR}/examples/run.sh"
  [ -f "${EXAMPLE_RUNNER}" ] || die "Example runner not found: ${EXAMPLE_RUNNER}"
  run_package_installer "${PACKAGE_PATH}"
  log INFO "Running ACLNN example (${EXAMPLE_MODE} mode)"
  (
    cd "${SCRIPT_DIR}/examples"
    bash ./run.sh "--${EXAMPLE_MODE}"
  )
fi

if [ "${RUN_UT}" = true ] || [ "${RUN_ST}" = true ] || [ "${RUN_EXAMPLE}" = true ]; then
  log INFO "Requested build/test pipeline completed"
fi