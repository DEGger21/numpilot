#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BUILD_OUT_DIR="${ROOT_DIR}/build_out"
PACKAGE_NAME="roll_custom"
TARGET_SOC_LABEL="Ascend910B3"
TARGET_CANN_VERSION="9.2.0-beta.2"

SUPPORTED_COMPUTE_UNITS=("ascend910b")

CORE_NUMS="$( (getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 4) | head -n1 )"
case "${CORE_NUMS}" in
  ''|*[!0-9]*) CORE_NUMS=4 ;;
esac
if [ "${CORE_NUMS}" -lt 1 ]; then
  CORE_NUMS=1
fi
if [ "${CORE_NUMS}" -gt 8 ]; then
  CORE_NUMS=8
fi
THREADS="${CORE_NUMS}"

COMPUTE_UNIT="ascend910b"
ENABLE_CLEAN=FALSE
ENABLE_COV=FALSE
RUN_UT=FALSE
RUN_ST=FALSE
RUN_EXAMPLE=FALSE
EXAMPLE_MODE="eager"

usage() {
  echo "Build script for the Roll custom operator"
  echo ""
  echo "Usage: bash build.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -h, --help                 Print this help message"
  echo "  --list-socs                List supported SoC versions"
  echo "  --soc=<soc_version>        Target compute unit (default: ascend910b)"
  echo "                             Accepted: ascend910b, ascend910b3 (normalized to ascend910b)"
  echo "  -j[n]                      Compile thread count, default: ${CORE_NUMS}, e.g. -j8"
  echo "  --make_clean               Remove isolated build/ and build_out/ directories"
  echo "  -u, --ut                   Build and run unit tests from tests/ut/"
  echo "      --cov                  Enable coverage mode for unit tests"
  echo "  -s, --st                   Build and run system tests from tests/st/"
  echo "  -e, --example              Build and run the aclnn example from examples/"
  echo "      --eager                Use eager aclnn example mode (default for -e)"
  echo "      --graph                Use GE graph example mode for -e"
  echo "  -a, --all                  Run unit tests and system tests"
  echo ""
  echo "Notes:"
  echo "  - This script only builds the operator package. It never installs or activates it."
  echo "  - ST and examples require a separately installed and explicitly approved package."
  echo "  - ASCEND_HOME_PATH must point at the target CANN ${TARGET_CANN_VERSION} installation."
  echo ""
  echo "Examples:"
  echo "  bash build.sh --soc=ascend910b -j8"
  echo "  bash build.sh --make_clean"
  echo "  bash build.sh --soc=ascend910b -j8 -u"
  echo "  bash build.sh --soc=ascend910b -j8 -s"
  echo "  bash build.sh --soc=ascend910b -j8 -e --eager"
}

info() {
  echo "[INFO] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

normalize_soc() {
  local unit
  unit="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  case "${unit}" in
    ascend910b|ascend910b3)
      echo "ascend910b"
      ;;
    *)
      return 1
      ;;
  esac
}

clean_workspace() {
  if [ -d "${BUILD_DIR}" ]; then
    rm -rf "${BUILD_DIR}"
  fi
  if [ -d "${BUILD_OUT_DIR}" ]; then
    rm -rf "${BUILD_OUT_DIR}"
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --list-socs)
      echo "Supported compute units for ${TARGET_SOC_LABEL}:"
      for soc in "${SUPPORTED_COMPUTE_UNITS[@]}"; do
        echo "  - ${soc}"
      done
      exit 0
      ;;
    --make_clean)
      ENABLE_CLEAN=TRUE
      shift
      ;;
    -j)
      shift
      [ "$#" -gt 0 ] || die "-j requires a numeric value"
      THREADS="$1"
      shift
      ;;
    -j[0-9]*)
      THREADS="${1#-j}"
      shift
      ;;
    -u|--ut)
      RUN_UT=TRUE
      shift
      ;;
    --cov)
      ENABLE_COV=TRUE
      shift
      ;;
    -s|--st)
      RUN_ST=TRUE
      shift
      ;;
    -a|--all)
      RUN_UT=TRUE
      RUN_ST=TRUE
      shift
      ;;
    -e|--example)
      RUN_EXAMPLE=TRUE
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
    --soc)
      shift
      [ "$#" -gt 0 ] || die "--soc requires a value"
      COMPUTE_UNIT="$1"
      shift
      ;;
    --soc=*)
      COMPUTE_UNIT="${1#*=}"
      shift
      ;;
    -*)
      die "Invalid option: $1"
      ;;
    *)
      die "Unexpected argument: $1"
      ;;
  esac
done

if [ "${ENABLE_CLEAN}" = "TRUE" ]; then
  clean_workspace
  info "Cleaned isolated build directories."
  exit 0
fi

if ! [[ "${THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  die "Thread count must be a positive integer, got: ${THREADS}"
fi

if ! NORMALIZED_SOC="$(normalize_soc "${COMPUTE_UNIT}")"; then
  die "Unsupported SoC '${COMPUTE_UNIT}'. Target is ${TARGET_SOC_LABEL} (compute unit ascend910b)."
fi
COMPUTE_UNIT="${NORMALIZED_SOC}"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  die "ASCEND_HOME_PATH is not set. Source the target CANN ${TARGET_CANN_VERSION} environment first."
fi
if [ ! -d "${ASCEND_HOME_PATH}" ]; then
  die "ASCEND_HOME_PATH does not point to a directory: ${ASCEND_HOME_PATH}"
fi

command -v cmake >/dev/null 2>&1 || die "cmake was not found in PATH"
[ -f "${ROOT_DIR}/CMakeLists.txt" ] || die "Missing root CMakeLists.txt in ${ROOT_DIR}"

info "Operator: Roll (asnumpy.roll)"
info "Target SoC: ${TARGET_SOC_LABEL}"
info "Compute unit: ${COMPUTE_UNIT}"
info "CANN version: ${TARGET_CANN_VERSION}"
info "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
info "Build thread count: ${THREADS}"

mkdir -p "${BUILD_DIR}" "${BUILD_OUT_DIR}"

info "Configuring project..."
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DASCEND_COMPUTE_UNIT="${COMPUTE_UNIT}" \
  -DCMAKE_BUILD_TYPE=Release

info "Building host libraries, kernel binary, and operator package..."
cmake --build "${BUILD_DIR}" --target all binary package -- -j "${THREADS}"

KERNEL_DIR="${BUILD_DIR}/op_kernel/ascendc_kernels/binary/${COMPUTE_UNIT}"
KERNEL_O=""
if [ -d "${KERNEL_DIR}" ]; then
  KERNEL_O="$(find "${KERNEL_DIR}" -type f -name '*.o' -print -quit 2>/dev/null || true)"
fi
if [ -z "${KERNEL_O}" ] && [ -d "${BUILD_DIR}/op_kernel" ]; then
  KERNEL_O="$(find "${BUILD_DIR}/op_kernel" -type f -name '*.o' -print -quit 2>/dev/null || true)"
fi
[ -n "${KERNEL_O}" ] || die "Kernel binary not found under ${BUILD_DIR}/op_kernel"

PKG_PATH="${BUILD_DIR}/custom_opp_ubuntu_aarch64.run"
if [ ! -f "${PKG_PATH}" ] || [ ! -s "${PKG_PATH}" ]; then
  PKG_PATH="$(find "${BUILD_DIR}" -maxdepth 2 -type f -name '*.run' -size +0c -print -quit 2>/dev/null || true)"
fi
[ -n "${PKG_PATH}" ] || die "Operator package not found under ${BUILD_DIR}"
[ -f "${PKG_PATH}" ] || die "Operator package path is not a file: ${PKG_PATH}"
[ -s "${PKG_PATH}" ] || die "Operator package is empty: ${PKG_PATH}"

MANIFEST="${BUILD_DIR}/roll_build_manifest.txt"
{
  echo "operator=Roll"
  echo "api=asnumpy.roll"
  echo "gap_id=roll-18cf7f900d51d230"
  echo "target_soc=${TARGET_SOC_LABEL}"
  echo "compute_unit=${COMPUTE_UNIT}"
  echo "cann_version=${TARGET_CANN_VERSION}"
  echo "claim_level=unverified_candidate_only"
  echo "kernel_binary=${KERNEL_O}"
  echo "package=${PKG_PATH}"
} > "${MANIFEST}"

info "Kernel binary: ${KERNEL_O}"
info "Operator package: ${PKG_PATH}"
info "Build manifest: ${MANIFEST}"
info "Build complete. The generated package remains unverified until deterministic gates pass."

if [ "${RUN_UT}" = "TRUE" ]; then
  UT_RUNNER="${ROOT_DIR}/tests/ut/run.sh"
  [ -f "${UT_RUNNER}" ] || die "UT runner not found: ${UT_RUNNER}"
  info "Running unit tests..."
  (
    cd "${ROOT_DIR}/tests/ut"
    ENABLE_COV="${ENABLE_COV}" \
    ASCEND_COMPUTE_UNIT="${COMPUTE_UNIT}" \
    ROLL_OPERATOR_PACKAGE="${PKG_PATH}" \
    bash ./run.sh
  )
  info "Unit tests completed."
fi

if [ "${RUN_ST}" = "TRUE" ]; then
  ST_RUNNER="${ROOT_DIR}/tests/st/run.sh"
  [ -f "${ST_RUNNER}" ] || die "ST runner not found: ${ST_RUNNER}"
  info "Running system tests."
  info "ST requires a separately installed and explicitly approved operator package; build.sh does not install or activate it."
  (
    cd "${ROOT_DIR}/tests/st"
    ASCEND_COMPUTE_UNIT="${COMPUTE_UNIT}" \
    ROLL_OPERATOR_PACKAGE="${PKG_PATH}" \
    bash ./run.sh
  )
  info "System tests completed."
fi

if [ "${RUN_EXAMPLE}" = "TRUE" ]; then
  EXAMPLE_RUNNER="${ROOT_DIR}/examples/run.sh"
  [ -f "${EXAMPLE_RUNNER}" ] || die "Example runner not found: ${EXAMPLE_RUNNER}"
  info "Running example in ${EXAMPLE_MODE} mode."
  info "Examples require a separately installed and explicitly approved operator package; build.sh does not install or activate it."
  (
    cd "${ROOT_DIR}/examples"
    ASCEND_COMPUTE_UNIT="${COMPUTE_UNIT}" \
    ROLL_OPERATOR_PACKAGE="${PKG_PATH}" \
    bash ./run.sh --"${EXAMPLE_MODE}"
  )
  info "Example completed."
fi

exit 0