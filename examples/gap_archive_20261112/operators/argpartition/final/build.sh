#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# Isolated build driver for the Argpartition custom Ascend C operator.
#
# Operator:  Argpartition (op type: Argpartition, API stem: argpartition)
# Target:    Ascend910B3 / ascend910b / CANN 9.2.0-beta.2
# Route:     registry-invoke (op_host + op_kernel + op_api)
#
# This script only writes inside the project directory (build/, build_out/).
# Installing the generated .run package into a CANN installation is NEVER
# implicit: it requires the explicit --install flag.
# ============================================================================

set -e

export BASE_PATH=$(
  cd "$(dirname "$0")"
  pwd
)
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"

# Only the SoC recorded in gap-spec.json is qualified for this candidate.
SUPPORT_COMPUTE_UNITS=("ascend910b")
DEFAULT_COMPUTE_UNIT="ascend910b"

if command -v nproc >/dev/null 2>&1; then
  CORE_NUMS=$(nproc)
else
  CORE_NUMS=$(grep -c '^processor' /proc/cpuinfo 2>/dev/null || echo 8)
fi
case "${CORE_NUMS}" in
  ''|*[!0-9]*) CORE_NUMS=8 ;;
esac
if [ "${CORE_NUMS}" -lt 1 ]; then
  CORE_NUMS=1
fi
if [ "${CORE_NUMS}" -gt 8 ]; then
  CORE_NUMS=8
fi

usage() {
  echo "Build script for the Argpartition custom Ascend C operator"
  echo "Target SoC: Ascend910B3 (ascend910b), CANN 9.2.0-beta.2"
  echo ""
  echo "Usage: bash build.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -h, --help              Print this help message"
  echo "  --list-socs             List qualified SoC versions"
  echo "  -j[n]                   Compile thread nums, default is ${CORE_NUMS}, eg: -j8"
  echo "  --soc=<soc_version>     Compile for the qualified Ascend SoC (ascend910b)"
  echo "  --make_clean            Clean build/ and build_out/ artifacts"
  echo "  -u, --ut                Run UT (Unit Tests) after building"
  echo "      --cov               Enable code coverage for UT"
  echo "  -s, --st                Run ST (System Tests) in Real mode (needs --install)"
  echo "      --mock              Run ST in Mock mode (CPU golden, no NPU, no install)"
  echo "  -e, --example           Run examples after building (needs --install)"
  echo "      --eager             Run aclnn example (default for -e)"
  echo "      --graph             Run graph mode (GE IR) example"
  echo "      --install           Explicitly install the built .run package"
  echo "  -a, --all               Run UT + ST (Real mode, needs --install)"
  echo ""
  echo "Examples:"
  echo "  bash build.sh --soc=ascend910b -j8"
  echo "  bash build.sh --soc=ascend910b -u"
  echo "  bash build.sh --soc=ascend910b --mock -s"
  echo "  bash build.sh --soc=ascend910b --install -s"
  echo "  bash build.sh --make_clean"
}

check_compute_unit() {
  local unit="$1"
  local support_unit
  for support_unit in "${SUPPORT_COMPUTE_UNITS[@]}"; do
    if [[ "$unit" == "$support_unit" ]]; then
      return 0
    fi
  done
  return 1
}

normalize_compute_unit() {
  local raw
  raw=$(echo "$1" | tr '[:upper:]' '[:lower:]')
  case "$raw" in
    ascend910b|ascend910b3)
      echo "ascend910b"
      ;;
    *)
      echo ""
      ;;
  esac
}

clean_build() {
  echo "[INFO] Cleaning build directory..."
  rm -rf "${BUILD_PATH}"
}

clean_build_out() {
  echo "[INFO] Cleaning build_out directory..."
  rm -rf "${BUILD_OUT_PATH}"
}

require_cann_env() {
  if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set."
    echo "[ERROR] Source the CANN environment script for CANN 9.2.0-beta.2 before building, e.g.:"
    echo "[ERROR]   source \${ASCEND_INSTALL_PATH}/set_env.sh"
    exit 1
  fi
  if [ ! -d "${ASCEND_HOME_PATH}" ]; then
    echo "[ERROR] ASCEND_HOME_PATH does not point to a directory: ${ASCEND_HOME_PATH}"
    exit 1
  fi
  if ! command -v cmake >/dev/null 2>&1; then
    echo "[ERROR] cmake was not found in PATH."
    exit 1
  fi
}

# Default values
THREAD_NUM=${CORE_NUMS}
COMPUTE_UNIT=""
COMPUTE_UNIT_GIVEN=FALSE
ENABLE_CLEAN=FALSE
ENABLE_COV=FALSE
RUN_UT=FALSE
RUN_ST=FALSE
RUN_ST_MOCK=FALSE
RUN_EXAMPLE=FALSE
DO_INSTALL=FALSE
EXAMPLE_MODE="eager"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --list-socs)
      echo "Qualified SoC versions for Argpartition:"
      for soc in "${SUPPORT_COMPUTE_UNITS[@]}"; do
        echo "  - ${soc}"
      done
      exit 0
      ;;
    -j*)
      THREAD_NUM="${1:2}"
      if [ -z "$THREAD_NUM" ]; then
        THREAD_NUM=${CORE_NUMS}
      fi
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
    --mock)
      RUN_ST_MOCK=TRUE
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
    --install)
      DO_INSTALL=TRUE
      shift
      ;;
    --make_clean)
      ENABLE_CLEAN=TRUE
      shift
      ;;
    --soc=*)
      COMPUTE_UNIT="${1#*=}"
      COMPUTE_UNIT_GIVEN=TRUE
      shift
      ;;
    --soc)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --soc requires a value"
        usage
        exit 1
      fi
      COMPUTE_UNIT="$2"
      COMPUTE_UNIT_GIVEN=TRUE
      shift 2
      ;;
    -*)
      echo "[ERROR] Invalid option: $1"
      usage
      exit 1
      ;;
    *)
      echo "[ERROR] Unexpected argument: $1"
      usage
      exit 1
      ;;
  esac
done

# Handle clean-only invocation
if [ "$ENABLE_CLEAN" = "TRUE" ] && [ "$COMPUTE_UNIT_GIVEN" = "FALSE" ] \
   && [ "$RUN_UT" = "FALSE" ] && [ "$RUN_ST" = "FALSE" ] \
   && [ "$RUN_EXAMPLE" = "FALSE" ]; then
  clean_build
  clean_build_out
  echo "[INFO] Clean completed."
  exit 0
fi

# Resolve default SoC
if [ -z "$COMPUTE_UNIT" ]; then
  COMPUTE_UNIT="${DEFAULT_COMPUTE_UNIT}"
  echo "[INFO] --soc not given, defaulting to ${COMPUTE_UNIT}"
fi

NORMALIZED_UNIT=$(normalize_compute_unit "$COMPUTE_UNIT")
if [ -z "$NORMALIZED_UNIT" ] || ! check_compute_unit "$NORMALIZED_UNIT"; then
  echo "[ERROR] Unsupported or unqualified SoC version: ${COMPUTE_UNIT}"
  echo "[INFO] This candidate is qualified only for Ascend910B3 (ascend910b)."
  echo "[INFO] Qualified SoC versions: ${SUPPORT_COMPUTE_UNITS[*]}"
  exit 1
fi
COMPUTE_UNIT="$NORMALIZED_UNIT"
echo "[INFO] Compile for SoC: ${COMPUTE_UNIT}"

# Validate thread count
case "${THREAD_NUM}" in
  ''|*[!0-9]*)
    echo "[ERROR] Invalid thread count: ${THREAD_NUM}"
    exit 1
    ;;
esac
if [ "${THREAD_NUM}" -lt 1 ]; then
  echo "[ERROR] Thread count must be >= 1"
  exit 1
fi

# Clean before build when explicitly requested together with other actions
if [ "$ENABLE_CLEAN" = "TRUE" ]; then
  clean_build
  clean_build_out
fi

require_cann_env

if [ ! -f "${BASE_PATH}/CMakeLists.txt" ]; then
  echo "[ERROR] CMakeLists.txt not found in ${BASE_PATH}"
  exit 1
fi

CMAKE_ARGS=("-DASCEND_COMPUTE_UNIT=${COMPUTE_UNIT}")

# Create build directory
mkdir -p "${BUILD_PATH}"
mkdir -p "${BUILD_OUT_PATH}"

# Clean CMakeCache so the selected SoC is always applied
if [ -f "${BUILD_PATH}/CMakeCache.txt" ]; then
  rm -f "${BUILD_PATH}/CMakeCache.txt"
fi

echo "----------------------------------------------------------------"
echo "[INFO] Configuring project..."
echo "[INFO] CMAKE_ARGS: ${CMAKE_ARGS[*]}"
cd "${BUILD_PATH}"
cmake "${CMAKE_ARGS[@]}" ..

echo "----------------------------------------------------------------"
echo "[INFO] Building project with ${THREAD_NUM} threads..."
cmake --build . --target all binary package -- -j "${THREAD_NUM}"

# Check kernel binary
KERNEL_O=$(find "${BUILD_PATH}/op_kernel/ascendc_kernels/binary/${COMPUTE_UNIT}" -name "*.o" 2>/dev/null | head -1)
if [ -z "${KERNEL_O}" ]; then
  echo "[ERROR] Kernel binary not found for ${COMPUTE_UNIT}"
  exit 1
fi

# Check package (name may vary with the configured package_name)
PKG_PATH=$(find "${BUILD_PATH}" -maxdepth 2 -type f -name "custom_opp_*.run" 2>/dev/null | head -1)
if [ -z "${PKG_PATH}" ] || [ ! -f "${PKG_PATH}" ] || [ ! -s "${PKG_PATH}" ]; then
  echo "[ERROR] Operator package not found or empty under ${BUILD_PATH}"
  exit 1
fi

echo "----------------------------------------------------------------"
echo "[INFO] Build completed successfully!"
echo "[INFO] Kernel binary: ${KERNEL_O}"
echo "[INFO] Package: ${PKG_PATH}"

install_package() {
  if [ "${DO_INSTALL}" != "TRUE" ]; then
    echo "[ERROR] Installing the operator package modifies a CANN installation and is never implicit."
    echo "[ERROR] Re-run with the explicit --install flag if you intend to install:"
    echo "[ERROR]   bash build.sh --soc=${COMPUTE_UNIT} --install${1:-}"
    exit 1
  fi
  echo "----------------------------------------------------------------"
  echo "[INFO] Explicit installation requested; installing operator package..."
  "${PKG_PATH}"
  echo "[INFO] Operator package installed."
}

# Run UT tests
if [ "$RUN_UT" = "TRUE" ]; then
  echo "----------------------------------------------------------------"
  if [ ! -f "${BASE_PATH}/tests/ut/run.sh" ]; then
    echo "[ERROR] UT runner not found: ${BASE_PATH}/tests/ut/run.sh"
    exit 1
  fi
  echo "[INFO] Running UT tests..."
  cd "${BASE_PATH}/tests/ut"
  if [ "$ENABLE_COV" = "TRUE" ]; then
    ENABLE_COV=1 ./run.sh
  else
    ENABLE_COV=0 ./run.sh
  fi
  cd - > /dev/null
  echo "[INFO] UT tests completed successfully!"
fi

# Run ST tests
if [ "$RUN_ST" = "TRUE" ]; then
  echo "----------------------------------------------------------------"
  if [ ! -f "${BASE_PATH}/tests/st/run.sh" ]; then
    echo "[ERROR] ST runner not found: ${BASE_PATH}/tests/st/run.sh"
    exit 1
  fi
  if [ "$RUN_ST_MOCK" = "TRUE" ]; then
    echo "[INFO] Running ST tests (Mock mode, CPU golden, no NPU)..."
    cd "${BASE_PATH}/tests/st"
    ./run.sh --mock
    cd - > /dev/null
  else
    install_package " -s"
    echo "[INFO] Running ST tests (Real mode)..."
    cd "${BASE_PATH}/tests/st"
    ./run.sh
    cd - > /dev/null
  fi
  echo "[INFO] ST tests completed successfully!"
fi

# Run examples
if [ "$RUN_EXAMPLE" = "TRUE" ]; then
  echo "----------------------------------------------------------------"
  if [ ! -f "${BASE_PATH}/examples/run.sh" ]; then
    echo "[ERROR] Example runner not found: ${BASE_PATH}/examples/run.sh"
    exit 1
  fi
  install_package " -e"
  echo "[INFO] Running examples (${EXAMPLE_MODE} mode)..."
  cd "${BASE_PATH}/examples"
  ./run.sh --"${EXAMPLE_MODE}"
  cd - > /dev/null
  echo "[INFO] Example completed successfully!"
fi

if [ "$RUN_UT" = "TRUE" ] || [ "$RUN_ST" = "TRUE" ] || [ "$RUN_EXAMPLE" = "TRUE" ]; then
  echo "----------------------------------------------------------------"
  echo "[INFO] All requested build/test steps completed successfully!"
fi