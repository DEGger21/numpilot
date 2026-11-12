#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

set -e

# The generated candidate is intentionally specialized to the observed target.
SUPPORT_COMPUTE_UNITS=("ascend910b")
DEFAULT_COMPUTE_UNIT="ascend910b"

export BASE_PATH=$(
  cd "$(dirname "$0")"
  pwd
)
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"

CORE_NUMS=$(nproc 2>/dev/null || echo 8)
if [ "${CORE_NUMS}" -gt 8 ]; then
  CORE_NUMS=8
fi

usage() {
  echo "Build script for the std operator candidate"
  echo "Usage: bash build.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -h, --help              Print this help message"
  echo "      --list-socs         List supported SoC versions"
  echo "  -j[n]                   Compile thread nums, default is ${CORE_NUMS}, eg: -j8"
  echo "      --jobs=n            Compile thread nums, eg: --jobs=8"
  echo "      --soc=soc_version   Compile for specified Ascend SoC (ascend910b)"
  echo "      --make_clean        Clean build artifacts"
  echo "  -u, --ut                Run UT (Unit Tests)"
  echo "      --cov               Enable code coverage for UT"
  echo "  -s, --st                Run ST (System Tests, requires installed operator package)"
  echo "  -e, --example           Run examples (requires NPU and installed operator package)"
  echo "      --eager             Run aclnn example (default for -e)"
  echo "      --graph             Run graph mode (GE IR) example"
  echo "  -a, --all               Run all tests (UT + ST)"
  echo ""
  echo "Examples:"
  echo "  bash build.sh --soc=ascend910b -j8"
  echo "  bash build.sh --make_clean"
  echo "  bash build.sh -u"
  echo "  bash build.sh -s"
  echo "  bash build.sh -e"
  echo "  bash build.sh -e --graph"
  echo "  bash build.sh -a"
}

check_compute_unit() {
  local unit="$1"
  for support_unit in "${SUPPORT_COMPUTE_UNITS[@]}"; do
    if [[ "${unit}" == "${support_unit}" ]]; then
      return 0
    fi
  done
  return 1
}

normalize_compute_unit() {
  local unit="$1"
  unit=$(echo "${unit}" | tr '[:upper:]' '[:lower:]')
  case "${unit}" in
    ascend910b3)
      echo "ascend910b"
      ;;
    *)
      echo "${unit}"
      ;;
  esac
}

clean_build() {
  if [ -d "${BUILD_PATH}" ]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_PATH}"
  fi
}

clean_build_out() {
  if [ -d "${BUILD_OUT_PATH}" ]; then
    echo "Cleaning build_out directory..."
    rm -rf "${BUILD_OUT_PATH}"
  fi
}

install_operator_package() {
  if [ ! -f "${PKG_PATH}" ]; then
    echo "[ERROR] Package not found: ${PKG_PATH}"
    exit 1
  fi
  echo "[INFO] Installing operator package: ${PKG_PATH}"
  "${PKG_PATH}"
}

# Default values
THREAD_NUM=${CORE_NUMS}
COMPUTE_UNIT=""
ENABLE_CLEAN=FALSE
ENABLE_COV=FALSE
RUN_UT=FALSE
RUN_ST=FALSE
RUN_EXAMPLE=FALSE
EXAMPLE_MODE="eager"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --list-socs)
      echo "Supported SoC versions:"
      for soc in "${SUPPORT_COMPUTE_UNITS[@]}"; do
        echo "  - ${soc}"
      done
      exit 0
      ;;
    -j*)
      THREAD_NUM="${1:2}"
      if [ -z "${THREAD_NUM}" ]; then
        THREAD_NUM=${CORE_NUMS}
      fi
      shift
      ;;
    --jobs=*)
      THREAD_NUM="${1#*=}"
      shift
      ;;
    -u|--ut)
      RUN_UT=true
      shift
      ;;
    --cov)
      ENABLE_COV=TRUE
      shift
      ;;
    -s|--st)
      RUN_ST=true
      shift
      ;;
    -a|--all)
      RUN_UT=true
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
    --make_clean)
      ENABLE_CLEAN=TRUE
      shift
      ;;
    --soc=*)
      COMPUTE_UNIT="${1#*=}"
      shift
      ;;
    --soc)
      if [ $# -lt 2 ]; then
        echo "[ERROR] --soc requires a value"
        usage
        exit 1
      fi
      COMPUTE_UNIT="$2"
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

# Handle clean
if [ "${ENABLE_CLEAN}" = "TRUE" ]; then
  clean_build
  clean_build_out
  exit 0
fi

# The target is fixed by the GapSpec. An omitted --soc uses the target SoC.
if [ -z "${COMPUTE_UNIT}" ]; then
  COMPUTE_UNIT="${DEFAULT_COMPUTE_UNIT}"
  echo "[INFO] --soc not provided; defaulting to target SoC: ${COMPUTE_UNIT}"
fi

COMPUTE_UNIT=$(normalize_compute_unit "${COMPUTE_UNIT}")
if ! check_compute_unit "${COMPUTE_UNIT}"; then
  echo "[ERROR] Unsupported SoC version: ${COMPUTE_UNIT}"
  echo "[INFO] Supported SoC versions: ${SUPPORT_COMPUTE_UNITS[*]}"
  exit 1
fi
echo "[INFO] Compile for SoC: ${COMPUTE_UNIT}"

if ! [[ "${THREAD_NUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] Invalid thread number: ${THREAD_NUM}"
  exit 1
fi

# Prepare cmake arguments
CMAKE_ARGS=(
  "-DASCEND_COMPUTE_UNIT=${COMPUTE_UNIT}"
)

# Create build directory
if [ ! -d "${BUILD_PATH}" ]; then
  mkdir -p "${BUILD_PATH}"
fi

# Clean CMakeCache so repeated invocations use the requested SoC.
if [ -f "${BUILD_PATH}/CMakeCache.txt" ]; then
  rm -f "${BUILD_PATH}/CMakeCache.txt"
fi

# Configure
echo "----------------------------------------------------------------"
echo "[INFO] Configuring project..."
echo "[INFO] CMAKE_ARGS: ${CMAKE_ARGS[*]}"
cd "${BUILD_PATH}" && cmake "${CMAKE_ARGS[@]}" ..

# Build (host + kernel binary + package)
echo "----------------------------------------------------------------"
echo "[INFO] Building project with ${THREAD_NUM} threads..."
cmake --build . --target all binary package -- -j "${THREAD_NUM}"

# Check kernel binary
KERNEL_DIR="${BUILD_PATH}/op_kernel/ascendc_kernels/binary/${COMPUTE_UNIT}"
KERNEL_O="$(find "${KERNEL_DIR}" -name '*.o' -type f 2>/dev/null | head -n 1 || true)"
if [ -z "${KERNEL_O}" ]; then
  echo "[ERROR] Kernel binary not found in ${KERNEL_DIR}"
  exit 1
fi

# Check package
PKG_PATH="${BUILD_PATH}/custom_opp_ubuntu_aarch64.run"
if [ ! -f "${PKG_PATH}" ] || [ ! -s "${PKG_PATH}" ]; then
  PKG_PATH="$(find "${BUILD_PATH}" -maxdepth 1 -type f -name 'custom_opp_*.run' 2>/dev/null | head -n 1 || true)"
fi
if [ -z "${PKG_PATH}" ] || [ ! -f "${PKG_PATH}" ] || [ ! -s "${PKG_PATH}" ]; then
  echo "[ERROR] Package not found or empty"
  exit 1
fi

echo "----------------------------------------------------------------"
echo "[INFO] Build completed successfully!"
echo "[INFO] Kernel binary: ${KERNEL_O}"
echo "[INFO] Package: ${PKG_PATH}"

# Run UT tests
if [ "${RUN_UT}" = true ]; then
  echo "----------------------------------------------------------------"
  echo "[INFO] Running UT tests..."
  if [ ! -f "${BASE_PATH}/tests/ut/run.sh" ]; then
    echo "[ERROR] UT runner not found: ${BASE_PATH}/tests/ut/run.sh"
    exit 1
  fi
  cd "${BASE_PATH}/tests/ut"
  ENABLE_COV=${ENABLE_COV} ./run.sh
  UT_RESULT=$?
  cd - > /dev/null
  if [ ${UT_RESULT} -ne 0 ]; then
    echo "[ERROR] UT tests failed"
    exit 1
  fi
  echo "[INFO] UT tests completed successfully!"
fi

# Run ST tests
if [ "${RUN_ST}" = true ]; then
  echo "----------------------------------------------------------------"
  echo "[INFO] Installing operator package for ST tests..."
  install_operator_package

  echo "----------------------------------------------------------------"
  echo "[INFO] Running ST tests (Real mode)..."
  if [ ! -f "${BASE_PATH}/tests/st/run.sh" ]; then
    echo "[ERROR] ST runner not found: ${BASE_PATH}/tests/st/run.sh"
    exit 1
  fi
  cd "${BASE_PATH}/tests/st"
  ./run.sh
  ST_RESULT=$?
  cd - > /dev/null
  if [ ${ST_RESULT} -ne 0 ]; then
    echo "[ERROR] ST tests failed"
    exit 1
  fi
  echo "[INFO] ST tests completed successfully!"
fi

# Run examples
if [ "${RUN_EXAMPLE}" = true ]; then
  echo "----------------------------------------------------------------"
  echo "[INFO] Installing operator package for examples..."
  install_operator_package

  echo "----------------------------------------------------------------"
  echo "[INFO] Running examples (${EXAMPLE_MODE} mode)..."
  if [ ! -f "${BASE_PATH}/examples/run.sh" ]; then
    echo "[ERROR] Example runner not found: ${BASE_PATH}/examples/run.sh"
    exit 1
  fi
  cd "${BASE_PATH}/examples"
  ./run.sh "--${EXAMPLE_MODE}"
  EXAMPLE_RESULT=$?
  cd - > /dev/null
  if [ ${EXAMPLE_RESULT} -ne 0 ]; then
    echo "[ERROR] Example execution failed"
    exit 1
  fi
  echo "[INFO] Example completed successfully!"
fi

if [ "${RUN_UT}" = true ] || [ "${RUN_ST}" = true ] || [ "${RUN_EXAMPLE}" = true ]; then
  echo "----------------------------------------------------------------"
  echo "[INFO] All requested build/test stages completed successfully!"
fi