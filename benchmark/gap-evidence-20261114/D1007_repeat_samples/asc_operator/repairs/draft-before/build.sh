#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SUPPORT_COMPUTE_UNITS=("ascend910b" "ascend910_93" "ascend950")

export BASE_PATH="$(
  cd "$(dirname "${BASH_SOURCE[0]}")"
  pwd
)"
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"

CORE_NUMS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
if ! [[ "${CORE_NUMS}" =~ ^[0-9]+$ ]] || [[ "${CORE_NUMS}" -le 0 ]]; then
  CORE_NUMS=4
fi
if [[ "${CORE_NUMS}" -gt 8 ]]; then
  CORE_NUMS=8
fi

usage() {
  echo "Build script for the repeat custom operator"
  echo "Usage: bash build.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -h, --help              Print this help message"
  echo "  --list-socs             List supported SoC versions"
  echo "  -j[n]                   Compile thread nums, default is ${CORE_NUMS}; examples: -j8 or -j 8"
  echo "  --soc=soc_version       Compile for specified Ascend SoC (ascend910b, ascend910_93, ascend950)"
  echo "  --make_clean            Clean build artifacts"
  echo "  -u, --ut                Run UT (Unit Tests)"
  echo "      --cov               Enable code coverage for UT"
  echo "  -s, --st                Run ST (System Tests); installs the generated package first"
  echo "  -e, --example           Run examples; installs the generated package first"
  echo "      --eager             Run ACLNN eager example (default for -e)"
  echo "      --graph             Run graph mode (GE IR) example"
  echo "  -a, --all               Run all tests (UT + ST)"
  echo ""
  echo "Examples:"
  echo "  bash build.sh --soc=ascend910b -j8"
  echo "  bash build.sh --soc=ascend910b -j8 -u"
  echo "  bash build.sh --soc=ascend910b -j8 -s"
  echo "  bash build.sh --soc=ascend910b -j8 -e --eager"
  echo "  bash build.sh --make_clean"
}

is_supported_soc() {
  local requested="$1"
  local supported
  for supported in "${SUPPORT_COMPUTE_UNITS[@]}"; do
    if [[ "${requested}" == "${supported}" ]]; then
      return 0
    fi
  done
  return 1
}

clean_artifacts() {
  echo "[INFO] Cleaning build directories..."
  rm -rf "${BUILD_PATH}" "${BUILD_OUT_PATH}"
}

require_file() {
  local path="$1"
  local description="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing ${description}: ${path}"
    exit 1
  fi
}

install_package() {
  require_file "${PKG_PATH}" "operator installation package"
  echo "----------------------------------------------------------------"
  echo "[INFO] Installing operator package: ${PKG_PATH}"
  (
    cd "${BUILD_PATH}"
    bash "${PKG_PATH}"
  )
  echo "[INFO] Operator package installed successfully."
}

run_ut() {
  local ut_dir="${BASE_PATH}/tests/ut"
  require_file "${ut_dir}/run.sh" "UT runner"
  echo "----------------------------------------------------------------"
  echo "[INFO] Running UT tests..."
  (
    cd "${ut_dir}"
    ENABLE_COV="${ENABLE_COV}" bash ./run.sh
  )
  echo "[INFO] UT tests completed successfully."
}

run_st() {
  local st_dir="${BASE_PATH}/tests/st"
  require_file "${st_dir}/run.sh" "ST runner"
  install_package
  echo "----------------------------------------------------------------"
  echo "[INFO] Running ST tests..."
  (
    cd "${st_dir}"
    bash ./run.sh
  )
  echo "[INFO] ST tests completed successfully."
}

run_example() {
  local example_dir="${BASE_PATH}/examples"
  require_file "${example_dir}/run.sh" "example runner"
  install_package
  echo "----------------------------------------------------------------"
  echo "[INFO] Running examples in ${EXAMPLE_MODE} mode..."
  (
    cd "${example_dir}"
    bash ./run.sh "--${EXAMPLE_MODE}"
  )
  echo "[INFO] Example execution completed successfully."
}

THREAD_NUM="${CORE_NUMS}"
COMPUTE_UNIT=""
ENABLE_CLEAN=FALSE
ENABLE_COV=FALSE
RUN_UT=FALSE
RUN_ST=FALSE
RUN_EXAMPLE=FALSE
EXAMPLE_MODE="eager"

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
      j_value="${1#-j}"
      if [[ -n "${j_value}" ]]; then
        THREAD_NUM="${j_value}"
        shift
      elif [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]]; then
        THREAD_NUM="$2"
        shift 2
      else
        THREAD_NUM="${CORE_NUMS}"
        shift
      fi
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
      if [[ $# -lt 2 ]]; then
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

if [[ "${ENABLE_CLEAN}" == "TRUE" ]]; then
  clean_artifacts
  exit 0
fi

if ! [[ "${THREAD_NUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] Invalid thread count: ${THREAD_NUM}"
  exit 1
fi

if [[ -z "${COMPUTE_UNIT}" ]]; then
  echo "[ERROR] --soc parameter is required"
  usage
  exit 1
fi

COMPUTE_UNIT="$(echo "${COMPUTE_UNIT}" | tr '[:upper:]' '[:lower:]')"
if ! is_supported_soc "${COMPUTE_UNIT}"; then
  echo "[ERROR] Invalid SoC version: ${COMPUTE_UNIT}"
  echo "[INFO] Supported SoC versions: ${SUPPORT_COMPUTE_UNITS[*]}"
  exit 1
fi

echo "[INFO] Base path: ${BASE_PATH}"
echo "[INFO] Build path: ${BUILD_PATH}"
echo "[INFO] Compile for SoC: ${COMPUTE_UNIT}"
echo "[INFO] Compile thread count: ${THREAD_NUM}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "[WARN] ASCEND_HOME_PATH is not set. CMake may fail to locate the CANN SDK."
else
  echo "[INFO] ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
fi

mkdir -p "${BUILD_PATH}" "${BUILD_OUT_PATH}"
rm -f "${BUILD_PATH}/CMakeCache.txt"

CMAKE_ARGS=(
  "-DASCEND_COMPUTE_UNIT=${COMPUTE_UNIT}"
)

echo "----------------------------------------------------------------"
echo "[INFO] Configuring project..."
cmake -S "${BASE_PATH}" -B "${BUILD_PATH}" "${CMAKE_ARGS[@]}"

echo "----------------------------------------------------------------"
echo "[INFO] Building host, kernel binary, and package..."
cmake --build "${BUILD_PATH}" --target all binary package -- -j"${THREAD_NUM}"

KERNEL_DIR="${BUILD_PATH}/op_kernel/ascendc_kernels/binary/${COMPUTE_UNIT}"
KERNEL_O="$(find "${KERNEL_DIR}" -type f -name "*.o" -print -quit 2>/dev/null || true)"
if [[ -z "${KERNEL_O}" ]]; then
  echo "[ERROR] Kernel binary not found under: ${KERNEL_DIR}"
  exit 1
fi

PKG_PATH="${BUILD_PATH}/custom_opp_ubuntu_aarch64.run"
if [[ ! -s "${PKG_PATH}" ]]; then
  FOUND_PKG="$(find "${BUILD_PATH}" -maxdepth 3 -type f -name "*.run" -print -quit 2>/dev/null || true)"
  if [[ -n "${FOUND_PKG}" ]]; then
    PKG_PATH="${FOUND_PKG}"
  fi
fi
if [[ -z "${PKG_PATH}" || ! -s "${PKG_PATH}" ]]; then
  echo "[ERROR] Operator package not found or empty under: ${BUILD_PATH}"
  exit 1
fi

echo "----------------------------------------------------------------"
echo "[INFO] Build completed successfully."
echo "[INFO] Kernel binary: ${KERNEL_O}"
echo "[INFO] Package: ${PKG_PATH}"

if [[ "${RUN_UT}" == "true" ]]; then
  run_ut
fi

if [[ "${RUN_ST}" == "true" ]]; then
  run_st
fi

if [[ "${RUN_EXAMPLE}" == "true" ]]; then
  run_example
fi

if [[ "${RUN_UT}" == "true" || "${RUN_ST}" == "true" || "${RUN_EXAMPLE}" == "true" ]]; then
  echo "----------------------------------------------------------------"
  echo "[INFO] Requested build and test steps completed successfully."
fi