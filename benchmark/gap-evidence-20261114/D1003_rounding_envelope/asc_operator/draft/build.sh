#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# Build script for the Nextafter custom Ascend C operator.
#
# Target platform: Ascend910B3
# Build SoC name: ascend910b
# Target CANN:    9.2.0-beta.2
#
# This script builds the isolated operator package in this directory tree.
# It does not modify ~/.bashrc, does not source a CANN environment implicitly,
# and only installs the generated custom operator package when the caller
# explicitly requests ST tests or example execution.
# ============================================================================

set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_PATH="${BASE_PATH}/build"
BUILD_OUT_PATH="${BASE_PATH}/build_out"
PACKAGE_PATH="${BUILD_PATH}/custom_opp_ubuntu_aarch64.run"

SUPPORT_COMPUTE_UNITS=("ascend910b")
DEFAULT_COMPUTE_UNIT="ascend910b"

CORE_NUMS="$(nproc 2>/dev/null || true)"
case "${CORE_NUMS}" in
  ''|*[!0-9]*)
    CORE_NUMS=4
    ;;
esac
if [ "${CORE_NUMS}" -gt 8 ]; then
  CORE_NUMS=8
fi
if [ "${CORE_NUMS}" -lt 1 ]; then
  CORE_NUMS=1
fi

THREAD_NUM="${CORE_NUMS}"
COMPUTE_UNIT=""
ENABLE_CLEAN="FALSE"
ENABLE_COV="FALSE"
RUN_UT="FALSE"
RUN_ST="FALSE"
RUN_EXAMPLE="FALSE"
EXAMPLE_MODE="eager"

usage() {
  echo "Build script for the Nextafter custom Ascend C operator"
  echo ""
  echo "Usage: bash build.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -h, --help              Print this help message"
  echo "  --list-socs             List supported SoC versions"
  echo "  -j[N], -j N             Compile thread count (default: ${CORE_NUMS})"
  echo "  --soc=<soc>, --soc <soc>"
  echo "                          Target Ascend SoC (supported: ${SUPPORT_COMPUTE_UNITS[*]})"
  echo "  --make_clean            Remove build/ and build_out/ artifacts"
  echo "  -u, --ut                Build, then run UT tests"
  echo "  --cov                   Enable UT coverage when -u/--ut is used"
  echo "  -s, --st                Build, install the custom operator package, and run ST tests"
  echo "  -e, --example           Build, install the custom operator package, and run examples"
  echo "  --eager                 Use eager aclnn example mode with -e (default)"
  echo "  --graph                 Use graph mode example with -e"
  echo "  -a, --all               Run UT and ST tests after building"
  echo ""
  echo "Examples:"
  echo "  bash build.sh --soc=ascend910b -j8"
  echo "  bash build.sh --soc=ascend910b -j8 -u"
  echo "  bash build.sh --soc=ascend910b -j8 -s"
  echo "  bash build.sh --soc=ascend910b -j8 -e --graph"
  echo "  bash build.sh --make_clean"
  echo ""
  echo "Notes:"
  echo "  - The caller must source the target CANN ${ASCEND_HOME_PATH:-<unset>} environment first."
  echo "  - -s/--st and -e/--example explicitly install build/custom_opp_ubuntu_aarch64.run."
  echo "  - This script does not perform implicit CPU fallback or implicit package activation."
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

check_compute_unit() {
  local unit="$1"
  local support
  for support in "${SUPPORT_COMPUTE_UNITS[@]}"; do
    if [ "${unit}" = "${support}" ]; then
      return 0
    fi
  done
  return 1
}

clean_artifacts() {
  echo "[INFO] Cleaning build artifacts under ${BASE_PATH}"
  rm -rf "${BUILD_PATH}" "${BUILD_OUT_PATH}"
}

install_package() {
  if [ ! -f "${PACKAGE_PATH}" ] || [ ! -s "${PACKAGE_PATH}" ]; then
    die "Custom operator package not found or empty: ${PACKAGE_PATH}"
  fi

  chmod +x "${PACKAGE_PATH}"
  echo "[INFO] Installing custom operator package because ST/example execution was explicitly requested."
  echo "[INFO] Package: ${PACKAGE_PATH}"
  if ! "${PACKAGE_PATH}"; then
    die "Custom operator package installation failed: ${PACKAGE_PATH}"
  fi
}

while [ "$#" -gt 0 ]; do
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
    --make_clean)
      ENABLE_CLEAN="TRUE"
      shift
      ;;
    -u|--ut)
      RUN_UT="TRUE"
      shift
      ;;
    --cov)
      ENABLE_COV="TRUE"
      shift
      ;;
    -s|--st)
      RUN_ST="TRUE"
      shift
      ;;
    -e|--example)
      RUN_EXAMPLE="TRUE"
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
      RUN_UT="TRUE"
      RUN_ST="TRUE"
      shift
      ;;
    -j)
      if [ "$#" -ge 2 ] && [[ "$2" =~ ^[0-9]+$ ]]; then
        THREAD_NUM="$2"
        shift 2
      else
        THREAD_NUM="${CORE_NUMS}"
        shift
      fi
      ;;
    -j*)
      THREAD_NUM="${1#-j}"
      if [ -z "${THREAD_NUM}" ]; then
        THREAD_NUM="${CORE_NUMS}"
      fi
      shift
      ;;
    --soc=*)
      COMPUTE_UNIT="${1#*=}"
      shift
      ;;
    --soc)
      if [ "$#" -lt 2 ]; then
        die "--soc requires a value"
      fi
      COMPUTE_UNIT="$2"
      shift 2
      ;;
    -*)
      echo "[ERROR] Invalid option: $1" >&2
      usage
      exit 1
      ;;
    *)
      echo "[ERROR] Unexpected argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [ "${ENABLE_CLEAN}" = "TRUE" ]; then
  clean_artifacts
  exit 0
fi

if [ -z "${COMPUTE_UNIT}" ]; then
  COMPUTE_UNIT="${DEFAULT_COMPUTE_UNIT}"
fi

COMPUTE_UNIT="$(printf '%s' "${COMPUTE_UNIT}" | tr '[:upper:]' '[:lower:]')"
if ! check_compute_unit "${COMPUTE_UNIT}"; then
  echo "[ERROR] Unsupported SoC version: ${COMPUTE_UNIT}" >&2
  echo "[INFO] Supported SoC versions: ${SUPPORT_COMPUTE_UNITS[*]}" >&2
  exit 1
fi

if ! [[ "${THREAD_NUM}" =~ ^[0-9]+$ ]] || [ "${THREAD_NUM}" -lt 1 ]; then
  die "Invalid thread count: ${THREAD_NUM}"
fi

if ! command -v cmake >/dev/null 2>&1; then
  die "cmake was not found in PATH"
fi

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  die "ASCEND_HOME_PATH is not set. Source the target CANN 9.2.0-beta.2 environment first."
fi

if [ ! -d "${ASCEND_HOME_PATH}" ]; then
  die "ASCEND_HOME_PATH does not point to an existing directory: ${ASCEND_HOME_PATH}"
fi

if [ ! -f "${BASE_PATH}/CMakeLists.txt" ]; then
  die "CMakeLists.txt was not found in ${BASE_PATH}"
fi

echo "[INFO] Operator: Nextafter"
echo "[INFO] Target SoC: Ascend910B3 (build SoC: ${COMPUTE_UNIT})"
echo "[INFO] CANN home: ${ASCEND_HOME_PATH}"
echo "[INFO] Build directory: ${BUILD_PATH}"
echo "[INFO] Thread count: ${THREAD_NUM}"

mkdir -p "${BUILD_PATH}"
rm -f "${BUILD_PATH}/CMakeCache.txt"

echo "----------------------------------------------------------------"
echo "[INFO] Configuring project..."
(
  cd "${BUILD_PATH}"
  cmake "-DASCEND_COMPUTE_UNIT=${COMPUTE_UNIT}" "${BASE_PATH}"
)

echo "----------------------------------------------------------------"
echo "[INFO] Building host, kernel binary, and package..."
(
  cd "${BUILD_PATH}"
  cmake --build . --target all binary package -- -j "${THREAD_NUM}"
)

KERNEL_DIR="${BUILD_PATH}/op_kernel/ascendc_kernels/binary/${COMPUTE_UNIT}"
KERNEL_O="$(find "${KERNEL_DIR}" -type f -name '*.o' -print -quit 2>/dev/null || true)"
if [ -z "${KERNEL_O}" ]; then
  die "Kernel binary was not generated under ${KERNEL_DIR}"
fi

if [ ! -f "${PACKAGE_PATH}" ] || [ ! -s "${PACKAGE_PATH}" ]; then
  PACKAGE_PATH="$(find "${BUILD_PATH}" -maxdepth 1 -type f -name 'custom_opp_*.run' -print -quit 2>/dev/null || true)"
fi
if [ -z "${PACKAGE_PATH}" ] || [ ! -f "${PACKAGE_PATH}" ] || [ ! -s "${PACKAGE_PATH}" ]; then
  die "Custom operator package was not generated under ${BUILD_PATH}"
fi

echo "[INFO] Kernel binary: ${KERNEL_O}"
echo "[INFO] Package: ${PACKAGE_PATH}"

if [ "${RUN_UT}" = "TRUE" ]; then
  if [ ! -f "${BASE_PATH}/tests/ut/run.sh" ]; then
    die "UT runner was not found: ${BASE_PATH}/tests/ut/run.sh"
  fi
  echo "----------------------------------------------------------------"
  echo "[INFO] Running UT tests..."
  (
    cd "${BASE_PATH}/tests/ut"
    ENABLE_COV="${ENABLE_COV}" bash ./run.sh
  )
fi

if [ "${RUN_ST}" = "TRUE" ]; then
  if [ ! -f "${BASE_PATH}/tests/st/run.sh" ]; then
    die "ST runner was not found: ${BASE_PATH}/tests/st/run.sh"
  fi
fi

if [ "${RUN_EXAMPLE}" = "TRUE" ]; then
  if [ ! -f "${BASE_PATH}/examples/run.sh" ]; then
    die "Example runner was not found: ${BASE_PATH}/examples/run.sh"
  fi
fi

if [ "${RUN_ST}" = "TRUE" ] || [ "${RUN_EXAMPLE}" = "TRUE" ]; then
  install_package
fi

if [ "${RUN_ST}" = "TRUE" ]; then
  echo "----------------------------------------------------------------"
  echo "[INFO] Running ST tests (Real mode)..."
  (
    cd "${BASE_PATH}/tests/st"
    bash ./run.sh
  )
fi

if [ "${RUN_EXAMPLE}" = "TRUE" ]; then
  echo "----------------------------------------------------------------"
  echo "[INFO] Running examples (${EXAMPLE_MODE} mode)..."
  (
    cd "${BASE_PATH}/examples"
    bash ./run.sh "--${EXAMPLE_MODE}"
  )
fi

echo "----------------------------------------------------------------"
echo "[INFO] Requested build/test steps completed."