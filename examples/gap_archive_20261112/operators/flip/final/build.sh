#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Unverified Flip candidate build driver.
#
# Project mapping established for later stages:
#   NumPy semantic source : numpy.flip
#   Public AsNumpy API    : asnumpy.flip
#   C++ OpType            : Flip
#   File/API stem         : flip
#   Target SoC            : Ascend910B3 (CMake compute unit: ascend910b)
#   Target CANN           : 9.2.0-beta.2
#
# This script builds an isolated project-local artifact tree. It does not install
# or activate an operator package unless --install-package is explicitly passed.
# A successful build is not a qualification result.

set -euo pipefail

BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
BUILD_PATH="${BASE_PATH}/build"
BUILD_OUT_PATH="${BASE_PATH}/build_out"

TARGET_SOC_DISPLAY="Ascend910B3"
TARGET_CANN_VERSION="9.2.0-beta.2"
DEFAULT_COMPUTE_UNIT="ascend910b"
SUPPORTED_COMPUTE_UNIT="ascend910b"

CORE_NUMS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '8\n')"
case "${CORE_NUMS}" in
  ''|*[!0-9]*)
    CORE_NUMS=8
    ;;
esac
if [ "${CORE_NUMS}" -gt 8 ]; then
  CORE_NUMS=8
fi
if [ "${CORE_NUMS}" -lt 1 ]; then
  CORE_NUMS=1
fi

THREAD_NUM="${CORE_NUMS}"
COMPUTE_UNIT_RAW="${DEFAULT_COMPUTE_UNIT}"
COMPUTE_UNIT="${DEFAULT_COMPUTE_UNIT}"

RUN_UT=false
RUN_ST=false
RUN_EXAMPLE=false
ENABLE_COV=false
INSTALL_PACKAGE=false
PACKAGE_INSTALLED=false
DO_CLEAN=false
EXAMPLE_MODE="eager"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

usage() {
  cat <<EOF
Build script for the Flip candidate operator

Target:
  SoC       : ${TARGET_SOC_DISPLAY}
  CMake unit: ${SUPPORTED_COMPUTE_UNIT}
  CANN      : ${TARGET_CANN_VERSION}

Usage:
  bash build.sh [OPTIONS]

Options:
  -h, --help                 Print this help message
  --list-socs                List supported SoC values
  -j[N]                      Compile threads, default ${CORE_NUMS}, maximum 8
  --soc=<version>            Target SoC: Ascend910B3 or ascend910b
  --make_clean               Remove project-local build/ and build_out/ trees
  -u, --ut                   Run UT after a successful build
  --cov                      Request coverage mode for UT
  -s, --st                   Run ST after a successful build
  -e, --example              Run the operator example after a successful build
  --eager                    Select eager example mode (default for -e)
  --graph                    Select graph example mode
  --install-package          Explicitly install the built custom operator package
  -a, --all                  Run UT and ST after a successful build

Examples:
  bash build.sh --soc=Ascend910B3 -j8
  bash build.sh --soc=ascend910b -j8
  bash build.sh --make_clean
  bash build.sh --soc=ascend910b -j8 -u
  bash build.sh --soc=ascend910b -j8 -s --install-package
  bash build.sh --soc=ascend910b -j8 -e --eager --install-package

Environment:
  ASCEND_HOME_PATH must point to the active CANN installation. This script
  does not source shell startup files and does not modify ~/.bashrc.
EOF
}

clean_all() {
  echo "[INFO] Cleaning project-local build directories..."
  rm -rf -- "${BUILD_PATH}"
  rm -rf -- "${BUILD_OUT_PATH}"
}

normalize_compute_unit() {
  local raw lower
  raw="$1"
  lower="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')"
  case "${lower}" in
    ascend910b|ascend910b3)
      printf '%s\n' "${SUPPORTED_COMPUTE_UNIT}"
      ;;
    *)
      return 1
      ;;
  esac
}

check_cann_environment() {
  if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    die "ASCEND_HOME_PATH is not set. Source the CANN 9.2.0-beta.2 environment before building."
  fi
  if [ ! -d "${ASCEND_HOME_PATH}" ]; then
    die "ASCEND_HOME_PATH does not point to a directory: ${ASCEND_HOME_PATH}"
  fi

  local version_line=""
  if [ -f "${ASCEND_HOME_PATH}/version.cfg" ]; then
    version_line="$(grep -m1 -E '^(Version|version)=' "${ASCEND_HOME_PATH}/version.cfg" 2>/dev/null || true)"
  fi

  echo "[INFO] CANN home: ${ASCEND_HOME_PATH}"
  if [ -n "${version_line}" ]; then
    echo "[INFO] CANN declaration: ${version_line}"
  else
    echo "[INFO] CANN target: ${TARGET_CANN_VERSION}"
  fi
}

check_source_layout() {
  [ -f "${BASE_PATH}/CMakeLists.txt" ] || die "Missing root CMakeLists.txt"
  [ -f "${BASE_PATH}/op_host/CMakeLists.txt" ] || die "Missing op_host/CMakeLists.txt"
  [ -f "${BASE_PATH}/op_kernel/CMakeLists.txt" ] || die "Missing op_kernel/CMakeLists.txt"

  command -v cmake >/dev/null 2>&1 || die "cmake is required but was not found in PATH"
  command -v find >/dev/null 2>&1 || die "find is required but was not found in PATH"
}

find_kernel_binary() {
  find "${BUILD_PATH}" \
    -type f \
    -path "*/ascendc_kernels/binary/${COMPUTE_UNIT}/*.o" \
    -print -quit 2>/dev/null || true
}

find_package_artifact() {
  find "${BUILD_PATH}" "${BUILD_OUT_PATH}" \
    -maxdepth 4 \
    -type f \
    \( -name 'custom_opp_*.run' -o -name 'flip_custom*.run' -o -name 'flip_*.run' \) \
    -print -quit 2>/dev/null || true
}

maybe_install_package() {
  if [ "${PACKAGE_INSTALLED}" = true ]; then
    return 0
  fi

  if [ "${INSTALL_PACKAGE}" != true ]; then
    echo "[INFO] Package installation is opt-in. Re-run with --install-package to install:" >&2
    echo "       ${PKG_PATH}" >&2
    return 0
  fi

  echo "[INFO] Explicitly installing the built custom operator package..."
  if [ ! -x "${PKG_PATH}" ]; then
    chmod u+x "${PKG_PATH}"
  fi
  "${PKG_PATH}"
  PACKAGE_INSTALLED=true
}

run_ut() {
  local runner="${BASE_PATH}/tests/ut/run.sh"
  local coverage_value="FALSE"

  [ -f "${runner}" ] || die "UT runner not found: ${runner}"
  if [ "${ENABLE_COV}" = true ]; then
    coverage_value="TRUE"
  fi

  echo "[INFO] Running UT..."
  (
    cd "${BASE_PATH}/tests/ut"
    ENABLE_COV="${coverage_value}" bash ./run.sh
  )
}

run_st() {
  local runner="${BASE_PATH}/tests/st/run.sh"

  [ -f "${runner}" ] || die "ST runner not found: ${runner}"
  maybe_install_package

  echo "[INFO] Running ST..."
  (
    cd "${BASE_PATH}/tests/st"
    bash ./run.sh
  )
}

run_example() {
  local runner="${BASE_PATH}/examples/run.sh"

  [ -f "${runner}" ] || die "Example runner not found: ${runner}"
  maybe_install_package

  echo "[INFO] Running example (${EXAMPLE_MODE} mode)..."
  (
    cd "${BASE_PATH}/examples"
    bash ./run.sh "--${EXAMPLE_MODE}"
  )
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --list-socs)
      echo "Supported target SoC:"
      echo "  - ${TARGET_SOC_DISPLAY} (CMake compute unit: ${SUPPORTED_COMPUTE_UNIT})"
      exit 0
      ;;
    --make_clean)
      DO_CLEAN=true
      shift
      ;;
    -j)
      [ "$#" -ge 2 ] || die "Option -j requires a thread count"
      THREAD_NUM="$2"
      shift 2
      ;;
    -j*)
      THREAD_NUM="${1#-j}"
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
    --install-package)
      INSTALL_PACKAGE=true
      shift
      ;;
    -a|--all)
      RUN_UT=true
      RUN_ST=true
      shift
      ;;
    --soc)
      [ "$#" -ge 2 ] || die "Option --soc requires a value"
      COMPUTE_UNIT_RAW="$2"
      shift 2
      ;;
    --soc=*)
      COMPUTE_UNIT_RAW="${1#*=}"
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      die "Unexpected argument: $1"
      ;;
  esac
done

if [ "${DO_CLEAN}" = true ]; then
  clean_all
  exit 0
fi

[ -n "${COMPUTE_UNIT_RAW}" ] || die "The target SoC value must not be empty"
if ! COMPUTE_UNIT="$(normalize_compute_unit "${COMPUTE_UNIT_RAW}")"; then
  die "Unsupported SoC '${COMPUTE_UNIT_RAW}'. This candidate targets ${TARGET_SOC_DISPLAY} only."
fi

[[ "${THREAD_NUM}" =~ ^[1-9][0-9]*$ ]] || die "Thread count must be a positive integer: ${THREAD_NUM}"

check_source_layout
check_cann_environment

mkdir -p "${BUILD_PATH}" "${BUILD_OUT_PATH}"
rm -f "${BUILD_PATH}/CMakeCache.txt"

echo "[INFO] Build root: ${BASE_PATH}"
echo "[INFO] Target SoC: ${TARGET_SOC_DISPLAY} (${COMPUTE_UNIT})"
echo "[INFO] Target CANN: ${TARGET_CANN_VERSION}"
echo "[INFO] Build directory: ${BUILD_PATH}"
echo "[INFO] Threads: ${THREAD_NUM}"

echo "----------------------------------------------------------------"
echo "[INFO] Configuring the Flip candidate..."
cmake \
  -S "${BASE_PATH}" \
  -B "${BUILD_PATH}" \
  -DASCEND_COMPUTE_UNIT="${COMPUTE_UNIT}"

echo "----------------------------------------------------------------"
echo "[INFO] Building host libraries, kernel binary, and package..."
cmake --build "${BUILD_PATH}" \
  --target all binary package \
  --parallel "${THREAD_NUM}"

KERNEL_O="$(find_kernel_binary)"
if [ -z "${KERNEL_O}" ]; then
  die "Kernel binary was not generated under */ascendc_kernels/binary/${COMPUTE_UNIT}/*.o"
fi
if [ ! -f "${KERNEL_O}" ]; then
  die "Kernel binary path is not a regular file: ${KERNEL_O}"
fi

PKG_PATH="$(find_package_artifact)"
if [ -z "${PKG_PATH}" ]; then
  die "Custom operator package was not generated under build/ or build_out/"
fi
if [ ! -f "${PKG_PATH}" ]; then
  die "Custom operator package path is not a regular file: ${PKG_PATH}"
fi
if [ ! -s "${PKG_PATH}" ]; then
  die "Custom operator package is empty: ${PKG_PATH}"
fi

echo "----------------------------------------------------------------"
echo "[INFO] Build artifacts:"
echo "[INFO]   Kernel binary: ${KERNEL_O}"
echo "[INFO]   Package      : ${PKG_PATH}"

if [ "${RUN_UT}" = true ]; then
  run_ut
fi

if [ "${RUN_ST}" = true ]; then
  run_st
fi

if [ "${RUN_EXAMPLE}" = true ]; then
  run_example
fi

if [ "${INSTALL_PACKAGE}" = true ]; then
  maybe_install_package
fi

echo "----------------------------------------------------------------"
echo "[INFO] Build driver finished."
echo "[INFO] The Flip candidate remains unverified until every deterministic gate passes."
echo "[INFO] No CPU/NumPy fallback, silent dtype cast, or package activation is performed by this script."