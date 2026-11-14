#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Draft build driver for the Diff registry-invoke operator.
# Target only: Ascend910B3 / compute unit ascend910b / CANN 9.2.0-beta.2.
# This script performs isolated local builds and optional local checks. It does not
# edit the CANN installation, ~/.bashrc, or any shared AsNumpy checkout.

set -euo pipefail

BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
readonly BASE_PATH
readonly BUILD_PATH="${BASE_PATH}/build"
readonly BUILD_OUT_PATH="${BASE_PATH}/build_out"
readonly TARGET_COMPUTE_UNIT="ascend910b"
readonly TARGET_SOC_LABEL="Ascend910B3"

if command -v nproc >/dev/null 2>&1; then
  CORE_NUMS="$(nproc)"
else
  CORE_NUMS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1\n')"
fi
case "${CORE_NUMS}" in
  ''|*[!0-9]*) CORE_NUMS=1 ;;
esac
if [ "${CORE_NUMS}" -gt 8 ]; then
  CORE_NUMS=8
fi
if [ "${CORE_NUMS}" -lt 1 ]; then
  CORE_NUMS=1
fi

THREAD_NUM="${CORE_NUMS}"
COMPUTE_UNIT_RAW="${TARGET_COMPUTE_UNIT}"
COMPUTE_UNIT="${TARGET_COMPUTE_UNIT}"
ENABLE_CLEAN=false
ENABLE_COV=FALSE
RUN_UT=false
RUN_ST=false
RUN_EXAMPLE=false
DO_INSTALL=false
EXAMPLE_MODE="eager"

info() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

die() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
Build script for the Diff operator draft (Ascend910B3, CANN 9.2.0-beta.2).

Usage:
  bash build.sh [OPTIONS]

Options:
  -h, --help              Print this help message.
  --list-socs             List bounded supported compute units.
  -j[n]                   Compile thread count, for example -j8 or -j 8.
                          Default is min(nproc, 8).
  --soc=<version>         Target compute unit. Only ascend910b is accepted.
                          alias: ascend910b3 (same bounded target).
  --make_clean            Remove local build/ and build_out/ contents and exit.
  -u, --ut                Run op_host/op_api unit tests after build.
  --cov                   Enable coverage for UT when the UT runner supports it.
  -s, --st                Run system tests. Requires an operator package that was
                          already installed explicitly after qualification gates.
  -e, --example           Run ACLNN examples. Requires the operator package to be
                          installed explicitly.
  --eager                 Use ACLNN eager example mode with -e (default).
  --graph                 Use GE graph example mode with -e.
  -a, --all               Run UT and ST. ST still requires explicit installation.
  --install               Explicitly install the built custom operator package
                          before ST/example execution. Refused unless
                          NUMPILOT_ALLOW_ACTIVATION=1 is set intentionally after
                          every operator-artifact.json gate passes.

Examples:
  bash build.sh --soc=ascend910b -j8
  bash build.sh --soc=ascend910b -j8 -u
  bash build.sh --soc=ascend910b -j8 -e
  bash build.sh --make_clean
  NUMPILOT_ALLOW_ACTIVATION=1 bash build.sh --soc=ascend910b -j8 --install -s

Notes:
  - The build directory is local to this workspace.
  - The script never sources or writes ~/.bashrc and never edits a shared AsNumpy
    checkout or the system CANN installation implicitly.
  - A successful local build does not by itself qualify this draft operator.
USAGE
}

require_cmd() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || die "Required command not found: ${cmd}"
}

normalize_compute_unit() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    ascend910b|ascend910b3)
      printf '%s\n' "${TARGET_COMPUTE_UNIT}"
      ;;
    *)
      return 1
      ;;
  esac
}

clean_artifacts() {
  if [ -d "${BUILD_PATH}" ]; then
    info "Cleaning ${BUILD_PATH}"
    find "${BUILD_PATH}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  fi
  if [ -d "${BUILD_OUT_PATH}" ]; then
    info "Cleaning ${BUILD_OUT_PATH}"
    find "${BUILD_OUT_PATH}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  fi
}

require_build_environment() {
  [ -f "${BASE_PATH}/CMakeLists.txt" ] || \
    die "CMakeLists.txt not found at ${BASE_PATH}/CMakeLists.txt"
  [ -n "${ASCEND_HOME_PATH:-}" ] || \
    die "ASCEND_HOME_PATH is not set. Source the CANN 9.2.0-beta.2 environment before building."
  [ -d "${ASCEND_HOME_PATH}" ] || \
    die "ASCEND_HOME_PATH does not point to a directory: ${ASCEND_HOME_PATH}"
  require_cmd cmake
  require_cmd make
}

find_kernel_object() {
  local found=""
  if [ -d "${BUILD_PATH}" ]; then
    found="$(find "${BUILD_PATH}" -type f \
      -path "*/op_kernel/ascendc_kernels/binary/${COMPUTE_UNIT}/*" \
      -name '*.o' -print -quit 2>/dev/null || true)"
  fi
  printf '%s\n' "${found}"
}

find_package_file() {
  local found=""
  if [ -d "${BUILD_PATH}" ]; then
    found="$(find "${BUILD_PATH}" -maxdepth 6 -type f -name '*.run' -print -quit 2>/dev/null || true)"
  fi
  if [ -z "${found}" ] && [ -d "${BUILD_OUT_PATH}" ]; then
    found="$(find "${BUILD_OUT_PATH}" -maxdepth 6 -type f -name '*.run' -print -quit 2>/dev/null || true)"
  fi
  printf '%s\n' "${found}"
}

build_operator() {
  require_build_environment

  info "Target SoC: ${TARGET_SOC_LABEL} (compute unit ${COMPUTE_UNIT})"
  info "Build directory: ${BUILD_PATH}"

  mkdir -p "${BUILD_PATH}"
  rm -f "${BUILD_PATH}/CMakeCache.txt"

  info "Configuring project with CMake"
  cmake -S "${BASE_PATH}" -B "${BUILD_PATH}" \
    "-DASCEND_COMPUTE_UNIT=${COMPUTE_UNIT}"

  info "Building host, kernel binary, and package with ${THREAD_NUM} threads"
  cmake --build "${BUILD_PATH}" \
    --target all binary package -- -j"${THREAD_NUM}"

  local kernel_object
  kernel_object="$(find_kernel_object)"
  [ -n "${kernel_object}" ] || \
    die "Kernel binary was not produced under ${BUILD_PATH} for ${COMPUTE_UNIT}."
  [ -s "${kernel_object}" ] || \
    die "Kernel binary is empty: ${kernel_object}"

  local package_file
  package_file="$(find_package_file)"
  [ -n "${package_file}" ] || \
    die "Custom operator package (*.run) was not produced under ${BUILD_PATH}."
  [ -s "${package_file}" ] || \
    die "Custom operator package is empty: ${package_file}"

  info "Kernel binary: ${kernel_object}"
  info "Operator package: ${package_file}"
}

install_package_explicitly() {
  [ "${NUMPILOT_ALLOW_ACTIVATION:-0}" = "1" ] || \
    die "Refusing to install the draft operator. Set NUMPILOT_ALLOW_ACTIVATION=1 only after every operator-artifact.json gate passes and activation is intentional."

  local package_file
  package_file="$(find_package_file)"
  [ -n "${package_file}" ] || \
    die "Cannot install: custom operator package (*.run) was not found."
  [ -s "${package_file}" ] || \
    die "Cannot install: custom operator package is empty: ${package_file}"

  info "Installing explicitly activated operator package: ${package_file}"
  (
    cd -- "$(dirname -- "${package_file}")"
    bash "$(basename -- "${package_file}")"
  )
}

run_unit_tests() {
  local runner="${BASE_PATH}/tests/ut/run.sh"
  [ -f "${runner}" ] || \
    die "UT runner not found: ${runner}"

  info "Running UT"
  (
    cd -- "$(dirname -- "${runner}")"
    ENABLE_COV="${ENABLE_COV}" bash ./run.sh
  )
}

run_st_tests() {
  local runner="${BASE_PATH}/tests/st/run.sh"
  [ -f "${runner}" ] || \
    die "ST runner not found: ${runner}"

  info "Running ST. This requires an explicitly installed operator package."
  (
    cd -- "$(dirname -- "${runner}")"
    bash ./run.sh
  )
}

run_examples() {
  local runner="${BASE_PATH}/examples/run.sh"
  [ -f "${runner}" ] || \
    die "Example runner not found: ${runner}"

  info "Running examples in ${EXAMPLE_MODE} mode"
  (
    cd -- "$(dirname -- "${runner}")"
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
      printf 'Supported compute units:\n'
      printf '  - %s (%s)\n' "${TARGET_COMPUTE_UNIT}" "${TARGET_SOC_LABEL}"
      exit 0
      ;;
    -j)
      if [ "$#" -ge 2 ] && [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
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
      COMPUTE_UNIT_RAW="${1#*=}"
      shift
      ;;
    --soc)
      [ "$#" -ge 2 ] || die "Option --soc requires a value."
      COMPUTE_UNIT_RAW="$2"
      shift 2
      ;;
    --make_clean|--clean)
      ENABLE_CLEAN=true
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
    --)
      shift
      break
      ;;
    -*)
      usage
      die "Invalid option: $1"
      ;;
    *)
      usage
      die "Unexpected argument: $1"
      ;;
  esac
done

if [ "${ENABLE_CLEAN}" = true ]; then
  clean_artifacts
  info "Clean completed."
  exit 0
fi

case "${THREAD_NUM}" in
  ''|*[!0-9]*)
    die "Invalid thread count: ${THREAD_NUM}"
    ;;
esac
[ "${THREAD_NUM}" -ge 1 ] || die "Thread count must be at least 1."

if ! COMPUTE_UNIT="$(normalize_compute_unit "${COMPUTE_UNIT_RAW}")"; then
  die "Unsupported SoC: ${COMPUTE_UNIT_RAW}. This draft is bounded to Ascend910B3 (ascend910b)."
fi

build_operator

if [ "${RUN_UT}" = true ]; then
  run_unit_tests
fi

if [ "${DO_INSTALL}" = true ]; then
  install_package_explicitly
fi

if [ "${RUN_ST}" = true ]; then
  run_st_tests
fi

if [ "${RUN_EXAMPLE}" = true ]; then
  run_examples
fi

if [ "${RUN_UT}" = true ] || [ "${RUN_ST}" = true ] || [ "${RUN_EXAMPLE}" = true ] || [ "${DO_INSTALL}" = true ]; then
  info "Requested local build and execution steps completed."
fi
info "Draft artifacts are available under ${BUILD_PATH}."
info "Qualification remains pending until all operator-artifact.json gates pass."