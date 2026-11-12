#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
#
# Isolated build driver for the Pad registry-invoke Ascend C operator.
# The generated candidate is unverified until the declared qualification
# gates and activation checks have actually passed.

set -euo pipefail

BASE_PATH="$(cd "$(dirname "$0")" && pwd)"
BUILD_PATH="${BASE_PATH}/build"
BUILD_OUT_PATH="${BASE_PATH}/build_out"

# This candidate is scoped to Ascend910B3. The CANN build spelling is
# ascend910b; no other SoC is claimed or accepted by this driver.
DEFAULT_COMPUTE_UNIT="ascend910b"
SUPPORT_COMPUTE_UNITS=("ascend910b")

DEFAULT_CORE_NUMS="$(nproc 2>/dev/null || echo 8)"
case "${DEFAULT_CORE_NUMS}" in
  ''|*[!0-9]*) DEFAULT_CORE_NUMS=8 ;;
esac
if [ "${DEFAULT_CORE_NUMS}" -gt 8 ]; then
  DEFAULT_CORE_NUMS=8
fi
if [ "${DEFAULT_CORE_NUMS}" -lt 1 ]; then
  DEFAULT_CORE_NUMS=1
fi

COMPUTE_UNIT_RAW=""
COMPUTE_UNIT="${DEFAULT_COMPUTE_UNIT}"
THREAD_NUM="${DEFAULT_CORE_NUMS}"
ENABLE_CLEAN=false
ENABLE_COV="FALSE"
RUN_UT=false
RUN_ST=false
RUN_EXAMPLE=false
EXAMPLE_MODE="eager"
SKIP_INSTALL=false

log() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*" >&2
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Build script for the Pad Ascend C operator candidate.

Usage:
  bash build.sh [OPTIONS]

Target:
  Ascend910B3 (CANN compute unit: ascend910b)

Options:
  -h, --help              Print this help message
  --list-socs             List the supported SoC version
  -j[n]                   Compile thread count, default is min(nproc, 8)
  -j n                    Compile thread count, explicit separate value
  --soc=<soc_version>     Build for Ascend910B3; accepts ascend910b or ascend910b3
  --make_clean            Remove build/ and build_out/ and exit
  -u, --ut                Run UT after a successful build
  --cov                   Enable coverage mode for UT where supported
  -s, --st                Install the built package and run ST (requires NPU)
  -e, --example           Install the built package and run an ACLNN example
  --eager                 Use eager ACLNN example mode (default for -e)
  --graph                 Use graph-mode example mode
  -a, --all               Run UT and ST after a successful build
  --no-install            With -s/-e, do not install the package; assume it is
                          already installed in the active isolated environment

Environment:
  ASCEND_HOME_PATH        Required for compilation; remaining CANN variables
                          are expected to be sourced by the caller.
  PAD_SKIP_PACKAGE_INSTALL=1
                          Equivalent to --no-install for package installation.

Examples:
  bash build.sh --soc=ascend910b -j8
  bash build.sh --make_clean
  bash build.sh -u --soc=ascend910b
  bash build.sh -s --soc=ascend910b
  bash build.sh -e --eager --soc=ascend910b
  bash build.sh -e --graph --soc=ascend910b
EOF
}

normalize_compute_unit() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    ascend910b|ascend910b3)
      printf '%s' "ascend910b"
      ;;
    *)
      printf '%s' "${value}"
      ;;
  esac
}

is_supported_compute_unit() {
  local unit="$1"
  local supported
  for supported in "${SUPPORT_COMPUTE_UNITS[@]}"; do
    if [ "${unit}" = "${supported}" ]; then
      return 0
    fi
  done
  return 1
}

clean_build_outputs() {
  log "Cleaning ${BUILD_PATH} and ${BUILD_OUT_PATH}"
  rm -rf "${BUILD_PATH}"
  rm -rf "${BUILD_OUT_PATH}"
}

require_source_layout() {
  [ -f "${BASE_PATH}/CMakeLists.txt" ] || \
    die "Missing CMakeLists.txt in ${BASE_PATH}"
  [ -d "${BASE_PATH}/op_host" ] || \
    die "Missing op_host directory in ${BASE_PATH}"
  [ -d "${BASE_PATH}/op_kernel" ] || \
    die "Missing op_kernel directory in ${BASE_PATH}"
  [ -d "${BASE_PATH}/op_api" ] || \
    die "Missing op_api directory in ${BASE_PATH}"
}

require_build_environment() {
  command -v cmake >/dev/null 2>&1 || \
    die "cmake is not available on PATH"
  [ -n "${ASCEND_HOME_PATH:-}" ] || \
    die "ASCEND_HOME_PATH is not set; source the pinned CANN environment first"
  [ -d "${ASCEND_HOME_PATH}" ] || \
    die "ASCEND_HOME_PATH does not point to a directory: ${ASCEND_HOME_PATH}"
}

find_built_package() {
  local candidate=""
  candidate="$(find "${BUILD_PATH}" -maxdepth 2 -type f -name 'custom_opp_*.run' -print -quit 2>/dev/null || true)"
  if [ -z "${candidate}" ] && [ -d "${BUILD_OUT_PATH}" ]; then
    candidate="$(find "${BUILD_OUT_PATH}" -maxdepth 2 -type f -name 'custom_opp_*.run' -print -quit 2>/dev/null || true)"
  fi
  printf '%s' "${candidate}"
}

install_operator_package() {
  if [ "${SKIP_INSTALL}" = true ] || [ "${PAD_SKIP_PACKAGE_INSTALL:-0}" = "1" ]; then
    warn "Skipping operator package installation; tests must use an already-installed package."
    return 0
  fi

  warn "Installing the built custom operator package for the explicitly requested test."
  if [ -x "${PACKAGE_PATH}" ]; then
    "${PACKAGE_PATH}"
  else
    bash "${PACKAGE_PATH}"
  fi
}

run_ut() {
  [ -f "${BASE_PATH}/tests/ut/run.sh" ] || \
    die "UT runner not found: ${BASE_PATH}/tests/ut/run.sh"
  log "Running UT"
  (
    cd "${BASE_PATH}/tests/ut"
    ENABLE_COV="${ENABLE_COV}" bash ./run.sh
  )
}

run_st() {
  [ -f "${BASE_PATH}/tests/st/run.sh" ] || \
    die "ST runner not found: ${BASE_PATH}/tests/st/run.sh"
  install_operator_package
  log "Running ST"
  (
    cd "${BASE_PATH}/tests/st"
    bash ./run.sh
  )
}

run_example() {
  [ -f "${BASE_PATH}/examples/run.sh" ] || \
    die "Example runner not found: ${BASE_PATH}/examples/run.sh"
  install_operator_package
  log "Running example in ${EXAMPLE_MODE} mode"
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
      echo "Supported SoC versions:"
      echo "  ascend910b  (Ascend910B3)"
      exit 0
      ;;
    -j)
      shift
      [ "$#" -gt 0 ] || die "-j requires an explicit thread count"
      THREAD_NUM="$1"
      shift
      ;;
    -j*)
      THREAD_NUM="${1#-j}"
      shift
      ;;
    --soc)
      shift
      [ "$#" -gt 0 ] || die "--soc requires a value"
      COMPUTE_UNIT_RAW="$1"
      shift
      ;;
    --soc=*)
      COMPUTE_UNIT_RAW="${1#*=}"
      shift
      ;;
    --make_clean)
      ENABLE_CLEAN=true
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
    --no-install)
      SKIP_INSTALL=true
      shift
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
  clean_build_outputs
  exit 0
fi

case "${THREAD_NUM}" in
  ''|*[!0-9]*)
    die "Thread count must be a positive integer, got: ${THREAD_NUM}"
    ;;
esac
[ "${THREAD_NUM}" -ge 1 ] || die "Thread count must be at least 1"

if [ -n "${COMPUTE_UNIT_RAW}" ]; then
  COMPUTE_UNIT="$(normalize_compute_unit "${COMPUTE_UNIT_RAW}")"
fi

if ! is_supported_compute_unit "${COMPUTE_UNIT}"; then
  die "Unsupported SoC '${COMPUTE_UNIT_RAW:-${COMPUTE_UNIT}}'; this candidate targets Ascend910B3 only (ascend910b)"
fi

require_source_layout
require_build_environment

log "Building Pad candidate for ASCEND_COMPUTE_UNIT=${COMPUTE_UNIT} using ${THREAD_NUM} threads"
mkdir -p "${BUILD_PATH}"
mkdir -p "${BUILD_OUT_PATH}"

CMAKE_ARGS=(
  "-DASCEND_COMPUTE_UNIT=${COMPUTE_UNIT}"
  "-DCMAKE_BUILD_TYPE=Release"
)

cmake -S "${BASE_PATH}" -B "${BUILD_PATH}" "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_PATH}" --target all binary package -- -j"${THREAD_NUM}"

KERNEL_DIR="${BUILD_PATH}/op_kernel/ascendc_kernels/binary/${COMPUTE_UNIT}"
KERNEL_O="$(find "${KERNEL_DIR}" -type f -name '*.o' -print -quit 2>/dev/null || true)"
if [ -z "${KERNEL_O}" ]; then
  die "Kernel binary was not produced under ${KERNEL_DIR}"
fi

PACKAGE_PATH="$(find_built_package)"
if [ -z "${PACKAGE_PATH}" ]; then
  die "Custom operator package custom_opp_*.run was not produced under ${BUILD_PATH}"
fi
[ -s "${PACKAGE_PATH}" ] || die "Custom operator package is empty: ${PACKAGE_PATH}"

log "Kernel object: ${KERNEL_O}"
log "Operator package: ${PACKAGE_PATH}"

if [ "${RUN_UT}" = true ]; then
  run_ut
fi

if [ "${RUN_ST}" = true ]; then
  run_st
fi

if [ "${RUN_EXAMPLE}" = true ]; then
  run_example
fi

log "Build completed and required build artifacts were found."
if [ "${RUN_UT}" = true ] || [ "${RUN_ST}" = true ] || [ "${RUN_EXAMPLE}" = true ]; then
  log "Requested build-time test/example steps completed successfully."
fi
warn "Candidate remains unverified until numerical, native-execution, whole-program, and activation gates pass."