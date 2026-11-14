#!/usr/bin/env bash
# Draft build driver for the argmax registry-invoke operator.
# This script builds an isolated operator package and runs tests only when
# explicitly requested. It never installs or activates the generated package.

set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_PATH="${BASE_PATH}/build"
BUILD_OUT_PATH="${BASE_PATH}/build_out"

SUPPORTED_COMPUTE_UNITS=("ascend910b")

log() {
    printf '[INFO] %s\n' "$*"
}

die() {
    printf '[ERROR] %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Build script for the argmax registry-invoke operator.

Usage:
  bash build.sh [OPTIONS]

Options:
  -h, --help              Print this help message.
  --list-socs             List supported SoC versions.
  -j[N], --jobs[=N]       Compile thread count. Defaults to detected CPU count, capped at 8.
  --soc=<soc>             Target SoC. Supported: ascend910b (Ascend910B3); ascend910b3 is accepted as an alias.
  --make_clean            Remove build/ and build_out/ artifacts.
  -u, --ut                Run unit tests through tests/ut/run.sh after building.
  --cov                   Enable coverage mode for UT when used with -u.
  -s, --st                Run system tests through tests/st/run.sh after building.
  -e, --example           Run ACLNN examples through examples/run.sh after building.
  --eager                 Use the eager ACLNN example mode. This is the default for -e.
  --graph                 Use the graph-mode example mode for -e.
  -a, --all               Run both UT and ST after building.

Notes:
  - The default and only supported compute unit is ascend910b, which covers Ascend910B3.
  - This script does not install the generated .run package and does not modify CANN,
    ~/.bashrc, or any shared AsNumpy checkout.
  - ST and example execution require an already prepared, explicitly managed runtime
    environment. The generated operator remains unverified until all project gates pass.

Examples:
  bash build.sh --soc=ascend910b -j8
  bash build.sh --soc=ascend910b3 -j8
  bash build.sh --soc=ascend910b -j8 -u
  bash build.sh --make_clean
EOF
}

detect_core_nums() {
    local cores
    cores="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
    if ! [[ "${cores}" =~ ^[1-9][0-9]*$ ]]; then
        cores="$(nproc 2>/dev/null || true)"
    fi
    if ! [[ "${cores}" =~ ^[1-9][0-9]*$ ]]; then
        cores=1
    fi
    if [ "${cores}" -gt 8 ]; then
        cores=8
    fi
    printf '%s\n' "${cores}"
}

normalize_soc() {
    local raw unit
    raw="$1"
    unit="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')"
    case "${unit}" in
        ascend910b|ascend910b3)
            printf '%s\n' "ascend910b"
            ;;
        *)
            return 1
            ;;
    esac
}

validate_thread_num() {
    case "$1" in
        ''|*[!0-9]*)
            die "Invalid thread count: $1"
            ;;
    esac
    if [ "$1" -lt 1 ]; then
        die "Thread count must be at least 1"
    fi
}

clean_build_dirs() {
    if [ -d "${BUILD_PATH}" ]; then
        log "Cleaning ${BUILD_PATH}"
        rm -rf "${BUILD_PATH:?}"/*
    fi
    if [ -d "${BUILD_OUT_PATH}" ]; then
        log "Cleaning ${BUILD_OUT_PATH}"
        rm -rf "${BUILD_OUT_PATH:?}"/*
    fi
}

run_ut() {
    local runner="${BASE_PATH}/tests/ut/run.sh"
    [ -f "${runner}" ] || die "UT runner not found: ${runner}"
    log "Running UT"
    (
        cd "${BASE_PATH}/tests/ut"
        ENABLE_COV="${ENABLE_COV}" bash ./run.sh
    )
}

run_st() {
    local runner="${BASE_PATH}/tests/st/run.sh"
    [ -f "${runner}" ] || die "ST runner not found: ${runner}"
    log "Running ST"
    (
        cd "${BASE_PATH}/tests/st"
        bash ./run.sh
    )
}

run_example() {
    local runner="${BASE_PATH}/examples/run.sh"
    [ -f "${runner}" ] || die "Example runner not found: ${runner}"
    log "Running examples in ${EXAMPLE_MODE} mode"
    (
        cd "${BASE_PATH}/examples"
        bash ./run.sh "--${EXAMPLE_MODE}"
    )
}

find_kernel_object() {
    local found
    found="$(find "${BUILD_PATH}" -type f -name '*.o' \
        -path "*ascendc_kernels/binary/${COMPUTE_UNIT}/*" \
        -print -quit 2>/dev/null || true)"
    if [ -z "${found}" ]; then
        found="$(find "${BUILD_PATH}" -type f -name '*.o' \
            -path '*ascendc_kernels/binary/*' \
            -print -quit 2>/dev/null || true)"
    fi
    printf '%s\n' "${found}"
}

find_package() {
    local found
    found="$(find "${BUILD_PATH}" "${BUILD_OUT_PATH}" -maxdepth 2 -type f -name '*.run' \
        -print -quit 2>/dev/null || true)"
    printf '%s\n' "${found}"
}

CORE_NUMS="$(detect_core_nums)"
THREAD_NUM="${CORE_NUMS}"
COMPUTE_UNIT="ascend910b"
ENABLE_CLEAN=FALSE
ENABLE_COV=FALSE
RUN_UT=FALSE
RUN_ST=FALSE
RUN_EXAMPLE=FALSE
EXAMPLE_MODE="eager"

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --list-socs)
            printf 'Supported SoC versions:\n'
            for soc in "${SUPPORTED_COMPUTE_UNITS[@]}"; do
                printf '  - %s\n' "${soc}"
            done
            printf '  - ascend910b3 (alias for ascend910b)\n'
            exit 0
            ;;
        --make_clean)
            ENABLE_CLEAN=TRUE
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
        -e|--example)
            RUN_EXAMPLE=TRUE
            shift
            ;;
        -a|--all)
            RUN_UT=TRUE
            RUN_ST=TRUE
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
        -j)
            if [ "$#" -lt 2 ]; then
                THREAD_NUM="${CORE_NUMS}"
                shift
            else
                THREAD_NUM="$2"
                shift 2
            fi
            validate_thread_num "${THREAD_NUM}"
            ;;
        -j*)
            THREAD_NUM="${1#-j}"
            if [ -z "${THREAD_NUM}" ]; then
                THREAD_NUM="${CORE_NUMS}"
            fi
            validate_thread_num "${THREAD_NUM}"
            shift
            ;;
        --jobs=*)
            THREAD_NUM="${1#*=}"
            validate_thread_num "${THREAD_NUM}"
            shift
            ;;
        --jobs)
            if [ "$#" -lt 2 ]; then
                THREAD_NUM="${CORE_NUMS}"
                shift
            else
                THREAD_NUM="$2"
                shift 2
            fi
            validate_thread_num "${THREAD_NUM}"
            ;;
        --soc)
            if [ "$#" -lt 2 ]; then
                die "--soc requires a value"
            fi
            if ! COMPUTE_UNIT="$(normalize_soc "$2")"; then
                die "Invalid SoC version: $2. Supported: ascend910b (Ascend910B3)"
            fi
            shift 2
            ;;
        --soc=*)
            raw_soc="${1#*=}"
            if ! COMPUTE_UNIT="$(normalize_soc "${raw_soc}")"; then
                die "Invalid SoC version: ${raw_soc}. Supported: ascend910b (Ascend910B3)"
            fi
            shift
            ;;
        *)
            die "Invalid option or argument: $1"
            ;;
    esac
done

if [ "${ENABLE_CLEAN}" = "TRUE" ]; then
    clean_build_dirs
    exit 0
fi

if [ ! -f "${BASE_PATH}/CMakeLists.txt" ]; then
    die "Missing root CMakeLists.txt at ${BASE_PATH}/CMakeLists.txt"
fi

command -v cmake >/dev/null 2>&1 || die "cmake is required but was not found in PATH"

if [ "${RUN_UT}" = "FALSE" ] && [ "${ENABLE_COV}" = "TRUE" ]; then
    die "--cov requires --ut or --all"
fi

mkdir -p "${BUILD_PATH}" "${BUILD_OUT_PATH}"
rm -f "${BUILD_PATH}/CMakeCache.txt"

CMAKE_ARGS=("-DASCEND_COMPUTE_UNIT=${COMPUTE_UNIT}")

log "Configuring argmax operator for ${COMPUTE_UNIT}"
cmake -S "${BASE_PATH}" -B "${BUILD_PATH}" "${CMAKE_ARGS[@]}"

log "Building with ${THREAD_NUM} thread(s)"
cmake --build "${BUILD_PATH}" --target all binary package -- -j"${THREAD_NUM}"

KERNEL_O="$(find_kernel_object)"
[ -n "${KERNEL_O}" ] || die "Kernel binary was not produced under ${BUILD_PATH}"
[ -s "${KERNEL_O}" ] || die "Kernel binary is empty: ${KERNEL_O}"

PKG_PATH="$(find_package)"
[ -n "${PKG_PATH}" ] || die "Operator package .run file was not produced"
[ -s "${PKG_PATH}" ] || die "Operator package is empty: ${PKG_PATH}"

log "Kernel binary: ${KERNEL_O}"
log "Operator package: ${PKG_PATH}"

if [ "${RUN_UT}" = "TRUE" ]; then
    run_ut
fi

if [ "${RUN_ST}" = "TRUE" ]; then
    run_st
fi

if [ "${RUN_EXAMPLE}" = "TRUE" ]; then
    run_example
fi

log "Build artifacts generated for ${COMPUTE_USER_UNIT:-${COMPUTE_UNIT}}"
log "The generated operator package was not installed or activated."
log "Qualification remains pending until the project gates pass."