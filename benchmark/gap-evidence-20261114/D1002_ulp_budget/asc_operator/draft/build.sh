#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Isolated build entry for the NumPilot Spacing operator.
# This script writes only under <repository>/build and <repository>/build_out.
# It does not install the generated package, modify CANN, or modify shell startup files.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BUILD_OUT_DIR="${ROOT_DIR}/build_out"

DEFAULT_SOC="ascend910b"
DEFAULT_JOBS="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
if [ "${DEFAULT_JOBS}" -gt 8 ] 2>/dev/null; then
    DEFAULT_JOBS=8
fi
if [ "${DEFAULT_JOBS}" -lt 1 ] 2>/dev/null; then
    DEFAULT_JOBS=1
fi

COMPUTE_UNIT="${DEFAULT_SOC}"
JOBS="${DEFAULT_JOBS}"
CLEAN_REQUESTED=false
RUN_UT=false

usage() {
    cat <<'EOF'
Build script for the NumPilot Spacing operator.

Usage:
  bash build.sh [OPTIONS]

Options:
  -h, --help              Print this help message.
  --list-socs             List supported SoC versions.
  -j[n], -j <n>           Compile thread count (default: min(nproc, 8)).
  --soc=<soc>             Target SoC. Supported: ascend910b (Ascend910B3).
  --make_clean            Remove local build/ and build_out/ directories, then exit.
  -u, --ut                Run tests/ut/run.sh after a successful build.

This script never installs the generated package or activates the operator.
EOF
}

list_socs() {
    echo "Supported SoC versions:"
    echo "  - ascend910b (Ascend910B3)"
}

normalize_soc() {
    local value="$1"
    value="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
    case "${value}" in
        ascend910b|ascend910b3)
            echo "ascend910b"
            ;;
        *)
            return 1
            ;;
    esac
}

clean_local_build() {
    if [ -n "${BUILD_DIR}" ] && [ "${BUILD_DIR}" != "/" ]; then
        rm -rf -- "${BUILD_DIR}"
    fi
    if [ -n "${BUILD_OUT_DIR}" ] && [ "${BUILD_OUT_DIR}" != "/" ]; then
        rm -rf -- "${BUILD_OUT_DIR}"
    fi
}

load_cann_environment() {
    if [ -n "${ASCEND_HOME_PATH:-}" ]; then
        if [ ! -d "${ASCEND_HOME_PATH}" ]; then
            echo "[ERROR] ASCEND_HOME_PATH is set but is not a directory: ${ASCEND_HOME_PATH}" >&2
            return 1
        fi
        return 0
    fi

    local candidates=(
        "/usr/local/Ascend/ascend-toolkit/latest"
        "/usr/local/Ascend/latest"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [ -f "${candidate}/set_env.sh" ]; then
            set +u
            # shellcheck disable=SC1090
            source "${candidate}/set_env.sh" >/dev/null 2>&1 || true
            set -u
            if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -d "${ASCEND_HOME_PATH}" ]; then
                return 0
            fi
        fi
    done

    echo "[ERROR] CANN environment is not available." >&2
    echo "[ERROR] Set ASCEND_HOME_PATH or source a CANN set_env.sh before building." >&2
    return 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --list-socs)
            list_socs
            exit 0
            ;;
        --make_clean|--clean)
            CLEAN_REQUESTED=true
            shift
            ;;
        -j)
            if [ "$#" -ge 2 ] && [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                JOBS="$2"
                shift 2
            else
                JOBS="${DEFAULT_JOBS}"
                shift
            fi
            ;;
        -j[0-9]*)
            JOBS="${1#-j}"
            shift
            ;;
        --soc)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --soc requires a value." >&2
                usage >&2
                exit 2
            fi
            if ! COMPUTE_UNIT="$(normalize_soc "$2")"; then
                echo "[ERROR] Invalid SoC: $2" >&2
                list_socs >&2
                exit 2
            fi
            shift 2
            ;;
        --soc=*)
            if ! COMPUTE_UNIT="$(normalize_soc "${1#*=}")"; then
                echo "[ERROR] Invalid SoC: ${1#*=}" >&2
                list_socs >&2
                exit 2
            fi
            shift
            ;;
        -u|--ut)
            RUN_UT=true
            shift
            ;;
        *)
            echo "[ERROR] Invalid option or unexpected argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ "${CLEAN_REQUESTED}" = true ]; then
    clean_local_build
    echo "[INFO] Removed local build artifacts under ${ROOT_DIR}."
    exit 0
fi

if ! [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Invalid compile thread count: ${JOBS}" >&2
    exit 2
fi

if [ "${COMPUTE_UNIT}" != "ascend910b" ]; then
    echo "[ERROR] This candidate supports only ascend910b (Ascend910B3)." >&2
    exit 2
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "[ERROR] cmake is required but was not found in PATH." >&2
    exit 1
fi

if [ ! -f "${ROOT_DIR}/CMakeLists.txt" ]; then
    echo "[ERROR] Missing root CMakeLists.txt at ${ROOT_DIR}/CMakeLists.txt" >&2
    exit 1
fi

if ! load_cann_environment; then
    exit 1
fi

mkdir -p "${BUILD_DIR}" "${BUILD_OUT_DIR}"

echo "================================================================"
echo "[INFO] Operator: spacing"
echo "[INFO] Target SoC: ${COMPUTE_UNIT} (Ascend910B3)"
echo "[INFO] Source: ${ROOT_DIR}"
echo "[INFO] Build: ${BUILD_DIR}"
echo "[INFO] Jobs: ${JOBS}"
echo "[INFO] CANN home: ${ASCEND_HOME_PATH}"
echo "================================================================"

echo "[INFO] Configuring CMake..."
cmake \
    -S "${ROOT_DIR}" \
    -B "${BUILD_DIR}" \
    -DASCEND_COMPUTE_UNIT="${COMPUTE_UNIT}" \
    -DCMAKE_BUILD_TYPE=Release

echo "[INFO] Building host libraries, kernel binaries, and package..."
cmake --build "${BUILD_DIR}" --target all binary package -- -j "${JOBS}"

KERNEL_DIR="${BUILD_DIR}/op_kernel/ascendc_kernels/binary/${COMPUTE_UNIT}"
KERNEL_O="$(find "${KERNEL_DIR}" -type f -name '*.o' -print -quit 2>/dev/null || true)"
if [ -z "${KERNEL_O}" ]; then
    echo "[ERROR] Kernel binary artifact was not produced under ${KERNEL_DIR}" >&2
    exit 1
fi

PACKAGE_PATH="${BUILD_DIR}/custom_opp_ubuntu_aarch64.run"
if [ ! -s "${PACKAGE_PATH}" ]; then
    PACKAGE_PATH="$(find "${BUILD_DIR}" "${BUILD_OUT_DIR}" -maxdepth 3 -type f -name '*.run' -print -quit 2>/dev/null || true)"
fi
if [ -z "${PACKAGE_PATH}" ] || [ ! -s "${PACKAGE_PATH}" ]; then
    echo "[ERROR] Operator package artifact was not produced." >&2
    exit 1
fi

echo "================================================================"
echo "[INFO] Build completed."
echo "[INFO] Kernel object: ${KERNEL_O}"
echo "[INFO] Package: ${PACKAGE_PATH}"
echo "[INFO] The package has not been installed or activated."
echo "================================================================"

if [ "${RUN_UT}" = true ]; then
    UT_DIR="${ROOT_DIR}/tests/ut"
    if [ ! -f "${UT_DIR}/run.sh" ]; then
        echo "[ERROR] UT requested but ${UT_DIR}/run.sh was not found." >&2
        exit 1
    fi
    echo "[INFO] Running isolated UT suite..."
    (
        cd "${UT_DIR}"
        bash ./run.sh
    )
    echo "[INFO] UT suite completed."
fi

exit 0