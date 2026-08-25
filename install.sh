#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
NumPilot source-first installer

Usage:
  ./install.sh --allow-network [options]

Required:
  --allow-network                 Allow pip and the explicit bootstrap to use the network.

Options:
  --venv PATH                    Virtual environment path (default: <repo>/.venv).
  --pip-index-url URL            Explicit package index for the initial Python dependencies.
  --cann-env PATH                CANN set_env.sh to source in this process.
  --asnumpy-source PATH          Use an existing exact pinned AsNumpy checkout.
  --with-cannbot                 Fetch or verify the pinned CANNBot Skills checkout.
  --cannbot-source PATH          Use an existing exact pinned CANNBot checkout.
  --accept-cannbot-license       Confirm acceptance of CANN Open Software License v2.0.
  --receipt PATH                 Bootstrap receipt path (default: NumPilot XDG state path).
  --help                         Show this help.

This script never runs sudo, edits shell startup files, stores credentials, or installs into
the system Python. CANN itself, a working compiler, Python development headers, and the system
fmt/spdlog development packages must already be present.
EOF
}

fail() {
    echo "[NumPilot install] $*" >&2
    exit 2
}

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
VENV_DIR="$SCRIPT_DIR/.venv"
RECEIPT_PATH=""
RECEIPT_PATH_EXPLICIT=false
PIP_INDEX_URL_OVERRIDE=""
CANN_ENV_FILE=""
ASNUMPY_SOURCE=""
CANNBOT_SOURCE=""
ALLOW_NETWORK=false
WITH_CANNBOT=false
ACCEPT_CANNBOT_LICENSE=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --allow-network)
            ALLOW_NETWORK=true
            shift
            ;;
        --venv)
            [ "$#" -ge 2 ] || fail "--venv requires a path"
            VENV_DIR=$2
            shift 2
            ;;
        --pip-index-url)
            [ "$#" -ge 2 ] || fail "--pip-index-url requires a URL"
            PIP_INDEX_URL_OVERRIDE=$2
            shift 2
            ;;
        --cann-env)
            [ "$#" -ge 2 ] || fail "--cann-env requires a path"
            CANN_ENV_FILE=$2
            shift 2
            ;;
        --asnumpy-source)
            [ "$#" -ge 2 ] || fail "--asnumpy-source requires a path"
            ASNUMPY_SOURCE=$2
            shift 2
            ;;
        --with-cannbot)
            WITH_CANNBOT=true
            shift
            ;;
        --cannbot-source)
            [ "$#" -ge 2 ] || fail "--cannbot-source requires a path"
            CANNBOT_SOURCE=$2
            WITH_CANNBOT=true
            shift 2
            ;;
        --accept-cannbot-license)
            ACCEPT_CANNBOT_LICENSE=true
            shift
            ;;
        --receipt)
            [ "$#" -ge 2 ] || fail "--receipt requires a path"
            RECEIPT_PATH=$2
            RECEIPT_PATH_EXPLICIT=true
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
done

[ "$ALLOW_NETWORK" = true ] || fail "refusing implicit network access; add --allow-network"
if [ "$WITH_CANNBOT" = true ] && [ "$ACCEPT_CANNBOT_LICENSE" != true ]; then
    fail "--with-cannbot requires --accept-cannbot-license after reviewing the license"
fi

command -v python3 >/dev/null 2>&1 || fail "python3 is required"
command -v git >/dev/null 2>&1 || fail "git is required"
command -v g++ >/dev/null 2>&1 || fail "g++ >= 11.2 is required"

python3 - <<'PY' || fail "Python >= 3.10 with development headers is required"
import pathlib
import sys
import sysconfig

if sys.version_info < (3, 10):
    raise SystemExit(1)
include = pathlib.Path(sysconfig.get_path("include"))
if not (include / "Python.h").is_file():
    raise SystemExit(1)
PY

if [ -n "$CANN_ENV_FILE" ]; then
    [ -f "$CANN_ENV_FILE" ] || fail "CANN environment file does not exist: $CANN_ENV_FILE"
    # shellcheck disable=SC1090
    . "$CANN_ENV_FILE"
elif [ -z "${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-}}" ]; then
    for CANDIDATE in \
        /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
        /usr/local/Ascend/cann-9.0.0/set_env.sh; do
        if [ -f "$CANDIDATE" ]; then
            # shellcheck disable=SC1090
            . "$CANDIDATE"
            break
        fi
    done
fi

CANN_ROOT=${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-}}
[ -n "$CANN_ROOT" ] && [ -d "$CANN_ROOT" ] || fail "source a valid CANN set_env.sh first"
[ -d "${ASCEND_OPP_PATH:-}" ] || fail "ASCEND_OPP_PATH is missing or invalid"
[ -f "$CANN_ROOT/include/acl/acl.h" ] || fail "CANN ACL headers are missing under $CANN_ROOT/include"
[ -f "$CANN_ROOT/lib64/libascendcl.so" ] || fail "CANN runtime library libascendcl.so is missing"

# Development-only CANN libraries can shadow the installed runtime and cause unresolved symbols.
# Keep every unrelated entry but remove all devlib paths for this installer process.
SANITIZED_LD_LIBRARY_PATH=""
OLD_IFS=$IFS
IFS=:
for ENTRY in ${LD_LIBRARY_PATH:-}; do
    case "$ENTRY" in
        */devlib|*/devlib/*) continue ;;
    esac
    if [ -n "$ENTRY" ]; then
        if [ -n "$SANITIZED_LD_LIBRARY_PATH" ]; then
            SANITIZED_LD_LIBRARY_PATH="$SANITIZED_LD_LIBRARY_PATH:$ENTRY"
        else
            SANITIZED_LD_LIBRARY_PATH=$ENTRY
        fi
    fi
done
IFS=$OLD_IFS
export LD_LIBRARY_PATH=$SANITIZED_LD_LIBRARY_PATH

find_cmake_package() {
    PACKAGE_PATTERN=$1
    for CMAKE_ROOT in /usr/lib64/cmake /usr/lib/cmake /usr/lib/*-linux-gnu/cmake; do
        [ -d "$CMAKE_ROOT" ] || continue
        if find "$CMAKE_ROOT" -maxdepth 2 -type f -iname "$PACKAGE_PATTERN" -print -quit 2>/dev/null \
            | grep -q .; then
            return 0
        fi
    done
    return 1
}

find_cmake_package 'fmt*config.cmake' || fail \
    "system fmt CMake package is missing (Ubuntu package: libfmt-dev); NumPilot will not run sudo"
find_cmake_package 'spdlog*config.cmake' || fail \
    "system spdlog CMake package is missing (Ubuntu package: libspdlog-dev); NumPilot will not run sudo"

if [ -e "$VENV_DIR" ] && [ ! -x "$VENV_DIR/bin/python" ]; then
    fail "refusing to overwrite a non-venv path: $VENV_DIR"
fi
if [ ! -x "$VENV_DIR/bin/python" ]; then
    python3 -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
PIP_ARGS=(install --disable-pip-version-check)
if [ -n "$PIP_INDEX_URL_OVERRIDE" ]; then
    PIP_ARGS+=(--index-url "$PIP_INDEX_URL_OVERRIDE")
fi
"$VENV_PYTHON" -m pip "${PIP_ARGS[@]}" -e "$SCRIPT_DIR[bootstrap]"

# Console tools installed into the project venv (especially cmake) must win over
# older system copies during bootstrap preflight and the subsequent source build.
export PATH="$VENV_DIR/bin:$PATH"

BOOTSTRAP_ARGS=(--build-asnumpy)
if [ -n "$ASNUMPY_SOURCE" ]; then
    BOOTSTRAP_ARGS+=(--asnumpy-cache "$ASNUMPY_SOURCE")
fi
if [ "$WITH_CANNBOT" = true ]; then
    BOOTSTRAP_ARGS+=(--with-cannbot --accept-cannbot-license)
    if [ -n "$CANNBOT_SOURCE" ]; then
        BOOTSTRAP_ARGS+=(--cannbot-cache "$CANNBOT_SOURCE")
    fi
fi

if [ -z "$RECEIPT_PATH" ]; then
    RECEIPT_PATH=$("$VENV_PYTHON" -m numpilot paths --get bootstrap_receipt)
fi

# Runtime provenance consumes this receipt later, so validate its location with
# the same private-owner boundary before any bootstrap command runs.  The
# default XDG state directory is created and normalized to 0700; an explicit
# custom parent must already exist and satisfy the policy.
if ! RECEIPT_PATH=$("$VENV_PYTHON" - "$RECEIPT_PATH" "$RECEIPT_PATH_EXPLICIT" <<'PY'
import os
from pathlib import Path
import stat
import sys

raw_path, explicit_text = sys.argv[1:]
if not raw_path or "\x00" in raw_path or "\n" in raw_path or "\r" in raw_path:
    raise SystemExit("receipt path is empty or contains unsafe characters")

receipt = Path(os.path.abspath(os.path.expanduser(raw_path)))
parent = receipt.parent
explicit = explicit_text == "true"
current_uid = os.getuid()

try:
    parent_status = parent.lstat()
except FileNotFoundError:
    if explicit:
        raise SystemExit("custom receipt parent must already exist")
    parent.mkdir(parents=True, mode=0o700)
    parent_status = parent.lstat()

if stat.S_ISLNK(parent_status.st_mode):
    raise SystemExit("receipt parent must not be a symlink")
if not stat.S_ISDIR(parent_status.st_mode):
    raise SystemExit("receipt parent must be a directory")
if parent_status.st_uid != current_uid:
    raise SystemExit("receipt parent must be owned by the current user")

if not explicit:
    os.chmod(parent, 0o700)
    parent_status = parent.lstat()
if parent_status.st_mode & 0o022:
    raise SystemExit("receipt parent must not be group/world writable")
if not explicit and stat.S_IMODE(parent_status.st_mode) != 0o700:
    raise SystemExit("default receipt parent must have mode 0700")

try:
    receipt_status = receipt.lstat()
except FileNotFoundError:
    pass
else:
    if stat.S_ISLNK(receipt_status.st_mode):
        raise SystemExit("receipt path must not be a symlink")
    if not stat.S_ISREG(receipt_status.st_mode):
        raise SystemExit("existing receipt path must be a regular file")
    if receipt_status.st_uid != current_uid:
        raise SystemExit("existing receipt must be owned by the current user")

print(receipt)
PY
); then
    fail "receipt path failed the private-owner safety policy"
fi

export NUMPILOT_BOOTSTRAP_RECEIPT=$RECEIPT_PATH
"$VENV_PYTHON" -m numpilot bootstrap --plan "${BOOTSTRAP_ARGS[@]}"
APPLY_ARGS=(--apply "${BOOTSTRAP_ARGS[@]}" --receipt "$RECEIPT_PATH")
"$VENV_PYTHON" -m numpilot bootstrap "${APPLY_ARGS[@]}"

VERIFY_ARGS=(--verify)
if [ "$WITH_CANNBOT" = true ]; then
    VERIFY_ARGS+=(--with-cannbot --accept-cannbot-license)
    if [ -n "$CANNBOT_SOURCE" ]; then
        VERIFY_ARGS+=(--cannbot-cache "$CANNBOT_SOURCE")
    fi
fi
"$VENV_PYTHON" -m numpilot bootstrap "${VERIFY_ARGS[@]}"
"$VENV_PYTHON" -m numpilot doctor --backend npu

echo "[NumPilot install] ready"
echo "[NumPilot install] activate: source $VENV_DIR/bin/activate"
echo "[NumPilot install] receipt: $RECEIPT_PATH"
printf '[NumPilot install] later shells: export NUMPILOT_BOOTSTRAP_RECEIPT=%q\n' \
    "$RECEIPT_PATH"
