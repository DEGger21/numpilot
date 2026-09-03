#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
NumPilot installer (frozen AsNumpy component by default)

Usage:
  ./install.sh --allow-network [options]

Required:
  --allow-network                 Allow pip and the explicit bootstrap to use the network.

Options:
  --venv PATH                    Virtual environment path (default: <repo>/.venv).
  --persistent-root PATH         Freeze state/config/cache under PATH and write
                                 PATH/activate-numpilot.sh for later shells.
  --pip-index-url URL            Explicit package index for the initial Python dependencies.
  --cann-env PATH                CANN set_env.sh to source in this process.
  --asnumpy-component PATH       Use the release-adjacent exact AsNumpy wheel component.
  --asnumpy-source PATH          (Maintainer-only) use an existing exact pinned source checkout.
  --with-cannbot                 Fetch or verify the pinned CANNBot Skills checkout.
  --cannbot-source PATH          Use an existing exact pinned CANNBot checkout.
  --accept-cannbot-license       Confirm acceptance of CANN Open Software License v2.0.
  --receipt PATH                 Bootstrap receipt path (default: NumPilot XDG state path).
  --help                         Show this help.

This script never runs sudo, edits shell startup files, stores credentials, or installs into
the system Python. CANN itself must already be present. A working compiler, Python development
headers, and the system fmt/spdlog development packages are required only for the explicit
maintainer source-build path.
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
ASNUMPY_COMPONENT=""
CANNBOT_SOURCE=""
PERSISTENT_ROOT=""
ACTIVATION_PATH=""
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
        --persistent-root)
            [ "$#" -ge 2 ] || fail "--persistent-root requires a path"
            PERSISTENT_ROOT=$2
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
        --asnumpy-component)
            [ "$#" -ge 2 ] || fail "--asnumpy-component requires a path"
            ASNUMPY_COMPONENT=$2
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
if [ -n "$ASNUMPY_SOURCE" ] && [ -n "$ASNUMPY_COMPONENT" ]; then
    fail "choose exactly one of --asnumpy-component or maintainer-only --asnumpy-source"
fi

command -v python3 >/dev/null 2>&1 || fail "python3 is required"

# A frozen component installs an already-built wheel and does not need a
# compiler toolchain. Keep these checks on explicit source/CANNBot paths so a
# release host is not forced to prepare build tools it will never use.
if [ -n "$ASNUMPY_SOURCE" ] || [ "$WITH_CANNBOT" = true ]; then
    command -v git >/dev/null 2>&1 || fail "git is required for source/CANNBot operations"
fi
if [ -n "$ASNUMPY_SOURCE" ]; then
    command -v g++ >/dev/null 2>&1 || fail "g++ >= 11.2 is required for source builds"

    python3 - <<'PY' || fail "Python >= 3.10 with development headers is required for source builds"
import pathlib
import sys
import sysconfig

if sys.version_info < (3, 10):
    raise SystemExit(1)
include = pathlib.Path(sysconfig.get_path("include"))
if not (include / "Python.h").is_file():
    raise SystemExit(1)
PY
fi

if [ -n "$PERSISTENT_ROOT" ]; then
    if ! PERSISTENT_ROOT=$(python3 - "$PERSISTENT_ROOT" <<'PY'
import os
from pathlib import Path
import sys

raw = sys.argv[1]
if not raw or "\x00" in raw or "\n" in raw or "\r" in raw:
    raise SystemExit(1)
# Walk the unnormalized spelling first.  ``abspath`` would collapse
# ``symlink/..`` even though the kernel follows the symlink before applying
# ``..``; ordinary ``nested/../root`` remains valid.
expanded = os.path.expanduser(raw)
current = Path(os.path.abspath(os.sep)) if os.path.isabs(expanded) else Path(os.getcwd())
for component in expanded.split(os.sep):
    if not component or component == ".":
        continue
    if component == "..":
        current = current.parent
        continue
    candidate = current / component
    if os.path.lexists(candidate) and candidate.is_symlink():
        raise SystemExit(1)
    current = candidate
path = Path(os.path.abspath(expanded))
if path == Path(path.anchor) or path.is_symlink() or (path.exists() and not path.is_dir()):
    raise SystemExit(1)
# A missing root may have several missing parents.  Inspect every existing
# component before mkdir -p so a symlinked ancestor cannot redirect runtime
# state outside the path the caller reviewed.
current = path
while True:
    if os.path.lexists(current) and current.is_symlink():
        raise SystemExit(1)
    if current == current.parent:
        break
    current = current.parent
print(path)
PY
    ); then
        fail "persistent root is invalid or unsafe"
    fi
    mkdir -p "$PERSISTENT_ROOT"
    export NUMPILOT_HOME=$PERSISTENT_ROOT
    export NUMPILOT_STATE_HOME="$PERSISTENT_ROOT/.numpilot-state"
    export NUMPILOT_CONFIG_HOME="$PERSISTENT_ROOT/.numpilot-config"
    export XDG_CACHE_HOME="$PERSISTENT_ROOT/.numpilot-cache"
    export NUMPILOT_CACHE_HOME="$XDG_CACHE_HOME/numpilot"
    export NUMPILOT_ASNUMPY_CACHE="$NUMPILOT_CACHE_HOME/asnumpy"
    export NUMPILOT_CANNBOT_CACHE="$NUMPILOT_CACHE_HOME/cannbot-skills"
    ACTIVATION_PATH="$PERSISTENT_ROOT/activate-numpilot.sh"
fi

if [ -z "$CANN_ENV_FILE" ]; then
    EXISTING_CANN_ROOT=${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-}}
    if [ -n "$EXISTING_CANN_ROOT" ] && [ -f "$EXISTING_CANN_ROOT/set_env.sh" ]; then
        CANN_ENV_FILE="$EXISTING_CANN_ROOT/set_env.sh"
    fi
fi
if [ -z "$CANN_ENV_FILE" ]; then
    for CANDIDATE in \
        /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
        /usr/local/Ascend/cann-9.0.0/set_env.sh; do
        if [ -f "$CANDIDATE" ]; then
            CANN_ENV_FILE=$CANDIDATE
            break
        fi
    done
fi
[ -f "$CANN_ENV_FILE" ] || fail "CANN environment file does not exist: $CANN_ENV_FILE"

_numpilot_installer_is_toolkit_path() {
    local lowered=${1,,}
    case "$lowered" in
        */ascend/cann|*/ascend/cann/*|*/ascend/cann-*|*/ascend/cann-*/*|*/ascend/cann_*|*/ascend/cann_*/*|*/ascend/ascend-toolkit/*) return 0 ;;
    esac
    return 1
}

_numpilot_installer_filter_path() {
    local name=$1 mode=$2 value entry key result= seen= lowered=
    local -a parts
    value=${!name-}
    IFS=: read -r -a parts <<< "$value"
    for entry in "${parts[@]}"; do
        [ -n "$entry" ] || continue
        if _numpilot_installer_is_toolkit_path "$entry"; then
            if [ "$mode" = toolkit ]; then
                continue
            fi
            lowered=${entry,,}
            case "$lowered" in */devlib|*/devlib/*) continue ;; esac
        fi
        key=$entry
        while [ "$key" != / ] && [ "${key%/}" != "$key" ]; do
            key=${key%/}
        done
        case :$seen: in
            *:"$key":*) ;;
            *)
                seen=${seen:+$seen:}$key
                result=${result:+$result:}$entry
                ;;
        esac
    done
    export "$name=$result"
}

for CANN_PATH_VARIABLE in PATH LD_LIBRARY_PATH PYTHONPATH CMAKE_PREFIX_PATH; do
    _numpilot_installer_filter_path "$CANN_PATH_VARIABLE" toolkit
done
# shellcheck disable=SC1090
. "$CANN_ENV_FILE"
_numpilot_installer_filter_path LD_LIBRARY_PATH devlib
unset CANN_PATH_VARIABLE EXISTING_CANN_ROOT
unset -f _numpilot_installer_is_toolkit_path _numpilot_installer_filter_path

CANN_ROOT=${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-}}
[ -n "$CANN_ROOT" ] && [ -d "$CANN_ROOT" ] || fail "source a valid CANN set_env.sh first"
[ -d "${ASCEND_OPP_PATH:-}" ] || fail "ASCEND_OPP_PATH is missing or invalid"
[ -f "$CANN_ROOT/include/acl/acl.h" ] || fail "CANN ACL headers are missing under $CANN_ROOT/include"
[ -f "$CANN_ROOT/lib64/libascendcl.so" ] || fail "CANN runtime library libascendcl.so is missing"
[ -f "$CANN_ENV_FILE" ] || fail "fixed CANN set_env.sh is missing: $CANN_ENV_FILE"

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

if [ -n "$ASNUMPY_SOURCE" ]; then
    find_cmake_package 'fmt*config.cmake' || fail \
        "system fmt CMake package is missing (Ubuntu package: libfmt-dev); NumPilot will not run sudo"
    find_cmake_package 'spdlog*config.cmake' || fail \
        "system spdlog CMake package is missing (Ubuntu package: libspdlog-dev); NumPilot will not run sudo"
fi

if ! VENV_DIR=$(python3 - "$VENV_DIR" "$SCRIPT_DIR" <<'PY'
import os
from pathlib import Path
import stat
import sys

raw_path, repository = sys.argv[1:]
if not raw_path or "\x00" in raw_path or "\n" in raw_path or "\r" in raw_path:
    raise SystemExit("venv path is empty or contains unsafe characters")

# Check raw components before normalizing so a path such as
# ``symlink/../venv`` cannot evade the symlink policy.
expanded = os.path.expanduser(raw_path)
current = Path(os.path.abspath(os.sep)) if os.path.isabs(expanded) else Path(os.getcwd())
for component in expanded.split(os.sep):
    if not component or component == ".":
        continue
    if component == "..":
        current = current.parent
        continue
    candidate = current / component
    if os.path.lexists(candidate) and candidate.is_symlink():
        raise SystemExit("venv path must not traverse a symlink")
    current = candidate
path = Path(os.path.abspath(expanded))
repository_path = Path(repository).resolve()
for forbidden in (
    Path("/bin"),
    Path("/etc"),
    Path("/lib"),
    Path("/lib64"),
    Path("/opt"),
    Path("/sbin"),
    Path("/usr"),
    Path("/var"),
):
    if path == forbidden or forbidden in path.parents:
        raise SystemExit("venv must not be created in a system hierarchy")
if path == Path(path.anchor) or path == repository_path:
    raise SystemExit("venv path is too broad")

current = path.parent
while current != Path(current.anchor):
    if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
        raise SystemExit("venv path must not traverse a symlink")
    current = current.parent

parent = path.parent
try:
    parent_status = parent.lstat()
except FileNotFoundError:
    raise SystemExit("venv parent must already exist")
if stat.S_ISLNK(parent_status.st_mode) or not stat.S_ISDIR(parent_status.st_mode):
    raise SystemExit("venv parent must be a real directory")
if parent_status.st_uid != os.getuid() or parent_status.st_mode & 0o022:
    raise SystemExit("venv parent must be user-owned and not group/world writable")

if os.path.lexists(path):
    target_status = path.lstat()
    if stat.S_ISLNK(target_status.st_mode) or not stat.S_ISDIR(target_status.st_mode):
        raise SystemExit("existing venv path must be a real directory")
    if target_status.st_uid != os.getuid() or target_status.st_mode & 0o022:
        raise SystemExit("existing venv must be user-owned and not group/world writable")
print(path)
PY
); then
    fail "venv path failed the private non-system safety policy"
fi

if [ -e "$VENV_DIR" ] && [ ! -x "$VENV_DIR/bin/python" ]; then
    fail "refusing to overwrite a non-venv path: $VENV_DIR"
fi
if [ ! -x "$VENV_DIR/bin/python" ]; then
    python3 -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" - "$VENV_DIR" <<'PY' || fail "--venv must name a genuine isolated virtual environment"
from pathlib import Path
import sys

expected = Path(sys.argv[1]).resolve()
configuration = expected / "pyvenv.cfg"
if (
    sys.prefix == sys.base_prefix
    or Path(sys.prefix).resolve() != expected
    or not configuration.is_file()
    or configuration.is_symlink()
):
    raise SystemExit(1)
PY
PIP_ARGS=(install --disable-pip-version-check)
if [ -n "$PIP_INDEX_URL_OVERRIDE" ]; then
    PIP_ARGS+=(--index-url "$PIP_INDEX_URL_OVERRIDE")
fi
# The frozen component path only needs NumPilot's runtime dependencies.  The
# heavier build extras (cmake/scikit-build/pybind11/build) are installed only
# for an explicit maintainer source build.
if [ -n "$ASNUMPY_SOURCE" ]; then
    "$VENV_PYTHON" -m pip "${PIP_ARGS[@]}" -e "$SCRIPT_DIR[bootstrap]"
else
    "$VENV_PYTHON" -m pip "${PIP_ARGS[@]}" -e "$SCRIPT_DIR"
fi

# Console tools installed into the project venv (especially cmake) must win over
# older system copies during bootstrap preflight and the subsequent source build.
export PATH="$VENV_DIR/bin:$PATH"

if [ -z "$ASNUMPY_COMPONENT" ]; then
    ASNUMPY_COMPONENT="$SCRIPT_DIR/components/asnumpy"
fi
BOOTSTRAP_ARGS=(--asnumpy-component "$ASNUMPY_COMPONENT")
if [ -n "$ASNUMPY_SOURCE" ]; then
    # Source builds remain a maintainer-only escape hatch.  The normal
    # installer never selects this path and therefore never fetches source.
    BOOTSTRAP_ARGS=(--build-asnumpy --asnumpy-cache "$ASNUMPY_SOURCE")
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

# Check the original spelling before ``abspath`` normalizes ``..``.  A
# symlinked component is unsafe even if the normalized path appears local.
expanded = os.path.expanduser(raw_path)
current = Path(os.path.abspath(os.sep)) if os.path.isabs(expanded) else Path(os.getcwd())
for component in expanded.split(os.sep):
    if not component or component == ".":
        continue
    if component == "..":
        current = current.parent
        continue
    candidate = current / component
    if os.path.lexists(candidate) and candidate.is_symlink():
        raise SystemExit("receipt path must not traverse a symlink")
    current = candidate
receipt = Path(os.path.abspath(os.path.expanduser(raw_path)))
parent = receipt.parent
explicit = explicit_text == "true"
current_uid = os.getuid()

# Validate the complete existing ancestor chain, not only receipt.parent.  A
# symlink several levels up would otherwise let the later default mkdir/write
# escape the intended private state root.
current = parent
while True:
    if os.path.lexists(current) and current.is_symlink():
        raise SystemExit("receipt path must not traverse a symlink")
    if current == current.parent:
        break
    current = current.parent

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
if [ -z "$ACTIVATION_PATH" ]; then
    ACTIVATION_PATH=$(
        "$VENV_PYTHON" -m numpilot paths --get state
    )/activate-numpilot.sh
fi
ACTIVATION_ARGS=(
    --destination "$ACTIVATION_PATH"
    --venv "$VENV_DIR"
    --cann-env "$CANN_ENV_FILE"
    --asnumpy-component "$ASNUMPY_COMPONENT"
    --receipt "$RECEIPT_PATH"
)
if [ -n "$PERSISTENT_ROOT" ]; then
    ACTIVATION_ARGS+=(--persistent-root "$PERSISTENT_ROOT")
fi
if [ -n "$CANNBOT_SOURCE" ]; then
    ACTIVATION_ARGS+=(--cannbot-checkout "$CANNBOT_SOURCE")
fi
"$VENV_PYTHON" -m numpilot.activation "${ACTIVATION_ARGS[@]}"
# Use the exact same credential-free environment that later shells receive.
# shellcheck disable=SC1090
. "$ACTIVATION_PATH"
hash -r

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
printf '[NumPilot install] activate: source %q\n' "$ACTIVATION_PATH"
printf '[NumPilot install] receipt: %q\n' "$RECEIPT_PATH"
