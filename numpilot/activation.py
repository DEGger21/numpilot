"""Generate a credential-free, repeatable Bash activation script."""

import os
from pathlib import Path
import shlex
import stat
import tempfile
from typing import Optional

from numpilot.paths import cache_home, config_home, state_home


class ActivationError(RuntimeError):
    """An activation target or fixed runtime input is unsafe."""


def _absolute(path: Path, label: str, *, must_be_file: bool = False) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    value = Path(os.path.abspath(value))
    if must_be_file and (value.is_symlink() or not value.is_file()):
        raise ActivationError("%s must be a regular file: %s" % (label, value))
    return value


def _shell_export(name: str, value: Path) -> str:
    return "export %s=%s" % (name, shlex.quote(str(value)))


def write_activation_script(
    *,
    destination: Path,
    persistent_root: Optional[Path],
    venv_dir: Path,
    cann_env_file: Path,
    asnumpy_cache: Path,
    asnumpy_component: Optional[Path] = None,
    cannbot_cache: Path,
    receipt_path: Path,
    cannbot_checkout: Optional[Path] = None,
) -> Path:
    """Atomically write a fixed, secret-free Bash activation script.

    ``asnumpy_cache`` and ``cannbot_cache`` are cache bases.  The latter is
    combined with NumPilot's pinned commit for the exact Skill checkout.
    """

    from numpilot.cannbot_skills import CANNBOT_COMMIT

    target = _absolute(destination, "activation destination")
    venv = _absolute(venv_dir, "virtual environment")
    activate = _absolute(
        venv / "bin" / "activate", "virtual environment activation", must_be_file=True
    )
    cann_env = _absolute(cann_env_file, "CANN environment", must_be_file=True)
    cann_root = cann_env.parent
    if (
        cann_env.name != "set_env.sh"
        or not (cann_root / "include" / "acl" / "acl.h").is_file()
        or not (cann_root / "lib64" / "libascendcl.so").is_file()
    ):
        raise ActivationError(
            "CANN environment must be the set_env.sh of a complete toolkit root"
        )
    asnumpy_base = _absolute(asnumpy_cache, "AsNumpy cache")
    if asnumpy_component is None:
        from numpilot.paths import asnumpy_component_dir

        asnumpy_component = asnumpy_component_dir()
    asnumpy_component_path = _absolute(
        asnumpy_component, "AsNumpy internal component"
    )
    cannbot_base = _absolute(cannbot_cache, "CANNBot cache")
    cannbot_exact = (
        _absolute(cannbot_checkout, "CANNBot checkout")
        if cannbot_checkout is not None
        else cannbot_base / CANNBOT_COMMIT
    )
    receipt = _absolute(receipt_path, "bootstrap receipt")

    if persistent_root is not None:
        root = _absolute(persistent_root, "persistent root")
        state = root / ".numpilot-state"
        config = root / ".numpilot-config"
        xdg_cache = root / ".numpilot-cache"
        app_cache = xdg_cache / "numpilot"
    else:
        root = None
        state = _absolute(state_home(), "state home")
        config = _absolute(config_home(), "config home")
        app_cache = _absolute(cache_home(), "NumPilot cache home")
        xdg_value = os.getenv("XDG_CACHE_HOME")
        xdg_cache = _absolute(
            Path(xdg_value).expanduser() if xdg_value else Path.home() / ".cache",
            "XDG cache home",
        )

    exports = []
    if root is not None:
        exports.append(_shell_export("NUMPILOT_HOME", root))
    exports.extend((
        _shell_export("NUMPILOT_STATE_HOME", state),
        _shell_export("NUMPILOT_CONFIG_HOME", config),
        _shell_export("XDG_CACHE_HOME", xdg_cache),
        _shell_export("NUMPILOT_CACHE_HOME", app_cache),
        _shell_export("NUMPILOT_BOOTSTRAP_RECEIPT", receipt),
        _shell_export("NUMPILOT_ASNUMPY_CACHE", asnumpy_base),
        _shell_export("NUMPILOT_ASNUMPY_COMPONENT_DIR", asnumpy_component_path),
        _shell_export("NUMPILOT_CANNBOT_CACHE", cannbot_base),
        _shell_export("NUMPILOT_CANNBOT_SKILLS", cannbot_exact),
    ))

    source = "\n".join((
        "# NumPilot generated activation; source this file from Bash.",
        "# Contains only fixed local paths and no remote-service configuration.",
        "",
        *exports,
        "",
        "_numpilot_is_toolkit_path() {",
        "    local _np_lower=${1,,}",
        "    case \"$_np_lower\" in",
        "        */ascend/cann|*/ascend/cann/*|*/ascend/cann-*|*/ascend/cann-*/*|*/ascend/cann_*|*/ascend/cann_*/*|*/ascend/ascend-toolkit/*) return 0 ;;",
        "    esac",
        "    return 1",
        "}",
        "",
        "_numpilot_filter_old_toolkit() {",
        "    local _np_name=$1 _np_value _np_entry _np_result=",
        "    local -a _np_parts",
        "    _np_value=${!_np_name-}",
        "    IFS=: read -r -a _np_parts <<< \"$_np_value\"",
        "    for _np_entry in \"${_np_parts[@]}\"; do",
        "        [ -n \"$_np_entry\" ] || continue",
        "        if _numpilot_is_toolkit_path \"$_np_entry\"; then continue; fi",
        "        case :$_np_result: in",
        "            *:\"$_np_entry\":*) ;;",
        "            *) _np_result=${_np_result:+$_np_result:}$_np_entry ;;",
        "        esac",
        "    done",
        "    export \"$_np_name=$_np_result\"",
        "}",
        "",
        "_numpilot_remove_cann_devlib() {",
        "    local _np_name=$1 _np_value _np_entry _np_result=",
        "    local -a _np_parts",
        "    _np_value=${!_np_name-}",
        "    IFS=: read -r -a _np_parts <<< \"$_np_value\"",
        "    for _np_entry in \"${_np_parts[@]}\"; do",
        "        [ -n \"$_np_entry\" ] || continue",
        "        case \"$_np_entry\" in",
        "            " + shlex.quote(str(cann_root / "devlib")) + "|" +
        shlex.quote(str(cann_root / "devlib")) + "/*) continue ;;",
        "        esac",
        "        case \"${_np_entry,,}\" in",
        "            */devlib|*/devlib/*)",
        "                if _numpilot_is_toolkit_path \"$_np_entry\"; then continue; fi ;;",
        "        esac",
        "        case :$_np_result: in",
        "            *:\"$_np_entry\":*) ;;",
        "            *) _np_result=${_np_result:+$_np_result:}$_np_entry ;;",
        "        esac",
        "    done",
        "    export \"$_np_name=$_np_result\"",
        "}",
        "",
        "_numpilot_dedupe() {",
        "    local _np_name=$1 _np_value _np_entry _np_key _np_result= _np_seen=",
        "    local -a _np_parts",
        "    _np_value=${!_np_name-}",
        "    IFS=: read -r -a _np_parts <<< \"$_np_value\"",
        "    for _np_entry in \"${_np_parts[@]}\"; do",
        "        [ -n \"$_np_entry\" ] || continue",
        "        _np_key=$_np_entry",
        "        while [ \"$_np_key\" != / ] && [ \"${_np_key%/}\" != \"$_np_key\" ]; do",
        "            _np_key=${_np_key%/}",
        "        done",
        "        case :$_np_seen: in",
        "            *:\"$_np_key\":*) ;;",
        "            *)",
        "                _np_seen=${_np_seen:+$_np_seen:}$_np_key",
        "                _np_result=${_np_result:+$_np_result:}$_np_entry",
        "                ;;",
        "        esac",
        "    done",
        "    export \"$_np_name=$_np_result\"",
        "}",
        "",
        "for _numpilot_path_name in PATH LD_LIBRARY_PATH PYTHONPATH CMAKE_PREFIX_PATH; do",
        "    _numpilot_filter_old_toolkit \"$_numpilot_path_name\"",
        "done",
        ". " + shlex.quote(str(cann_env)),
        "_numpilot_remove_cann_devlib LD_LIBRARY_PATH",
        ". " + shlex.quote(str(activate)),
        "for _numpilot_path_name in PATH LD_LIBRARY_PATH PYTHONPATH CMAKE_PREFIX_PATH; do",
        "    _numpilot_dedupe \"$_numpilot_path_name\"",
        "done",
        "unset _numpilot_path_name",
        "unset -f _numpilot_is_toolkit_path _numpilot_filter_old_toolkit _numpilot_remove_cann_devlib _numpilot_dedupe",
        "",
    ))

    if os.path.lexists(target) and target.is_symlink():
        raise ActivationError("activation destination must not be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ActivationError("activation parent must be a real directory")

    descriptor = -1
    temporary_name = ""
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".%s." % target.name,
            suffix=".tmp",
            dir=str(target.parent),
        )
        os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(source)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        temporary_name = ""
        os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as exc:
        raise ActivationError("cannot write activation script: %s" % type(exc).__name__) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
    return target


def main(argv=None) -> int:
    """CLI used by the source-first installer after all verification gates."""

    import argparse

    from numpilot.paths import (
        asnumpy_cache_base,
        asnumpy_component_dir,
        cannbot_cache_base,
        default_bootstrap_receipt,
    )

    parser = argparse.ArgumentParser(
        description="write a credential-free NumPilot activation script"
    )
    parser.add_argument("--destination", required=True)
    parser.add_argument("--persistent-root")
    parser.add_argument("--venv", required=True)
    parser.add_argument("--cann-env", required=True)
    parser.add_argument("--asnumpy-cache")
    parser.add_argument("--asnumpy-component")
    parser.add_argument("--cannbot-cache")
    parser.add_argument("--cannbot-checkout")
    parser.add_argument("--receipt")
    args = parser.parse_args(argv)
    written = write_activation_script(
        destination=Path(args.destination),
        persistent_root=(
            Path(args.persistent_root) if args.persistent_root else None
        ),
        venv_dir=Path(args.venv),
        cann_env_file=Path(args.cann_env),
        asnumpy_cache=(
            Path(args.asnumpy_cache)
            if args.asnumpy_cache
            else asnumpy_cache_base()
        ),
        asnumpy_component=(
            Path(args.asnumpy_component)
            if args.asnumpy_component
            else asnumpy_component_dir()
        ),
        cannbot_cache=(
            Path(args.cannbot_cache)
            if args.cannbot_cache
            else cannbot_cache_base()
        ),
        receipt_path=(
            Path(args.receipt) if args.receipt else default_bootstrap_receipt()
        ),
        cannbot_checkout=(
            Path(args.cannbot_checkout) if args.cannbot_checkout else None
        ),
    )
    print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
