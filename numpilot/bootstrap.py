"""Read-only planning and explicit bootstrap for the NumPilot NPU demo.

Nothing in this module runs at package installation or import time.  Planning
never fetches; applying requires an explicit AsNumpy component/wheel/source choice
and an explicit CANNBot license decision.
"""

from __future__ import annotations

import base64
import binascii
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import Parser
import hashlib
import hmac
import io
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from zipfile import BadZipFile, ZipFile

from numpilot import __version__
from numpilot.asnumpy_component import (
    AsNumpyComponent,
    AsNumpyComponentError,
    component_receipt_identity,
    load_component_manifest,
    resolve_component_wheel,
    verify_component_attachments,
)
from numpilot.cannbot_skills import (
    CANNBOT_COMMIT,
    CANNBOT_LICENSE,
    CANNBOT_REPOSITORY,
    default_skill_checkout,
    fetch_pinned_skill,
    load_skill_bundle,
)


ASNUMPY_REPOSITORY = "https://gitcode.com/cann/asnumpy.git"
ASNUMPY_COMMIT = "1e7177f12039b2d87e3cdff6693459d895744f68"
ASNUMPY_TREE = "a5fb1f3a70be9fc17ecf0dd928776924d4c78e1e"
ASNUMPY_VERSION = "0.4.1.dev12+gc9115e8ce.d20261031"
ASNUMPY_LICENSE = "Apache-2.0"

DEMO_P0_WHEEL_SHA256 = (
    "afe845c45d17e7f5cd619831f555fba3c2344296ffe02f1d251017b04cf0282d"
)
_AUDITED_WHEEL_PROFILES = {
    DEMO_P0_WHEEL_SHA256: {
        "profile_id": "demo-v1-p0-move-cp310-aarch64",
        "version": "0.1.dev106+g21368abf9.d20260822",
        "required_tag": "cp310-cp310-linux_aarch64",
        "runtime_constraints": {
            "python_implementation": "CPython",
            "python_major_minor": "3.10",
            "machine": "aarch64",
            "packages": {"numpy": "1.26.4", "loguru": "0.7.3"},
        },
        "base_repository": "https://gitcode.com/li-xingHIT/asnumpyoplogin.git",
        "base_commit": "21368abf95f578f40ea82dc87f6f3b11ae74e45b",
        "exact_source_tree_claimed": False,
        "capability_audit_id": (
            "asnumpy-0.1.dev106-g21368abf9-poisson-move-"
            "init4f4e8083-array16c901f5-math62e0ba09-direct4479c588-"
            "utils1b298145-core4465ce6e-v3"
        ),
        "qualification_scope": "profile-bound; not a global AsNumpy capability claim",
    }
}

CANNBOT_ASCENDC_SKILL_CATALOG = (
    {
        "phase": "工程模板",
        "name": "ascendc-registry-invoke-template",
        "path": "ops/ascendc-registry-invoke-template/SKILL.md",
    },
    {
        "phase": "生成流程",
        "name": "ops-registry-invoke-workflow",
        "path": "plugins-official/ops-registry-invoke/workflow/SKILL.md",
    },
    {
        "phase": "API 与编译修复",
        "name": "ascendc-api-best-practices",
        "path": "ops/ascendc-api-best-practices/SKILL.md",
    },
    {
        "phase": "单元测试",
        "name": "ascendc-ut-develop",
        "path": "ops/ascendc-ut-develop/SKILL.md",
    },
    {
        "phase": "系统测试",
        "name": "ascendc-st-design",
        "path": "ops/ascendc-st-design/SKILL.md",
    },
    {
        "phase": "精度调试",
        "name": "ascendc-precision-debug",
        "path": "ops/ascendc-precision-debug/SKILL.md",
    },
    {
        "phase": "运行时调试",
        "name": "ascendc-runtime-debug",
        "path": "ops/ascendc-runtime-debug/SKILL.md",
    },
    {
        "phase": "性能优化",
        "name": "ascendc-perf-optimize",
        "path": "ops/ascendc-perf-optimize/SKILL.md",
    },
)

_ASNUMPY_README_SMOKE = """\
import hashlib
from importlib import metadata
import json
from pathlib import Path
import asnumpy as ap
arr = ap.ones((1000, 1000), dtype=ap.float32)
if tuple(arr.shape) != (1000, 1000):
    raise RuntimeError("AsNumpy README smoke returned the wrong shape")
if arr.dtype != ap.float32:
    raise RuntimeError("AsNumpy README smoke returned the wrong dtype")
module = Path(ap.__file__).resolve()
native = []
for path in sorted(module.parent.glob("*.so")):
    native.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
payload = {
    "asnumpy_version": metadata.version("asnumpy"),
    "numpy_version": metadata.version("numpy"),
    "loguru_version": metadata.version("loguru"),
    "module_path": str(module),
    "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
    "native_modules": native,
    "shape": [1000, 1000],
    "dtype": str(arr.dtype),
}
print("NUMPILOT_ASNUMPY_SMOKE=" + json.dumps(payload, sort_keys=True))
"""

_RUNTIME_PACKAGES = {
    "numpy": (1, 26, 0),
    "loguru": (0, 7, 0),
}
_SOURCE_BUILD_PACKAGES = {
    "numpy": (2, 0, 0),
    "build": (1, 0, 0),
    "scikit-build-core": (1, 0, 0),
    "pybind11": (3, 0, 0),
    "setuptools-scm": (9, 2, 2),
    "towncrier": (24, 0, 0),
}

_KEY_FILES = (
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "CMakeLists.txt",
    "bindings/python/CMakeLists.txt",
    "csrc/CMakeLists.txt",
)
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_VERSION = re.compile(r"(?<![\w.])(\d+)\.(\d+)(?:\.(\d+))?(?![\w.])")
_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_WHEEL_FILES = 4096
_MAX_WHEEL_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_MAX_OUTPUT = 4096
_COMMAND_TIMEOUT = 120
_BUILD_TIMEOUT = 30 * 60
_CANN_VERSION_MAX_BYTES = 64 * 1024
_CANN_VERSION_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}$")

Runner = Callable[..., Any]
Finder = Callable[[str], Optional[str]]
VersionReader = Callable[[str], str]


class BootstrapError(RuntimeError):
    """Bootstrap input, provenance, environment, or execution failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class AsNumpyCheckout:
    root: str
    repository: str
    commit: str
    tree: str
    license: str
    files_sha256: Mapping[str, str]

    def manifest(self) -> Dict[str, Any]:
        return {
            "repository": self.repository,
            "commit": self.commit,
            "tree": self.tree,
            "license": self.license,
            "files_sha256": dict(sorted(self.files_sha256.items())),
        }


def _canonical_repo(value: str) -> str:
    return value.strip().rstrip("/").removesuffix(".git")


def _run(
    arguments: Sequence[str],
    *,
    runner: Optional[Runner] = None,
    cwd: Optional[Path] = None,
    timeout: int = _COMMAND_TIMEOUT,
    environment: Optional[Mapping[str, str]] = None,
) -> str:
    command = [str(item) for item in arguments]
    if not command or any("\x00" in item for item in command):
        raise BootstrapError("invalid subprocess argument list")
    options: Dict[str, Any] = {
        "check": False,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "timeout": timeout,
    }
    if cwd is not None:
        options["cwd"] = str(cwd)
    if environment is not None:
        options["env"] = dict(environment)
    try:
        result = (runner or subprocess.run)(command, **options)
    except subprocess.TimeoutExpired as exc:
        raise BootstrapError("subprocess timed out: %s" % command[0]) from exc
    except OSError as exc:
        raise BootstrapError("cannot execute subprocess: %s" % command[0]) from exc
    code = int(getattr(result, "returncode", 0))
    if code:
        # Never copy command output into a receipt: tools may echo inherited
        # proxy URLs or credentials.
        raise BootstrapError(
            "subprocess failed: %s (exit=%d)" % (Path(command[0]).name, code)
        )
    output = getattr(result, "stdout", "") or ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    return str(output)[:_MAX_OUTPUT].strip()


def _git(root: Path, *args: str, runner: Optional[Runner] = None) -> str:
    return _run(["git", "-C", str(root), *args], runner=runner)


def default_asnumpy_checkout(env: Optional[Mapping[str, str]] = None) -> Path:
    from numpilot.paths import asnumpy_cache_base

    return asnumpy_cache_base(env) / ASNUMPY_COMMIT


def _default_cannbot_checkout(env: Mapping[str, str]) -> Path:
    return default_skill_checkout(env)


def load_asnumpy_checkout(
    root: Path,
    *,
    expected_repository: str = ASNUMPY_REPOSITORY,
    expected_commit: str = ASNUMPY_COMMIT,
    expected_tree: Optional[str] = None,
    key_files: Tuple[str, ...] = _KEY_FILES,
    runner: Optional[Runner] = None,
) -> AsNumpyCheckout:
    """Validate a real, exact, clean AsNumpy checkout without networking."""

    checkout = Path(root).expanduser()
    if checkout.is_symlink() or not checkout.is_dir():
        raise BootstrapError("AsNumpy checkout is not a real directory: %s" % checkout)
    commit = _git(checkout, "rev-parse", "HEAD", runner=runner)
    if commit != expected_commit:
        raise BootstrapError(
            "AsNumpy commit mismatch: expected %s, got %s" % (expected_commit, commit)
        )
    origin = _git(checkout, "remote", "get-url", "origin", runner=runner)
    if _canonical_repo(origin) != _canonical_repo(expected_repository):
        raise BootstrapError("AsNumpy origin is not the pinned official repository")
    if _git(
        checkout,
        "status",
        "--porcelain",
        "--untracked-files=no",
        runner=runner,
    ):
        raise BootstrapError("AsNumpy checkout has dirty tracked files")
    _git(
        checkout,
        "ls-files",
        "--error-unmatch",
        "--",
        *key_files,
        runner=runner,
    )

    digests: Dict[str, str] = {}
    for relative in key_files:
        candidate = checkout.joinpath(*Path(relative).parts)
        if candidate.is_symlink() or not candidate.is_file():
            raise BootstrapError("required AsNumpy source file is missing: %s" % relative)
        try:
            payload = candidate.read_bytes()
        except OSError as exc:
            raise BootstrapError("cannot read AsNumpy source file: %s" % relative) from exc
        if len(payload) > _MAX_FILE_BYTES:
            raise BootstrapError("AsNumpy source file exceeds size limit: %s" % relative)
        digests[relative] = hashlib.sha256(payload).hexdigest()

    tree = _git(checkout, "rev-parse", "HEAD^{tree}", runner=runner)
    tree_gate = expected_tree
    if tree_gate is None and expected_commit == ASNUMPY_COMMIT:
        tree_gate = ASNUMPY_TREE
    if tree_gate is not None and tree != tree_gate:
        raise BootstrapError(
            "AsNumpy tree mismatch: expected %s, got %s" % (tree_gate, tree)
        )
    return AsNumpyCheckout(
        root=str(checkout.resolve()),
        repository=ASNUMPY_REPOSITORY,
        commit=commit,
        tree=tree,
        license=ASNUMPY_LICENSE,
        files_sha256=digests,
    )


def fetch_asnumpy_checkout(
    destination: Optional[Path] = None,
    *,
    runner: Optional[Runner] = None,
) -> AsNumpyCheckout:
    """Fetch only the audited commit and atomically publish its verified cache."""

    target = Path(destination).expanduser() if destination else default_asnumpy_checkout()
    if target.exists() or target.is_symlink():
        return load_asnumpy_checkout(target, runner=runner)
    parent = target.parent
    if parent.is_symlink():
        raise BootstrapError("AsNumpy cache parent must not be a symlink")
    try:
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            os.chmod(parent, 0o700)
    except OSError as exc:
        raise BootstrapError("cannot create private AsNumpy cache") from exc

    temporary = Path(tempfile.mkdtemp(prefix=".asnumpy-", dir=str(parent)))
    try:
        _git(temporary, "init", runner=runner)
        _git(
            temporary,
            "remote",
            "add",
            "origin",
            ASNUMPY_REPOSITORY,
            runner=runner,
        )
        _git(
            temporary,
            "fetch",
            "--depth",
            "1",
            "origin",
            ASNUMPY_COMMIT,
            runner=runner,
        )
        _git(temporary, "checkout", "--detach", "FETCH_HEAD", runner=runner)
        load_asnumpy_checkout(temporary, runner=runner)
        try:
            os.replace(temporary, target)
        except OSError:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(temporary, ignore_errors=True)
                return load_asnumpy_checkout(target, runner=runner)
            raise
        return load_asnumpy_checkout(target, runner=runner)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parse_version(value: str) -> Optional[Tuple[int, int, int]]:
    match = _VERSION.search(value)
    if not match:
        return None
    return tuple(int(part or 0) for part in match.groups())  # type: ignore[return-value]


def _tool(
    name: str,
    minimum: Tuple[int, int, int],
    runner: Optional[Runner],
    finder: Finder,
) -> Dict[str, Any]:
    executable = finder(name)
    report: Dict[str, Any] = {
        "path": executable,
        "version": None,
        "minimum": ".".join(map(str, minimum)),
        "ok": False,
    }
    if not executable:
        return report
    try:
        parsed = _parse_version(_run([executable, "--version"], runner=runner, timeout=15))
    except BootstrapError:
        return report
    report["version"] = ".".join(map(str, parsed)) if parsed else None
    report["ok"] = bool(parsed and parsed >= minimum)
    return report


def _installed_version(reader: VersionReader) -> Optional[str]:
    try:
        return str(reader("asnumpy"))
    except (metadata.PackageNotFoundError, LookupError, ValueError):
        return None


def _python_package(
    name: str,
    minimum: Tuple[int, int, int],
    reader: VersionReader,
) -> Dict[str, Any]:
    try:
        value = str(reader(name))
    except (metadata.PackageNotFoundError, LookupError, ValueError):
        value = None
    parsed = _parse_version(value) if value else None
    return {
        "version": value,
        "minimum": ".".join(map(str, minimum)),
        "ok": bool(parsed and parsed >= minimum),
    }


def _cache(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_symlink": path.is_symlink(),
        "is_directory": path.is_dir() and not path.is_symlink(),
    }


def _read_cann_version_file(path: Path) -> Optional[str]:
    """Read a conventional CANN ``version.info`` without importing CANN.

    CANN installations have used both ``Version=...`` and ``version=...`` in
    different package metadata files.  This deliberately reads only a small
    bounded prefix and accepts a conservative token, so a malformed or
    user-controlled file cannot become a runtime identity by accident.
    """

    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _CANN_VERSION_MAX_BYTES:
            return None
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()[:128]
    except (OSError, UnicodeError):
        return None
    for line in lines:
        key, separator, value = line.partition("=")
        if not separator or key.strip().lower() != "version":
            continue
        value = value.strip().strip("\"'")
        if _CANN_VERSION_VALUE.fullmatch(value):
            return value
    return None


def _detect_cann_version(
    environment: Mapping[str, str],
    toolkit: Optional[Path],
    opp: Optional[Path],
) -> Dict[str, Any]:
    """Return a side-effect-free CANN version observation for a plan.

    The supplied environment mapping is used instead of reading process
    globals, which keeps plans deterministic in tests and in CI adapters.  A
    disagreement between metadata files is represented explicitly and is
    treated as a failure by the component runtime gate.
    """

    observations: list[tuple[str, str]] = []
    env_value = str(environment.get("CANN_VERSION", "")).strip()
    if env_value.lower() not in {"", "unknown", "auto"} and _CANN_VERSION_VALUE.fullmatch(env_value):
        observations.append(("CANN_VERSION", env_value))

    candidates: list[tuple[str, Optional[Path]]] = [
        ("toolkit/version.info", toolkit / "version.info" if toolkit else None),
        ("toolkit/opp/version.info", toolkit / "opp" / "version.info" if toolkit else None),
        ("opp/version.info", opp / "version.info" if opp else None),
        (
            "toolkit/aarch64-linux/ascend_toolkit_install.info",
            toolkit / "aarch64-linux" / "ascend_toolkit_install.info" if toolkit else None,
        ),
        (
            "toolkit/x86_64-linux/ascend_toolkit_install.info",
            toolkit / "x86_64-linux" / "ascend_toolkit_install.info" if toolkit else None,
        ),
    ]
    for source, path in candidates:
        if path is None:
            continue
        value = _read_cann_version_file(path)
        if value is not None:
            observations.append((source, value))

    values = sorted({value for _source, value in observations})
    return {
        "version": values[0] if len(values) == 1 else None,
        "version_sources": [source for source, _value in observations],
        "version_values": values,
        "version_conflict": len(values) > 1,
    }


def build_bootstrap_plan(
    *,
    asnumpy_component: Optional[Path] = None,
    asnumpy_wheel: Optional[Path] = None,
    expected_wheel_sha256: Optional[str] = None,
    build_asnumpy: bool = False,
    use_cannbot: bool = False,
    accept_cannbot_license: bool = False,
    asnumpy_cache: Optional[Path] = None,
    cannbot_cache: Optional[Path] = None,
    environment: Optional[Mapping[str, str]] = None,
    runner: Optional[Runner] = None,
    tool_finder: Optional[Finder] = None,
    version_reader: Optional[VersionReader] = None,
) -> Dict[str, Any]:
    """Return a JSON-safe local plan.  It never clones, fetches, builds, or installs."""

    env = dict(os.environ if environment is None else environment)
    finder = tool_finder or shutil.which
    source_cache = Path(asnumpy_cache) if asnumpy_cache else default_asnumpy_checkout(env)
    skill_cache = (
        Path(cannbot_cache) if cannbot_cache else _default_cannbot_checkout(env)
    )
    source_cached = source_cache.is_dir() and not source_cache.is_symlink()
    skill_cached = skill_cache.is_dir() and not skill_cache.is_symlink()
    prefix = str(sys.prefix)
    base_prefix = str(getattr(sys, "base_prefix", sys.prefix))
    toolkit_value = env.get("ASCEND_TOOLKIT_HOME") or env.get("ASCEND_HOME_PATH")
    opp_value = env.get("ASCEND_OPP_PATH")
    toolkit = Path(toolkit_value).expanduser() if toolkit_value else None
    opp = Path(opp_value).expanduser() if opp_value else None
    cann_version = _detect_cann_version(env, toolkit, opp)
    npu_smi = finder("npu-smi")
    compiler = "g++" if finder("g++") else "gcc"
    package_reader = version_reader or metadata.version
    installed = _installed_version(package_reader)

    component_selected = asnumpy_component is not None
    wheel_selected = asnumpy_wheel is not None or expected_wheel_sha256 is not None
    component: Optional[AsNumpyComponent] = None
    component_error: Optional[str] = None
    if component_selected:
        try:
            component = load_component_manifest(asnumpy_component)
        except AsNumpyComponentError as exc:
            component_error = str(exc)
    requested_profile = (
        _AUDITED_WHEEL_PROFILES.get(expected_wheel_sha256.lower())
        if expected_wheel_sha256 and _SHA256.fullmatch(expected_wheel_sha256)
        else None
    )
    expected_installed_version = (
        component.version
        if component is not None
        else requested_profile["version"]
        if requested_profile is not None
        else ASNUMPY_VERSION if build_asnumpy else None
    )
    issues = []
    if component_selected and (wheel_selected or build_asnumpy):
        issues.append(
            "select exactly one AsNumpy mode: internal component, local wheel, or pinned source build"
        )
    if component_selected and component_error:
        issues.append("internal AsNumpy component is invalid: %s" % component_error)
    if component_selected and component is not None:
        try:
            resolve_component_wheel(component)
            verify_component_attachments(component)
        except AsNumpyComponentError as exc:
            issues.append(str(exc))
    if wheel_selected and (asnumpy_wheel is None or expected_wheel_sha256 is None):
        issues.append("local wheel and expected SHA-256 must be supplied together")
    # An internal component is already a complete, independently validated
    # installation mode.  Do not apply the legacy wheel-vs-source XOR check
    # to it; otherwise ``--plan --asnumpy-component`` is rejected before the
    # component's own manifest/wheel checks can run.
    if not component_selected and wheel_selected == bool(build_asnumpy):
        issues.append("select exactly one AsNumpy mode: local wheel or pinned source build")
    if expected_wheel_sha256 and not _SHA256.fullmatch(expected_wheel_sha256):
        issues.append("expected wheel SHA-256 must be 64 hexadecimal characters")
    if use_cannbot and not accept_cannbot_license:
        issues.append("CANNBot license acceptance is required before acquisition")

    actions = [{"action": "inspect_environment", "network": False}]
    if component_selected:
        actions += [
            {"action": "verify_internal_asnumpy_component", "network": False},
            {"action": "pip_install_verified_component_wheel", "network": False},
        ]
    elif asnumpy_wheel is not None:
        actions += [
            {"action": "verify_local_wheel_sha256", "network": False},
            {"action": "pip_install_verified_wheel", "network": False},
        ]
    elif build_asnumpy:
        actions += [
            {
                "action": "verify_or_fetch_pinned_asnumpy_source",
                "network": not source_cached,
            },
            {
                "action": "build_asnumpy_wheel",
                # The upstream CMake/build backend is explicit, but NumPilot
                # does not yet execute it inside an OS network sandbox.
                "network": True,
                "network_reason": "build subprocess is not OS-network-isolated",
            },
            {"action": "pip_install_built_wheel", "network": False},
        ]
    else:
        actions.append({"action": "select_asnumpy_install_mode", "network": False})
    if use_cannbot:
        actions.append(
            {
                "action": "verify_or_fetch_pinned_cannbot_bundle",
                "network": not skill_cached,
                "license_acceptance_required": True,
            }
        )
    actions.append({"action": "write_atomic_receipt", "network": False})

    plan = {
        "schema_version": 1,
        "mode": "read-only-plan",
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "major_minor": "%d.%d" % sys.version_info[:2],
            "minimum": "3.10",
            "ok": sys.version_info >= (3, 10),
        },
        "machine": {"system": platform.system(), "machine": platform.machine()},
        "venv": {
            "active": prefix != base_prefix or bool(getattr(sys, "real_prefix", None)),
            "prefix": prefix,
            "base_prefix": base_prefix,
        },
        "cann": {
            "toolkit_path": str(toolkit) if toolkit else None,
            "toolkit_directory": bool(toolkit and toolkit.is_dir()),
            "opp_path": str(opp) if opp else None,
            "opp_directory": bool(opp and opp.is_dir()),
            **cann_version,
        },
        "npu_smi": {"path": npu_smi, "available": bool(npu_smi)},
        "asnumpy_installation": {
            "installed": installed is not None,
            "version": installed,
            "expected_version": expected_installed_version,
            "version_matches_request": bool(
                expected_installed_version and installed == expected_installed_version
            ),
            "identity_scope": "distribution-metadata-only",
        },
        "repositories": {
            "asnumpy": (
                {
                    "mode": "internal-component",
                    "component": component.summary(),
                    "runtime_constraints": dict(component.runtime),
                    "source_repository_claimed": True,
                }
                if component is not None
                else {
                    "mode": "pinned-source-build",
                    "repository": ASNUMPY_REPOSITORY,
                    "commit": ASNUMPY_COMMIT,
                    "tree": ASNUMPY_TREE,
                    "license": ASNUMPY_LICENSE,
                }
                if build_asnumpy
                else {
                    "mode": "provided-wheel" if wheel_selected else "not-selected",
                    "source_repository_claimed": False,
                    "audited_profile": (
                        requested_profile["profile_id"]
                        if requested_profile is not None
                        else None
                    ),
                    "wheel_profile": (
                        dict(requested_profile)
                        if requested_profile is not None
                        else None
                    ),
                }
            ),
            "cannbot": {
                "repository": CANNBOT_REPOSITORY,
                "commit": CANNBOT_COMMIT,
                "license": CANNBOT_LICENSE,
            },
        },
        "caches": {"asnumpy": _cache(source_cache), "cannbot": _cache(skill_cache)},
        "build_tools": {
            "cmake": _tool("cmake", (3, 26, 0), runner, finder),
            "gcc": _tool(compiler, (11, 2, 0), runner, finder),
            "git": _tool("git", (0, 0, 0), runner, finder),
        },
        "python_packages": {
            "runtime": {
                name: _python_package(name, minimum, package_reader)
                for name, minimum in _RUNTIME_PACKAGES.items()
            },
            "source_build": {
                name: _python_package(name, minimum, package_reader)
                for name, minimum in _SOURCE_BUILD_PACKAGES.items()
            },
        },
        "request": {
            "asnumpy_mode": (
                "component"
                if component_selected
                else "wheel" if asnumpy_wheel else "source" if build_asnumpy else None
            ),
            "wheel_path": str(asnumpy_wheel) if asnumpy_wheel else None,
            "expected_wheel_sha256": (
                expected_wheel_sha256.lower() if expected_wheel_sha256 else None
            ),
            "component_path": str(asnumpy_component) if asnumpy_component else None,
            "use_cannbot": bool(use_cannbot),
            "cannbot_license_accepted": bool(accept_cannbot_license),
        },
        "planned_actions": actions,
        "network_required": any(action["network"] for action in actions),
        "network_policy": {
            "local_wheel_pip": "--isolated --no-index --no-deps",
            "source_fetch": "disabled for internal component; maintainer-only source mode is explicit",
            "source_build_subprocess_network_isolated": False,
            "note": (
                "source build is conservatively marked network=true because the "
                "upstream build subprocess is explicit but not OS-network-sandboxed; "
                "review its CMake before apply"
            ),
        },
        "issues": issues,
    }
    json.dumps(plan, sort_keys=True)
    return plan


def _environment_issues(plan: Mapping[str, Any], require_venv: bool) -> list[str]:
    issues = []
    if not plan["python"]["ok"]:
        issues.append("Python 3.10 or newer is required")
    if require_venv and not plan["venv"]["active"]:
        issues.append("an active virtual environment is required")
    if not plan["cann"]["toolkit_directory"]:
        issues.append("a valid CANN toolkit environment is required")
    if not plan["cann"]["opp_directory"]:
        issues.append("a valid ASCEND_OPP_PATH is required")
    if not plan["npu_smi"]["available"]:
        issues.append("npu-smi is required for a ready NPU environment")
    issues += [
        "AsNumpy runtime requires %s >= %s" % (name, report["minimum"])
        for name, report in plan["python_packages"]["runtime"].items()
        if not report["ok"]
    ]
    asnumpy_repository = plan["repositories"]["asnumpy"]
    profile = asnumpy_repository.get("wheel_profile")
    constraints = profile.get("runtime_constraints") if profile else None
    constraint_label = "audited AsNumpy profile"
    if asnumpy_repository.get("mode") == "internal-component":
        component = asnumpy_repository.get("component")
        if isinstance(component, Mapping):
            candidate = component.get("runtime")
            if isinstance(candidate, Mapping):
                # The component lock uses flat dependency names so it remains
                # easy to inspect in a release receipt.  Accept an explicit
                # ``packages`` map as a backwards-compatible extension, but
                # never infer a version from a minimum requirement.
                normalized: Dict[str, Any] = dict(candidate)
                packages = normalized.get("packages")
                if isinstance(packages, Mapping):
                    normalized_packages = dict(packages)
                else:
                    normalized_packages = {}
                for package_name in ("numpy", "loguru"):
                    key = "%s_version" % package_name
                    if key in normalized:
                        normalized_packages[package_name] = normalized[key]
                if normalized_packages:
                    normalized["packages"] = normalized_packages
                constraints = normalized
                constraint_label = "frozen AsNumpy component"
                if component.get("artifact_required") is True:
                    required_runtime = {
                        "cann_version",
                        "machine",
                        "loguru_version",
                        "numpy_version",
                        "python_implementation",
                        "python_major_minor",
                    }
                    missing_runtime = sorted(required_runtime - set(normalized))
                    if missing_runtime:
                        issues.append(
                            "frozen AsNumpy component is missing runtime constraints: %s"
                            % ", ".join(missing_runtime)
                        )
            elif component.get("artifact_required") is True:
                issues.append(
                    "frozen AsNumpy component has no runtime constraints"
                )
    if constraints:
        expected_implementation = constraints.get("python_implementation")
        if expected_implementation and plan["python"]["implementation"] != expected_implementation:
            issues.append(
                "%s requires %s" % (constraint_label, expected_implementation)
            )
        expected_python = constraints.get("python_major_minor")
        if expected_python and plan["python"]["major_minor"] != expected_python:
            issues.append(
                "%s requires Python %s" % (constraint_label, expected_python)
            )
        expected_machine = constraints.get("machine")
        if expected_machine and plan["machine"]["machine"] != expected_machine:
            issues.append(
                "%s requires machine %s" % (constraint_label, expected_machine)
            )
        packages = plan["python_packages"]["runtime"]
        expected_packages = constraints.get("packages", {})
        if isinstance(expected_packages, Mapping):
            for name, expected in expected_packages.items():
                if name not in packages:
                    issues.append(
                        "%s requires package %s == %s, but it was not inspected"
                        % (constraint_label, name, expected)
                    )
                    continue
                if not isinstance(expected, str) or packages[name]["version"] != expected:
                    issues.append(
                        "%s requires %s == %s" % (constraint_label, name, expected)
                    )
        expected_cann = constraints.get("cann_version")
        if expected_cann:
            observed_cann = plan.get("cann", {}).get("version")
            if plan.get("cann", {}).get("version_conflict"):
                issues.append(
                    "%s cannot use CANN metadata with conflicting versions" % constraint_label
                )
            elif observed_cann != expected_cann:
                if observed_cann is None:
                    issues.append(
                        "%s requires CANN %s, but the installed CANN version could not be determined"
                        % (constraint_label, expected_cann)
                    )
                else:
                    issues.append(
                        "%s requires CANN %s (observed %s)"
                        % (constraint_label, expected_cann, observed_cann)
                    )
    return issues


def _build_issues(plan: Mapping[str, Any]) -> list[str]:
    issues = [
        "source build requires a supported %s" % name
        for name in ("cmake", "gcc", "git")
        if not plan["build_tools"][name]["ok"]
    ]
    issues += [
        "source build requires %s >= %s" % (name, report["minimum"])
        for name, report in plan["python_packages"]["source_build"].items()
        if not report["ok"]
    ]
    return issues


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BootstrapError("cannot read local wheel") from exc
    return digest.hexdigest()


def _verify_wheel(path: Path, expected: str) -> Dict[str, Any]:
    wheel = Path(path).expanduser()
    if wheel.is_symlink() or not wheel.is_file() or wheel.suffix != ".whl":
        raise BootstrapError("local AsNumpy wheel must be a real .whl file")
    if not _SHA256.fullmatch(expected):
        raise BootstrapError("expected wheel SHA-256 must be 64 hexadecimal characters")
    observed = _file_sha256(wheel)
    if not hmac.compare_digest(observed, expected.lower()):
        raise BootstrapError("AsNumpy wheel SHA-256 mismatch")
    try:
        with ZipFile(wheel) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > _MAX_WHEEL_FILES:
                raise BootstrapError("AsNumpy wheel has an invalid file count")
            if sum(item.file_size for item in infos) > _MAX_WHEEL_UNCOMPRESSED_BYTES:
                raise BootstrapError("AsNumpy wheel exceeds the uncompressed size limit")
            for item in infos:
                raw_name = item.filename
                if not raw_name or "\x00" in raw_name or "\\" in raw_name:
                    raise BootstrapError("AsNumpy wheel contains an unsafe path")
                name = raw_name[:-1] if raw_name.endswith("/") else raw_name
                member = PurePosixPath(name)
                if (
                    not name
                    or member.is_absolute()
                    or ".." in member.parts
                    or "" in member.parts
                    or ":" in member.parts[0]
                    or member.as_posix() != name
                ):
                    raise BootstrapError("AsNumpy wheel contains an unsafe path")
                mode = item.external_attr >> 16
                if stat.S_ISLNK(mode) or item.flag_bits & 0x1:
                    raise BootstrapError(
                        "AsNumpy wheel contains a symbolic link or encrypted member"
                    )
            names = [
                item.filename[:-1] if item.filename.endswith("/") else item.filename
                for item in infos
            ]
            if len(names) != len(set(names)):
                raise BootstrapError("AsNumpy wheel contains duplicate member paths")
            metadata_names = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            wheel_names = [
                name for name in names if name.endswith(".dist-info/WHEEL")
            ]
            record_names = [
                name for name in names if name.endswith(".dist-info/RECORD")
            ]
            if not (
                len(metadata_names) == len(wheel_names) == len(record_names) == 1
            ):
                raise BootstrapError(
                    "AsNumpy wheel must contain one METADATA, WHEEL, and RECORD"
                )
            metadata_bytes = archive.read(metadata_names[0])
            wheel_bytes = archive.read(wheel_names[0])
            record_bytes = archive.read(record_names[0])
            member_digests: Dict[str, bytes] = {}
            member_sizes: Dict[str, int] = {}
            for item in infos:
                digest = hashlib.sha256()
                with archive.open(item, "r") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                member_digests[item.filename] = digest.digest()
                member_sizes[item.filename] = item.file_size
    except (BadZipFile, KeyError, OSError) as exc:
        raise BootstrapError("local AsNumpy wheel is not a readable wheel archive") from exc

    if max(map(len, (metadata_bytes, wheel_bytes, record_bytes))) > _MAX_FILE_BYTES:
        raise BootstrapError("AsNumpy wheel metadata exceeds the size limit")
    try:
        package_metadata = Parser().parsestr(metadata_bytes.decode("utf-8"))
        wheel_metadata = Parser().parsestr(wheel_bytes.decode("utf-8"))
        record_text = record_bytes.decode("utf-8")
    except UnicodeError as exc:
        raise BootstrapError("AsNumpy wheel metadata is not UTF-8") from exc

    try:
        record_rows = list(csv.reader(io.StringIO(record_text, newline="")))
    except csv.Error as exc:
        raise BootstrapError("AsNumpy wheel RECORD is not valid CSV") from exc
    if any(len(row) != 3 for row in record_rows):
        raise BootstrapError("AsNumpy wheel RECORD rows must have three fields")
    record_map: Dict[str, Tuple[str, str]] = {}
    for name, digest_field, size_field in record_rows:
        if name in record_map:
            raise BootstrapError("AsNumpy wheel RECORD contains duplicate paths")
        record_map[name] = (digest_field, size_field)
    if set(record_map) != set(names):
        raise BootstrapError("AsNumpy wheel RECORD does not cover the archive exactly")
    record_name = record_names[0]
    for name in names:
        digest_field, size_field = record_map[name]
        if name == record_name:
            if digest_field or size_field:
                raise BootstrapError("AsNumpy wheel RECORD self-entry must be unhashed")
            continue
        if not digest_field.startswith("sha256=") or not size_field.isdigit():
            raise BootstrapError("AsNumpy wheel RECORD entry is not SHA-256 bound")
        encoded = digest_field.split("=", 1)[1]
        try:
            expected_digest = base64.b64decode(
                encoded + "=" * (-len(encoded) % 4),
                altchars=b"-_",
                validate=True,
            )
        except (binascii.Error, ValueError) as exc:
            raise BootstrapError("AsNumpy wheel RECORD hash is invalid") from exc
        if (
            len(expected_digest) != hashlib.sha256().digest_size
            or not hmac.compare_digest(member_digests[name], expected_digest)
            or int(size_field) != member_sizes[name]
        ):
            raise BootstrapError("AsNumpy wheel RECORD hash or size mismatch")
    distribution = str(package_metadata.get("Name", ""))
    if re.sub(r"[-_.]+", "-", distribution).lower() != "asnumpy":
        raise BootstrapError("local wheel distribution is not AsNumpy")
    version = str(package_metadata.get("Version", "")).strip()
    tags = [str(value) for value in wheel_metadata.get_all("Tag", [])]
    if not version or not tags:
        raise BootstrapError("AsNumpy wheel version or compatibility tag is missing")

    module_member = "asnumpy/__init__.py"
    native_members = sorted(
        name
        for name in names
        if PurePosixPath(name).parent == PurePosixPath("asnumpy")
        and name.endswith(".so")
    )
    if module_member not in member_digests or not native_members:
        raise BootstrapError(
            "AsNumpy wheel must contain package __init__.py and a native module"
        )
    installed_files = {
        "module": {
            "path": module_member,
            "sha256": member_digests[module_member].hex(),
        },
        "native_modules": [
            {"path": name, "sha256": member_digests[name].hex()}
            for name in native_members
        ],
    }

    profile = _AUDITED_WHEEL_PROFILES.get(observed)
    if profile is not None:
        if version != profile["version"]:
            raise BootstrapError("audited AsNumpy wheel version does not match its profile")
        if profile["required_tag"] not in tags:
            raise BootstrapError("audited AsNumpy wheel tag does not match its profile")
        trust = {
            "kind": "numpilot-audited-profile",
            **dict(profile),
        }
    else:
        trust = {
            "kind": "caller-supplied-hash",
            "capability_qualification_required": True,
            "source_repository_claimed": False,
        }
    return {
        "filename": wheel.name,
        "distribution": distribution,
        "version": version,
        "requires_python": package_metadata.get("Requires-Python"),
        "requires_dist": package_metadata.get_all("Requires-Dist", []),
        "tags": tags,
        "sha256": observed,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "installed_files": installed_files,
        "trust": trust,
    }


def _verify_internal_component(
    component: AsNumpyComponent,
    *,
    wheel_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify one release-adjacent wheel against the immutable component lock."""

    try:
        wheel = resolve_component_wheel(component, wheel_path)
        attachment_identity = verify_component_attachments(component)
    except AsNumpyComponentError as exc:
        raise BootstrapError(str(exc)) from exc
    identity = _verify_wheel(wheel, component.wheel_sha256)
    if identity["version"] != component.version:
        raise BootstrapError(
            "internal AsNumpy component version mismatch: expected %s, got %s"
            % (component.version, identity["version"])
        )
    observed_tags = set(identity.get("tags") or ())
    expected_tags = set(component.wheel_tags)
    if observed_tags != expected_tags:
        raise BootstrapError(
            "internal AsNumpy component wheel tags mismatch: expected %s, got %s"
            % (sorted(expected_tags), sorted(observed_tags))
        )
    if (
        component.wheel_metadata_sha256 is not None
        and identity["metadata_sha256"] != component.wheel_metadata_sha256
    ):
        raise BootstrapError("internal AsNumpy component METADATA hash mismatch")
    if (
        component.wheel_record_sha256 is not None
        and identity["record_sha256"] != component.wheel_record_sha256
    ):
        raise BootstrapError("internal AsNumpy component RECORD hash mismatch")
    expected_files = component.installed_files
    if expected_files:
        if dict(identity.get("installed_files") or {}) != dict(expected_files):
            raise BootstrapError(
                "internal AsNumpy component installed-file identity mismatch"
            )
    identity["component"] = component_receipt_identity(component)
    if attachment_identity:
        identity["component"]["verified_attachments"] = attachment_identity
    identity["trust"] = {
        "kind": "numpilot-internal-component",
        "component_id": component.component_id,
        "manifest_sha256": identity["component"]["manifest_sha256"],
        "repository": component.repository,
        "commit": component.commit,
        "tree": component.tree,
    }
    return identity


def _install_wheel(wheel: Path, runner: Optional[Runner]) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            str(wheel),
        ],
        runner=runner,
        timeout=_BUILD_TIMEOUT,
    )


def _build_environment(env: Mapping[str, str]) -> Dict[str, str]:
    allowed = {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "CC",
        "CXX",
        "LD_LIBRARY_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_HOME_PATH",
        "ASCEND_OPP_PATH",
    }
    result = {name: value for name, value in env.items() if name in allowed}
    result["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ASNUMPY"] = ASNUMPY_VERSION
    return result


def _build_wheel(
    checkout: AsNumpyCheckout,
    output: Path,
    runner: Optional[Runner],
    env: Mapping[str, str],
) -> Path:
    output.mkdir(mode=0o700)
    source = output.parent / "source"
    # This clone is from the verified local cache, not the network.  A private
    # clone keeps build products out of the content-addressed cache.
    _run(
        ["git", "clone", "--no-hardlinks", "--no-checkout", "--", checkout.root, str(source)],
        runner=runner,
    )
    _git(source, "checkout", "--detach", checkout.commit, runner=runner)
    _git(source, "remote", "set-url", "origin", ASNUMPY_REPOSITORY, runner=runner)
    load_asnumpy_checkout(source, runner=runner)
    _run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(output)],
        runner=runner,
        cwd=source,
        timeout=_BUILD_TIMEOUT,
        environment=_build_environment(env),
    )
    wheels = [
        path
        for path in output.glob("asnumpy-*.whl")
        if path.is_file() and not path.is_symlink()
    ]
    if len(wheels) != 1:
        raise BootstrapError("source build must produce exactly one AsNumpy wheel")
    return wheels[0]


def _cannbot_manifest(bundle: Any) -> Dict[str, Any]:
    try:
        manifest = dict(bundle.manifest())
    except (AttributeError, TypeError, ValueError) as exc:
        raise BootstrapError("CANNBot acquirer returned an invalid bundle") from exc
    if manifest.get("commit") != CANNBOT_COMMIT:
        raise BootstrapError("CANNBot bundle commit mismatch")
    if _canonical_repo(str(manifest.get("repository", ""))) != _canonical_repo(
        CANNBOT_REPOSITORY
    ):
        raise BootstrapError("CANNBot bundle origin mismatch")
    return manifest


def _cannbot_skill_smoke(bundle: Any) -> Dict[str, Any]:
    """Prove the pinned bundle can answer the documented Skill-discovery smoke."""

    manifest = _cannbot_manifest(bundle)
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise BootstrapError("CANNBot bundle manifest has no verified file inventory")
    missing = [
        item["path"]
        for item in CANNBOT_ASCENDC_SKILL_CATALOG
        if item["path"] not in files
    ]
    if missing:
        raise BootstrapError(
            "CANNBot bundle is missing the Ascend C Skill smoke input: %s"
            % missing[0]
        )
    return {
        "status": "passed",
        "question": "现在写 Ascend C 算子有哪些能参考的 Skill？",
        "answer_source": "verified-local-bundle",
        "bundle_sha256": manifest.get("bundle_sha256"),
        "skills": [dict(item) for item in CANNBOT_ASCENDC_SKILL_CATALOG],
    }


def verify_asnumpy_installation(
    *,
    runner: Optional[Runner] = None,
    expected_wheel: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the official README installation smoke on the current interpreter."""

    if runner is None:
        try:
            metadata.version("asnumpy")
        except metadata.PackageNotFoundError as exc:
            raise BootstrapError(
                "AsNumpy is not installed in the current interpreter; "
                "run explicit bootstrap --apply first"
            ) from exc

    output = _run(
        [sys.executable, "-I", "-c", _ASNUMPY_README_SMOKE],
        runner=runner,
        timeout=_COMMAND_TIMEOUT,
    )
    matches = re.findall(r"^NUMPILOT_ASNUMPY_SMOKE=(\{.*\})$", output, re.MULTILINE)
    if len(matches) != 1:
        raise BootstrapError("AsNumpy smoke did not return one identity record")
    try:
        identity = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise BootstrapError("AsNumpy smoke identity is invalid JSON") from exc
    if not isinstance(identity, dict):
        raise BootstrapError("AsNumpy smoke identity is not an object")
    if identity.get("shape") != [1000, 1000] or identity.get("dtype") != "float32":
        raise BootstrapError("AsNumpy official README smoke returned the wrong result")
    version = identity.get("asnumpy_version")
    if expected_wheel is not None and version != expected_wheel.get("version"):
        raise BootstrapError(
            "installed AsNumpy version does not match the verified wheel metadata"
        )
    module_path = Path(str(identity.get("module_path", "")))
    module_hash = str(identity.get("module_sha256", ""))
    if (
        module_path.is_symlink()
        or not module_path.is_file()
        or not _SHA256.fullmatch(module_hash)
        or not hmac.compare_digest(_file_sha256(module_path), module_hash)
    ):
        raise BootstrapError("installed AsNumpy module identity is not verifiable")
    native = identity.get("native_modules")
    if not isinstance(native, list) or not native:
        raise BootstrapError("installed AsNumpy has no verified native module")
    for item in native:
        if not isinstance(item, dict):
            raise BootstrapError("installed AsNumpy native identity is invalid")
        name = str(item.get("path", ""))
        digest = str(item.get("sha256", ""))
        candidate = module_path.parent / name
        if (
            Path(name).name != name
            or candidate.is_symlink()
            or not candidate.is_file()
            or not _SHA256.fullmatch(digest)
            or not hmac.compare_digest(_file_sha256(candidate), digest)
        ):
            raise BootstrapError("installed AsNumpy native module hash mismatch")
    wheel_file_identity_verified = False
    if expected_wheel is not None:
        installed_files = expected_wheel.get("installed_files")
        if not isinstance(installed_files, Mapping):
            raise BootstrapError("verified wheel has no installed-file identity")
        expected_module = installed_files.get("module")
        expected_native = installed_files.get("native_modules")
        if not isinstance(expected_module, Mapping) or not isinstance(
            expected_native, list
        ):
            raise BootstrapError("verified wheel installed-file identity is invalid")
        if not hmac.compare_digest(
            module_hash, str(expected_module.get("sha256", ""))
        ):
            raise BootstrapError(
                "installed AsNumpy module does not match the verified wheel RECORD"
            )
        expected_native_hashes = sorted(
            str(item.get("sha256", ""))
            for item in expected_native
            if isinstance(item, Mapping)
        )
        actual_native_hashes = sorted(str(item["sha256"]) for item in native)
        if (
            len(expected_native_hashes) != len(expected_native)
            or expected_native_hashes != actual_native_hashes
        ):
            raise BootstrapError(
                "installed AsNumpy native modules do not match the verified wheel RECORD"
            )
        wheel_file_identity_verified = True
    return {
        "status": "passed",
        "source": "official AsNumpy README.md installation verification",
        **identity,
        "wheel_file_identity_verified": wheel_file_identity_verified,
        "interpreter": sys.executable,
    }


def verify_bootstrap_installation(
    *,
    use_cannbot: bool = False,
    accept_cannbot_license: bool = False,
    cannbot_cache: Optional[Path] = None,
    environment: Optional[Mapping[str, str]] = None,
    runner: Optional[Runner] = None,
    cannbot_loader: Optional[Callable[[Path], Any]] = None,
) -> Dict[str, Any]:
    """Re-run local installation smokes; this function never fetches."""

    env = dict(os.environ if environment is None else environment)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "mode": "local-installation-verification",
        "network_requested_by_numpilot": False,
        "asnumpy": verify_asnumpy_installation(runner=runner),
        "cannbot": {"status": "not-requested"},
    }
    if use_cannbot:
        if not accept_cannbot_license:
            raise BootstrapError(
                "CANNBot license acceptance is required before Skill verification"
            )
        root = (
            Path(cannbot_cache).expanduser()
            if cannbot_cache is not None
            else _default_cannbot_checkout(env)
        )
        if root.is_symlink() or not root.is_dir():
            raise BootstrapError(
                "verified CANNBot cache is missing; run explicit bootstrap --apply first"
            )
        bundle = (cannbot_loader or load_skill_bundle)(root)
        report["cannbot"] = _cannbot_skill_smoke(bundle)
    return report


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).expanduser()
    if target.is_symlink() or target.parent.is_symlink():
        raise BootstrapError("bootstrap receipt path must not use a symlink")
    try:
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise BootstrapError("cannot create bootstrap receipt directory") from exc
    if not target.parent.is_dir():
        raise BootstrapError("bootstrap receipt parent is not a directory")
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    descriptor, name = tempfile.mkstemp(
        prefix=".%s." % target.name,
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(name)
    try:
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except OSError as exc:
        raise BootstrapError("cannot atomically write bootstrap receipt") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def apply_bootstrap(
    *,
    receipt_path: Path,
    asnumpy_component: Optional[Path] = None,
    asnumpy_wheel: Optional[Path] = None,
    expected_wheel_sha256: Optional[str] = None,
    build_asnumpy: bool = False,
    use_cannbot: bool = False,
    accept_cannbot_license: bool = False,
    asnumpy_cache: Optional[Path] = None,
    cannbot_cache: Optional[Path] = None,
    environment: Optional[Mapping[str, str]] = None,
    runner: Optional[Runner] = None,
    tool_finder: Optional[Finder] = None,
    version_reader: Optional[VersionReader] = None,
    asnumpy_acquirer: Optional[Callable[[Path], AsNumpyCheckout]] = None,
    asnumpy_builder: Optional[
        Callable[
            [AsNumpyCheckout, Path, Optional[Runner], Mapping[str, str]],
            Path,
        ]
    ] = None,
    cannbot_acquirer: Optional[Callable[[Path], Any]] = None,
) -> Dict[str, Any]:
    """Apply one explicit request and atomically write ``ready`` or ``failed``."""

    env = dict(os.environ if environment is None else environment)
    source_cache = Path(asnumpy_cache) if asnumpy_cache else default_asnumpy_checkout(env)
    skill_cache = Path(cannbot_cache) if cannbot_cache else _default_cannbot_checkout(env)
    component_requested = asnumpy_component is not None
    wheel_requested = asnumpy_wheel is not None or expected_wheel_sha256 is not None
    if component_requested:
        asnumpy_identity = {
            "mode": "internal-component",
            "source_repository_claimed": True,
        }
    elif wheel_requested:
        asnumpy_identity = {
            "mode": "provided-wheel",
            "source_repository_claimed": False,
        }
    elif build_asnumpy:
        asnumpy_identity = {
            "mode": "pinned-source-build",
            "repository": ASNUMPY_REPOSITORY,
            "commit": ASNUMPY_COMMIT,
            "tree": ASNUMPY_TREE,
        }
    else:
        asnumpy_identity = {
            "mode": "invalid-request",
            "source_repository_claimed": False,
        }
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "producer": "numpilot-bootstrap-v1",
        "numpilot_version": __version__,
        "started_at_utc": _utc_now(),
        "status": "applying",
        "ready": False,
        "repositories": {
            "asnumpy": asnumpy_identity,
            "cannbot": {
                "repository": CANNBOT_REPOSITORY,
                "commit": CANNBOT_COMMIT,
                "requested": use_cannbot,
            },
        },
        "actions": [],
        "wheel": None,
        "verification": {},
    }
    _write_receipt(receipt_path, receipt)
    try:
        if component_requested and (wheel_requested or build_asnumpy):
            raise BootstrapError(
                "select exactly one AsNumpy mode: internal component, local wheel, or pinned source build"
            )
        wheel_mode = asnumpy_wheel is not None or expected_wheel_sha256 is not None
        if wheel_mode and (asnumpy_wheel is None or expected_wheel_sha256 is None):
            raise BootstrapError("local wheel and expected SHA-256 must be supplied together")
        if not component_requested and wheel_mode == bool(build_asnumpy):
            raise BootstrapError(
                "select exactly one AsNumpy mode: local wheel or pinned source build"
            )
        if use_cannbot and not accept_cannbot_license:
            raise BootstrapError("CANNBot license acceptance is required before acquisition")

        plan = build_bootstrap_plan(
            asnumpy_component=asnumpy_component,
            asnumpy_wheel=asnumpy_wheel,
            expected_wheel_sha256=expected_wheel_sha256,
            build_asnumpy=build_asnumpy,
            use_cannbot=use_cannbot,
            accept_cannbot_license=accept_cannbot_license,
            asnumpy_cache=source_cache,
            cannbot_cache=skill_cache,
            environment=env,
            runner=runner,
            tool_finder=tool_finder,
            version_reader=version_reader,
        )
        receipt["plan_sha256"] = hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        problems = _environment_issues(plan, require_venv=runner is None)
        if build_asnumpy:
            problems += _build_issues(plan)
        if problems:
            raise BootstrapError("; ".join(problems))
        receipt["environment"] = {
            name: plan[name]
            for name in ("python", "machine", "venv", "cann", "npu_smi", "asnumpy_installation")
        }
        receipt["actions"].append("environment_gates_passed")

        provided_identity = None
        component_manifest = None
        if component_requested:
            try:
                component_manifest = load_component_manifest(asnumpy_component)
            except AsNumpyComponentError as exc:
                raise BootstrapError(str(exc)) from exc
            provided_identity = _verify_internal_component(component_manifest)
            receipt["repositories"]["asnumpy"].update(
                {
                    "mode": "internal-component",
                    "component": component_receipt_identity(component_manifest),
                    "repository": component_manifest.repository,
                    "commit": component_manifest.commit,
                    "tree": component_manifest.tree,
                    "license": component_manifest.license,
                }
            )
            receipt["actions"].append("internal_component_verified")
        elif asnumpy_wheel is not None and expected_wheel_sha256 is not None:
            provided_identity = _verify_wheel(
                asnumpy_wheel, expected_wheel_sha256
            )
            receipt["repositories"]["asnumpy"].update(
                {
                    "wheel_profile": provided_identity["trust"],
                    "source_repository_claimed": False,
                }
            )
            receipt["actions"].append("local_wheel_identity_verified")

        if use_cannbot:
            if cannbot_acquirer:
                bundle = cannbot_acquirer(skill_cache)
            elif skill_cache.is_dir() and not skill_cache.is_symlink():
                bundle = load_skill_bundle(skill_cache)
            else:
                bundle = fetch_pinned_skill(skill_cache)
            receipt["repositories"]["cannbot"].update(_cannbot_manifest(bundle))
            receipt["repositories"]["cannbot"]["license_accepted"] = True
            receipt["actions"].append("cannbot_bundle_verified")
            receipt["verification"]["cannbot"] = _cannbot_skill_smoke(bundle)

        if component_requested and provided_identity is not None:
            wheel = Path(str(resolve_component_wheel(component_manifest)))
            with tempfile.TemporaryDirectory(prefix="numpilot-component-") as temporary:
                staged = Path(temporary) / wheel.name
                shutil.copyfile(wheel, staged)
                if not hmac.compare_digest(
                    _file_sha256(staged), provided_identity["sha256"]
                ):
                    raise BootstrapError("AsNumpy component wheel changed while staging")
                _install_wheel(staged, runner)
            wheel_identity, wheel_kind = provided_identity, "internal-component"
            receipt["actions"].append("verified_component_wheel_installed")
        elif asnumpy_wheel is not None and provided_identity is not None:
            with tempfile.TemporaryDirectory(prefix="numpilot-wheel-") as temporary:
                staged = Path(temporary) / asnumpy_wheel.name
                shutil.copyfile(asnumpy_wheel, staged)
                if not hmac.compare_digest(
                    _file_sha256(staged), provided_identity["sha256"]
                ):
                    raise BootstrapError("AsNumpy wheel changed while staging")
                _install_wheel(staged, runner)
            wheel_identity, wheel_kind = provided_identity, "provided"
            receipt["actions"].append("verified_wheel_installed")
        else:
            if asnumpy_acquirer:
                checkout = asnumpy_acquirer(source_cache)
            elif source_cache.is_dir() and not source_cache.is_symlink():
                checkout = load_asnumpy_checkout(source_cache, runner=runner)
            else:
                checkout = fetch_asnumpy_checkout(source_cache, runner=runner)
            if checkout.commit != ASNUMPY_COMMIT or checkout.tree != ASNUMPY_TREE:
                raise BootstrapError("AsNumpy acquirer returned an unpinned checkout")
            receipt["repositories"]["asnumpy"].update(checkout.manifest())
            receipt["actions"].append("pinned_asnumpy_source_verified")
            with tempfile.TemporaryDirectory(prefix="numpilot-asnumpy-build-") as temporary:
                output = Path(temporary) / "wheel"
                wheel = (
                    asnumpy_builder(checkout, output, runner, env)
                    if asnumpy_builder
                    else _build_wheel(checkout, output, runner, env)
                )
                if wheel.is_symlink() or not wheel.is_file():
                    raise BootstrapError("AsNumpy builder returned no real wheel")
                wheel_hash = _file_sha256(wheel)
                wheel_identity = _verify_wheel(wheel, wheel_hash)
                if wheel_identity["version"] != ASNUMPY_VERSION:
                    raise BootstrapError(
                        "pinned AsNumpy source produced an unexpected version"
                    )
                wheel_identity["trust"] = {
                    "kind": "built-from-pinned-source",
                    "repository": ASNUMPY_REPOSITORY,
                    "commit": ASNUMPY_COMMIT,
                    "tree": ASNUMPY_TREE,
                }
                _install_wheel(wheel, runner)
            wheel_kind = "built-from-pinned-source"
            receipt["actions"] += ["asnumpy_wheel_built", "built_wheel_installed"]

        receipt["verification"]["asnumpy"] = verify_asnumpy_installation(
            runner=runner,
            expected_wheel=wheel_identity,
        )
        receipt["actions"].append("asnumpy_official_readme_smoke_passed")
        receipt["wheel"] = {"kind": wheel_kind, **wheel_identity}
        receipt["completed_at_utc"] = _utc_now()
        receipt["status"], receipt["ready"] = "ready", True
        _write_receipt(receipt_path, receipt)
        return receipt
    except Exception as exc:
        receipt["status"], receipt["ready"] = "failed", False
        receipt["completed_at_utc"] = _utc_now()
        receipt["error"] = {
            "type": type(exc).__name__,
            "message": " ".join(str(exc).splitlines())[:512],
        }
        _write_receipt(receipt_path, receipt)
        if isinstance(exc, BootstrapError):
            raise
        raise BootstrapError("bootstrap apply failed: %s" % type(exc).__name__) from exc


def apply_bootstrap_plan(**kwargs: Any) -> Dict[str, Any]:
    return apply_bootstrap(**kwargs)
