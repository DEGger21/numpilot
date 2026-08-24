#!/usr/bin/env python3
"""Trusted float32 qualification harness for the cbrt showcase operator.

This tool is the execution boundary that NumPilot's admission code deliberately
does not cross.  It copies and rebinds one sealed cbrt artifact, rebuilds it in
an external temporary directory, installs it into a new private OPP root, runs
both model-produced test suites and independent NPU checks, emits all six
schema-v2 gate reports, and publishes one operator-specific capability overlay.

The output directory must not exist.  Failed runs retain their raw diagnostics,
but never retain a final qualification or capability overlay.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Dict, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

from numpilot.backends.npu import CBRT_ASNUMPY_AUDIT_ID


PRODUCER = "numpilot-qualification-harness-v1"
REGISTRY_KEY = "cbrt"
VENDOR = "cbrt_custom"
SOC_BUILD_NAME = "ascend910b"
SOC_RUNTIME_NAME = "Ascend910B"
CANN_VERSION = "9.0.0"
ASNUMPY_VERSION = "0.0.1"
SOURCE_SHA256 = "9022b2174d5e1dde99eb110728edb6708b42b7cbc86374accc2fe3a2c94f1453"
WHEEL_SHA256 = "e0f31b683e217d9c5f5b87fd67721bf97263c79c0597c1be6b00835408a27db3"
ORIGINAL_SPEC_SHA256 = (
    "916657ac1b76fdb46212de6808ca22423d115da14e6c2e04d81742a845c08e6e"
)
ARTIFACT_MANIFEST_SHA256 = (
    "d2598da78d67269658e95af05db30a5f3afb9486b40fa6a8e5cfe2a03f5dab4b"
)
ARTIFACT_ID = "draft-49771d28f566082534e4"
ARTIFACT_TEST_SHA256 = (
    "9f6b85c74cfe6ef77226d02315880e07552faa7c85548e3996a52b4c7abeb758"
)
ASNUMPY_TEST_SHA256 = (
    "78623ee05bdcb4360b97ed47a49d411a979991237df00defa4e4d3c8d7b11917"
)
RUNTIME_FILE_SHA256 = {
    "__init__.py": "4f4e8083755dde87bd6f0dc8c765da71984cb1346c97e8b6dbb97d714be34c04",
    "math.py": "62e0ba09576ead5d3b44f6b0b33311e849efd47ab0bb11f7875c0a43f499085c",
    "custom_ops.py": "78937935889331e5f24f92abf805c93e3cad6f77f1fffd921edbc7961995d793",
    "_custom_ops_registry.py": (
        "01370707cc5d7f5586d054029806021b9cac2cb0e2c013bcae3e6f4015259b04"
    ),
    "utils.py": "1b2981456b2e4258b2f3f9611a07c98dc5a79acb8d67d35e3f2595af13911435",
    "_core": "338dd5ad32008ee12337d64f882a81ce0334b083a60007d648df37995fd1f42b",
}
SOURCE_RELATIVE = "source/cube_root_sensor_calibration.py"
QUALIFICATION_SPEC_RELATIVE = "cbrt/spec/gap-spec.json"
ORIGINAL_SPEC_RELATIVE = "cbrt/spec/original-gap-spec.json"
RESULT_PREFIX = "NUMPILOT_CBRT_QUALIFICATION_RESULT="
NEGATIVE_STARTED = "NUMPILOT_CBRT_NEGATIVE_CALL_STARTED=1"
NEGATIVE_FAILED = "NUMPILOT_CBRT_NEGATIVE_CALL_FAILED="
NEGATIVE_UNEXPECTED_SUCCESS = "NUMPILOT_CBRT_NEGATIVE_UNEXPECTED_SUCCESS=1"
CONSTRAINTS = (
    "dtype=float32",
    "rank>=1",
    "contiguous=true",
    "numel>=1",
)
SHAPES = ((8,), (4, 16), (1024,), (3, 5, 7), (8193,))
RTOL = 1e-5
ATOL = 1e-6

_CANN_ROOT = Path("/usr/local/Ascend/cann-9.0.0")
_DRIVER_ROOT = Path("/usr/local/Ascend/driver/lib64")
_TRUSTED_SYSTEM_PATH = (
    _CANN_ROOT / "bin",
    _CANN_ROOT / "tools/ccec_compiler/bin",
    _CANN_ROOT / "tools/bisheng_compiler/bin",
    _CANN_ROOT / "tools/show_kernel_debug_data",
    _CANN_ROOT / "tools/msobjdump",
    Path("/usr/local/sbin"),
    Path("/usr/local/bin"),
    Path("/usr/sbin"),
    Path("/usr/bin"),
    Path("/sbin"),
    Path("/bin"),
)
_CANN_PYTHON_PATH = (
    _CANN_ROOT / "python/site-packages",
    _CANN_ROOT / "opp/built-in/op_impl/ai_core/tbe",
)
_CANN_LIBRARY_PATH = (
    _DRIVER_ROOT,
    _DRIVER_ROOT / "common",
    _DRIVER_ROOT / "driver",
    _CANN_ROOT / "aarch64-linux/lib64",
    _CANN_ROOT / "lib64/plugin/opskernel",
    _CANN_ROOT / "lib64/plugin/nnengine",
    _CANN_ROOT / "opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64",
)
_CANN_CMAKE_PREFIX_PATH = (
    _CANN_ROOT / "lib64/cmake",
    _CANN_ROOT / "toolkit/tools/tikicpulib/lib/cmake",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(data: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    path.write_bytes(_json_bytes(data))
    path.chmod(0o600)


def _publish_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    """Atomically publish JSON within a private evidence directory."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + "-", suffix=".tmp", dir=str(path.parent)
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(_json_bytes(data))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_fd = os.open(str(path.parent), directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _require_file(path: str, label: str, expected_sha256: str | None = None) -> Path:
    invoked = Path(path).absolute()
    if invoked.is_symlink() or not invoked.is_file():
        raise RuntimeError(f"{label} is not an ordinary file: {invoked}")
    target = invoked.resolve(strict=True)
    if expected_sha256 is not None:
        observed = _sha256_file(target)
        if observed != expected_sha256:
            raise RuntimeError(
                f"{label} SHA-256 drift: expected {expected_sha256}, got {observed}"
            )
    return target


def _require_python(path: str) -> Path:
    # Keep the invoked virtualenv path so Python can discover pyvenv.cfg, while
    # separately verifying that the link target is an executable regular file.
    invoked = Path(path).absolute()
    target = invoked.resolve(strict=True)
    if not target.is_file() or not os.access(invoked, os.X_OK):
        raise RuntimeError(f"AsNumpy Python is not executable: {invoked}")
    return invoked


def _require_artifact(path: str) -> tuple[Path, Dict[str, Any]]:
    invoked = Path(path).absolute()
    if invoked.is_symlink() or not invoked.is_dir():
        raise RuntimeError(f"sealed artifact is not an ordinary directory: {invoked}")
    root = invoked.resolve(strict=True)
    manifest_path = root / "operator-artifact.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("sealed artifact manifest is missing or unsafe")
    observed_manifest_sha = _sha256_file(manifest_path)
    if observed_manifest_sha != ARTIFACT_MANIFEST_SHA256:
        raise RuntimeError(
            "sealed artifact manifest SHA-256 drift: expected %s, got %s"
            % (ARTIFACT_MANIFEST_SHA256, observed_manifest_sha)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_id") != ARTIFACT_ID
        or manifest.get("gap_id") != "cbrt-a48e7704b679eb02"
        or manifest.get("status") != "generated_unverified"
    ):
        raise RuntimeError("sealed artifact identity/lifecycle drift")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or len(expected_files) != 18:
        raise RuntimeError("sealed artifact file manifest is invalid")
    actual_files = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise RuntimeError(f"sealed artifact contains a symlink: {candidate}")
        if candidate.is_file() and candidate != manifest_path:
            actual_files.add(candidate.relative_to(root).as_posix())
    if actual_files != set(expected_files):
        raise RuntimeError("sealed artifact file set drift")
    for relative, expected in expected_files.items():
        candidate = root.joinpath(*Path(relative).parts)
        if _sha256_file(candidate) != expected:
            raise RuntimeError(f"sealed artifact file hash drift: {relative}")
    return root, manifest


def _private_environment_directory(path: Path) -> Path:
    """Create one non-symlinked 0700 directory for untrusted subprocess state."""

    invoked = path.absolute()
    if os.path.lexists(invoked):
        if invoked.is_symlink() or not invoked.is_dir():
            raise RuntimeError(f"subprocess environment path is unsafe: {invoked}")
    else:
        invoked.mkdir(parents=True, mode=0o700)
    invoked.chmod(0o700)
    return invoked.resolve(strict=True)


def _command_env(
    *,
    sandbox_root: Path,
    python: Path,
    custom_opp_path: Path | None = None,
) -> Dict[str, str]:
    """Return a credential-free, deterministic CANN build/runtime environment."""

    sandbox = _private_environment_directory(sandbox_root)
    private_paths = {
        "HOME": _private_environment_directory(sandbox / "home"),
        "TMPDIR": _private_environment_directory(sandbox / "tmp"),
        "XDG_CACHE_HOME": _private_environment_directory(sandbox / "xdg-cache"),
        "XDG_CONFIG_HOME": _private_environment_directory(sandbox / "xdg-config"),
        "XDG_DATA_HOME": _private_environment_directory(sandbox / "xdg-data"),
        "XDG_STATE_HOME": _private_environment_directory(sandbox / "xdg-state"),
        "XDG_RUNTIME_DIR": _private_environment_directory(sandbox / "xdg-runtime"),
        "ASNUMPY_LOG_DIR": _private_environment_directory(sandbox / "asnumpy-logs"),
    }
    repository = Path(__file__).resolve().parents[1]
    trusted_path = (python.absolute().parent, *_TRUSTED_SYSTEM_PATH)
    python_path = (repository, *_CANN_PYTHON_PATH)
    env = {
        "ASCEND_HOME_PATH": str(_CANN_ROOT),
        "ASCEND_TOOLKIT_HOME": str(_CANN_ROOT),
        "ASCEND_OPP_PATH": str(_CANN_ROOT / "opp"),
        "ASCEND_AICPU_PATH": str(_CANN_ROOT),
        "TOOLCHAIN_HOME": str(_CANN_ROOT / "toolkit"),
        "ASCEND_SOC_VERSION": SOC_RUNTIME_NAME,
        # This showcase was qualified specifically on physical device 0.  Do
        # not let a caller's environment silently change that identity.
        "ASCEND_VISIBLE_DEVICES": "0",
        "CANN_VERSION": CANN_VERSION,
        "CMAKE_PREFIX_PATH": os.pathsep.join(
            str(path) for path in _CANN_CMAKE_PREFIX_PATH
        ),
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "ASNUMPY_DEBUG": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "-p no:cacheprovider",
        "LD_LIBRARY_PATH": os.pathsep.join(
            str(path) for path in _CANN_LIBRARY_PATH
        ),
        "PATH": os.pathsep.join(str(path) for path in trusted_path),
        "PYTHONPATH": os.pathsep.join(str(path) for path in python_path),
        **{name: str(path) for name, path in private_paths.items()},
    }
    env["TMP"] = env["TMPDIR"]
    env["TEMP"] = env["TMPDIR"]
    if custom_opp_path is not None:
        env["ASCEND_CUSTOM_OPP_PATH"] = str(custom_opp_path)
    return env


def _write_log(path: Path, output: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    path.write_text(output, encoding="utf-8")
    path.chmod(0o600)


def _run_logged(
    command: Iterable[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    timeout: int,
) -> str:
    try:
        completed = subprocess.run(
            tuple(str(item) for item in command),
            cwd=str(cwd),
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        _write_log(log_path, output)
        raise RuntimeError(f"command timed out; see {log_path}") from exc
    _write_log(log_path, completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}; see {log_path}"
        )
    return completed.stdout


def _run_expected_failure(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    timeout: int,
) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            tuple(str(item) for item in command),
            cwd=str(cwd),
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        _write_log(log_path, output)
        raise RuntimeError(f"negative cbrt process timed out; see {log_path}") from exc
    _write_log(log_path, completed.stdout)
    if completed.returncode == 0:
        raise RuntimeError("cbrt unexpectedly succeeded with the wrong custom OPP path")
    if (
        NEGATIVE_STARTED not in completed.stdout
        or NEGATIVE_FAILED not in completed.stdout
        or NEGATIVE_UNEXPECTED_SUCCESS in completed.stdout
    ):
        raise RuntimeError(
            "negative process did not prove that the cbrt call itself failed; "
            f"see {log_path}"
        )
    return {
        "exit_code": completed.returncode,
        "call_started": True,
        "call_failed": True,
        "log_path": str(log_path),
    }


def _copy_file(source: Path, destination: Path, *, executable: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.parent.chmod(0o700)
    shutil.copy2(source, destination)
    destination.chmod(0o700 if executable else 0o600)


def _copy_artifact(source: Path, destination: Path) -> Dict[str, Any]:
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    for directory in (destination, *tuple(
        candidate for candidate in destination.rglob("*") if candidate.is_dir()
    )):
        directory.chmod(0o700)
    manifest_path = destination / "operator-artifact.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["workspace"] = str(destination)
    _write_json(manifest_path, manifest)
    return manifest


def _qualification_spec(
    original: Mapping[str, Any], *, source_relative: str = SOURCE_RELATIVE
) -> Dict[str, Any]:
    """Mechanically narrow the immutable discovery spec to the tested scope."""

    expected_identity = {
        "schema_version": 1,
        "gap_id": "cbrt-a48e7704b679eb02",
        "api_name": "cbrt",
        "qualified_api": "numpy.cbrt",
        "asnumpy_api": "asnumpy.cbrt",
        "source_sha256": SOURCE_SHA256,
        "target_soc": SOC_RUNTIME_NAME,
        "cann_version": CANN_VERSION,
        "semantic_source": "numpy.cbrt",
    }
    for key, expected in expected_identity.items():
        if original.get(key) != expected:
            raise RuntimeError(f"original GapSpec identity drift: {key}")
    if original.get("callsites") != [{
        "call_expr": "numpy.cbrt(<arg0>)",
        "col_offset": 17,
        "lineno": 18,
        "qualified_api": "numpy.cbrt",
        "reason": "API 不在静态覆盖集",
    }]:
        raise RuntimeError("original GapSpec callsite drift")
    if original.get("signature_variants") != [{
        "keyword_args": [],
        "positional_args": 1,
    }]:
        raise RuntimeError("original GapSpec signature drift")

    qualified = json.loads(json.dumps(original, ensure_ascii=False))
    qualified["source_path"] = source_relative
    qualified["dtype_rules"] = ["float32"]
    qualified["shape_rules"] = ["rank>=1", "contiguous=true", "numel>=1"]
    unchanged_original = dict(original)
    unchanged_qualified = dict(qualified)
    for key in ("source_path", "dtype_rules", "shape_rules"):
        unchanged_original.pop(key, None)
        unchanged_qualified.pop(key, None)
    if unchanged_original != unchanged_qualified:
        raise RuntimeError("qualification GapSpec changed non-scope fields")
    return qualified


def _build_and_install(
    artifact: Path,
    *,
    build_root: Path,
    output_root: Path,
    private_opp: Path,
    python: Path,
) -> Dict[str, Any]:
    build_dir = build_root / REGISTRY_KEY
    raw_dir = output_root / "raw" / "build"
    env = _command_env(
        sandbox_root=build_root / ".subprocess-environment",
        python=python,
    )
    _run_logged(
        (
            "cmake",
            "-S",
            str(artifact),
            "-B",
            str(build_dir),
            f"-DASCEND_COMPUTE_UNIT={SOC_BUILD_NAME}",
        ),
        cwd=output_root,
        env=env,
        log_path=raw_dir / "configure.log",
        timeout=600,
    )
    _run_logged(
        (
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "all",
            "binary",
            "package",
            "--",
            "-j1",
        ),
        cwd=output_root,
        env=env,
        log_path=raw_dir / "build.log",
        timeout=1800,
    )
    package = build_dir / "custom_opp_ubuntu_aarch64.run"
    kernels = tuple(build_dir.glob(
        f"op_kernel/ascendc_kernels/binary/{SOC_BUILD_NAME}/**/*.o"
    ))
    if not package.is_file() or package.stat().st_size == 0 or not kernels:
        raise RuntimeError("cbrt build output is incomplete")
    package_copy = output_root / "runtime" / "packages" / "cbrt.run"
    _copy_file(package, package_copy, executable=True)
    _run_logged(
        (
            "bash",
            str(package_copy),
            f"--install-path={private_opp}",
            "--quiet",
        ),
        cwd=output_root,
        env=env,
        log_path=raw_dir / "install.log",
        timeout=600,
    )
    vendor = private_opp / "vendors" / VENDOR
    if vendor.is_symlink() or not vendor.is_dir():
        raise RuntimeError(f"installer did not create expected vendor: {vendor}")
    return {
        "package": package_copy,
        "vendor": vendor,
        "kernel_count": len(kernels),
        "output_artifacts": 1 + len(kernels),
    }


def _materialize_private_opp_symlinks(private_opp: Path) -> None:
    """Replace installer-created relative links with evidence-bound bytes."""

    root = private_opp.resolve(strict=True)
    for link in sorted(path for path in private_opp.rglob("*") if path.is_symlink()):
        target = link.resolve(strict=True)
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"private OPP symlink escapes its root: {link}") from exc
        if not target.is_file():
            raise RuntimeError(f"private OPP symlink target is not a file: {link}")
        link.unlink()
        shutil.copy2(target, link)


def _junit_counts(path: Path) -> Dict[str, int]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise RuntimeError(f"pytest JUnit report is unreadable: {path}") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        suites = list(root.iter("testsuite"))
    if not suites:
        raise RuntimeError(f"pytest JUnit report has no testsuite: {path}")
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in suites:
        for key in counts:
            raw = suite.attrib.get(key, "0")
            try:
                value = int(raw)
            except ValueError as exc:
                raise RuntimeError(f"pytest JUnit {key} count is invalid") from exc
            if value < 0:
                raise RuntimeError(f"pytest JUnit {key} count is negative")
            counts[key] += value
    return counts


def _run_pytest_suite(
    *,
    python: Path,
    test: Path,
    label: str,
    expected_tests: int,
    output_root: Path,
    env: Mapping[str, str],
) -> Dict[str, int]:
    raw_dir = output_root / "raw" / "tests"
    junit = raw_dir / f"{label}-junit.xml"
    _run_logged(
        (
            str(python),
            "-m",
            "pytest",
            "-q",
            str(test),
            f"--junitxml={junit}",
        ),
        cwd=output_root,
        env=env,
        log_path=raw_dir / f"{label}.log",
        timeout=600,
    )
    junit.chmod(0o600)
    counts = _junit_counts(junit)
    if counts != {
        "tests": expected_tests,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }:
        raise RuntimeError(f"{label} pytest result is not an exact all-pass: {counts}")
    return counts


def _wrapper_has_host_fallback(wrapper: Any) -> bool:
    source = textwrap.dedent(inspect.getsource(wrapper))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr == "to_numpy":
            return True
        if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
            if function.value.id in {"np", "numpy"}:
                return True
    return False


def _upload_showcase_inputs(asnumpy_module: Any, *values: Any) -> tuple[Any, ...]:
    """Mirror AstRewriteEngine's forward-entry upload for named parameters."""

    import numpy as np

    return tuple(
        asnumpy_module.ndarray.from_numpy(value)
        if isinstance(value, np.ndarray)
        else value
        for value in values
    )


def _runtime_worker() -> int:
    """Run only in the explicitly selected AsNumpy virtual environment."""

    import numpy as np
    import asnumpy
    import asnumpy.custom_ops as custom_ops

    from numpilot.backends.npu import NpuBackend
    from numpilot.capabilities import ExecutionProvenance

    if getattr(asnumpy, "__version__", None) != ASNUMPY_VERSION:
        raise RuntimeError("installed AsNumpy version drift")
    public = getattr(asnumpy, "cbrt", None)
    wrapper = getattr(custom_ops, "cbrt", None)
    backing = custom_ops.__dict__.get("_cbrt")
    if not callable(public) or public is not wrapper:
        raise RuntimeError("asnumpy.cbrt is not asnumpy.custom_ops.cbrt")
    if not callable(backing) or getattr(backing, "__module__", None) != (
        "asnumpy._core.custom_ops"
    ):
        raise RuntimeError("cbrt wrapper does not bind asnumpy._core.custom_ops")
    if _wrapper_has_host_fallback(wrapper):
        raise RuntimeError("cbrt wrapper contains a NumPy/to_numpy host fallback call")

    max_abs_error = 0.0
    for shape in SHAPES:
        count = int(np.prod(shape))
        values = np.linspace(-64.0, 64.0, count, dtype=np.float32).reshape(shape)
        values = np.ascontiguousarray(values, dtype=np.float32)
        expected = np.cbrt(values)
        actual = public(values)
        if type(actual) is not asnumpy.ndarray:
            raise RuntimeError(f"cbrt did not return asnumpy.ndarray for shape {shape}")
        if tuple(actual.shape) != shape or np.dtype(actual.dtype) != np.dtype("float32"):
            raise RuntimeError(f"cbrt output metadata mismatch for shape {shape}")
        host = actual.to_numpy()
        if host.shape != shape or host.dtype != np.dtype("float32"):
            raise RuntimeError(f"cbrt host result metadata mismatch for shape {shape}")
        np.testing.assert_allclose(host, expected, rtol=RTOL, atol=ATOL)
        max_abs_error = max(
            max_abs_error,
            float(np.max(np.abs(host.astype(np.float64) - expected.astype(np.float64)))),
        )

    signed_input = np.array([0.0, -0.0, -8.0, 8.0], dtype=np.float32)
    signed_expected = np.cbrt(signed_input)
    signed_actual = public(signed_input)
    if type(signed_actual) is not asnumpy.ndarray:
        raise RuntimeError("signed-zero cbrt result is not asnumpy.ndarray")
    signed_host = signed_actual.to_numpy()
    np.testing.assert_allclose(signed_host, signed_expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(np.signbit(signed_host), np.signbit(signed_expected))

    signal = np.linspace(-64.0, 64.0, 8193, dtype=np.float32)
    gain = np.array([0.75], dtype=np.float32)
    offset = np.array([-0.125], dtype=np.float32)
    # AstRewriteEngine inserts this exact idempotent host-to-device boundary
    # for every named forward parameter before rewriting the three calls.
    signal_device, gain_device, offset_device = _upload_showcase_inputs(
        asnumpy, signal, gain, offset
    )
    compressed = public(signal_device)
    scaled = asnumpy.multiply(compressed, gain_device)
    chained = asnumpy.add(scaled, offset_device)
    if type(chained) is not asnumpy.ndarray:
        raise RuntimeError("cbrt->multiply->add did not return asnumpy.ndarray")
    if tuple(chained.shape) != (8193,) or np.dtype(chained.dtype) != np.dtype("float32"):
        raise RuntimeError("real showcase chain output metadata mismatch")
    chain_host = chained.to_numpy()
    chain_expected = np.add(np.multiply(np.cbrt(signal), gain), offset)
    np.testing.assert_allclose(chain_host, chain_expected, rtol=RTOL, atol=ATOL)
    max_abs_error = max(
        max_abs_error,
        float(np.max(np.abs(
            chain_host.astype(np.float64) - chain_expected.astype(np.float64)
        ))),
    )

    maps = Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace")
    vendor_lines = sorted({
        line for line in maps.splitlines()
        if VENDOR in line and ".so" in line
    })
    required_libraries = {
        "libcust_opapi.so",
        "libcust_opmaster_rt2.0.so",
        "libcust_opsproto_rt2.0.so",
    }
    observed_libraries = {
        name for name in required_libraries
        if any(name in line for line in vendor_lines)
    }
    if observed_libraries != required_libraries:
        raise RuntimeError(
            "cbrt custom vendor libraries are not all mapped: %s"
            % sorted(observed_libraries)
        )

    base_manifest = NpuBackend(module=asnumpy).capability_manifest((REGISTRY_KEY,))
    capability = base_manifest.operators.get(REGISTRY_KEY)
    if base_manifest.audit_id != CBRT_ASNUMPY_AUDIT_ID:
        raise RuntimeError("base manifest does not have the exact cbrt runtime audit ID")
    if (
        capability is None
        or not capability.available
        or capability.provenance is not ExecutionProvenance.UNKNOWN
        or capability.strict_npu
    ):
        raise RuntimeError("base cbrt capability is not available+unknown+non-strict")
    if tuple(base_manifest.operators) != (REGISTRY_KEY,):
        raise RuntimeError("overlay base manifest contains unrelated operators")

    module_root = Path(asnumpy.__file__).resolve().parent
    core_paths = tuple(module_root.glob("_core*.so"))
    if len(core_paths) != 1:
        raise RuntimeError("installed AsNumpy core identity is ambiguous")
    identity_paths = {
        "__init__.py": module_root / "__init__.py",
        "math.py": module_root / "math.py",
        "custom_ops.py": module_root / "custom_ops.py",
        "_custom_ops_registry.py": module_root / "_custom_ops_registry.py",
        "utils.py": module_root / "utils.py",
        "_core": core_paths[0],
    }
    observed_identity = {
        key: _sha256_file(path) for key, path in identity_paths.items()
    }
    if observed_identity != RUNTIME_FILE_SHA256:
        raise RuntimeError("installed AsNumpy runtime identity files drifted")

    payload = {
        "api_identity": True,
        "backing_callable_module": "asnumpy._core.custom_ops",
        "base_manifest": base_manifest.to_dict(),
        "cpu_fallback_detected": False,
        "identity_files": {key: str(path) for key, path in identity_paths.items()},
        "kernel_launches": len(SHAPES) + 1 + 3,
        "mapped_vendor_libraries": sorted(required_libraries),
        "max_abs_error": max_abs_error,
        "numerical_cases": len(SHAPES) + 1 + 1,
        "public_callable": True,
        "shapes": [list(shape) for shape in SHAPES],
        "signed_zero_verified": True,
    }
    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True))
    return 0


def _negative_worker() -> int:
    """Prove that cbrt cannot run when its custom OPP vendor is unavailable."""

    import numpy as np
    import asnumpy

    print(NEGATIVE_STARTED, flush=True)
    try:
        result = asnumpy.cbrt(np.array([-8.0, 0.0, 8.0], dtype=np.float32))
        result.to_numpy()
    except Exception as exc:
        print(NEGATIVE_FAILED + type(exc).__name__, flush=True)
        raise
    print(NEGATIVE_UNEXPECTED_SUCCESS, flush=True)
    return 0


def _extract_worker_result(output: str) -> Dict[str, Any]:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_PREFIX):
            data = json.loads(line[len(RESULT_PREFIX):])
            if isinstance(data, dict):
                return data
            break
    raise RuntimeError("runtime worker did not emit a structured result")


def _hash_tree(root: Path, *, exclude: Iterable[Path] = ()) -> Dict[str, str]:
    excluded = {path.absolute() for path in exclude}
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"evidence tree contains a symlink: {path}")
        if not path.is_file() or path.absolute() in excluded:
            continue
        result[path.relative_to(root).as_posix()] = _sha256_file(path)
    return result


def _create_analysis_profiles(output_root: Path, source_sha256: str) -> Path:
    import numpy as np

    from numpilot.capabilities import (
        AnalysisCallsiteProfile,
        AnalysisProfileSet,
        profile_operator_call,
    )

    signal = np.linspace(-64.0, 64.0, 8193, dtype=np.float32)
    profile = profile_operator_call((signal,), {})
    callsite = AnalysisCallsiteProfile(
        qualified_name="numpy.cbrt",
        lineno=18,
        col_offset=17,
        profile=profile,
    )
    profiles = AnalysisProfileSet(
        source_sha256=source_sha256,
        callsites=(callsite,),
    )
    data = profiles.to_dict()
    tensor = data["callsites"][0]["profile"]["inputs"][0]
    if (
        data["source_sha256"] != SOURCE_SHA256
        or tensor != {
            "dtype": "float32",
            "shape": [8193],
            "rank": 1,
            "c_contiguous": True,
        }
        or data["callsites"][0]["profile"]["positional_args"] != 1
        or data["callsites"][0]["profile"]["keyword_args"] != []
    ):
        raise RuntimeError("analysis profile API produced an unexpected cbrt profile")
    path = output_root / "analysis-profiles.json"
    _write_json(path, data)
    return path


def _gate_claims(
    name: str,
    *,
    artifact_files: int,
    build: Mapping[str, Any],
    tests_passed: int,
    runtime_result: Mapping[str, Any],
    signature_count: int,
    signature_sha256: str,
) -> Dict[str, Any]:
    if name == "artifact_integrity":
        return {"verified_files": artifact_files, "mismatches": 0}
    if name == "build":
        return {"exit_code": 0, "output_artifacts": build["output_artifacts"]}
    if name == "unit_test":
        claims: Dict[str, Any] = {"passed": tests_passed, "failed": 0}
    elif name == "numerical_validation":
        claims = {
            "oracle": "numpy",
            "cases": runtime_result["numerical_cases"],
            "failed": 0,
            "cpu_fallback_detected": runtime_result["cpu_fallback_detected"],
        }
    elif name == "public_asnumpy_api":
        claims = {
            "api": "asnumpy.cbrt",
            "callable": runtime_result["public_callable"],
            "cpu_fallback_detected": runtime_result["cpu_fallback_detected"],
        }
    elif name == "npu_execution":
        claims = {
            "backend": "npu",
            "device_count": 1,
            "kernel_launches": runtime_result["kernel_launches"],
            "cpu_fallback_detected": runtime_result["cpu_fallback_detected"],
        }
    else:
        raise RuntimeError(f"unknown qualification gate: {name}")
    claims.update({
        "covered_signature_variants": signature_count,
        "signature_variants_sha256": signature_sha256,
    })
    return claims


def _assert_profile_admitted(capability: Any) -> None:
    """Require strict NPU admission for the exact showcased runtime profile."""

    import numpy as np

    from numpilot.capabilities import (
        ExecutionProvenance,
        match_capability_profile,
        profile_operator_call,
    )

    if (
        capability.provenance is not ExecutionProvenance.CUSTOM_NPU
        or capability.constraints != CONSTRAINTS
        or len(capability.signature_variants) != 1
        or capability.signature_variants[0].positional_args != 1
        or capability.signature_variants[0].keyword_args
    ):
        raise RuntimeError("admitted cbrt capability identity/scope mismatch")
    signal = np.linspace(-64.0, 64.0, 8193, dtype=np.float32)
    match = match_capability_profile(
        capability,
        profile_operator_call((signal,), {}),
    )
    if not match.matched:
        raise RuntimeError(
            "cbrt profile-aware strict-NPU admission failed: "
            + "; ".join(match.reasons)
        )


def _remove_final_publications(paths: Iterable[Path]) -> None:
    failures = []
    for path in paths:
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
        except OSError as exc:
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError(
            "failed to roll back qualification publication: " + " | ".join(failures)
        )


def _main(args: argparse.Namespace) -> int:
    from numpilot.capabilities import (
        CapabilityManifest,
        ExecutionProvenance,
        OperatorSignature,
    )
    from numpilot.gaps.qualification import (
        CapabilityOverlay,
        QualificationEvidence,
        QualificationRef,
        REQUIRED_QUALIFICATION_GATES,
        ZERO_SHA256,
        apply_overlay,
        capability_manifest_sha256,
        compute_qualification_id,
        publish_overlay_atomic,
        qualification_constraints_sha256,
        qualification_signatures_sha256,
        validate_qualification,
    )

    source = _require_file(args.source, "source", SOURCE_SHA256)
    original_spec = _require_file(
        args.original_spec, "original GapSpec", ORIGINAL_SPEC_SHA256
    )
    artifact, original_artifact_manifest = _require_artifact(args.artifact)
    asnumpy_python = _require_python(args.asnumpy_python)
    asnumpy_wheel = _require_file(
        args.asnumpy_wheel, "AsNumpy wheel", WHEEL_SHA256
    )
    asnumpy_test = _require_file(
        args.asnumpy_test, "AsNumpy cbrt test", ASNUMPY_TEST_SHA256
    )
    harness_source = _require_file(str(Path(__file__).absolute()), "qualification harness")

    output_root = Path(args.out).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError(f"output must not exist: {output_root}")
    output_root = output_root.parent.resolve(strict=True) / output_root.name
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)

    qualification_path = output_root / "cbrt" / "qualification.json"
    pending_qualification_path = output_root / "cbrt" / ".qualification.pending.json"
    overlay_path = output_root / "overlay" / "capability-overlay.json"
    admitted_path = output_root / "overlay" / "admitted-capability-manifest.json"
    summary_path = output_root / "qualification-summary.json"
    public_paths = (
        qualification_path,
        overlay_path,
        admitted_path,
        summary_path,
        pending_qualification_path,
    )

    copied_source = output_root / SOURCE_RELATIVE
    _copy_file(source, copied_source)
    if _sha256_file(copied_source) != SOURCE_SHA256:
        raise RuntimeError("source copy hash mismatch")

    original_spec_copy = output_root / ORIGINAL_SPEC_RELATIVE
    _copy_file(original_spec, original_spec_copy)
    original_spec_data = json.loads(original_spec_copy.read_text(encoding="utf-8"))
    qualification_spec_data = _qualification_spec(original_spec_data)
    qualification_spec_path = output_root / QUALIFICATION_SPEC_RELATIVE
    _write_json(qualification_spec_path, qualification_spec_data)

    artifact_copy = output_root / "cbrt" / "artifact"
    artifact_manifest = _copy_artifact(artifact, artifact_copy)
    if artifact_manifest["artifact_id"] != original_artifact_manifest["artifact_id"]:
        raise RuntimeError("artifact identity changed during evidence copy")

    artifact_test_copy = output_root / "tests" / "artifact" / "test_cbrt.py"
    _copy_file(artifact / "tests" / "test_cbrt.py", artifact_test_copy)
    if _sha256_file(artifact_test_copy) != ARTIFACT_TEST_SHA256:
        raise RuntimeError("artifact test copy hash mismatch")
    asnumpy_test_copy = output_root / "tests" / "asnumpy" / "test_cbrt.py"
    _copy_file(asnumpy_test, asnumpy_test_copy)
    harness_copy = output_root / "tools" / "cbrt_qualification_harness.py"
    _copy_file(harness_source, harness_copy, executable=True)
    wheel_copy = output_root / "runtime" / "asnumpy-wheel.whl"
    _copy_file(asnumpy_wheel, wheel_copy)

    private_opp = output_root / "runtime" / "private-opp"
    private_opp.mkdir(parents=True, mode=0o700)
    private_opp.chmod(0o700)
    with tempfile.TemporaryDirectory(
        prefix="numpilot-cbrt-qualification-build-",
        dir=str(output_root.parent),
    ) as temporary_build:
        build = _build_and_install(
            artifact_copy,
            build_root=Path(temporary_build),
            output_root=output_root,
            private_opp=private_opp,
            python=asnumpy_python,
        )

    _materialize_private_opp_symlinks(private_opp)
    for directory in (private_opp, *tuple(
        candidate for candidate in private_opp.rglob("*") if candidate.is_dir()
    )):
        directory.chmod(0o700)
    vendor = private_opp / "vendors" / VENDOR
    runtime_env = _command_env(
        sandbox_root=output_root / "runtime" / "subprocess-environment" / "positive",
        python=asnumpy_python,
        custom_opp_path=vendor,
    )

    artifact_test_counts = _run_pytest_suite(
        python=asnumpy_python,
        test=artifact_test_copy,
        label="sealed-artifact-test-cbrt",
        expected_tests=21,
        output_root=output_root,
        env=runtime_env,
    )
    asnumpy_test_counts = _run_pytest_suite(
        python=asnumpy_python,
        test=asnumpy_test_copy,
        label="asnumpy-float32-test-cbrt",
        expected_tests=4,
        output_root=output_root,
        env=runtime_env,
    )

    worker_output = _run_logged(
        (str(asnumpy_python), str(harness_copy), "--runtime-worker"),
        cwd=Path(__file__).resolve().parents[1],
        env=runtime_env,
        log_path=output_root / "raw" / "runtime-worker.log",
        timeout=600,
    )
    worker = _extract_worker_result(worker_output)
    base_manifest = CapabilityManifest.from_dict(worker["base_manifest"])
    if tuple(base_manifest.operators) != (REGISTRY_KEY,):
        raise RuntimeError("base manifest was not scoped to the overlay entry")

    negative_opp = output_root / "runtime" / "negative-empty-opp"
    negative_opp.mkdir(mode=0o700)
    negative_log = output_root / "raw" / "negative-wrong-opp.log"
    negative = _run_expected_failure(
        (str(asnumpy_python), str(harness_copy), "--negative-worker"),
        cwd=Path(__file__).resolve().parents[1],
        env=_command_env(
            sandbox_root=(
                output_root / "runtime" / "subprocess-environment" / "negative"
            ),
            python=asnumpy_python,
            custom_opp_path=negative_opp,
        ),
        log_path=negative_log,
        timeout=120,
    )
    negative["log_path"] = negative_log.relative_to(output_root).as_posix()
    negative["wrong_opp_path"] = negative_opp.relative_to(output_root).as_posix()
    _write_json(output_root / "runtime" / "negative-opp-result.json", negative)

    runtime_identity_files: Dict[str, str] = {}
    for key, raw_path in worker["identity_files"].items():
        source_path = Path(raw_path)
        destination_name = (
            source_path.name if key == "_core" else key
        )
        destination = output_root / "runtime" / "asnumpy" / destination_name
        _copy_file(source_path, destination)
        observed = _sha256_file(destination)
        if observed != RUNTIME_FILE_SHA256[key]:
            raise RuntimeError(f"copied AsNumpy runtime identity drift: {key}")
        runtime_identity_files[
            destination.relative_to(output_root).as_posix()
        ] = observed
    _write_json(
        output_root / "runtime" / "base-capability-manifest.json",
        base_manifest.to_dict(),
    )

    analysis_profiles_path = _create_analysis_profiles(output_root, SOURCE_SHA256)
    _write_json(output_root / "runtime" / "test-summary.json", {
        "schema_version": 1,
        "producer": PRODUCER,
        "sealed_artifact": artifact_test_counts,
        "asnumpy_float32": asnumpy_test_counts,
        "negative_wrong_opp": negative,
    })

    opp_files = {}
    for path in sorted(private_opp.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"private OPP still contains a symlink: {path}")
        if path.is_file():
            opp_files[path.relative_to(private_opp).as_posix()] = _sha256_file(path)
    opp_manifest = {
        "schema_version": 1,
        "producer": PRODUCER,
        "soc": SOC_RUNTIME_NAME,
        "cann": CANN_VERSION,
        "vendor": VENDOR,
        "files": opp_files,
        "package": {
            "artifact_id": artifact_manifest["artifact_id"],
            "path": build["package"].relative_to(output_root).as_posix(),
            "sha256": _sha256_file(build["package"]),
        },
        "asnumpy_wheel": {
            "path": wheel_copy.relative_to(output_root).as_posix(),
            "sha256": _sha256_file(wheel_copy),
        },
        "runtime_identity_files": runtime_identity_files,
    }
    opp_manifest_path = output_root / "runtime" / "opp-manifest.json"
    _write_json(opp_manifest_path, opp_manifest)
    _write_json(output_root / "input-manifest.json", {
        "schema_version": 1,
        "producer": PRODUCER,
        "source": {"path": SOURCE_RELATIVE, "sha256": SOURCE_SHA256},
        "original_gap_spec": {
            "path": ORIGINAL_SPEC_RELATIVE,
            "sha256": ORIGINAL_SPEC_SHA256,
        },
        "qualification_gap_spec": {
            "path": QUALIFICATION_SPEC_RELATIVE,
            "sha256": _sha256_file(qualification_spec_path),
        },
        "sealed_artifact_input_manifest_sha256": ARTIFACT_MANIFEST_SHA256,
        "artifact_test_sha256": ARTIFACT_TEST_SHA256,
        "asnumpy_test_sha256": ASNUMPY_TEST_SHA256,
        "asnumpy_wheel_sha256": WHEEL_SHA256,
        "qualification_harness_sha256": _sha256_file(harness_copy),
        "analysis_profiles": {
            "path": analysis_profiles_path.relative_to(output_root).as_posix(),
            "sha256": _sha256_file(analysis_profiles_path),
        },
    })

    future_paths = (
        qualification_path,
        pending_qualification_path,
        overlay_path,
        admitted_path,
        summary_path,
    )
    runtime_artifacts = _hash_tree(output_root, exclude=future_paths)
    runtime = {
        "backend": "npu",
        "backend_version": base_manifest.backend_version,
        "backend_audit_id": base_manifest.audit_id,
        "backend_module_sha256": RUNTIME_FILE_SHA256["__init__.py"],
        "cann_version": base_manifest.environment["cann_version"],
        "soc_version": base_manifest.environment["soc_version"],
        "python_version": base_manifest.environment["python_version"],
        "opp_manifest_path": opp_manifest_path.relative_to(output_root).as_posix(),
        "opp_manifest_sha256": _sha256_file(opp_manifest_path),
    }
    if (
        runtime["backend_version"] != ASNUMPY_VERSION
        or runtime["backend_audit_id"] != CBRT_ASNUMPY_AUDIT_ID
        or runtime["cann_version"] != CANN_VERSION
        or "ascend910b" not in "".join(
            character.lower() for character in runtime["soc_version"]
            if character.isalnum()
        )
        or not str(runtime["python_version"]).startswith("3.10")
    ):
        raise RuntimeError("runtime target identity does not match the qualification scope")

    signatures = (OperatorSignature(positional_args=1),)
    signature_sha256 = qualification_signatures_sha256(signatures)
    constraints_sha256 = qualification_constraints_sha256(CONSTRAINTS)
    spec_sha256 = _sha256_file(qualification_spec_path)
    artifact_manifest_path = artifact_copy / "operator-artifact.json"
    artifact_manifest_sha256 = _sha256_file(artifact_manifest_path)
    gate_metadata = {}
    total_tests = artifact_test_counts["tests"] + asnumpy_test_counts["tests"]
    for gate in sorted(REQUIRED_QUALIFICATION_GATES):
        report = {
            "schema_version": 2,
            "gate": gate,
            "status": "passed",
            "producer": PRODUCER,
            "artifact_id": artifact_manifest["artifact_id"],
            "gap_spec_sha256": spec_sha256,
            "artifact_manifest_sha256": artifact_manifest_sha256,
            "constraints_sha256": constraints_sha256,
            "target": runtime,
            "claims": _gate_claims(
                gate,
                artifact_files=len(artifact_manifest["files"]),
                build=build,
                tests_passed=total_tests,
                runtime_result=worker,
                signature_count=len(signatures),
                signature_sha256=signature_sha256,
            ),
        }
        report_path = output_root / "cbrt" / "reports" / f"{gate}.json"
        _write_json(report_path, report)
        gate_metadata[gate] = {
            "status": "passed",
            "report_path": report_path.relative_to(output_root).as_posix(),
            "report_sha256": _sha256_file(report_path),
            "producer": PRODUCER,
            "format_version": 2,
        }

    evidence = {
        "schema_version": 1,
        "qualification_id": "qual-" + "0" * 20,
        "gap_id": qualification_spec_data["gap_id"],
        "registry_key": REGISTRY_KEY,
        "qualified_api": "numpy.cbrt",
        "asnumpy_api": "asnumpy.cbrt",
        "gap_spec_path": QUALIFICATION_SPEC_RELATIVE,
        "gap_spec_sha256": spec_sha256,
        "source_path": SOURCE_RELATIVE,
        "source_sha256": SOURCE_SHA256,
        "artifact_manifest_path": artifact_manifest_path.relative_to(
            output_root
        ).as_posix(),
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "artifact_id": artifact_manifest["artifact_id"],
        "artifact_files": artifact_manifest["files"],
        "runtime_artifacts": runtime_artifacts,
        "runtime": runtime,
        "constraints": list(CONSTRAINTS),
        "gates": gate_metadata,
    }
    evidence["qualification_id"] = compute_qualification_id(evidence)
    parsed = QualificationEvidence.from_dict(evidence)
    capability = validate_qualification(
        parsed,
        evidence_root=str(output_root),
        current_manifest=base_manifest,
    )
    if (
        capability.provenance is not ExecutionProvenance.CUSTOM_NPU
        or capability.constraints != CONSTRAINTS
        or capability.signature_variants != signatures
    ):
        raise RuntimeError("validated cbrt capability is outside the requested scope")

    try:
        _write_json(pending_qualification_path, evidence)
        pending_ref = QualificationRef(
            registry_key=REGISTRY_KEY,
            evidence_path=pending_qualification_path.relative_to(
                output_root
            ).as_posix(),
            evidence_sha256=_sha256_file(pending_qualification_path),
        )
        pending_overlay = CapabilityOverlay(
            revision=1,
            parent_sha256=ZERO_SHA256,
            base_manifest_sha256=capability_manifest_sha256(base_manifest),
            entries=(pending_ref,),
        )
        pending_admitted = apply_overlay(
            base_manifest,
            pending_overlay,
            evidence_root=str(output_root),
        )
        _assert_profile_admitted(pending_admitted.operators[REGISTRY_KEY])

        _publish_json_atomic(qualification_path, evidence)
        pending_qualification_path.unlink()
        qualification_ref = QualificationRef(
            registry_key=REGISTRY_KEY,
            evidence_path=qualification_path.relative_to(output_root).as_posix(),
            evidence_sha256=_sha256_file(qualification_path),
        )
        overlay = CapabilityOverlay(
            revision=1,
            parent_sha256=ZERO_SHA256,
            base_manifest_sha256=capability_manifest_sha256(base_manifest),
            entries=(qualification_ref,),
        )
        admitted = apply_overlay(
            base_manifest,
            overlay,
            evidence_root=str(output_root),
        )
        admitted_capability = admitted.operators[REGISTRY_KEY]
        _assert_profile_admitted(admitted_capability)
        _publish_json_atomic(admitted_path, admitted.to_dict())
        expected_overlay_sha256 = _sha256_bytes(overlay.to_bytes())
        summary = {
            "schema_version": 1,
            "producer": PRODUCER,
            "status": "qualified_and_overlay_published",
            "operator": REGISTRY_KEY,
            "artifact_id": artifact_manifest["artifact_id"],
            "qualification_id": parsed.qualification_id,
            "qualification_path": qualification_path.relative_to(
                output_root
            ).as_posix(),
            "qualification_sha256": _sha256_file(qualification_path),
            "overlay_path": overlay_path.relative_to(output_root).as_posix(),
            "overlay_sha256": expected_overlay_sha256,
            "admitted_manifest_path": admitted_path.relative_to(
                output_root
            ).as_posix(),
            "admitted_manifest_sha256": _sha256_file(admitted_path),
            "analysis_profiles_path": analysis_profiles_path.relative_to(
                output_root
            ).as_posix(),
            "constraints": list(CONSTRAINTS),
            "signature": {"positional_args": 1, "keyword_args": []},
            "tested_shapes": [list(shape) for shape in SHAPES],
            "rtol": RTOL,
            "atol": ATOL,
            "signed_zero_verified": worker["signed_zero_verified"],
            "sealed_artifact_tests_passed": artifact_test_counts["tests"],
            "asnumpy_float32_tests_passed": asnumpy_test_counts["tests"],
            "numerical_cases": worker["numerical_cases"],
            "wrong_opp_call_failed": negative["call_failed"],
            "mapped_vendor_libraries": worker["mapped_vendor_libraries"],
        }
        _publish_json_atomic(summary_path, summary)
        # The overlay is the activation commit and is intentionally published
        # last.  Before this call, partial metadata cannot activate cbrt.
        published_overlay_sha256 = publish_overlay_atomic(str(overlay_path), overlay)
        if published_overlay_sha256 != expected_overlay_sha256:
            raise RuntimeError("published overlay SHA-256 is inconsistent")
    except BaseException as exc:
        try:
            _remove_final_publications(public_paths)
        except RuntimeError as rollback_exc:
            raise rollback_exc from exc
        raise

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--negative-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source")
    parser.add_argument("--original-spec")
    parser.add_argument("--artifact")
    parser.add_argument("--asnumpy-python")
    parser.add_argument("--asnumpy-wheel")
    parser.add_argument("--asnumpy-test")
    parser.add_argument("--out")
    parser.add_argument("--jobs", type=int, default=1)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.runtime_worker and args.negative_worker:
        raise RuntimeError("worker modes are mutually exclusive")
    if args.runtime_worker:
        return _runtime_worker()
    if args.negative_worker:
        return _negative_worker()
    required = (
        "source",
        "original_spec",
        "artifact",
        "asnumpy_python",
        "asnumpy_wheel",
        "asnumpy_test",
        "out",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise RuntimeError("missing required arguments: " + ", ".join(missing))
    if args.jobs != 1:
        raise RuntimeError("--jobs must be exactly 1 for this qualification")
    return _main(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"qualification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
