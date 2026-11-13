"""Private worker for hash-bound ordinary-script resident benchmarks.

The parent process is responsible for creating immutable source, migrated, and
migration-report snapshots.  This worker consumes those snapshots through a
bounded JSON contract and publishes only write-once ``.npy`` files plus a
write-once ``manifest.json``.  It is deliberately not a public API.

The command-line arguments are parsed and the process is pinned to the
requested CPU before NumPy or NumPilot is imported.  This is important: BLAS
libraries commonly inspect their thread environment while they are imported.
Process isolation is useful for repeatability and timeouts, but it is not an OS
security sandbox; benchmark scripts must already have passed NumPilot's
ordinary-script migration audit.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager, nullcontext
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import stat
import statistics
import sys
import time
import types
from typing import Any, Mapping


# The benchmark runs from an isolated side directory.  Keep the repository
# source tree importable when NumPilot is being used directly from a checkout
# (the supported development and demo workflow), even with Python -I.
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

MARKER = "# NUMPILOT_DEVICE_INPUTS_END"
THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_MAX_CONTRACT_BYTES = 256 * 1024
_MAX_SCRIPT_BYTES = 4 * 1024 * 1024
_MAX_REPORT_BYTES = 4 * 1024 * 1024
_MAX_OUTPUT_NAMES = 128
_MAX_OUTPUT_BYTES = 1024 * 1024 * 1024
_MAX_ERROR_BYTES = 16 * 1024
_ALLOWED_DTYPE_KINDS = frozenset("biufc")
_FORBIDDEN_SUFFIX_CALLS = frozenset({
    "array",
    "asarray",
    "from_numpy",
    "item",
    "to_numpy",
})
_FORBIDDEN_SUFFIX_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)


class _WorkerFailure(RuntimeError):
    """Stable, deliberately message-free worker failure."""

    def __init__(self, code: str, phase: str, exception_type: str | None = None):
        super().__init__(code)
        self.code = code
        self.phase = phase
        self.exception_type = exception_type


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json(payload: bytes, *, code: str, phase: str) -> Any:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=object_pairs)
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise _WorkerFailure(code, phase, type(exc).__name__) from exc


def _read_regular(path: Path, limit: int, *, phase: str = "worker_input") -> bytes:
    descriptor = -1
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(str(path), flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise _WorkerFailure("invalid_input_file", phase)
        if metadata.st_size > limit:
            raise _WorkerFailure("input_too_large", phase)
        chunks = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > limit:
            raise _WorkerFailure("input_too_large", phase)
        return payload
    except _WorkerFailure:
        raise
    except OSError as exc:
        raise _WorkerFailure("invalid_input_file", phase, type(exc).__name__) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        payload = (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _WorkerFailure(
            "manifest_serialization_failed", "manifest", type(exc).__name__
        ) from exc
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = -1
    try:
        descriptor = os.open(str(path), flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise _WorkerFailure("unexpected_output_file", "manifest") from exc
    except OSError as exc:
        raise _WorkerFailure(
            "manifest_publish_failed", "manifest", type(exc).__name__
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _absolute_path(value: object) -> Path | None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value) > 4096
        or not os.path.isabs(value)
    ):
        return None
    return Path(os.path.abspath(value))


def _migration_report_path(value: object) -> Path | None:
    if isinstance(value, dict) and set(value) == {"path"}:
        value = value["path"]
    return _absolute_path(value)


def _validate_contract(value: Any, backend: str) -> dict[str, Any]:
    """Validate the bounded worker contract without importing NumPy."""

    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise _WorkerFailure("invalid_contract", "worker_input")
    if value.get("backend") != backend:
        raise _WorkerFailure("invalid_contract", "worker_input")
    matching_roles = {
        "cpu": frozenset({"cpu", "reference"}),
        "npu": frozenset({"npu", "candidate"}),
    }
    if value.get("role") not in matching_roles[backend]:
        raise _WorkerFailure("invalid_contract", "worker_input")
    for field in ("script_sha256", "source_sha256"):
        if not _is_sha256(value.get(field)):
            raise _WorkerFailure("invalid_contract", "worker_input")
    expected_worker = value.get("worker_sha256")
    if expected_worker is not None and not _is_sha256(expected_worker):
        raise _WorkerFailure("invalid_contract", "worker_input")
    if value.get("marker") != MARKER:
        raise _WorkerFailure("invalid_contract", "worker_input")
    names = value.get("output_names")
    if (
        not isinstance(names, list)
        or not names
        or len(names) > _MAX_OUTPUT_NAMES
        or len(set(names)) != len(names)
        or any(
            not isinstance(name, str)
            or not name.isidentifier()
            or len(name) > 256
            for name in names
        )
    ):
        raise _WorkerFailure("invalid_contract", "worker_input")
    warmup = value.get("warmup")
    samples = value.get("samples")
    if (
        isinstance(warmup, bool)
        or not isinstance(warmup, int)
        or not 0 <= warmup <= 100
        or isinstance(samples, bool)
        or not isinstance(samples, int)
        or not 1 <= samples <= 1000
    ):
        raise _WorkerFailure("invalid_contract", "worker_input")
    if value.get("cpu_threads") != 1:
        raise _WorkerFailure("invalid_contract", "worker_input")
    max_output_bytes = value.get("max_output_bytes")
    if (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or not 1024 <= max_output_bytes <= _MAX_OUTPUT_BYTES
    ):
        raise _WorkerFailure("invalid_contract", "worker_input")

    report_path = _migration_report_path(value.get("migration_report"))
    reference_path = _absolute_path(value.get("reference_source"))
    report_sha256 = value.get("migration_report_sha256")
    if backend == "npu":
        if (
            report_path is None
            or reference_path is None
            or not _is_sha256(report_sha256)
        ):
            raise _WorkerFailure("invalid_contract", "worker_input")
    elif (
        value.get("migration_report") is not None
        or report_sha256 is not None
        or value.get("reference_source") is not None
    ):
        raise _WorkerFailure("invalid_contract", "worker_input")

    result = dict(value)
    result.setdefault("migration_report", None)
    result.setdefault("migration_report_sha256", None)
    result.setdefault("reference_source", None)
    result["_migration_report_path"] = report_path
    result["_reference_source_path"] = reference_path
    return result


def _load_contract(path: Path, backend: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular(path, _MAX_CONTRACT_BYTES)
    value = _strict_json(payload, code="invalid_contract", phase="worker_input")
    return _validate_contract(value, backend), payload


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return tuple(reversed(parts))


def _asnumpy_aliases(tree: ast.AST) -> set[str]:
    aliases = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import):
            continue
        for item in node.names:
            if item.name == "asnumpy":
                aliases.add(item.asname or "asnumpy")
    return aliases


def _numpy_aliases(tree: ast.AST) -> set[str]:
    aliases = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import):
            continue
        for item in node.names:
            if item.name == "numpy":
                aliases.add(item.asname or "numpy")
    return aliases


def _imports_module(tree: ast.AST, module_name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            item.name == module_name or item.name.startswith(module_name + ".")
            for item in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and (
            node.module == module_name
            or (node.module or "").startswith(module_name + ".")
        ):
            return True
    return False


def _asnumpy_call_counts(
    source: str, filename: str, aliases: set[str]
) -> Counter[str]:
    tree = ast.parse(source, filename=filename)
    counts: Counter[str] = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path = _attribute_path(node.func)
        if not path or path[0] not in aliases or len(path) < 2:
            continue
        canonical = ".".join(path[1:])
        # Host/device input preparation is permitted before the marker, but it
        # is not an NPU operator and therefore is not sent to the profile guard.
        if canonical not in {"from_numpy", "ndarray.from_numpy"}:
            counts[canonical] += 1
    return counts


def _simple_assignment_target(target: ast.AST) -> bool:
    if isinstance(target, ast.Name):
        return True
    if isinstance(target, (ast.Tuple, ast.List)) and target.elts:
        return all(_simple_assignment_target(item) for item in target.elts)
    return False


def _audit_suffix(
    source: str, filename: str, *, permitted_call_alias: str
) -> ast.Module:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise _WorkerFailure(
            "invalid_benchmark_script", "script_audit", type(exc).__name__
        ) from exc
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_SUFFIX_NODES):
            raise _WorkerFailure("forbidden_suffix_node", "script_audit")
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if (
                len(statement.targets) != 1
                or not _simple_assignment_target(statement.targets[0])
            ):
                raise _WorkerFailure("forbidden_suffix_statement", "script_audit")
        elif isinstance(statement, ast.AnnAssign):
            if (
                statement.value is None
                or not _simple_assignment_target(statement.target)
            ):
                raise _WorkerFailure("forbidden_suffix_statement", "script_audit")
        else:
            raise _WorkerFailure("forbidden_suffix_statement", "script_audit")
    for node in ast.walk(tree):
        if isinstance(node, (ast.AugAssign, ast.NamedExpr)):
            raise _WorkerFailure("forbidden_suffix_statement", "script_audit")
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
            raise _WorkerFailure("untraced_suffix_operation", "script_audit")
        if isinstance(node, ast.Call):
            path = _attribute_path(node.func)
            if path and path[-1] in _FORBIDDEN_SUFFIX_CALLS:
                raise _WorkerFailure("forbidden_suffix_transfer", "script_audit")
            if not path or path[0] != permitted_call_alias or len(path) < 2:
                raise _WorkerFailure("untraced_suffix_call", "script_audit")
            if any(keyword.arg in {None, "out"} for keyword in node.keywords):
                raise _WorkerFailure("forbidden_suffix_call", "script_audit")
    return tree


def _build_script_plan(
    payload: bytes, path: Path, backend: str, marker: str = MARKER
) -> dict[str, Any]:
    try:
        source = payload.decode("utf-8")
    except UnicodeError as exc:
        raise _WorkerFailure(
            "invalid_benchmark_script", "script_audit", type(exc).__name__
        ) from exc
    if source.count(marker) != 1:
        raise _WorkerFailure("invalid_timing_marker", "script_audit")
    try:
        full_tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise _WorkerFailure(
            "invalid_benchmark_script", "script_audit", type(exc).__name__
        ) from exc
    prefix, suffix = source.split(marker, 1)
    aliases = _asnumpy_aliases(full_tree)
    numpy_aliases = _numpy_aliases(full_tree)
    if (
        backend == "npu"
        and (aliases != {"ap"} or _imports_module(full_tree, "numpy"))
    ):
        raise _WorkerFailure("invalid_asnumpy_import", "script_audit")
    if backend == "cpu" and _imports_module(full_tree, "asnumpy"):
        raise _WorkerFailure("cpu_uses_asnumpy", "script_audit")
    if backend == "cpu" and numpy_aliases != {"np"}:
        raise _WorkerFailure("invalid_numpy_import", "script_audit")
    _audit_suffix(
        suffix,
        str(path) + ":suffix",
        permitted_call_alias="ap" if backend == "npu" else "np",
    )
    try:
        compile_start = time.perf_counter_ns()
        prefix_code = compile(
            prefix, str(path) + ":prefix", "exec", dont_inherit=True
        )
        suffix_code = compile(
            suffix, str(path) + ":suffix", "exec", dont_inherit=True
        )
        compile_ns = time.perf_counter_ns() - compile_start
    except (SyntaxError, ValueError) as exc:
        raise _WorkerFailure(
            "invalid_benchmark_script", "script_compile", type(exc).__name__
        ) from exc
    prefix_counts = _asnumpy_call_counts(prefix, str(path) + ":prefix", aliases)
    suffix_counts = _asnumpy_call_counts(suffix, str(path) + ":suffix", aliases)
    if backend == "npu" and not suffix_counts:
        raise _WorkerFailure("no_resident_npu_operations", "script_audit")
    return {
        "prefix_code": prefix_code,
        "suffix_code": suffix_code,
        "compile_ns": compile_ns,
        "prefix_call_counts": prefix_counts,
        "suffix_call_counts": suffix_counts,
    }


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        raise ValueError("percentile input is empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


class _AclDeviceSynchronizer:
    def __init__(self) -> None:
        try:
            self._library = ctypes.CDLL("libascendcl.so")
            self._function = self._library.aclrtSynchronizeDevice
            self._function.argtypes = []
            self._function.restype = ctypes.c_int
        except (AttributeError, OSError) as exc:
            raise _WorkerFailure(
                "synchronizer_unavailable", "npu_preflight", type(exc).__name__
            ) from exc
        self.calls = 0

    @property
    def identity(self) -> str:
        return str(self._library._name)

    def __call__(self) -> None:
        try:
            result = int(self._function())
        except Exception as exc:
            raise _WorkerFailure(
                "device_synchronization_failed", "resident_execution",
                type(exc).__name__,
            ) from exc
        if result != 0:
            raise _WorkerFailure(
                "device_synchronization_failed", "resident_execution"
            )
        self.calls += 1


class _NestedRuntimeProxy:
    def __init__(self, root, prefix, targets, guard, counts, cache):
        self._root = root
        self._prefix = prefix
        self._targets = targets
        self._guard = guard
        self._counts = counts
        self._cache = cache

    def _resolve_original(self, path: str) -> Any:
        value = self._root
        for part in path.split("."):
            value = getattr(value, part)
        return value

    def __getattr__(self, name: str) -> Any:
        canonical = f"{self._prefix}.{name}" if self._prefix else name
        if canonical in self._targets:
            if canonical not in self._cache:
                target = self._targets[canonical]

                def traced(*args, **kwargs):
                    self._guard.validate_call(canonical, args, kwargs)
                    self._counts[canonical] += 1
                    return target(*args, **kwargs)

                traced.__name__ = canonical
                self._cache[canonical] = traced
            return self._cache[canonical]
        if any(key.startswith(canonical + ".") for key in self._targets):
            if canonical not in self._cache:
                self._cache[canonical] = _NestedRuntimeProxy(
                    self._root,
                    canonical,
                    self._targets,
                    self._guard,
                    self._counts,
                    self._cache,
                )
            return self._cache[canonical]
        return self._resolve_original(canonical)


def _guarded_module(module, guard, operation_names):
    """Return a nested API proxy bound to the guard's exact target snapshot."""

    targets = dict(guard.runtime_targets or {})
    if set(targets) != set(operation_names):
        raise _WorkerFailure("profile_target_mismatch", "npu_preflight")
    counts = {name: 0 for name in sorted(operation_names)}
    cache = {}
    root_proxy = _NestedRuntimeProxy(
        module, "", targets, guard, counts, cache
    )

    class _RootProxy(types.ModuleType):
        def __getattr__(self, name):
            return root_proxy.__getattr__(name)

    proxy = _RootProxy(getattr(module, "__name__", "asnumpy"))
    for attribute in (
        "__file__", "__package__", "__path__", "__spec__", "__version__"
    ):
        if hasattr(module, attribute):
            setattr(proxy, attribute, getattr(module, attribute))
    return proxy, counts


def _identity_receipt(identity: Any) -> dict[str, Any]:
    if hasattr(identity, "to_dict"):
        value = identity.to_dict()
        if isinstance(value, dict):
            return value
    return {
        "status": getattr(identity, "status", None),
        "identity_sha256": getattr(identity, "identity_sha256", None),
        "audit_id": getattr(identity, "audit_id", None),
        "asnumpy_version": getattr(identity, "asnumpy_version", None),
        "issues": list(getattr(identity, "issues", ())),
    }


def _qualification_receipt(analysis: Any) -> dict[str, Any]:
    value = analysis.to_dict() if hasattr(analysis, "to_dict") else analysis
    if not isinstance(value, dict):
        raise _WorkerFailure("qualification_failed", "npu_preflight")
    calls = value.get("calls")
    all_eligible = (
        isinstance(calls, list)
        and bool(calls)
        and all(
            isinstance(call, dict)
            and call.get("decision") == "DIRECT"
            and call.get("performance_eligible") is True
            for call in calls
        )
    )
    safe_calls = calls if all_eligible else []
    receipt = {
        "source_sha256": value.get("source_sha256"),
        "decision": value.get("decision"),
        "runtime_identity_status": value.get("runtime_identity_status"),
        "runtime_identity_sha256": value.get("runtime_identity_sha256"),
        "all_calls_performance_eligible": all_eligible,
        "calls": [
            {
                "canonical_key": call.get("canonical_key"),
                "decision": call.get("decision"),
                "effective_route": call.get("effective_route"),
                "performance_eligible": call.get("performance_eligible"),
            }
            for call in safe_calls
        ],
        "issues": value.get("issues"),
    }
    if (
        receipt["decision"] != "DIRECT"
        or receipt["runtime_identity_status"] != "matched"
        or not receipt["all_calls_performance_eligible"]
    ):
        raise _WorkerFailure("qualification_failed", "npu_preflight")
    return receipt


def _validate_migration_report(
    report: Any,
    *,
    source_sha256: str,
    script_sha256: str,
    runtime_identity_sha256: str,
) -> dict[str, Any]:
    """Return explicit strict-NPU gates for a parsed migration report."""

    if not isinstance(report, dict):
        raise _WorkerFailure("invalid_migration_report", "npu_preflight")
    qualified = report.get("qualified_analysis")
    receipt = report.get("llm_receipt")
    calls = qualified.get("calls") if isinstance(qualified, dict) else None
    gates = {
        "schema": report.get("schema_version") == 2,
        "mode": report.get("mode") == "script",
        "status": report.get("status") in {
            "migrated_strict_npu",
            "migrated_scoped_npu",
        },
        # Scoped reports are admissible only when their bound analysis proves
        # every call is inside a native performance profile. The worker then
        # rechecks the runtime guard and call provenance before publishing.
        "strict_npu": (
            report.get("strict_npu") is True
            or (
                report.get("status") == "migrated_scoped_npu"
                and isinstance(qualified, dict)
                and qualified.get("decision") == "DIRECT"
                and all(
                    isinstance(call, dict)
                    and call.get("decision") == "DIRECT"
                    and call.get("performance_eligible") is True
                    for call in calls or ()
                )
            )
        ),
        "actual_remote_call": report.get("actual_remote_call") is True,
        "ast_audit": report.get("ast_audit") == "passed",
        "no_claimed_execution": report.get("execution_performed") is False,
        "numerical_pending": report.get("numerical_verification") == "not_run",
        "performance_pending": report.get("performance_valid") is False,
        "published_files": report.get("published_files") == ["migrated.py"],
        "source_hash": report.get("source_sha256") == source_sha256,
        "migrated_hash": report.get("migrated_sha256") == script_sha256,
        "qualified_source_hash": (
            isinstance(qualified, dict)
            and qualified.get("source_sha256") == source_sha256
        ),
        "qualified_decision": (
            isinstance(qualified, dict)
            and qualified.get("decision") == "DIRECT"
        ),
        "qualified_runtime": (
            isinstance(qualified, dict)
            and qualified.get("runtime_identity_status") == "matched"
            and qualified.get("runtime_identity_sha256")
            == runtime_identity_sha256
        ),
        "qualified_calls": (
            isinstance(calls, list)
            and bool(calls)
            and all(
                isinstance(call, dict)
                and call.get("decision") == "DIRECT"
                and call.get("performance_eligible") is True
                for call in calls
            )
        ),
        "llm_http": (
            isinstance(receipt, dict) and receipt.get("http_status") == 200
        ),
        "llm_tls": (
            isinstance(receipt, dict)
            and receipt.get("tls_verification") == "enabled"
        ),
        "llm_model_exact": (
            isinstance(receipt, dict)
            and receipt.get("response_model_identity") == "exact"
        ),
    }
    if not all(gates.values()):
        raise _WorkerFailure("migration_report_gate_failed", "npu_preflight")
    return gates


@contextmanager
def _silence_process_output():
    """Redirect both Python and native stdout/stderr while user code runs."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    null_descriptor = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_descriptor, 1)
        os.dup2(null_descriptor, 2)
        yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(null_descriptor)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _run_resident(
    plan: Mapping[str, Any],
    *,
    warmup: int,
    samples: int,
    filename: str,
    execution_context=None,
    synchronizer=None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute prefix once and a straight-line resident suffix repeatedly."""

    namespace = {
        "__name__": "__main__",
        "__file__": filename,
        "__package__": None,
        "__cached__": None,
        "__builtins__": __builtins__,
    }
    warmup_ns: list[int] = []
    samples_ns: list[int] = []
    context = execution_context if execution_context is not None else nullcontext()
    was_gc_enabled = gc.isenabled()
    try:
        with context:
            prefix_start = time.perf_counter_ns()
            exec(plan["prefix_code"], namespace, namespace)
            if synchronizer is not None:
                synchronizer()
            prefix_ns = time.perf_counter_ns() - prefix_start
            if was_gc_enabled:
                gc.disable()
            for index in range(warmup + samples):
                if synchronizer is not None:
                    synchronizer()
                start = time.perf_counter_ns()
                exec(plan["suffix_code"], namespace, namespace)
                if synchronizer is not None:
                    synchronizer()
                elapsed = time.perf_counter_ns() - start
                (warmup_ns if index < warmup else samples_ns).append(elapsed)
    except _WorkerFailure:
        raise
    except BaseException as exc:
        raise _WorkerFailure(
            "script_exception", "resident_execution", type(exc).__name__
        ) from exc
    finally:
        if was_gc_enabled:
            gc.enable()
    timing = {
        "clock": "time.perf_counter_ns",
        "scope": (
            "marker suffix only; prefix/input preparation executes once; "
            "resident inputs; output allocation and device completion included"
        ),
        "excludes": [
            "compile",
            "prefix/input preparation",
            "host-to-device input preparation",
            "final device-to-host output transfer",
            "output serialization",
        ],
        "compile_ns": int(plan["compile_ns"]),
        "prefix_input_preparation_ns": prefix_ns,
        "warmup_ns": warmup_ns,
        "samples_ns": samples_ns,
        "median_ns": float(statistics.median(samples_ns)),
        "p10_ns": _percentile(samples_ns, 0.10),
        "p90_ns": _percentile(samples_ns, 0.90),
    }
    return namespace, timing


def _normalise_output(value: Any, name: str, np: Any, backend: str):
    transfer_calls = 0
    start = time.perf_counter_ns()
    try:
        if backend == "npu":
            transfer = getattr(value, "to_numpy", None)
            if not callable(transfer):
                raise _WorkerFailure(
                    "npu_output_not_device_resident", "output_conversion:" + name
                )
            value = transfer()
            transfer_calls = 1
        elif callable(getattr(value, "to_numpy", None)):
            raise _WorkerFailure("cpu_output_is_device_value", "output_conversion:" + name)
        if not isinstance(
            value, (np.ndarray, np.generic, bool, int, float, complex)
        ):
            raise _WorkerFailure("unsupported_dtype", "output_conversion:" + name)
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.kind not in _ALLOWED_DTYPE_KINDS:
            raise _WorkerFailure("unsupported_dtype", "output_conversion:" + name)
        result = np.array(
            array, dtype=array.dtype, copy=True, order="C", subok=False
        )
    except _WorkerFailure:
        raise
    except Exception as exc:
        raise _WorkerFailure(
            "output_conversion_failed", "output_conversion:" + name,
            type(exc).__name__,
        ) from exc
    transfer_ns = time.perf_counter_ns() - start
    return result, transfer_calls, transfer_ns


def _save_outputs(
    output_dir: Path,
    namespace: Mapping[str, Any],
    names: list[str],
    np: Any,
    backend: str,
    max_output_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    missing = [name for name in names if name not in namespace]
    if missing:
        raise _WorkerFailure("missing_output", "output_collection")
    entries = []
    total_data_bytes = 0
    total_serialized_bytes = 0
    transfer_calls = 0
    transfer_ns = []
    for index, name in enumerate(names):
        array, calls, elapsed = _normalise_output(
            namespace[name], name, np, backend
        )
        transfer_calls += calls
        transfer_ns.append(elapsed)
        total_data_bytes += int(array.nbytes)
        if total_data_bytes > max_output_bytes:
            raise _WorkerFailure("output_too_large", "output_serialization")
        filename = "output-%04d.npy" % index
        path = output_dir / filename
        try:
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            path.chmod(0o600)
        except FileExistsError as exc:
            raise _WorkerFailure(
                "unexpected_output_file", "output_serialization:" + name
            ) from exc
        except Exception as exc:
            raise _WorkerFailure(
                "output_serialization_failed", "output_serialization:" + name,
                type(exc).__name__,
            ) from exc
        payload = _read_regular(
            path, max_output_bytes, phase="output_serialization:" + name
        )
        total_serialized_bytes += len(payload)
        if total_serialized_bytes > max_output_bytes:
            raise _WorkerFailure("output_too_large", "output_serialization")
        entries.append({
            "name": name,
            "filename": filename,
            "sha256": _sha256(payload),
            "shape": [int(item) for item in array.shape],
            "dtype": array.dtype.str,
            "elements": int(array.size),
            "data_bytes": int(array.nbytes),
            "serialized_bytes": len(payload),
            "finite": bool(np.isfinite(array).all())
            if array.dtype.kind in "fc" else True,
            "device_to_host_calls": calls,
            "device_to_host_ns": elapsed,
        })
    if backend == "npu" and transfer_calls != len(names):
        raise _WorkerFailure("output_transfer_count_mismatch", "output_conversion")
    return entries, {
        "total_data_bytes": total_data_bytes,
        "total_serialized_bytes": total_serialized_bytes,
        "device_to_host_calls": transfer_calls,
        "expected_device_to_host_calls": len(names) if backend == "npu" else 0,
        "per_output_ns": transfer_ns,
        "total_ns": sum(transfer_ns),
    }


def _threadpool_info() -> list[dict[str, Any]]:
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        observed = []
    else:
        observed = [
            {
                "source": "threadpoolctl",
                "user_api": item.get("user_api"),
                "internal_api": item.get("internal_api"),
                "prefix": item.get("prefix"),
                "filepath": item.get("filepath"),
                "num_threads": item.get("num_threads"),
                "version": item.get("version"),
            }
            for item in threadpool_info()
        ]
    if observed:
        return observed
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    libraries = set()
    try:
        lines = maps.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        fields = line.split(None, 5)
        if len(fields) == 6 and "openblas" in fields[5].lower():
            libraries.add(fields[5])
    for library_path in sorted(libraries):
        try:
            library = ctypes.CDLL(library_path)
        except OSError:
            continue
        getter = None
        for symbol in (
            "scipy_openblas_get_num_threads64_",
            "scipy_openblas_get_num_threads_64_",
            "openblas_get_num_threads64_",
            "openblas_get_num_threads_64_",
            "openblas_get_num_threads",
        ):
            try:
                getter = getattr(library, symbol)
            except AttributeError:
                continue
            break
        if getter is None:
            continue
        getter.argtypes = []
        getter.restype = ctypes.c_int
        observed.append({
            "source": "openblas_runtime_api",
            "user_api": "blas",
            "internal_api": "openblas",
            "prefix": "openblas",
            "filepath": library_path,
            "num_threads": int(getter()),
            "version": None,
        })
    return observed


def _prepare_process_controls(core: int) -> dict[str, Any]:
    if isinstance(core, bool) or not isinstance(core, int) or core < 0:
        raise _WorkerFailure("invalid_core", "process_preflight")
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise _WorkerFailure("affinity_unavailable", "process_preflight")
    try:
        affinity_before = sorted(os.sched_getaffinity(0))
        if core not in affinity_before:
            raise _WorkerFailure("core_not_available", "process_preflight")
        os.sched_setaffinity(0, {core})
        affinity_bound = sorted(os.sched_getaffinity(0))
    except _WorkerFailure:
        raise
    except OSError as exc:
        raise _WorkerFailure(
            "affinity_failed", "process_preflight", type(exc).__name__
        ) from exc
    if affinity_bound != [core]:
        raise _WorkerFailure("affinity_failed", "process_preflight")
    environment = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    if any(value != "1" for value in environment.values()):
        raise _WorkerFailure("thread_environment_mismatch", "process_preflight")
    return {
        "requested_core": core,
        "before": affinity_before,
        "after_binding": affinity_bound,
        "after_execution": None,
    }, environment


def _assert_bound_inputs_unchanged(bindings: Mapping[str, Any]) -> None:
    for item in bindings.values():
        if item is None:
            continue
        payload = _read_regular(item["path"], item["limit"], phase="input_drift")
        if _sha256(payload) != item["sha256"]:
            raise _WorkerFailure("input_drift", "input_drift")


def _execute(
    backend_name: str,
    script_path: Path,
    contract_path: Path,
    output_dir: Path,
    affinity: dict[str, Any],
    thread_environment: Mapping[str, str | None],
) -> None:
    contract, contract_payload = _load_contract(contract_path, backend_name)
    worker_path = Path(__file__).resolve()
    worker_payload = _read_regular(worker_path, _MAX_SCRIPT_BYTES)
    worker_sha256 = _sha256(worker_payload)
    if (
        contract.get("worker_sha256") is not None
        and contract["worker_sha256"] != worker_sha256
    ):
        raise _WorkerFailure("worker_hash_mismatch", "worker_input")
    script_payload = _read_regular(script_path, _MAX_SCRIPT_BYTES)
    script_sha256 = _sha256(script_payload)
    if script_sha256 != contract["script_sha256"]:
        raise _WorkerFailure("script_hash_mismatch", "worker_input")
    if backend_name == "cpu" and script_sha256 != contract["source_sha256"]:
        raise _WorkerFailure("source_hash_mismatch", "worker_input")

    source_path = contract["_reference_source_path"]
    report_path = contract["_migration_report_path"]
    source_payload = script_payload
    report_payload = None
    if backend_name == "npu":
        source_payload = _read_regular(source_path, _MAX_SCRIPT_BYTES)
        if _sha256(source_payload) != contract["source_sha256"]:
            raise _WorkerFailure("source_hash_mismatch", "worker_input")
        report_payload = _read_regular(report_path, _MAX_REPORT_BYTES)
        if _sha256(report_payload) != contract["migration_report_sha256"]:
            raise _WorkerFailure("migration_report_hash_mismatch", "worker_input")

    plan = _build_script_plan(
        script_payload, script_path, backend_name, contract["marker"]
    )
    bindings = {
        "contract": {
            "path": contract_path,
            "limit": _MAX_CONTRACT_BYTES,
            "sha256": _sha256(contract_payload),
        },
        "worker": {
            "path": worker_path,
            "limit": _MAX_SCRIPT_BYTES,
            "sha256": worker_sha256,
        },
        "script": {
            "path": script_path,
            "limit": _MAX_SCRIPT_BYTES,
            "sha256": script_sha256,
        },
        "source": None if backend_name == "cpu" else {
            "path": source_path,
            "limit": _MAX_SCRIPT_BYTES,
            "sha256": contract["source_sha256"],
        },
        "migration_report": None if backend_name == "cpu" else {
            "path": report_path,
            "limit": _MAX_REPORT_BYTES,
            "sha256": contract["migration_report_sha256"],
        },
    }

    try:
        import numpy as np
    except Exception as exc:
        raise _WorkerFailure(
            "numpy_unavailable", "worker_import", type(exc).__name__
        ) from exc

    backend = None
    identity_before = None
    identity_after = None
    report_gates = None
    current_qualification = None
    profile_receipt = None
    expected_counts = None
    runtime_counts = None
    synchronizer = None
    execution_context = nullcontext()
    with _silence_process_output():
        if backend_name == "npu":
            try:
                from numpilot.backends.npu import NpuBackend
                from numpilot.module_injection import temporary_module
                from numpilot.qualified_analysis import analyze_qualified_script
            except Exception as exc:
                raise _WorkerFailure(
                    "numpilot_unavailable", "worker_import", type(exc).__name__
                ) from exc
            backend = NpuBackend()
            identity_before = backend.runtime_identity()
            if getattr(identity_before, "status", None) != "matched":
                raise _WorkerFailure("runtime_identity_mismatch", "npu_preflight")
            report = _strict_json(
                report_payload,
                code="invalid_migration_report",
                phase="npu_preflight",
            )
            report_gates = _validate_migration_report(
                report,
                source_sha256=contract["source_sha256"],
                script_sha256=script_sha256,
                runtime_identity_sha256=identity_before.identity_sha256,
            )
            try:
                source_text = source_payload.decode("utf-8")
                analysis = analyze_qualified_script(
                    source_text,
                    str(source_path),
                    runtime_identity=identity_before,
                )
            except Exception as exc:
                raise _WorkerFailure(
                    "qualification_failed", "npu_preflight", type(exc).__name__
                ) from exc
            current_qualification = _qualification_receipt(analysis)
            if (
                current_qualification["source_sha256"] != contract["source_sha256"]
                or current_qualification["runtime_identity_sha256"]
                != identity_before.identity_sha256
            ):
                raise _WorkerFailure("qualification_failed", "npu_preflight")
            prefix_counts = plan["prefix_call_counts"]
            suffix_counts = plan["suffix_call_counts"]
            operation_names = tuple(sorted(prefix_counts | suffix_counts))
            try:
                guard = backend.prepare_profile_guard(operation_names)
                proxy, runtime_counts = _guarded_module(
                    backend.get_module(), guard, operation_names
                )
            except _WorkerFailure:
                raise
            except Exception as exc:
                raise _WorkerFailure(
                    "profile_guard_failed", "npu_preflight", type(exc).__name__
                ) from exc
            execution_context = temporary_module("asnumpy", proxy)
            synchronizer = _AclDeviceSynchronizer()

        namespace, timing = _run_resident(
            plan,
            warmup=contract["warmup"],
            samples=contract["samples"],
            filename=str(script_path),
            execution_context=execution_context,
            synchronizer=synchronizer,
        )
        if backend_name == "npu":
            iterations = contract["warmup"] + contract["samples"]
            expected_counts = {
                name: (
                    plan["prefix_call_counts"][name]
                    + plan["suffix_call_counts"][name] * iterations
                )
                for name in sorted(
                    plan["prefix_call_counts"] | plan["suffix_call_counts"]
                )
            }
            if dict(runtime_counts) != expected_counts:
                raise _WorkerFailure(
                    "runtime_call_count_mismatch", "resident_execution"
                )
            profile_receipt = guard.to_dict()
            if any(
                profile_receipt.get(field) is not True
                for field in (
                    "strict_npu",
                    "runtime_profile_ok",
                    "runtime_target_integrity_ok",
                )
            ):
                raise _WorkerFailure("profile_guard_failed", "resident_execution")

        outputs, transfer = _save_outputs(
            output_dir,
            namespace,
            contract["output_names"],
            np,
            backend_name,
            contract["max_output_bytes"],
        )
        del namespace
        if backend_name == "npu":
            identity_after = backend.runtime_identity()
            if (
                getattr(identity_after, "status", None) != "matched"
                or identity_after.identity_sha256 != identity_before.identity_sha256
            ):
                raise _WorkerFailure(
                    "runtime_identity_drift", "runtime_identity"
                )

    pools = _threadpool_info()
    blas_pools = [pool for pool in pools if pool.get("user_api") == "blas"]
    if backend_name == "cpu" and any(
        pool.get("num_threads") != 1 for pool in blas_pools
    ):
        raise _WorkerFailure("cpu_threadpool_mismatch", "threadpool_gate")
    try:
        affinity_after = sorted(os.sched_getaffinity(0))
    except OSError as exc:
        raise _WorkerFailure(
            "affinity_failed", "process_postflight", type(exc).__name__
        ) from exc
    if affinity_after != [affinity["requested_core"]]:
        raise _WorkerFailure("affinity_drift", "process_postflight")
    affinity["after_execution"] = affinity_after

    _assert_bound_inputs_unchanged(bindings)
    iterations = contract["warmup"] + contract["samples"]
    sync_observed = 0 if synchronizer is None else synchronizer.calls
    sync_expected = 0 if synchronizer is None else 1 + 2 * iterations
    if sync_observed != sync_expected:
        raise _WorkerFailure(
            "device_synchronization_count_mismatch", "resident_execution"
        )
    timing["final_output_transfer"] = transfer
    manifest = {
        "schema_version": 1,
        "producer": "numpilot-script-benchmark-worker-v1",
        "status": "pass",
        "role": contract["role"],
        "backend": backend_name,
        "pid": os.getpid(),
        "python_executable": sys.executable,
        "contract_sha256": _sha256(contract_payload),
        "script_sha256": script_sha256,
        "source_sha256": contract["source_sha256"],
        "migration_report_sha256": contract["migration_report_sha256"],
        "worker_sha256": worker_sha256,
        "affinity": affinity,
        "thread_environment": dict(thread_environment),
        "threadpools": pools,
        "cpu_blas_thread_gate": {
            "required": backend_name == "cpu",
            "observed_pool_count": len(blas_pools),
            "passed": backend_name != "cpu" or all(
                pool.get("num_threads") == 1 for pool in blas_pools
            ),
        },
        "protocol": {
            "marker": MARKER,
            "prefix_executions": 1,
            "suffix_warmup_executions": contract["warmup"],
            "suffix_sample_executions": contract["samples"],
            "resident_compute_only": True,
        },
        "timing": timing,
        "outputs": outputs,
        "total_data_bytes": transfer["total_data_bytes"],
        "total_serialized_bytes": transfer["total_serialized_bytes"],
        "migration_report": None if report_path is None else {
            "path": str(report_path),
            "sha256": contract["migration_report_sha256"],
            "gates": report_gates,
        },
        "current_qualification": current_qualification,
        "runtime_identity_before": None
        if identity_before is None else _identity_receipt(identity_before),
        "runtime_identity_after": None
        if identity_after is None else _identity_receipt(identity_after),
        "profile_guard": profile_receipt,
        "runtime_call_counts": None if runtime_counts is None else {
            "expected": expected_counts,
            "observed": dict(runtime_counts),
        },
        "synchronization": {
            "api": None if synchronizer is None else "aclrtSynchronizeDevice",
            "library": None if synchronizer is None else synchronizer.identity,
            "prefix_completion_calls": 0 if synchronizer is None else 1,
            "before_and_after_each_resident_execution": synchronizer is not None,
            "observed_calls": sync_observed,
            "expected_calls": sync_expected,
        },
    }
    _write_exclusive_json(output_dir / "manifest.json", manifest)


def _publish_error(output_dir: Path, failure: _WorkerFailure) -> None:
    value = {
        "schema_version": 1,
        "status": "failed",
        "code": failure.code,
        "phase": failure.phase,
        "exception_type": failure.exception_type,
    }
    rendered = json.dumps(value, sort_keys=True).encode("utf-8")
    if len(rendered) > _MAX_ERROR_BYTES:
        value = {
            "schema_version": 1,
            "status": "failed",
            "code": "worker_error_metadata_too_large",
            "phase": "worker_error",
            "exception_type": None,
        }
    try:
        _write_exclusive_json(output_dir / "error.json", value)
    except Exception:
        pass


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", choices=("cpu", "npu"), required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--core", type=int, required=True)
    return parser.parse_args(argv)


def _absolute_cli_path(value: str) -> Path:
    path = _absolute_path(value)
    if path is None:
        raise _WorkerFailure("invalid_path", "worker_input")
    return path


def main(argv: list[str] | None = None) -> int:
    # Keep this order stable.  No NumPy/NumPilot import may move above process
    # affinity and thread-environment validation.
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    output_dir: Path | None = None
    try:
        affinity, thread_environment = _prepare_process_controls(args.core)
        script_path = _absolute_cli_path(args.script)
        contract_path = _absolute_cli_path(args.contract)
        output_dir = _absolute_cli_path(args.out)
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise _WorkerFailure("invalid_output_directory", "worker_input")
        if any(output_dir.iterdir()):
            raise _WorkerFailure("unexpected_output_file", "worker_input")
        _execute(
            args.backend,
            script_path,
            contract_path,
            output_dir,
            affinity,
            thread_environment,
        )
    except _WorkerFailure as exc:
        if output_dir is not None and output_dir.is_dir():
            _publish_error(output_dir, exc)
        return 2
    except BaseException as exc:
        if output_dir is not None and output_dir.is_dir():
            _publish_error(
                output_dir,
                _WorkerFailure(
                    "worker_internal_error", "worker_internal", type(exc).__name__
                ),
            )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
