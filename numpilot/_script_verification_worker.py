"""Private subprocess worker for ordinary-script numerical verification.

This module is intentionally not a public API.  The parent process invokes the
file with ``python -I`` and communicates only through bounded JSON manifests and
``.npy`` files written with ``allow_pickle=False``.  User stdout/stderr is not a
result channel.

Process isolation is useful for state separation and timeouts, but it is not an
OS security sandbox.  NumPilot only executes migration artifacts that have
already passed the ordinary-script AST audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


_MAX_CONTRACT_BYTES = 256 * 1024
_MAX_ERROR_BYTES = 16 * 1024
_MAX_MODULE_FILE_BYTES = 256 * 1024 * 1024
_ALLOWED_DTYPE_KINDS = frozenset("biufc")


class _WorkerFailure(RuntimeError):
    def __init__(self, code: str, phase: str, exception_type: str | None = None):
        super().__init__(code)
        self.code = code
        self.phase = phase
        self.exception_type = exception_type


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise _WorkerFailure("invalid_input_file", "worker_input")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _WorkerFailure(
            "invalid_input_file", "worker_input", type(exc).__name__
        ) from exc
    if size > limit:
        raise _WorkerFailure("input_too_large", "worker_input")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise _WorkerFailure(
            "input_read_failed", "worker_input", type(exc).__name__
        ) from exc
    if len(payload) > limit:
        raise _WorkerFailure("input_too_large", "worker_input")
    return payload


def _write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _load_contract(path: Path) -> dict[str, Any]:
    raw = _read_regular(path, _MAX_CONTRACT_BYTES)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise _WorkerFailure(
            "invalid_contract", "worker_input", type(exc).__name__
        ) from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise _WorkerFailure("invalid_contract", "worker_input")
    names = value.get("output_names")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name.isidentifier() for name in names)
        or len(set(names)) != len(names)
    ):
        raise _WorkerFailure("invalid_contract", "worker_input")
    expected_sha256 = value.get("script_sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise _WorkerFailure("invalid_contract", "worker_input")
    worker_sha256 = value.get("worker_sha256")
    if (
        not isinstance(worker_sha256, str)
        or len(worker_sha256) != 64
        or any(character not in "0123456789abcdef" for character in worker_sha256)
    ):
        raise _WorkerFailure("invalid_contract", "worker_input")
    invocation_path = value.get("python_invocation_path")
    if (
        not isinstance(invocation_path, str)
        or not invocation_path
        or "\x00" in invocation_path
        or len(invocation_path) > 4096
        or not os.path.isabs(invocation_path)
    ):
        raise _WorkerFailure("invalid_contract", "worker_input")
    invocation_realpath = value.get("python_invocation_realpath")
    if (
        not isinstance(invocation_realpath, str)
        or not invocation_realpath
        or "\x00" in invocation_realpath
        or len(invocation_realpath) > 4096
        or not os.path.isabs(invocation_realpath)
    ):
        raise _WorkerFailure("invalid_contract", "worker_input")
    invocation_sha256 = value.get("python_invocation_sha256")
    if (
        not isinstance(invocation_sha256, str)
        or len(invocation_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in invocation_sha256
        )
    ):
        raise _WorkerFailure("invalid_contract", "worker_input")
    invocation_size = value.get("python_invocation_size")
    if (
        not isinstance(invocation_size, int)
        or isinstance(invocation_size, bool)
        or invocation_size <= 0
        or invocation_size > _MAX_MODULE_FILE_BYTES
    ):
        raise _WorkerFailure("invalid_contract", "worker_input")
    max_output_bytes = value.get("max_output_bytes")
    if (
        not isinstance(max_output_bytes, int)
        or isinstance(max_output_bytes, bool)
        or max_output_bytes < 1024
    ):
        raise _WorkerFailure("invalid_contract", "worker_input")
    max_script_bytes = value.get("max_script_bytes")
    if (
        not isinstance(max_script_bytes, int)
        or isinstance(max_script_bytes, bool)
        or max_script_bytes < 1024
    ):
        raise _WorkerFailure("invalid_contract", "worker_input")
    return value


def _normalise_output(value: Any, name: str, np: Any) -> Any:
    phase = "output_conversion:" + name
    try:
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()
        if not isinstance(
            value, (np.ndarray, np.generic, bool, int, float, complex)
        ):
            raise _WorkerFailure("unsupported_dtype", phase)
        array = np.asarray(value)
    except _WorkerFailure:
        raise
    except Exception as exc:
        raise _WorkerFailure(
            "output_conversion_failed", phase, type(exc).__name__
        ) from exc
    if array.dtype.hasobject or array.dtype.kind not in _ALLOWED_DTYPE_KINDS:
        raise _WorkerFailure("unsupported_dtype", phase)
    try:
        # Detach the receipt value from mutable user-authored storage while
        # preserving scalar shape and the exact dtype.
        return np.array(array, dtype=array.dtype, copy=True, order="C", subok=False)
    except Exception as exc:
        raise _WorkerFailure(
            "output_conversion_failed", phase, type(exc).__name__
        ) from exc


def _file_evidence(raw_path: Any) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "path": None,
        "realpath": None,
        "file_size": None,
        "file_sha256": None,
        "file_hash_status": "unavailable",
    }
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        return evidence
    path = os.path.abspath(raw_path)
    realpath = os.path.realpath(path)
    evidence["path"] = path
    evidence["realpath"] = realpath
    descriptor = -1
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(realpath, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            return evidence
        evidence["file_size"] = int(metadata.st_size)
        if metadata.st_size > _MAX_MODULE_FILE_BYTES:
            evidence["file_hash_status"] = "too_large"
            return evidence
        digest = hashlib.sha256()
        total = 0
        while total <= _MAX_MODULE_FILE_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_MODULE_FILE_BYTES + 1 - total),
            )
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        if total > _MAX_MODULE_FILE_BYTES:
            evidence["file_hash_status"] = "too_large"
            evidence["file_sha256"] = None
            return evidence
        evidence["file_sha256"] = digest.hexdigest()
        evidence["file_hash_status"] = "hashed"
    except OSError:
        evidence["file_hash_status"] = "unavailable"
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return evidence


def _module_file_evidence(module: Any) -> dict[str, Any]:
    evidence = _file_evidence(getattr(module, "__file__", None))
    version = getattr(module, "__version__", None)
    evidence["version"] = (
        str(version)[:256]
        if isinstance(version, (str, int, float))
        else None
    )
    return evidence


def _freeze_runtime_identity(np: Any, contract: dict[str, Any]) -> dict[str, Any]:
    executable = os.path.abspath(sys.executable)
    executable_evidence = _file_evidence(executable)
    if (
        executable_evidence["file_hash_status"] != "hashed"
        or executable_evidence["realpath"]
        != contract["python_invocation_realpath"]
        or executable_evidence["file_sha256"]
        != contract["python_invocation_sha256"]
        or executable_evidence["file_size"]
        != contract["python_invocation_size"]
    ):
        raise _WorkerFailure("python_identity_mismatch", "worker_identity")
    return {
        "pid": os.getpid(),
        "python_invocation_path": contract["python_invocation_path"],
        "python_executable": executable,
        "python_executable_realpath": executable_evidence["realpath"],
        "python_executable_sha256": executable_evidence["file_sha256"],
        "python_executable_size": executable_evidence["file_size"],
        "python_prefix": os.path.abspath(sys.prefix),
        "python_base_prefix": os.path.abspath(
            getattr(sys, "base_prefix", sys.prefix)
        ),
        "python_version": "%d.%d.%d" % sys.version_info[:3],
        "numpy": _module_file_evidence(np),
    }


def _loaded_runtime_modules(frozen_numpy: dict[str, Any]) -> dict[str, Any]:
    evidence = {"numpy": dict(frozen_numpy)}
    for name in ("asnumpy", "asnumpy._core"):
        module = sys.modules.get(name)
        if module is not None:
            evidence[name] = _module_file_evidence(module)
    return evidence


def _execute(script: Path, contract: dict[str, Any], output_dir: Path) -> None:
    worker_payload = _read_regular(Path(__file__), int(contract["max_script_bytes"]))
    if _sha256(worker_payload) != contract["worker_sha256"]:
        raise _WorkerFailure("worker_hash_mismatch", "worker_input")
    script_payload = _read_regular(script, int(contract["max_script_bytes"]))
    if _sha256(script_payload) != contract["script_sha256"]:
        raise _WorkerFailure("script_hash_mismatch", "worker_input")

    try:
        import numpy as np
    except Exception as exc:
        raise _WorkerFailure(
            "numpy_unavailable", "worker_import", type(exc).__name__
        ) from exc
    runtime_identity = _freeze_runtime_identity(np, contract)

    namespace = {
        "__name__": "__main__",
        "__file__": str(script),
        "__package__": None,
        "__cached__": None,
        "__builtins__": __builtins__,
    }
    try:
        # Compile exactly the bytes whose digest was checked above.  Reopening
        # the pathname through runpy would create a validation/use race.
        code = compile(script_payload, str(script), "exec", dont_inherit=True)
        exec(code, namespace, namespace)
    except BaseException as exc:
        raise _WorkerFailure(
            "script_exception", "script_execution", type(exc).__name__
        ) from exc

    missing = [name for name in contract["output_names"] if name not in namespace]
    if missing:
        raise _WorkerFailure("missing_output", "output_collection")

    entries: list[dict[str, Any]] = []
    total_data_bytes = 0
    total_serialized_bytes = 0
    for index, name in enumerate(contract["output_names"]):
        array = _normalise_output(namespace[name], name, np)
        next_data_bytes = total_data_bytes + int(array.nbytes)
        if next_data_bytes > int(contract["max_output_bytes"]):
            raise _WorkerFailure("output_too_large", "output_serialization:" + name)
        total_data_bytes = next_data_bytes
        filename = "output-%04d.npy" % index
        output_path = output_dir / filename
        try:
            with output_path.open("xb") as handle:
                np.save(handle, array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise _WorkerFailure(
                "unexpected_output_file", "output_serialization:" + name
            ) from exc
        except Exception as exc:
            raise _WorkerFailure(
                "output_serialization_failed",
                "output_serialization:" + name,
                type(exc).__name__,
            ) from exc
        payload = _read_regular(output_path, int(contract["max_output_bytes"]))
        total_serialized_bytes += len(payload)
        if total_serialized_bytes > int(contract["max_output_bytes"]):
            raise _WorkerFailure("output_too_large", "output_serialization")
        entries.append({
            "name": name,
            "filename": filename,
            "sha256": _sha256(payload),
            "serialized_bytes": len(payload),
            "data_bytes": int(array.nbytes),
            "dtype": array.dtype.str,
            "shape": [int(item) for item in array.shape],
        })

    manifest = {
        "schema_version": 1,
        "status": "completed",
        "script_sha256": contract["script_sha256"],
        "worker_sha256": contract["worker_sha256"],
        "pid": runtime_identity["pid"],
        "python_invocation_path": runtime_identity["python_invocation_path"],
        "python_executable": runtime_identity["python_executable"],
        "python_executable_realpath": runtime_identity[
            "python_executable_realpath"
        ],
        "python_executable_sha256": runtime_identity[
            "python_executable_sha256"
        ],
        "python_executable_size": runtime_identity["python_executable_size"],
        "python_prefix": runtime_identity["python_prefix"],
        "python_base_prefix": runtime_identity["python_base_prefix"],
        "python_version": runtime_identity["python_version"],
        "loaded_runtime_modules": _loaded_runtime_modules(
            runtime_identity["numpy"]
        ),
        "output_names": list(contract["output_names"]),
        "total_data_bytes": total_data_bytes,
        "total_serialized_bytes": total_serialized_bytes,
        "outputs": entries,
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
    # Keep the error channel deliberately tiny and free from exception messages,
    # tracebacks, stdout, and stderr.
    rendered = json.dumps(value, ensure_ascii=True, sort_keys=True).encode("utf-8")
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
    parser.add_argument("--script", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    output_dir = Path(args.out)
    try:
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise _WorkerFailure("invalid_output_directory", "worker_input")
        if any(output_dir.iterdir()):
            raise _WorkerFailure("unexpected_output_file", "worker_input")
        contract = _load_contract(Path(args.contract))
        _execute(Path(args.script), contract, output_dir)
    except _WorkerFailure as exc:
        _publish_error(output_dir, exc)
        return 2
    except BaseException as exc:
        failure = _WorkerFailure(
            "worker_internal_error", "worker_internal", type(exc).__name__
        )
        _publish_error(output_dir, failure)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
