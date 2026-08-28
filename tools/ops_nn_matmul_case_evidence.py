"""Execute and compare the NumPy and migrated ops-nn MatMul case artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import stat
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = path.absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _write_once_private(path: Path, payload: dict[str, Any]) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
        raise RuntimeError("evidence output is not private")


def verify(source: Path, migrated: Path, migration_report: Path) -> dict[str, Any]:
    import asnumpy as ap
    from asnumpy.linalg import direct

    from numpilot.backends.npu import NpuBackend

    runtime = NpuBackend().runtime_identity()
    if runtime.status != "matched":
        raise RuntimeError(
            "MatMul runtime identity is not matched: " + "; ".join(runtime.issues)
        )

    source_scope = runpy.run_path(str(source))
    expected = source_scope.get("result")
    if type(expected) is not np.ndarray:
        raise TypeError("original script did not produce a NumPy ndarray result")

    native_calls = 0
    original_native = direct._matmul
    original_host = direct._as_host_array

    def counted_native(left, right):
        nonlocal native_calls
        native_calls += 1
        return original_native(left, right)

    def forbidden_host(_array):
        raise AssertionError("migrated qualified matmul attempted a host fallback")

    direct._matmul = counted_native
    direct._as_host_array = forbidden_host
    try:
        migrated_scope = runpy.run_path(str(migrated))
    finally:
        direct._matmul = original_native
        direct._as_host_array = original_host

    device_result = migrated_scope.get("result")
    if type(device_result) is not ap.ndarray:
        raise TypeError("migrated script did not produce an AsNumPy ndarray result")
    actual = device_result.to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    if native_calls != 1:
        raise AssertionError(f"expected one native matmul call, observed {native_calls}")
    if actual.shape != (8, 4) or actual.dtype != np.dtype(np.float32):
        raise AssertionError("migrated result shape or dtype is outside the case contract")

    receipt = os.getenv("NUMPILOT_OPS_NN_MATMUL_AUDIT_RECEIPT")
    receipt_path = _regular_file(receipt or "", "qualification receipt")
    core_path = Path(ap._core.__file__).resolve(strict=True)
    error = np.abs(actual - expected)
    return {
        "schema_version": 1,
        "producer": "numpilot-ops-nn-matmul-case-evidence-v1",
        "status": "passed",
        "execution_performed": True,
        "performance_valid": False,
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
            "result_type": "numpy.ndarray",
        },
        "migration": {
            "path": str(migrated),
            "sha256": _sha256(migrated),
            "report_path": str(migration_report),
            "report_sha256": _sha256(migration_report),
        },
        "runtime": {
            "asnumpy_version": ap.__version__,
            "audit_id": runtime.audit_id,
            "identity_sha256": runtime.identity_sha256,
            "qualification_receipt_path": str(receipt_path),
            "qualification_receipt_sha256": _sha256(receipt_path),
            "module_path": str(Path(ap.__file__).resolve(strict=True)),
            "native_module_path": str(core_path),
            "native_module_sha256": _sha256(core_path),
        },
        "result": {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "min": float(actual.min()),
            "max": float(actual.max()),
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
            "rtol": 1e-5,
            "atol": 1e-6,
        },
        "route": {
            "public_api": "asnumpy.matmul",
            "native_binding": "asnumpy._core.matmul",
            "native_call_count": native_calls,
            "host_fallback_attempted": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--migrated", required=True)
    parser.add_argument("--migration-report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = _regular_file(args.source, "source")
    migrated = _regular_file(args.migrated, "migrated artifact")
    report = _regular_file(args.migration_report, "migration report")
    payload = verify(source, migrated, report)
    payload["harness"] = {
        "path": str(Path(__file__).resolve(strict=True)),
        "sha256": _sha256(Path(__file__).resolve(strict=True)),
    }
    _write_once_private(Path(args.out), payload)
    print(
        "status=passed native_calls=%d max_abs_error=%.9g evidence=%s"
        % (
            payload["route"]["native_call_count"],
            payload["result"]["max_abs_error"],
            Path(args.out).absolute(),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
