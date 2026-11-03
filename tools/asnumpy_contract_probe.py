#!/usr/bin/env python3
"""Probe the locked AsNumpy runtime against NumPy boundary contracts.

The probe is deliberately local and deterministic.  It does not read llm.json,
contact a model, fetch files, compile code, or change the installed runtime.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Callable

import numpy as np


PROBE_SCHEMA = "numpilot.asnumpy-contract-probe"
PROBE_SCHEMA_VERSION = 1


def _manifest_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        return {"status": "unavailable", "path": str(path)}
    payload = path.read_bytes()
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {
            "status": "invalid",
            "path": str(path),
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        }
    if not isinstance(manifest, dict):
        return {"status": "invalid", "path": str(path)}
    return {
        "status": "bound",
        "path": str(path),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "component_id": manifest.get("component_id"),
        "version": manifest.get("version"),
        "repository": manifest.get("repository"),
        "commit": manifest.get("commit"),
        "tree": manifest.get("tree"),
        "wheel": (manifest.get("wheel") or {}).get("filename"),
        "wheel_sha256": (manifest.get("wheel") or {}).get("sha256"),
    }


def _cann_version() -> str:
    """Resolve the CANN toolkit version without trusting a caller export."""

    explicit = os.getenv("CANN_VERSION")
    if explicit:
        return explicit
    for name in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_LATEST_HOME"):
        raw = os.getenv(name)
        if not raw:
            continue
        path = Path(raw)
        try:
            path = path.resolve()
        except OSError:
            pass
        basename = path.name
        if basename.startswith("cann-") and len(basename) > len("cann-"):
            return basename[len("cann-"):]
    return "unknown"


def _soc_name() -> str:
    """Resolve the SoC name through pyACL when no explicit export exists."""

    explicit = os.getenv("ASCEND_SOC_VERSION")
    if explicit:
        return explicit
    try:
        import acl  # type: ignore
    except Exception:
        return "unknown"
    try:
        name = acl.get_soc_name()
    except Exception:
        return "unknown"
    return name if isinstance(name, str) and name else "unknown"


def _normalise(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalise(value.item())
        return [_normalise(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "+Inf"
        if np.isneginf(value):
            return "-Inf"
        if value == 0.0 and np.signbit(value):
            return "-0.0"
        return value
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return value


def _semantic_equal(expected: np.ndarray, observed: np.ndarray) -> bool:
    if expected.shape != observed.shape or expected.dtype != observed.dtype:
        return False
    if not np.array_equal(expected, observed, equal_nan=True):
        return False
    zero_expected = expected == 0
    zero_observed = observed == 0
    if not np.array_equal(zero_expected, zero_observed):
        return False
    if np.any(zero_expected):
        if not np.array_equal(
            np.signbit(expected[zero_expected]),
            np.signbit(observed[zero_observed]),
        ):
            return False
    return True


def _run_case(
    name: str,
    numpy_api: str,
    input_contract: dict[str, Any],
    numpy_fn: Callable[[], np.ndarray],
    asnumpy_fn: Callable[[], Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "numpy_api": numpy_api,
        "input_contract": input_contract,
    }
    try:
        expected = np.asarray(numpy_fn())
        result["numpy"] = {
            "shape": list(expected.shape),
            "dtype": str(expected.dtype),
            "value": _normalise(expected),
        }
    except Exception as exc:
        result["status"] = "numpy_error"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:512]}
        return result
    try:
        observed = np.asarray(asnumpy_fn())
        result["asnumpy"] = {
            "shape": list(observed.shape),
            "dtype": str(observed.dtype),
            "value": _normalise(observed),
        }
    except Exception as exc:
        result["status"] = "asnumpy_error"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:512]}
        return result
    result["shape_match"] = expected.shape == observed.shape
    result["dtype_match"] = expected.dtype == observed.dtype
    result["semantic_match"] = _semantic_equal(expected, observed)
    result["status"] = "verified_match" if result["semantic_match"] else "semantic_mismatch"
    return result


def _cases(ap_module: Any) -> list[dict[str, Any]]:
    def npu(value: np.ndarray) -> Any:
        return ap_module.ndarray.from_numpy(value)

    return [
        {
            "name": "fmax_nan_ignores_nan",
            "numpy_api": "numpy.fmax",
            "input_contract": {
                "dtype": "float32",
                "shape": [1],
                "left": ["NaN"],
                "right": [2.0],
            },
            "numpy_fn": lambda: np.fmax(
                np.array([np.nan], dtype=np.float32),
                np.array([2.0], dtype=np.float32),
            ),
            "asnumpy_fn": lambda: ap_module.fmax(
                npu(np.array([np.nan], dtype=np.float32)),
                npu(np.array([2.0], dtype=np.float32)),
            ).to_numpy(),
        },
        {
            "name": "copysign_preserves_negative_zero",
            "numpy_api": "numpy.copysign",
            "input_contract": {
                "dtype": "float32",
                "shape": [1],
                "magnitude": [1.0],
                "sign": ["-0.0"],
            },
            "numpy_fn": lambda: np.copysign(
                np.array([1.0], dtype=np.float32),
                np.array([-0.0], dtype=np.float32),
            ),
            "asnumpy_fn": lambda: ap_module.copysign(
                npu(np.array([1.0], dtype=np.float32)),
                npu(np.array([-0.0], dtype=np.float32)),
            ).to_numpy(),
        },
        {
            "name": "add_zero_dimensional_shape",
            "numpy_api": "numpy.add",
            "input_contract": {
                "dtype": "float32",
                "shape": [],
                "left": 1.0,
                "right": 2.0,
            },
            "numpy_fn": lambda: np.add(
                np.array(1.0, dtype=np.float32),
                np.array(2.0, dtype=np.float32),
            ),
            "asnumpy_fn": lambda: ap_module.add(
                npu(np.array(1.0, dtype=np.float32)),
                npu(np.array(2.0, dtype=np.float32)),
            ).to_numpy(),
        },
        {
            "name": "matmul_fp32_2d",
            "numpy_api": "numpy.matmul",
            "input_contract": {
                "dtype": "float32",
                "left_shape": [2, 3],
                "right_shape": [3, 2],
                "strict_profile": "fp32-2d-contiguous-inner-dim-match",
            },
            "numpy_fn": lambda: np.matmul(
                np.arange(6, dtype=np.float32).reshape(2, 3),
                np.arange(6, dtype=np.float32).reshape(3, 2),
            ),
            "asnumpy_fn": lambda: ap_module.matmul(
                npu(np.arange(6, dtype=np.float32).reshape(2, 3)),
                npu(np.arange(6, dtype=np.float32).reshape(3, 2)),
            ).to_numpy(),
        },
    ]


def build_report(manifest_path: Path) -> dict[str, Any]:
    import asnumpy as ap_module

    cases = []
    for case in _cases(ap_module):
        cases.append(_run_case(**case))
    return {
        "schema": PROBE_SCHEMA,
        "schema_version": PROBE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runtime": {
            "python": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "asnumpy_version": getattr(ap_module, "__version__", "unknown"),
            "asnumpy_module": str(Path(getattr(ap_module, "__file__", "")).resolve()),
            "cann_version": _cann_version(),
            "soc": _soc_name(),
        },
        "component": _manifest_identity(manifest_path),
        "cases": cases,
        "summary": {
            "total": len(cases),
            "verified_match": sum(item["status"] == "verified_match" for item in cases),
            "semantic_mismatch": sum(item["status"] == "semantic_mismatch" for item in cases),
            "errors": sum(item["status"].endswith("_error") for item in cases),
        },
        "execution_boundary": {
            "network_used": False,
            "model_called": False,
            "source_executed": False,
            "runtime_imported": True,
        },
    }


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("output already exists: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="components/asnumpy/manifest.json",
        help="locked AsNumpy manifest used for provenance binding",
    )
    parser.add_argument("--out", help="write a 0600 JSON report without overwriting")
    parser.add_argument(
        "--require-match", action="store_true",
        help="return non-zero when any contract case has a semantic mismatch",
    )
    args = parser.parse_args(argv)
    report = build_report(Path(args.manifest).expanduser())
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if args.out:
        _write_once(Path(args.out).expanduser(), payload)
    print(payload.decode("utf-8"), end="")
    if report["summary"]["errors"]:
        return 2
    if args.require_match and report["summary"]["semantic_mismatch"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

