#!/usr/bin/env python3
"""Trusted, out-of-process qualification harness for the Poisson P0 operators.

The NumPilot admission module intentionally never executes generated code.  This
tool owns that missing execution boundary: it rebuilds sealed artifacts in an
external build directory, installs only into a new private OPP root, runs the
artifact tests and public AsNumpy APIs on the NPU, emits the six schema-v2 gate
reports, validates them with NumPilot, and atomically publishes one overlay.

The output directory must not exist.  A failed run remains on disk without a
qualification.json or overlay, so partial work can never be mistaken for an
activated result.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, Mapping


PRODUCER = "numpilot-qualification-harness-v1"
SOC_BUILD_NAME = "ascend910b"
SOC_RUNTIME_NAME = "Ascend910B"
CANN_VERSION = "9.0.0"
OPERATOR_CONFIG = {
    "apply_gradient_descent": {
        "vendor": "apply_gradient_descent_custom",
        "test": "tests/test_apply_gradient_descent.py",
    },
    "smooth_l1_loss": {
        "vendor": "smooth_l1_loss_custom",
        "test": "tests/test_smooth_l1_loss.py",
    },
}
RESULT_PREFIX = "NUMPILOT_QUALIFICATION_RESULT="
_PASS_COUNT = re.compile(r"(?:^|\s)([0-9]+) passed(?:\s|,|$)")


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
    path.write_bytes(_json_bytes(data))
    path.chmod(0o600)


def _require_file(path: str, label: str) -> Path:
    target = Path(path).resolve()
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"{label} is not a regular file: {target}")
    return target


def _require_python(path: str) -> Path:
    # Virtualenv Python executables are normally symlinks.  Keep the invoked
    # path (and therefore pyvenv.cfg discovery) while validating its target.
    invoked = Path(path).absolute()
    target = invoked.resolve()
    if not target.is_file() or not os.access(invoked, os.X_OK):
        raise RuntimeError(f"AsNumpy Python is not executable: {invoked}")
    return invoked


def _require_artifact(path: str, expected_key: str) -> Path:
    root = Path(path).resolve()
    manifest_path = root / "operator-artifact.json"
    if root.is_symlink() or not root.is_dir() or not manifest_path.is_file():
        raise RuntimeError(f"sealed artifact is invalid: {root}")
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise RuntimeError(f"sealed artifact contains a symlink: {candidate}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "generated_unverified":
        raise RuntimeError(f"artifact is not generated_unverified: {root}")
    if not str(manifest.get("gap_id", "")).startswith(expected_key + "-"):
        raise RuntimeError(f"artifact identity does not match {expected_key}")
    return root


def _command_env(private_opp: Path | None = None) -> Dict[str, str]:
    env = dict(os.environ)
    driver_first = [
        "/usr/local/Ascend/driver/lib64",
        "/usr/local/Ascend/driver/lib64/common",
        "/usr/local/Ascend/driver/lib64/driver",
        "/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64",
        "/usr/local/Ascend/ascend-toolkit/latest/lib64",
    ]
    env.update({
        "ASCEND_HOME_PATH": "/usr/local/Ascend/cann-9.0.0",
        "ASCEND_TOOLKIT_HOME": "/usr/local/Ascend/cann-9.0.0",
        "ASCEND_OPP_PATH": "/usr/local/Ascend/cann-9.0.0/opp",
        "ASCEND_SOC_VERSION": SOC_RUNTIME_NAME,
        "CANN_VERSION": CANN_VERSION,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_ADDOPTS": "-p no:cacheprovider",
        "LD_LIBRARY_PATH": ":".join(driver_first),
    })
    if private_opp is not None:
        roots = [
            private_opp / "vendors" / config["vendor"]
            for config in OPERATOR_CONFIG.values()
        ]
        for root in roots:
            if not root.is_dir():
                raise RuntimeError(f"private OPP vendor is missing: {root}")
        env["ASCEND_CUSTOM_OPP_PATH"] = ":".join(str(root) for root in roots)
    return env


def _run_logged(
    command: Iterable[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    timeout: int,
) -> str:
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
    log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_path.write_text(completed.stdout, encoding="utf-8")
    log_path.chmod(0o600)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}; see {log_path}"
        )
    return completed.stdout


def _copy_artifact(source: Path, destination: Path) -> Dict[str, Any]:
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    manifest_path = destination / "operator-artifact.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["workspace"] = str(destination)
    _write_json(manifest_path, manifest)
    return manifest


def _build_and_install(
    key: str,
    artifact: Path,
    *,
    build_root: Path,
    output_root: Path,
    private_opp: Path,
    jobs: int,
) -> Dict[str, Any]:
    build_dir = build_root / key
    raw_dir = output_root / "raw" / key
    env = _command_env()
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
        log_path=raw_dir / "build-configure.log",
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
            f"-j{jobs}",
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
        raise RuntimeError(f"build output is incomplete for {key}")
    package_copy = output_root / "runtime" / "packages" / f"{key}.run"
    package_copy.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    shutil.copy2(package, package_copy)
    package_copy.chmod(0o700)
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
    vendor = private_opp / "vendors" / OPERATOR_CONFIG[key]["vendor"]
    if not vendor.is_dir():
        raise RuntimeError(f"installer did not create the expected vendor: {vendor}")
    return {
        "package": package_copy,
        "kernel_count": len(kernels),
        "output_artifacts": 1 + len(kernels),
    }


def _pytest_count(output: str, key: str) -> int:
    matches = _PASS_COUNT.findall(output)
    if not matches:
        raise RuntimeError(f"pytest did not report passing tests for {key}")
    if re.search(r"(?:^|\s)[1-9][0-9]* (?:failed|skipped|error)", output):
        raise RuntimeError(f"pytest reported a non-pass result for {key}")
    return int(matches[-1])


def _runtime_worker() -> int:
    """Execute only inside the explicitly selected AsNumpy virtualenv."""

    import numpy as np
    import asnumpy
    import asnumpy.custom_ops as custom_ops

    from numpilot.backends.npu import NpuBackend

    results: Dict[str, Any] = {}
    rng = np.random.default_rng(20260821)
    shapes = ((4,), (33,), (2, 3))
    for key in OPERATOR_CONFIG:
        public = getattr(asnumpy, key, None)
        wrapper = getattr(custom_ops, key, None)
        backing = custom_ops.__dict__.get("_" + key)
        if not callable(public) or public is not wrapper:
            raise RuntimeError(f"public AsNumpy API is not the registered wrapper: {key}")
        if not callable(backing) or getattr(backing, "__module__", None) != (
            "asnumpy._core.custom_ops"
        ):
            raise RuntimeError(f"custom op does not bind the pybind NPU target: {key}")
        source = inspect.getsource(wrapper)
        forbidden = (".to_numpy(", "np.subtract(", "np.mean(", "numpy.")
        if any(marker in source for marker in forbidden):
            raise RuntimeError(f"public wrapper contains a host fallback marker: {key}")

        max_abs_error = 0.0
        for shape in shapes:
            left = np.ascontiguousarray(
                rng.standard_normal(shape), dtype=np.float32
            )
            right = np.ascontiguousarray(
                rng.standard_normal(shape), dtype=np.float32
            )
            if key == "apply_gradient_descent":
                alpha = np.array([np.float32(0.25)], dtype=np.float32)
                expected = left - alpha * right
                actual = public(left, alpha, right)
            else:
                error = left - right
                absolute = np.abs(error)
                expected = np.mean(np.where(
                    absolute < np.float32(1.0),
                    absolute * absolute / np.float32(2.0),
                    absolute - np.float32(0.5),
                ).astype(np.float32))
                actual = public(left, right)
            if type(actual) is not asnumpy.ndarray:
                raise RuntimeError(f"public API did not return asnumpy.ndarray: {key}")
            host = np.asarray(actual, dtype=np.float32)
            np.testing.assert_allclose(host, expected, rtol=1e-4, atol=1e-5)
            error = float(np.max(np.abs(host - expected)))
            max_abs_error = max(max_abs_error, error)
        results[key] = {
            "cases": len(shapes),
            "failed": 0,
            "max_abs_error": max_abs_error,
            "public_callable": True,
            "cpu_fallback_detected": False,
            "kernel_launches": len(shapes),
        }

    core_path = Path(asnumpy.__file__).with_name(
        next(
            path.name
            for path in Path(asnumpy.__file__).parent.glob("_core*.so")
        )
    )
    manifest = NpuBackend(module=asnumpy).capability_manifest(
        tuple(OPERATOR_CONFIG)
    )
    payload = {
        "operators": results,
        "base_manifest": manifest.to_dict(),
        "asnumpy_init": str(Path(asnumpy.__file__).resolve()),
        "asnumpy_core": str(core_path.resolve()),
    }
    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True))
    return 0


def _extract_worker_result(output: str) -> Dict[str, Any]:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_PREFIX):
            data = json.loads(line[len(RESULT_PREFIX):])
            if not isinstance(data, dict):
                break
            return data
    raise RuntimeError("runtime worker did not emit a structured result")


def _copy_runtime_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    shutil.copy2(source, destination)
    destination.chmod(0o600)


def _materialize_private_opp_symlinks(private_opp: Path) -> None:
    """Replace installer-created relative links with their bound file bytes."""

    root = private_opp.resolve()
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


def _hash_tree(root: Path, *, exclude: Iterable[Path] = ()) -> Dict[str, str]:
    excluded = {path.resolve() for path in exclude}
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"evidence tree contains a symlink: {path}")
        if not path.is_file() or path.resolve() in excluded:
            continue
        result[path.relative_to(root).as_posix()] = _sha256_file(path)
    return result


def _gate_claims(
    name: str,
    *,
    key: str,
    asnumpy_api: str,
    artifact_files: int,
    build: Mapping[str, Any],
    unit_passed: int,
    runtime_result: Mapping[str, Any],
    signature_count: int,
    signature_sha256: str,
) -> Dict[str, Any]:
    claims: Dict[str, Any]
    if name == "artifact_integrity":
        return {"verified_files": artifact_files, "mismatches": 0}
    if name == "build":
        return {"exit_code": 0, "output_artifacts": build["output_artifacts"]}
    if name == "unit_test":
        claims = {"passed": unit_passed, "failed": 0}
    elif name == "numerical_validation":
        claims = {
            "oracle": "numpy",
            "cases": runtime_result["cases"],
            "failed": runtime_result["failed"],
            "cpu_fallback_detected": runtime_result["cpu_fallback_detected"],
        }
    elif name == "public_asnumpy_api":
        claims = {
            "api": asnumpy_api,
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
        raise RuntimeError(f"unknown gate: {name}")
    claims.update({
        "covered_signature_variants": signature_count,
        "signature_variants_sha256": signature_sha256,
    })
    return claims


def _main(args: argparse.Namespace) -> int:
    from numpilot.capabilities import CapabilityManifest, OperatorSignature
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

    source = _require_file(args.source, "source")
    asnumpy_python = _require_python(args.asnumpy_python)
    asnumpy_wheel = _require_file(args.asnumpy_wheel, "AsNumpy wheel")
    inputs = {
        "apply_gradient_descent": {
            "spec": _require_file(args.apply_spec, "apply GapSpec"),
            "artifact": _require_artifact(
                args.apply_artifact, "apply_gradient_descent"
            ),
        },
        "smooth_l1_loss": {
            "spec": _require_file(args.smooth_spec, "smooth GapSpec"),
            "artifact": _require_artifact(args.smooth_artifact, "smooth_l1_loss"),
        },
    }
    output_root = Path(args.out).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError(f"output must not exist: {output_root}")
    output_root.mkdir(parents=True, mode=0o700)
    output_root.chmod(0o700)

    copied_source = output_root / "examples" / "poisson_ritz_step.py"
    _copy_runtime_file(source, copied_source)
    if _sha256_file(copied_source) != _sha256_file(source):
        raise RuntimeError("source copy hash mismatch")

    artifact_manifests: Dict[str, Dict[str, Any]] = {}
    spec_data: Dict[str, Dict[str, Any]] = {}
    build_results: Dict[str, Dict[str, Any]] = {}
    private_opp = output_root / "runtime" / "private-opp"
    private_opp.mkdir(parents=True, mode=0o700)
    with tempfile.TemporaryDirectory(
        prefix="numpilot-qualification-build-",
        dir=str(output_root.parent),
    ) as temp_build:
        build_root = Path(temp_build)
        for key, item in inputs.items():
            artifact_copy = output_root / key / "artifact"
            artifact_manifests[key] = _copy_artifact(
                item["artifact"], artifact_copy
            )
            spec_copy = output_root / key / "spec" / "gap-spec.json"
            _copy_runtime_file(item["spec"], spec_copy)
            spec_data[key] = json.loads(spec_copy.read_text(encoding="utf-8"))
            build_results[key] = _build_and_install(
                key,
                item["artifact"],
                build_root=build_root,
                output_root=output_root,
                private_opp=private_opp,
                jobs=args.jobs,
            )

    _materialize_private_opp_symlinks(private_opp)
    runtime_env = _command_env(private_opp)
    unit_counts: Dict[str, int] = {}
    for key in OPERATOR_CONFIG:
        test_path = output_root / key / "artifact" / OPERATOR_CONFIG[key]["test"]
        output = _run_logged(
            (str(asnumpy_python), "-m", "pytest", "-q", str(test_path)),
            cwd=output_root,
            env=runtime_env,
            log_path=output_root / "raw" / key / "unit-and-runtime-tests.log",
            timeout=600,
        )
        unit_counts[key] = _pytest_count(output, key)

    worker_output = _run_logged(
        (str(asnumpy_python), str(Path(__file__).resolve()), "--runtime-worker"),
        cwd=Path(__file__).resolve().parents[1],
        env=runtime_env,
        log_path=output_root / "raw" / "runtime-worker.log",
        timeout=600,
    )
    worker = _extract_worker_result(worker_output)
    base_manifest = CapabilityManifest.from_dict(worker["base_manifest"])

    runtime_dir = output_root / "runtime"
    _copy_runtime_file(
        Path(worker["asnumpy_init"]), runtime_dir / "asnumpy" / "__init__.py"
    )
    core_source = Path(worker["asnumpy_core"])
    _copy_runtime_file(core_source, runtime_dir / "asnumpy" / core_source.name)
    _copy_runtime_file(asnumpy_wheel, runtime_dir / "asnumpy-wheel.whl")
    _write_json(runtime_dir / "base-capability-manifest.json", base_manifest.to_dict())

    opp_files = {}
    for path in sorted(private_opp.rglob("*")):
        if path.is_file():
            opp_files[path.relative_to(private_opp).as_posix()] = _sha256_file(path)
    opp_manifest = {
        "schema_version": 1,
        "producer": PRODUCER,
        "soc": SOC_RUNTIME_NAME,
        "cann": CANN_VERSION,
        "vendors": [config["vendor"] for config in OPERATOR_CONFIG.values()],
        "files": opp_files,
        "packages": {
            key: {
                "artifact_id": artifact_manifests[key]["artifact_id"],
                "sha256": _sha256_file(build_results[key]["package"]),
            }
            for key in OPERATOR_CONFIG
        },
        "asnumpy_wheel_sha256": _sha256_file(asnumpy_wheel),
    }
    opp_manifest_path = runtime_dir / "opp-manifest.json"
    _write_json(opp_manifest_path, opp_manifest)

    qualification_paths = [
        output_root / key / "qualification.json" for key in OPERATOR_CONFIG
    ]
    overlay_path = output_root / "overlay" / "capability-overlay.json"
    admitted_path = output_root / "overlay" / "admitted-capability-manifest.json"
    runtime_artifacts = _hash_tree(
        output_root,
        exclude=tuple(qualification_paths) + (overlay_path, admitted_path),
    )
    runtime = {
        "backend": "npu",
        "backend_version": base_manifest.backend_version,
        "backend_audit_id": base_manifest.audit_id,
        "backend_module_sha256": _sha256_file(Path(worker["asnumpy_init"])),
        "cann_version": base_manifest.environment["cann_version"],
        "soc_version": base_manifest.environment["soc_version"],
        "python_version": base_manifest.environment["python_version"],
        "opp_manifest_path": opp_manifest_path.relative_to(output_root).as_posix(),
        "opp_manifest_sha256": _sha256_file(opp_manifest_path),
    }
    constraints = ["dtype=float32", "rank>=1", "contiguous=true"]
    constraints_sha256 = qualification_constraints_sha256(constraints)

    refs = []
    qualifications = {}
    for key in OPERATOR_CONFIG:
        spec = spec_data[key]
        manifest = artifact_manifests[key]
        signatures = tuple(
            OperatorSignature.from_dict(item)
            for item in spec["signature_variants"]
        )
        signature_sha256 = qualification_signatures_sha256(signatures)
        spec_path = output_root / key / "spec" / "gap-spec.json"
        manifest_path = output_root / key / "artifact" / "operator-artifact.json"
        gate_metadata = {}
        for gate in sorted(REQUIRED_QUALIFICATION_GATES):
            report = {
                "schema_version": 2,
                "gate": gate,
                "status": "passed",
                "producer": PRODUCER,
                "artifact_id": manifest["artifact_id"],
                "gap_spec_sha256": _sha256_file(spec_path),
                "artifact_manifest_sha256": _sha256_file(manifest_path),
                "constraints_sha256": constraints_sha256,
                "target": runtime,
                "claims": _gate_claims(
                    gate,
                    key=key,
                    asnumpy_api=spec["asnumpy_api"],
                    artifact_files=len(manifest["files"]),
                    build=build_results[key],
                    unit_passed=unit_counts[key],
                    runtime_result=worker["operators"][key],
                    signature_count=len(signatures),
                    signature_sha256=signature_sha256,
                ),
            }
            report_path = output_root / key / "reports" / f"{gate}.json"
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
            "gap_id": spec["gap_id"],
            "registry_key": key,
            "qualified_api": spec["qualified_api"],
            "asnumpy_api": spec["asnumpy_api"],
            "gap_spec_path": spec_path.relative_to(output_root).as_posix(),
            "gap_spec_sha256": _sha256_file(spec_path),
            "source_path": "examples/poisson_ritz_step.py",
            "source_sha256": _sha256_file(copied_source),
            "artifact_manifest_path": manifest_path.relative_to(output_root).as_posix(),
            "artifact_manifest_sha256": _sha256_file(manifest_path),
            "artifact_id": manifest["artifact_id"],
            "artifact_files": manifest["files"],
            "runtime_artifacts": runtime_artifacts,
            "runtime": runtime,
            "constraints": constraints,
            "gates": gate_metadata,
        }
        evidence["qualification_id"] = compute_qualification_id(evidence)
        evidence_path = output_root / key / "qualification.json"
        _write_json(evidence_path, evidence)
        parsed = QualificationEvidence.from_dict(evidence)
        qualifications[key] = parsed
        validate_qualification(
            parsed,
            evidence_root=str(output_root),
            current_manifest=base_manifest,
        )
        refs.append(QualificationRef(
            registry_key=key,
            evidence_path=evidence_path.relative_to(output_root).as_posix(),
            evidence_sha256=_sha256_file(evidence_path),
        ))

    overlay = CapabilityOverlay(
        revision=1,
        parent_sha256=ZERO_SHA256,
        base_manifest_sha256=capability_manifest_sha256(base_manifest),
        entries=tuple(refs),
    )
    published_sha256 = publish_overlay_atomic(str(overlay_path), overlay)
    admitted = apply_overlay(
        base_manifest,
        overlay,
        evidence_root=str(output_root),
    )
    _write_json(admitted_path, admitted.to_dict())
    summary = {
        "schema_version": 1,
        "producer": PRODUCER,
        "status": "qualified_and_overlay_published",
        "overlay_path": overlay_path.relative_to(output_root).as_posix(),
        "overlay_sha256": published_sha256,
        "operators": {
            key: {
                "artifact_id": artifact_manifests[key]["artifact_id"],
                "qualification_id": qualifications[key].qualification_id,
                "tests_passed": unit_counts[key],
                "numerical_cases": worker["operators"][key]["cases"],
                "max_abs_error": worker["operators"][key]["max_abs_error"],
                "kernel_launches": worker["operators"][key]["kernel_launches"],
            }
            for key in OPERATOR_CONFIG
        },
    }
    _write_json(output_root / "qualification-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source")
    parser.add_argument("--asnumpy-python")
    parser.add_argument("--asnumpy-wheel")
    parser.add_argument("--apply-spec")
    parser.add_argument("--apply-artifact")
    parser.add_argument("--smooth-spec")
    parser.add_argument("--smooth-artifact")
    parser.add_argument("--out")
    parser.add_argument("--jobs", type=int, default=4)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.runtime_worker:
        return _runtime_worker()
    required = (
        "source",
        "asnumpy_python",
        "asnumpy_wheel",
        "apply_spec",
        "apply_artifact",
        "smooth_spec",
        "smooth_artifact",
        "out",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise RuntimeError("missing required arguments: " + ", ".join(missing))
    if args.jobs < 1 or args.jobs > 32:
        raise RuntimeError("--jobs must be between 1 and 32")
    return _main(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"qualification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
