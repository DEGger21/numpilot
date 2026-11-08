"""Offline integrity audits for NumPilot operator-artifact stores and runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from numpilot.operator_artifacts import (
    ACTIVATION_SCHEMA,
    BINDING_SCHEMA,
    CANDIDATE_SCHEMA,
    OperatorArtifactStoreError,
    OperatorKey,
    QUALIFICATION_SCHEMA,
    STORE_SCHEMA,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OperatorArtifactStoreError(f"missing or symlinked artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OperatorArtifactStoreError(f"artifact must be an object: {path}")
    return value


def _add(errors: list[str], message: str) -> None:
    errors.append(message)


def _safe_relative(root: Path, value: Any) -> Path | None:
    """Resolve a receipt path while rejecting absolute/traversal escapes."""
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    try:
        resolved = (root / candidate).resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return resolved


def audit_store(root: str | Path) -> dict[str, Any]:
    """Return a deterministic, read-only integrity report for one store."""
    root = Path(root).expanduser().absolute()
    errors: list[str] = []
    counts = {"operators": 0, "candidates": 0, "qualifications": 0, "activations": 0, "samples": 0}
    if root.is_symlink() or not root.is_dir():
        return {"schema": "numpilot.operator-artifact-audit", "schema_version": 1, "ok": False, "root": str(root), "errors": ["store root is missing or symlinked"], "counts": counts}
    try:
        store = _read(root / "store.json")
        if store.get("schema") != STORE_SCHEMA:
            _add(errors, "store schema mismatch")
        if store.get("schema_version") != 1:
            _add(errors, "store schema version mismatch")
    except Exception as exc:
        return {"schema": "numpilot.operator-artifact-audit", "schema_version": 1, "ok": False, "root": str(root), "errors": [str(exc)], "counts": counts}

    operators_root = root / "operators"
    samples_root = root / "samples"
    if operators_root.is_symlink() or not operators_root.is_dir():
        _add(errors, "operators directory is missing or symlinked")
    if samples_root.is_symlink() or not samples_root.is_dir():
        _add(errors, "samples directory is missing or symlinked")
    operator_ids: set[str] = set()
    candidate_ids: set[tuple[str, str]] = set()
    qualification_ids: set[tuple[str, str]] = set()
    activation_ids: set[tuple[str, str]] = set()
    if operators_root.is_dir() and not operators_root.is_symlink():
        for operator_dir in sorted(operators_root.glob("*")):
            if operator_dir.is_symlink() or not operator_dir.is_dir():
                _add(errors, f"invalid operator directory: {operator_dir.name}")
                continue
            operator_id = operator_dir.name
            operator_ids.add(operator_id)
            counts["operators"] += 1
            try:
                record = _read(operator_dir / "operator.json")
                if record.get("operator_id") != operator_id:
                    _add(errors, f"operator id mismatch: {operator_id}")
                key_payload = record.get("key")
                if not isinstance(key_payload, dict):
                    _add(errors, f"operator key missing: {operator_id}")
                else:
                    try:
                        if OperatorKey(**key_payload).operator_id != operator_id:
                            _add(errors, f"operator key hash mismatch: {operator_id}")
                    except Exception as exc:
                        _add(errors, f"invalid operator key {operator_id}: {type(exc).__name__}")
            except Exception as exc:
                _add(errors, str(exc)); continue
            for group, filename, seen, status_name in (
                ("candidates", "candidate.json", candidate_ids, "candidate"),
                ("qualifications", "qualification.json", qualification_ids, "qualification"),
                ("activations", "activation.json", activation_ids, "activation"),
            ):
                group_root = operator_dir / group
                if group_root.is_symlink():
                    _add(errors, f"{status_name} directory is symlinked: {operator_id}")
                    continue
                if not group_root.is_dir():
                    continue
                for artifact_dir in sorted(group_root.glob("*")):
                    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
                        _add(errors, f"invalid {status_name} directory: {artifact_dir}"); continue
                    artifact_id = artifact_dir.name
                    try:
                        payload = _read(artifact_dir / filename)
                        expected = payload.get(f"{status_name}_id")
                        if expected != artifact_id:
                            _add(errors, f"{status_name} id mismatch: {operator_id}/{artifact_id}")
                        if payload.get("operator_id") != operator_id:
                            _add(errors, f"{status_name} operator mismatch: {operator_id}/{artifact_id}")
                        if payload.get("schema") != {"candidate": CANDIDATE_SCHEMA, "qualification": QUALIFICATION_SCHEMA, "activation": ACTIVATION_SCHEMA}[status_name]:
                            _add(errors, f"{status_name} schema mismatch: {operator_id}/{artifact_id}")
                        if status_name == "candidate":
                            source_name = payload.get("source_name")
                            source = _safe_relative(artifact_dir, source_name)
                            if source is None or not source.is_file() or source.is_symlink():
                                _add(errors, f"candidate source missing: {operator_id}/{artifact_id}")
                            elif payload.get("source_sha256") != _sha256(source):
                                _add(errors, f"candidate source hash mismatch: {operator_id}/{artifact_id}")
                            files_hashes = payload.get("files_sha256")
                            if files_hashes is not None:
                                if not isinstance(files_hashes, dict):
                                    _add(errors, f"candidate files manifest is invalid: {operator_id}/{artifact_id}")
                                else:
                                    manifest = artifact_dir / "files-manifest.json"
                                    if manifest.is_file() and not manifest.is_symlink():
                                        try:
                                            if json.loads(manifest.read_text(encoding="utf-8")) != files_hashes:
                                                _add(errors, f"candidate files manifest mismatch: {operator_id}/{artifact_id}")
                                        except Exception:
                                            _add(errors, f"candidate files manifest unreadable: {operator_id}/{artifact_id}")
                                    for relative, expected_sha in sorted(files_hashes.items()):
                                        file_path = _safe_relative(artifact_dir / "files", relative)
                                        if file_path is None or not file_path.is_file() or file_path.is_symlink():
                                            _add(errors, f"candidate file missing: {operator_id}/{artifact_id}/{relative}")
                                        elif expected_sha != _sha256(file_path):
                                            _add(errors, f"candidate file hash mismatch: {operator_id}/{artifact_id}/{relative}")
                            candidate_ids.add((operator_id, artifact_id)); counts["candidates"] += 1
                        elif status_name == "qualification":
                            candidate_id = payload.get("candidate_id")
                            if (operator_id, candidate_id) not in candidate_ids:
                                _add(errors, f"qualification candidate missing: {operator_id}/{artifact_id}")
                            status = payload.get("status")
                            receipt = payload.get("receipt") or {}
                            if status == "qualified" and receipt.get("interface_available") is not True:
                                _add(errors, f"qualified receipt missing interface_available: {operator_id}/{artifact_id}")
                            qualification_ids.add((operator_id, artifact_id)); counts["qualifications"] += 1
                        else:
                            qualification_id = payload.get("qualification_id")
                            if (operator_id, qualification_id) not in qualification_ids:
                                _add(errors, f"activation qualification missing: {operator_id}/{artifact_id}")
                            qualification = operator_dir / "qualifications" / str(qualification_id) / "qualification.json"
                            if not qualification.is_file() or _read(qualification).get("status") != "qualified":
                                _add(errors, f"activation qualification invalid: {operator_id}/{artifact_id}")
                            if payload.get("status") == "activated":
                                receipt = payload.get("receipt") or {}
                                if receipt.get("activated") is not True or receipt.get("upstream_modified") is not False:
                                    _add(errors, f"activated receipt boundary invalid: {operator_id}/{artifact_id}")
                            activation_ids.add((operator_id, artifact_id)); counts["activations"] += 1
                    except Exception as exc:
                        _add(errors, str(exc))

    bindings: dict[str, dict[str, Any]] = {}
    if samples_root.is_dir() and not samples_root.is_symlink():
        for sample in sorted(samples_root.glob("*.json")):
            counts["samples"] += 1
            try:
                binding = _read(sample)
                if binding.get("schema") != BINDING_SCHEMA:
                    _add(errors, f"sample schema mismatch: {sample.name}")
                sample_id = binding.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    _add(errors, f"sample id missing: {sample.name}")
                elif sample_id in bindings:
                    _add(errors, f"duplicate sample id: {sample_id}")
                else:
                    bindings[sample_id] = binding
                operator_id = binding.get("operator_id")
                candidate_id = binding.get("candidate_id")
                qualification_id = binding.get("qualification_id")
                activation_id = binding.get("activation_id")
                status = binding.get("status")
                if operator_id is None:
                    if any(value is not None for value in (candidate_id, qualification_id, activation_id)):
                        _add(errors, f"no-operator sample contains artifact refs: {sample.name}")
                else:
                    if operator_id not in operator_ids or (operator_id, candidate_id) not in candidate_ids:
                        _add(errors, f"sample candidate ref missing: {sample.name}")
                    if qualification_id is not None and (operator_id, qualification_id) not in qualification_ids:
                        _add(errors, f"sample qualification ref missing: {sample.name}")
                    if activation_id is not None and (operator_id, activation_id) not in activation_ids:
                        _add(errors, f"sample activation ref missing: {sample.name}")
                    candidate = root / "operators" / str(operator_id) / "candidates" / str(candidate_id) / "candidate.json"
                    if candidate.is_file() and binding.get("candidate_source_sha256") != _read(candidate).get("source_sha256"):
                        _add(errors, f"sample candidate source hash mismatch: {sample.name}")
                    if status == "qualified" and (operator_id, qualification_id) not in qualification_ids:
                        _add(errors, f"qualified sample dependency missing: {sample.name}")
                    if status == "activated" and (operator_id, activation_id) not in activation_ids:
                        _add(errors, f"activated sample dependency missing: {sample.name}")
            except Exception as exc:
                _add(errors, str(exc))
    return {
        "schema": "numpilot.operator-artifact-audit",
        "schema_version": 1,
        "ok": not errors,
        "root": str(root),
        "counts": counts,
        "errors": sorted(set(errors)),
    }


def audit_run(root: str | Path, *, expected_total: int = 150) -> dict[str, Any]:
    """Audit a complete benchmark run without executing any user/model code.

    ``expected_total`` defaults to the competition contract of 150 cases;
    focused unit tests may pass a smaller value while preserving the same
    reference checks.
    """
    if not isinstance(expected_total, int) or expected_total < 0:
        raise ValueError("expected_total must be a non-negative integer")
    root = Path(root).expanduser().absolute()
    errors: list[str] = []
    store_report = audit_store(root / "operator-artifacts")
    errors.extend(store_report.get("errors", []))
    store_root = root / "operator-artifacts"
    bindings: dict[str, tuple[Path, dict[str, Any]]] = {}
    if store_root.is_dir() and (store_root / "samples").is_dir():
        for path in sorted((store_root / "samples").glob("*.json")):
            try:
                binding = _read(path)
                sample_id = binding.get("sample_id")
                if isinstance(sample_id, str):
                    bindings[sample_id] = (path, binding)
            except Exception:
                continue
    refs: dict[str, tuple[Path, dict[str, Any]]] = {}
    cases_root = root / "cases"
    if cases_root.is_symlink() or not cases_root.is_dir():
        errors.append("cases directory is missing or symlinked")
    else:
        for path in sorted(cases_root.rglob("operator-ref.json")):
            try:
                ref = _read(path)
            except Exception as exc:
                errors.append(str(exc)); continue
            sample_id = ref.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                errors.append(f"operator ref sample id missing: {path}"); continue
            if sample_id in refs:
                errors.append(f"duplicate operator ref sample: {sample_id}")
            refs[sample_id] = (path, ref)
            binding_rel = _safe_relative(root, ref.get("sample_binding"))
            if binding_rel is None:
                errors.append(f"operator ref binding path is unsafe: {path}"); continue
            try:
                binding_rel.relative_to(store_root.resolve())
            except ValueError:
                errors.append(f"operator ref binding escapes store: {path}"); continue
            if not binding_rel.is_file():
                errors.append(f"operator ref binding missing: {path}"); continue
            try:
                binding = _read(binding_rel)
            except Exception as exc:
                errors.append(str(exc)); continue
            if ref.get("operator_store") != "operator-artifacts":
                errors.append(f"operator ref store mismatch: {path}")
            for field in ("sample_id", "status", "operator_id", "candidate_id", "qualification_id", "activation_id"):
                if ref.get(field) != binding.get(field):
                    errors.append(f"operator ref/binding mismatch {field}: {path}")
            if binding.get("sample_id") not in bindings:
                errors.append(f"operator ref binding not indexed: {path}")
        for sample_id in sorted(set(bindings) - set(refs)):
            errors.append(f"sample binding has no operator ref: {sample_id}")
    summary_path = root / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file() and not summary_path.is_symlink():
        try:
            summary = _read(summary_path)
            total = summary.get("total")
            if not isinstance(total, int) or total != expected_total:
                errors.append(f"summary total does not match expected total: {total}/{expected_total}")
            if total != len(refs):
                errors.append(f"summary total/ref count mismatch: {total}/{len(refs)}")
            sample_count = store_report.get("counts", {}).get("samples", 0)
            if sample_count != expected_total:
                errors.append(f"sample binding count does not match expected total: {sample_count}/{expected_total}")
            artifact_summary = summary.get("operator_artifacts") or {}
            if artifact_summary.get("sample_bindings") != sample_count:
                errors.append("summary operator artifact count mismatch")
        except Exception as exc:
            errors.append(str(exc))
    else:
        errors.append("summary.json is missing or symlinked")
    counts = {
        "operator_refs": len(refs),
        "sample_bindings": store_report.get("counts", {}).get("samples", 0),
        "receipts": sum(1 for _ in cases_root.rglob("receipt.json")) if cases_root.is_dir() else 0,
        "summary_total": summary.get("total"),
    }
    return {
        "schema": "numpilot.operator-artifact-run-audit",
        "schema_version": 1,
        "ok": not errors,
        "root": str(root),
        "counts": counts,
        "store": store_report,
        "errors": sorted(set(errors)),
    }
