"""Hash-bound, exact-sample qualification of an official overlay callsite.

This is an offline verifier, never an execution engine.  It does not import
AsNumpy, execute source, or promote a runtime's global audit status.  Archive
both the case evidence directory and its official qualification bundle.  When
restoring an archive, explicit directory overrides relocate those two roots;
all file hashes and semantic bindings are still checked.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from numpilot.agents.analysis import numpy_bindings_for_forward, resolve_numpy_function
from numpilot.benchmark_lowering import lower_benchmark_program
from numpilot.capabilities import (AnalysisProfileSet, CapabilityManifest,
    RuntimeIdentity, match_capability_profile)
from numpilot.contracts import OffloadPlan
from numpilot.engines.ast_rewrite import AstRewriteEngine, audit_forward_surface
from numpilot.gaps.official_qualification import (
    OFFICIAL_PRODUCER, apply_official_overlay, load_official_overlay,
)

SCOPE_SCHEMA = "numpilot.scoped-runtime-qualification"
SCOPE_REF_SCHEMA = "numpilot.scoped-runtime-qualification-ref"
IDENTITY_SCOPE = "official_overlay_callsite"
_FILES = frozenset({"original.py", "analysis_source.py", "migrated.py",
    "profiles.json", "overlay-admission.json", "runtime-before.json",
    "runtime-after.json", "validation-evidence.json", "mainline-result.json",
    "base-manifest.json", "capability-overlay.json"})
_IDENTITIES = ("sample_id", "source_sha256", "analysis_source_sha256",
    "generated_source_sha256", "profile_sha256", "runtime_identity_sha256",
    "qualification_id", "overlay_sha256", "required_ops", "runtime_identity_status")
_REF_FIELDS = frozenset({"schema", "schema_version", "receipt_path",
    "receipt_sha256", "identity_scope", "global_runtime_audit_promoted", *_IDENTITIES})
_SCOPE_FIELDS = frozenset({"schema", "schema_version", "identity_scope",
    "global_runtime_audit_promoted", "performance_valid", "status", "evidence",
    "official_evidence_root", *_IDENTITIES})
_BASELINE_IDENTITIES = ("baseline_qualification_id", "baseline_certificate_sha256")


def _identity_fields(value):
    return _IDENTITIES + (_BASELINE_IDENTITIES if value.get("schema_version") == 2 else ())


_SHA = re.compile(r"[0-9a-f]{64}\Z")


class ScopedQualificationError(ValueError):
    """Evidence is missing, inconsistent, or outside the qualified scope."""


def _require(ok: Any, reason: str) -> None:
    if not ok:
        raise ScopedQualificationError(reason)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: Any) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _regular(path: Path, *, limit: int = 8 * 1024 * 1024) -> bytes:
    _require(path.is_absolute(), "evidence path must be absolute")
    for part in (path, *path.parents):
        _require(not part.is_symlink(), "symlink evidence path rejected")
    _require(path.is_file(), "evidence file missing: " + str(path))
    _require(path.stat().st_size <= limit, "evidence file exceeds size limit")
    return path.read_bytes()


def _object(payload: bytes, label: str) -> dict:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON field: " + label)
            result[key] = value
        return result
    try:
        data = json.loads(payload, object_pairs_hook=no_duplicates,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ScopedQualificationError("nonfinite JSON value")))
    except (ValueError, UnicodeError) as exc:
        raise ScopedQualificationError("invalid JSON: " + label) from exc
    _require(isinstance(data, dict), "expected JSON object: " + label)
    return data


def _identity(value: Any) -> dict:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    try:
        return RuntimeIdentity.from_dict(value).to_dict()
    except (TypeError, ValueError, KeyError) as exc:
        raise ScopedQualificationError("invalid runtime identity: " + str(exc)) from exc


def _load_official_admission(root: Path, official_root: str, before: dict):
    """Recheck the same official boundary as the live backend, without imports."""
    overlay = load_official_overlay(str(root / "capability-overlay.json"))
    base = CapabilityManifest.from_dict(_object(_regular(root / "base-manifest.json"), "base manifest"))
    official_names = {entry.registry_key for entry in overlay.entries}
    official_base = CapabilityManifest(backend=base.backend, backend_version=base.backend_version,
        operators={name: base.operators[name] for name in official_names},
        environment=dict(base.environment), schema_version=base.schema_version, audit_id=base.audit_id)
    admitted = apply_official_overlay(official_base, overlay, evidence_root=official_root,
        trusted_producers=(OFFICIAL_PRODUCER,), current_identity=before)
    combined = CapabilityManifest(backend=base.backend, backend_version=base.backend_version,
        operators={**base.operators, **admitted.operators}, environment=dict(base.environment),
        schema_version=base.schema_version, audit_id=admitted.audit_id)
    return overlay, combined


def _check_evidence(scope: Mapping[str, Any], root: Path, official_root: str,
                    expected_runtime_identity: Any = None, baseline_root=None) -> None:
    files = scope["evidence"]["files"]
    expected_files = _FILES | ({"baseline-api-certificate.json"} if scope["schema_version"] == 2 else set())
    _require(isinstance(files, dict) and set(files) == expected_files,
             "scope evidence file set is incomplete or unknown")
    raw = {}
    for name, sha in files.items():
        _require(isinstance(sha, str) and _SHA.fullmatch(sha), "invalid evidence SHA-256")
        raw[name] = _regular(root / name)
        _require(_sha(raw[name]) == sha, "evidence hash mismatch: " + name)
    data = {name: _object(payload, name) for name, payload in raw.items() if name.endswith(".json")}
    source = raw["original.py"].decode("utf-8")
    analysis = raw["analysis_source.py"].decode("utf-8")
    generated = raw["migrated.py"].decode("utf-8")
    for name, key in (("original.py", "source_sha256"), ("analysis_source.py", "analysis_source_sha256"),
                      ("migrated.py", "generated_source_sha256"), ("profiles.json", "profile_sha256"),
                      ("capability-overlay.json", "overlay_sha256")):
        _require(files[name] == scope[key], key + " mismatch")
    _require(lower_benchmark_program(source).source == analysis,
             "analysis source is not the deterministic lowering of the original")
    before = _identity(data["runtime-before.json"])
    after = _identity(data["runtime-after.json"])
    _require(before == after, "runtime changed during execution")
    _require(before["status"] in {"audit_required", "matched"}, "runtime status cannot be scoped")
    _require(before["identity_sha256"] == scope["runtime_identity_sha256"], "runtime identity mismatch")
    _require(before["status"] == scope["runtime_identity_status"], "global runtime status mismatch")
    if expected_runtime_identity is not None:
        _require(before == _identity(expected_runtime_identity), "expected runtime identity drift")
    overlay, manifest = _load_official_admission(root, official_root, before)
    _require(scope["qualification_id"] == overlay.qualification_id, "qualification ID mismatch")
    overlay_ops = sorted(entry.registry_key for entry in overlay.entries)
    qualified_names = sorted(entry.qualified_api for entry in overlay.entries)
    admission = data["overlay-admission.json"]
    if scope["schema_version"] == 2:
        from numpilot.baseline_api_qualification import load_baseline_api_certificate, apply_baseline_api_certificate
        _require(files["baseline-api-certificate.json"] == scope["baseline_certificate_sha256"],
                 "baseline certificate file binding mismatch")
        certificate = load_baseline_api_certificate(root / "baseline-api-certificate.json")
        _require(certificate.qualification_id == scope["baseline_qualification_id"], "baseline certificate ID mismatch")
        manifest = apply_baseline_api_certificate(manifest, certificate, evidence_root=baseline_root,
            current_identity=before, verify_live_files=False)
        baseline_admission = {
            "qualification_id": certificate.qualification_id,
            "certificate_sha256": scope["baseline_certificate_sha256"],
            "runtime_identity_sha256": certificate.runtime.identity_sha256,
            "operators": list(certificate.entry_keys),
            "qualified_names": sorted(entry.qualified_api for entry in certificate.entries),
        }
        _require(admission.get("baseline_api") == baseline_admission, "baseline admission mismatch")
        overlay_ops = sorted(set(overlay_ops) | set(certificate.entry_keys))
        qualified_names = sorted(set(qualified_names) | set(baseline_admission["qualified_names"]))
    else:
        _require("baseline_api" not in admission, "v1 scope cannot silently include baseline API admission")
    _require(admission.get("execution") == "strict_npu_mainline" and
             admission.get("admission") == "official_qualification_overlay", "non-NPU overlay admission")
    for name in ("qualification_id", "overlay_sha256", "runtime_identity_sha256"):
        _require(admission.get(name) == scope[name], "overlay admission " + name + " mismatch")
    _require(admission.get("operators") == overlay_ops and
             admission.get("qualified_names") == qualified_names,
             "overlay admitted operator set mismatch")
    profiles_doc = data["profiles.json"]
    profiles = AnalysisProfileSet.from_dict(profiles_doc.get("profiles"))
    _require(profiles.source_sha256 == scope["analysis_source_sha256"], "profile source mismatch")
    sites = list(profiles.callsites)
    _require(bool(sites) and type(profiles_doc.get("matched_callsites")) is int and
             type(profiles_doc.get("attempted_callsites")) is int and
             profiles_doc["matched_callsites"] == profiles_doc.get("attempted_callsites") == len(sites),
             "incomplete static callsite profiles")
    tree = ast.parse(analysis)
    forwards = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "forward"]
    _require(len(forwards) == 1, "missing unique forward")
    bindings = numpy_bindings_for_forward(tree, forwards[0])
    static_sites = []
    for node in ast.walk(forwards[0]):
        if isinstance(node, ast.Call):
            resolved = resolve_numpy_function(node.func, bindings)
            if resolved is not None:
                static_sites.append((resolved[1], node.lineno, node.col_offset,
                                     len(node.args), tuple(sorted(k.arg or "**" for k in node.keywords))))
    recorded_sites = [(s.qualified_name, s.lineno, s.col_offset, s.profile.positional_args,
                       tuple(s.profile.keyword_args)) for s in sites]
    _require(sorted(recorded_sites) == sorted(static_sites), "static profile does not cover exact source calls")
    counts = dict(sorted(Counter(s.qualified_name.removeprefix("numpy.") for s in sites).items()))
    required = sorted(counts)
    _require(scope["required_ops"] == required and set(required) <= set(overlay_ops),
             "required operations outside official scope")
    for site in sites:
        name = site.qualified_name.removeprefix("numpy.")
        _require(match_capability_profile(manifest.operators[name], site.profile).matched,
                 "callsite profile is outside official constraints")
    expected_generated = AstRewriteEngine().rewrite(analysis, OffloadPlan({name: "NPU" for name in required}, 0))
    _require(generated == expected_generated, "generated source is not the deterministic AST rewrite")
    audit = audit_forward_surface(generated)
    _require(audit.ok and audit.ap_calls == counts, "generated forward surface is not strictly device-backed")
    result = data["mainline-result.json"]
    _require(result.get("status") == "pass" and result.get("engine") == "ast" and
             result.get("rewrite_integrity") is True, "mainline AST migration did not pass")
    for key in ("rewrites", "expected_rewrites"):
        declared = result.get(key)
        _require(isinstance(declared, dict) and all(type(v) is int for v in declared.values())
                 and declared == counts, "mainline " + key + " mismatch")
    for key in ("source_sha256", "analysis_source_sha256"):
        _require(result.get(key) == scope[key], "mainline " + key + " mismatch")
    if "migrated_source" in result:
        _require(result["migrated_source"] == generated, "mainline generated source mismatch")
    validation = data["validation-evidence.json"]
    if "validation_evidence" in result:
        _require(result["validation_evidence"] == validation, "mainline validation mismatch")
    _require(validation.get("status") == "pass" and validation.get("backend") == "npu",
             "validation is not passing NPU execution")
    for key in ("numerically_consistent", "input_consistent", "kernel_consistent",
                "artifact_consistent", "artifact_inputs_executed"):
        _require(validation.get(key) is True, "validation fact failed: " + key)
    _require(validation.get("source_sha256") == scope["source_sha256"] and
             validation.get("migrated_sha256") == scope["generated_source_sha256"], "validation source binding mismatch")
    comparisons = validation.get("comparisons", {})
    _require(isinstance(comparisons, dict) and set(comparisons) == {"inputs", "kernel", "artifact"},
             "missing numerical comparison paths")
    for comparison in comparisons.values():
        _require(isinstance(comparison, dict) and comparison.get("consistent") is True,
                 "numerical comparison did not pass")
        _require(type(comparison.get("n_elements")) is int and comparison["n_elements"] > 0,
                 "comparison has no verified elements")
        for key in ("max_abs_diff", "max_rel_diff", "atol", "rtol"):
            value = comparison.get(key)
            _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                     "comparison has invalid numeric evidence")
        _require(all(comparison.get(key) is None for key in ("failure_path", "failure_index", "reason")),
                 "comparison contains failure detail")
    guard = validation.get("backend_provenance", {})
    _require(guard.get("schema_version") == "1.1" and guard.get("backend") == "npu",
             "unknown or non-NPU runtime guard")
    for key in ("strict_npu", "ok", "preflight_ok", "preflight_strict_npu", "runtime_profile_ok",
                "runtime_target_integrity_ok", "runtime_execution_ok", "runtime_trace_ok"):
        _require(guard.get(key) is True, "runtime guard failed: " + key)
    _require(guard.get("profile_status") == "matched", "runtime profile did not match")
    for key in ("required_ops", "profile_scoped"):
        _require(guard.get(key) == required, "runtime guard scope mismatch")
    for key in ("missing", "non_strict", "runtime_target_integrity_errors"):
        _require(guard.get(key) == [], "runtime guard contains errors")
    call_counts = guard.get("runtime_call_counts")
    _require(isinstance(call_counts, dict) and set(call_counts) == {"kernel", "artifact"}
             and all(isinstance(values, dict) and all(type(v) is int for v in values.values())
                     for values in call_counts.values())
             and call_counts == {"kernel": counts, "artifact": counts},
             "actual runtime calls do not match source callsites")
    checks = guard.get("runtime_profile_checks")
    _require(isinstance(checks, dict) and set(checks) == set(required), "runtime profile check set mismatch")
    for name in required:
        expected = Counter(json.dumps(s.profile.to_dict(), sort_keys=True) for s in sites
                           if s.qualified_name == "numpy." + name)
        actual = Counter()
        _require(isinstance(checks[name], list), "invalid runtime profile checks")
        for check in checks[name]:
            _require(isinstance(check, dict) and check.get("matched") is True and check.get("reasons") == [],
                     "runtime call profile failed")
            actual[json.dumps(check.get("profile"), sort_keys=True)] += 1
        _require(actual == Counter({key: count * 2 for key, count in expected.items()}),
                 "actual runtime profiles differ from original source")
    _require(guard.get("audit_id") == manifest.audit_id and guard.get("backend_version") == before["asnumpy_version"],
             "guard runtime audit/version mismatch")
    environment = guard.get("environment", {})
    for key, identity_key in (("module_path", "module_path"), ("cann_version", "cann_version"),
                              ("python_version", "python_version"), ("soc_version", "soc_model")):
        _require(environment.get(key) == before[identity_key], "guard environment drift: " + key)
    _require(guard.get("capabilities") == {name: manifest.operators[name].to_dict() for name in required},
             "guard capability provenance mismatch")


def _validate_scope(scope, root, official_root, expected_runtime_identity=None, baseline_root=None):
    _require(isinstance(scope, dict), "scope receipt must be an object")
    version = scope.get("schema_version")
    _require(scope.get("schema") == SCOPE_SCHEMA and type(version) is int and version in {1, 2},
             "unknown scoped qualification schema")
    fields = _SCOPE_FIELDS | (set(_BASELINE_IDENTITIES) | {"baseline_evidence_root"} if version == 2 else set())
    _require(set(scope) == fields, "unknown scope receipt fields")
    if version == 2:
        baseline_root = baseline_root or scope.get("baseline_evidence_root")
        _require(isinstance(baseline_root, str) and Path(baseline_root).is_absolute(), "invalid baseline evidence root")
    _require(scope["identity_scope"] == IDENTITY_SCOPE and scope["global_runtime_audit_promoted"] is False
             and scope["performance_valid"] is False and scope["status"] == "qualified", "invalid scoped claim")
    _require(isinstance(scope["sample_id"], str) and bool(scope["sample_id"]), "invalid sample ID")
    for name in _identity_fields(scope):
        if name.endswith("sha256"):
            _require(isinstance(scope[name], str) and _SHA.fullmatch(scope[name]), "invalid " + name)
    _require(isinstance(scope["evidence"], dict) and set(scope["evidence"]) == {"root", "files"}, "invalid evidence root")
    try:
        _check_evidence(scope, root, official_root, expected_runtime_identity, baseline_root)
    except ScopedQualificationError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OSError, RuntimeError) as exc:
        raise ScopedQualificationError("scoped evidence rejected: " + str(exc)) from exc


def build_scoped_qualification(evidence_dir, *, sample_id, source_sha256,
        analysis_source_sha256, overlay_path, overlay_evidence_root,
        expected_runtime_identity, baseline_certificate_path=None, baseline_evidence_root=None) -> dict:
    """Seal evidence only after rechecking it; return a portable hash-bound ref."""
    root = Path(evidence_dir).expanduser().absolute()
    overlay_bytes = _regular(Path(overlay_path).expanduser().absolute())
    target = root / "capability-overlay.json"
    if target.exists():
        _require(_regular(target) == overlay_bytes, "scope overlay already differs")
    else:
        with target.open("xb") as stream:
            stream.write(overlay_bytes)
        target.chmod(0o600)
    extra_files = set()
    if baseline_certificate_path is not None:
        _require(baseline_evidence_root is not None, "baseline evidence root required")
        baseline_bytes = _regular(Path(baseline_certificate_path).expanduser().absolute())
        baseline_target = root / "baseline-api-certificate.json"
        if baseline_target.exists():
            _require(_regular(baseline_target) == baseline_bytes, "scope baseline certificate already differs")
        else:
            with baseline_target.open("xb") as stream:
                stream.write(baseline_bytes)
            baseline_target.chmod(0o600)
        extra_files.add("baseline-api-certificate.json")
    else:
        _require(baseline_evidence_root is None, "baseline root without certificate")
    files = {name: _sha(_regular(root / name)) for name in sorted(_FILES | extra_files)}
    admission = _object(_regular(root / "overlay-admission.json"), "admission")
    guard = _object(_regular(root / "validation-evidence.json"), "validation").get("backend_provenance", {})
    scope = {"schema": SCOPE_SCHEMA, "schema_version": 1, "status": "qualified",
        "identity_scope": IDENTITY_SCOPE, "global_runtime_audit_promoted": False,
        "performance_valid": False, "sample_id": sample_id, "source_sha256": source_sha256,
        "analysis_source_sha256": analysis_source_sha256,
        "generated_source_sha256": files["migrated.py"], "profile_sha256": files["profiles.json"],
        "runtime_identity_sha256": _identity(expected_runtime_identity)["identity_sha256"],
        "runtime_identity_status": _identity(expected_runtime_identity)["status"],
        "qualification_id": admission.get("qualification_id"), "overlay_sha256": files["capability-overlay.json"],
        "required_ops": guard.get("required_ops"), "evidence": {"root": str(root), "files": files},
        "official_evidence_root": str(Path(overlay_evidence_root).expanduser().absolute())}
    if baseline_certificate_path is not None:
        from numpilot.baseline_api_qualification import load_baseline_api_certificate
        certificate = load_baseline_api_certificate(root / "baseline-api-certificate.json")
        scope.update({"schema_version": 2, "baseline_qualification_id": certificate.qualification_id,
            "baseline_certificate_sha256": files["baseline-api-certificate.json"],
            "baseline_evidence_root": str(Path(baseline_evidence_root).expanduser().absolute())})
    _validate_scope(scope, root, scope["official_evidence_root"], expected_runtime_identity)
    payload = _json(scope)
    receipt = root / "scope.json"
    if receipt.exists():
        _require(_regular(receipt) == payload, "immutable scope receipt already differs")
    else:
        with receipt.open("xb") as stream:
            stream.write(payload)
        receipt.chmod(0o600)
    return {"schema": SCOPE_REF_SCHEMA, "schema_version": scope["schema_version"],
        "receipt_path": str(receipt), "receipt_sha256": _sha(payload),
        "identity_scope": IDENTITY_SCOPE, "global_runtime_audit_promoted": False,
        **{key: scope[key] for key in _identity_fields(scope)}}


def validate_scoped_qualification(ref, *, expected_sample_id=None,
        expected_source_sha256=None, expected_runtime_identity=None,
        expected_runtime_identity_sha256=None, evidence_dir=None,
        official_evidence_root=None, baseline_evidence_root=None) -> dict:
    """Read and revalidate the complete chain, optionally against a live identity.

    ``evidence_dir`` / ``official_evidence_root`` relocate archived evidence;
    neither changes any hash or admits unverified source. Offline validation is
    historical evidence verification, not a fresh device execution.
    """
    _require(isinstance(ref, Mapping), "scope reference must be an object")
    version = ref.get("schema_version")
    _require(ref.get("schema") == SCOPE_REF_SCHEMA and type(version) is int and version in {1, 2},
             "unknown scoped qualification reference schema")
    fields = _REF_FIELDS | (set(_BASELINE_IDENTITIES) if version == 2 else set())
    _require(set(ref) == fields, "unknown scope reference fields")
    _require(ref["identity_scope"] == IDENTITY_SCOPE and ref["global_runtime_audit_promoted"] is False,
             "invalid scope reference claim")
    path = Path(evidence_dir).expanduser().absolute() / "scope.json" if evidence_dir is not None else Path(ref["receipt_path"])
    payload = _regular(path)
    _require(_sha(payload) == ref["receipt_sha256"], "scope receipt hash mismatch")
    scope = _object(payload, "scope receipt")
    _require(scope.get("schema_version") == version, "scope/reference version mismatch")
    for key in _identity_fields(ref):
        _require(scope.get(key) == ref[key], "scope reference " + key + " mismatch")
    for expected, key in ((expected_sample_id, "sample_id"), (expected_source_sha256, "source_sha256"),
                          (expected_runtime_identity_sha256, "runtime_identity_sha256")):
        if expected is not None:
            _require(scope.get(key) == expected, "expected " + key + " mismatch")
    root = path.parent
    if evidence_dir is None:
        _require(scope.get("evidence", {}).get("root") == str(root), "scope root binding mismatch")
    official_root = str(Path(official_evidence_root).absolute()) if official_evidence_root is not None else scope.get("official_evidence_root")
    _require(isinstance(official_root, str) and Path(official_root).is_absolute(), "invalid official evidence root")
    _validate_scope(scope, root, official_root, expected_runtime_identity, baseline_evidence_root)
    if evidence_dir is not None:
        scope = {**scope, "evidence": {**scope["evidence"], "root": str(root)}}
    return scope
