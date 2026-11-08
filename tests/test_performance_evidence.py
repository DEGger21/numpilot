import hashlib
import json

from numpilot.performance_evidence import PerformanceEvidenceCatalog


def _receipt(source_sha="a" * 64, runtime_sha="b" * 64):
    return {
        "schema_version": 1,
        "receipt_type": "numpilot_ordinary_script_benchmark",
        "status": "passed",
        "binding": {
            "source": {"path": "source.py", "sha256": source_sha},
            "migrated": {"path": "migrated.py", "sha256": "c" * 64},
            "migration_report": {"path": "report.json", "sha256": "d" * 64},
        },
        "workers": {"npu": {
            "runtime_identity_before": {"status": "matched", "identity_sha256": runtime_sha},
            "runtime_identity_after": {"status": "matched", "identity_sha256": runtime_sha},
        }},
        "gates": {"strict": True, "numeric": True},
        "comparisons": {"output": {"consistent": True}},
        "protocol": {"scope": "steady_device_resident_marker_suffix"},
        "evidence_boundary": {"steady_state": True, "device_resident": True, "end_to_end": False},
        "performance": {"performance_valid": True, "issues": [], "median_speedup": 2.0, "cpu_median_ms": 2.0, "npu_median_ms": 1.0},
    }


def _catalog(tmp_path, receipt=None):
    receipt = receipt or _receipt()
    receipt_path = tmp_path / "A001.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    manifest = {"schema": "numpilot.performance-evidence-manifest", "schema_version": 1, "entries": {"A_direct_simple/A001": {"path": "A001.json", "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(), "selector": {"profile": "default"}}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return PerformanceEvidenceCatalog(tmp_path)


def test_performance_catalog_accepts_hash_bound_receipt(tmp_path):
    result = _catalog(tmp_path).resolve("A_direct_simple/A001", source_sha256="a" * 64, backend="npu", strict_npu_qualified=True, numerical_verified=True, runtime_identity_sha256="b" * 64)
    assert result.qualified is True
    assert result.ref["schema"] == "numpilot.performance-evidence-ref"
    assert result.ref["receipt_sha256"]


def test_performance_catalog_never_qualifies_shim_or_missing_entry(tmp_path):
    catalog = _catalog(tmp_path)
    shim = catalog.resolve("A_direct_simple/A001", source_sha256="a" * 64, backend="shim", strict_npu_qualified=None, numerical_verified=True, runtime_identity_sha256="b" * 64)
    assert shim.qualified is False
    missing = catalog.resolve("A_direct_simple/A002", source_sha256="a" * 64, backend="npu", strict_npu_qualified=True, numerical_verified=True, runtime_identity_sha256="b" * 64)
    assert missing.qualified is None
    assert missing.available is False


def test_performance_catalog_rejects_tampered_receipt(tmp_path):
    _catalog(tmp_path)
    (tmp_path / "A001.json").write_text("{}", encoding="utf-8")
    catalog = PerformanceEvidenceCatalog(tmp_path)
    result = catalog.resolve("A_direct_simple/A001", source_sha256="a" * 64, backend="npu", strict_npu_qualified=True, numerical_verified=True, runtime_identity_sha256="b" * 64)
    assert result.qualified is False
    assert "hash mismatch" in result.reason
