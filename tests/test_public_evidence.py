import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"


def _json(relative):
    return json.loads((EVIDENCE / relative).read_text(encoding="utf-8"))


def test_public_evidence_manifest_is_complete_and_valid():
    manifest = EVIDENCE / "MANIFEST.sha256"
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        assert relative.startswith("evidence/")
        target = ROOT / relative
        assert target.is_file()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        entries.append(relative)

    expected = sorted(
        str(path.relative_to(ROOT))
        for path in EVIDENCE.rglob("*")
        if path.is_file() and path != manifest
    )
    assert entries == expected


def test_curated_evidence_preserves_fail_closed_verdicts():
    heat = _json("heat-diffusion-20260813/bench/npu_benchmark.json")
    assert heat["status"] == "pass"
    assert heat["performance_valid"] is True
    assert heat["provenance"]["ok"] is True

    qualification = _json(
        "poisson-ritz-p0-20260821/qualification-evidence-v5/"
        "qualification-summary.json"
    )
    assert qualification["status"] == "qualified_and_overlay_published"

    iteration = _json(
        "poisson-ritz-p0-20260821/runs/"
        "poisson-full-iteration-strict-npu-v2/report.json"
    )
    assert iteration["status"] == "pass"
    assert iteration["total_rewrites"] == 117

    multiround = _json(
        "poisson-ritz-p0-20260821/runs/"
        "poisson-multiround-strict-npu-v3/evidence.json"
    )
    assert multiround["status"] == "pass"
    assert multiround["strict_npu"] is True
    assert multiround["iterations"] == 250
    assert multiround["total_runtime_calls"] == 29250

    profiler = _json(
        "poisson-ritz-p0-20260821/profiler/evidence-run/evidence.json"
    )
    assert profiler["status"] == "pass"
    assert profiler["strict_npu"] is True
    assert profiler["iterations"] == 10
    assert profiler["total_runtime_calls"] == 1170

    move_multiround = _json(
        "poisson-ritz-p0-20260821/runs/"
        "poisson-multiround-strict-npu-v4-move/evidence.json"
    )
    assert move_multiround["status"] == "pass"
    assert move_multiround["strict_npu"] is True
    assert move_multiround["iterations"] == 250
    assert move_multiround["total_runtime_calls"] == 29250

    copy_audit = _json(
        "poisson-ritz-p0-20260821/profiler-copy-elision/audit/evidence.json"
    )
    assert copy_audit["status"] == "pass"
    assert copy_audit["before"]["rows"] == 1235
    assert copy_audit["after"]["rows"] == 35
    assert copy_audit["eliminated"]["copies"] == 1200
    assert all(copy_audit["checks"].values())
