import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"


def _json(relative):
    return json.loads((EVIDENCE / relative).read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_ops_nn_matmul_evidence_binds_the_rebuild_patch_and_execution():
    case = EVIDENCE / "ops-nn-matmul-20260827"
    receipt = _json("ops-nn-matmul-20260827/qualification-receipt.json")
    execution = _json("ops-nn-matmul-20260827/execution-evidence.json")
    migration = case / "final-migration" / "migrated.py"
    migration_report = case / "final-migration" / "script-migration-report.json"
    source = ROOT / "examples" / "ops_nn_matmul_inference.py"
    harness = ROOT / "tools" / "ops_nn_matmul_case_evidence.py"
    patch = (
        ROOT
        / "integrations"
        / "asnumpy"
        / "0002-ops-nn-matmul-native-fp32.patch"
    )

    assert receipt["status"] == "qualified"
    assert receipt["qualification"]["all_gates_passed"] is True
    assert _sha256(patch) == receipt["asnumpy"]["source_patch_sha256"]
    assert _sha256(source) == execution["source"]["sha256"]
    assert _sha256(migration) == execution["migration"]["sha256"]
    assert _sha256(migration_report) == execution["migration"]["report_sha256"]
    assert _sha256(harness) == execution["harness"]["sha256"]
    assert execution["status"] == "passed"
    assert execution["performance_valid"] is False
    assert execution["route"] == {
        "host_fallback_attempted": False,
        "native_binding": "asnumpy._core.matmul",
        "native_call_count": 1,
        "public_api": "asnumpy.matmul",
    }


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

    performance = _json(
        "poisson-ritz-p0-20260821/performance-original-scale/"
        "final/evidence.json"
    )
    assert performance["status"] == "pass"
    assert performance["performance"]["published"] is True
    assert performance["performance"]["winner"] == "cpu"
    assert performance["cpu"]["threadpools"][0]["num_threads"] == 1
    assert performance["npu"]["strict_npu"] is True
    assert performance["npu"]["total_runtime_calls"] == 361530
    assert performance["workload"]["convergence_iterations"] == 3000
    assert performance["comparison"]["consistent"] is True
    assert performance["profiler_audit"]["status"] == "pass"
    assert all(performance["profiler_audit"]["checks"].values())

    performance_root = EVIDENCE / (
        "poisson-ritz-p0-20260821/performance-original-scale"
    )
    assert performance["runner_sha256"] == hashlib.sha256(
        (ROOT / "tools/poisson_performance_evidence.py").read_bytes()
    ).hexdigest()
    assert performance["cpu"]["output_sha256"] == hashlib.sha256(
        (performance_root / "final/cpu-convergence-output.npz").read_bytes()
    ).hexdigest()
    assert performance["npu"]["output_sha256"] == hashlib.sha256(
        (performance_root / "final/npu-convergence-output.npz").read_bytes()
    ).hexdigest()
    profiler_artifacts = performance["profiler_audit"]["artifacts"]
    for key, relative in (
        ("run_evidence_sha256", "profiler/profile-run.json"),
        ("runtime_db_sha256", "profiler/runtime.db"),
        ("task_db_sha256", "profiler/rts_track.db"),
    ):
        assert profiler_artifacts[key] == hashlib.sha256(
            (performance_root / relative).read_bytes()
        ).hexdigest()

    optimized = _json(
        "poisson-ritz-p0-20260821/performance-broadcast-optimized/"
        "final/evidence.json"
    )
    assert optimized["status"] == "pass"
    assert optimized["performance"]["published"] is True
    assert optimized["performance"]["winner"] == "cpu"
    assert optimized["performance"]["issues"] == []
    assert optimized["cpu"]["threadpools"][0]["num_threads"] == 1
    assert optimized["npu"]["strict_npu"] is True
    assert optimized["npu"]["runtime_profile_ok"] is True
    assert optimized["npu"]["runtime_target_integrity_ok"] is True
    assert sum(optimized["npu"]["per_iteration_calls"].values()) == 101
    assert optimized["npu"]["per_iteration_calls"]["outer"] == 2
    assert optimized["npu"]["total_runtime_calls"] == 312090
    assert optimized["workload"] == performance["workload"]
    assert optimized["scale"] == performance["scale"]
    assert optimized["comparison"] == performance["comparison"]
    assert (
        performance["performance"]["npu_median_per_iteration_ms"]
        / optimized["performance"]["npu_median_per_iteration_ms"]
        > 1.2
    )

    optimized_audit = optimized["profiler_audit"]
    assert optimized_audit["status"] == "pass"
    assert all(optimized_audit["checks"].values())
    assert optimized_audit["kernel_tasks"]["total"] == 505
    assert optimized_audit["memcpy"]["rows"] == 34

    optimized_root = EVIDENCE / (
        "poisson-ritz-p0-20260821/performance-broadcast-optimized"
    )
    assert optimized["runner_sha256"] == hashlib.sha256(
        (ROOT / "tools/poisson_broadcast_performance_evidence.py").read_bytes()
    ).hexdigest()
    assert optimized["npu"]["source_sha256"] == hashlib.sha256(
        (ROOT / "examples/poisson_ritz_iteration_broadcast.py").read_bytes()
    ).hexdigest()
    assert optimized["npu"]["migrated_sha256"] == hashlib.sha256(
        (optimized_root / "migration/migrated.py").read_bytes()
    ).hexdigest()
    assert optimized["npu"]["migration_report_sha256"] == hashlib.sha256(
        (optimized_root / "migration/report.json").read_bytes()
    ).hexdigest()
    for worker, filename in (
        ("cpu", "cpu-convergence-output.npz"),
        ("npu", "npu-convergence-output.npz"),
    ):
        optimized_output = optimized_root / "final" / filename
        baseline_output = performance_root / "final" / filename
        digest = hashlib.sha256(optimized_output.read_bytes()).hexdigest()
        assert optimized[worker]["output_sha256"] == digest
        assert optimized_output.read_bytes() == baseline_output.read_bytes()

    optimized_profiler_artifacts = optimized_audit["artifacts"]
    for key, relative in (
        ("run_evidence_sha256", "profiler/profile-run.json"),
        ("runtime_db_sha256", "profiler/runtime.db"),
        ("task_db_sha256", "profiler/rts_track.db"),
    ):
        assert optimized_profiler_artifacts[key] == hashlib.sha256(
            (optimized_root / relative).read_bytes()
        ).hexdigest()
