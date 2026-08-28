import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "evidence" / "competition-ordinary-20260828"
POISSON_CASE = ROOT / "evidence" / "competition-poisson-stage-20260828"
AUDITOR = ROOT / "tools" / "audit_script_benchmark_receipt.py"


def _run(case):
    return subprocess.run(
        [sys.executable, str(AUDITOR), str(case), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )


def _replace_manifest_digest(case, relative):
    manifest = case / "MANIFEST.sha256"
    digest = hashlib.sha256((case / relative).read_bytes()).hexdigest()
    needle = f"  ./{relative}"
    lines = [
        f"{digest}{needle}" if line.endswith(needle) else line
        for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_packaged_ordinary_evidence_recomputes_published_statistics():
    completed = _run(CASE)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)

    assert result["status"] == "passed"
    assert result["actual_remote_call"] is True
    assert result["configured_model"] == "glm-5.3"
    assert result["hardware"] == "Ascend910B4"
    assert result["samples_per_backend"] == 20
    assert result["all_benchmark_gates_passed"] is True
    assert result["resident"]["median_speedup"] == 8.078965110765662
    assert result["resident"]["conservative_speedup"] == 7.5026021068715885
    assert result["resident"]["bootstrap_95ci"] == [
        7.954381451900704,
        8.146640628459657,
    ]
    assert result["evidence_boundary"] == {
        "development_tree_dirty_at_capture": True,
        "direct_end_to_end_sample": False,
        "kernel_profiler_trace": False,
        "numeric_receipt_and_output_hashes_preserved": True,
        "numerical_arrays_in_public_bundle": False,
        "numerical_values_independently_recomputed": False,
        "steady_device_resident": True,
    }


def test_packaged_poisson_stage_binds_bootstrap_and_recomputes_statistics():
    completed = _run(POISSON_CASE)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)

    assert result["status"] == "passed"
    assert result["bootstrap_receipt_bound"] is True
    assert result["actual_remote_call"] is True
    assert result["configured_model"] == "glm-5.3"
    assert result["hardware"] == "Ascend910B4"
    assert result["samples_per_backend"] == 20
    assert result["all_benchmark_gates_passed"] is True
    assert result["resident"] == {
        "bootstrap_95ci": [2.5379188859894697, 2.592045265736361],
        "conservative_speedup": 2.4712193248528562,
        "cpu_median_ms": 458.6982805,
        "median_speedup": 2.5674650455003585,
        "npu_median_ms": 178.658043,
    }


def test_auditor_rejects_relabelled_speedup_even_with_updated_case_manifest(tmp_path):
    copied = tmp_path / "case"
    shutil.copytree(CASE, copied)
    report_path = copied / "script-benchmark-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["performance"]["median_speedup"] = 80.0
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _replace_manifest_digest(copied, "script-benchmark-report.json")

    completed = _run(copied)
    assert completed.returncode == 1
    assert "median speedup mismatch" in completed.stderr


def test_auditor_rejects_rebound_source_even_with_updated_case_manifest(tmp_path):
    copied = tmp_path / "case"
    shutil.copytree(CASE, copied)
    source = copied / "source.py"
    source.write_text(
        source.read_text(encoding="utf-8") + "\n# tampered\n",
        encoding="utf-8",
    )
    _replace_manifest_digest(copied, "source.py")

    completed = _run(copied)
    assert completed.returncode == 1
    assert "benchmark does not bind the packaged source" in completed.stderr


def test_auditor_rejects_rebound_bootstrap_even_with_updated_case_manifest(
    tmp_path,
):
    copied = tmp_path / "case"
    shutil.copytree(POISSON_CASE, copied)
    receipt_path = copied / "bootstrap-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["numpilot_version"] = "99.0"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _replace_manifest_digest(copied, "bootstrap-receipt.json")

    completed = _run(copied)
    assert completed.returncode == 1
    assert "bootstrap receipt hash is not runtime-bound" in completed.stderr
