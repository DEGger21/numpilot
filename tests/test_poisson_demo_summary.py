import hashlib
import json
from pathlib import Path

import pytest

from tools.poisson_demo_summary import (
    DemoEvidenceError,
    load_published_summary,
    render_affinity_performance,
    render_live_migration,
    render_live_performance,
    render_published,
    verify_public_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_published_poisson_showcase_is_fail_closed_and_truthful():
    summary = load_published_summary(ROOT)
    rendered = render_published(summary)

    assert summary["rewrites"] == 101
    assert summary["baseline_calls"] == 117
    assert summary["optimized_calls"] == 101
    assert summary["iterations"] == 3000
    assert summary["cpu_faster"] > 1.0
    assert "CPU 仍约快" in rendered
    assert "不宣称 NPU 超越 CPU" in rendered


def test_public_manifest_verifier_covers_curated_evidence():
    count, total_bytes = verify_public_manifest(ROOT)

    assert count > 20
    assert total_bytes > 0


def test_public_manifest_verifier_rejects_parent_traversal(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("not evidence", encoding="utf-8")
    digest = hashlib.sha256(secret.read_bytes()).hexdigest()
    (evidence / "MANIFEST.sha256").write_text(
        f"{digest}  evidence/../secret.txt\n",
        encoding="utf-8",
    )

    with pytest.raises(DemoEvidenceError, match="越界"):
        verify_public_manifest(tmp_path)


def test_live_renderers_accept_curated_passing_receipts():
    migration = ROOT / (
        "evidence/poisson-ritz-p0-20260821/performance-broadcast-optimized/"
        "migration/report.json"
    )
    performance = ROOT / (
        "evidence/poisson-ritz-p0-20260821/performance-broadcast-optimized/"
        "final/evidence.json"
    )

    assert "strict-NPU=yes" in render_live_migration(migration)
    assert "全部发布门通过" in render_live_performance(performance)


def test_live_migration_renderer_fails_closed_on_non_strict_receipt(tmp_path):
    source = ROOT / (
        "evidence/poisson-ritz-p0-20260821/performance-broadcast-optimized/"
        "migration/report.json"
    )
    receipt = json.loads(source.read_text(encoding="utf-8"))
    receipt["backend_provenance"]["strict_npu"] = False
    target = tmp_path / "report.json"
    target.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(DemoEvidenceError, match="strict-NPU"):
        render_live_migration(target)


def _affinity_receipt(tmp_path):
    source = ROOT / (
        "evidence/poisson-ritz-p0-20260821/performance-broadcast-optimized/"
        "final/evidence.json"
    )
    receipt = json.loads(source.read_text(encoding="utf-8"))
    receipt["cpu"]["cpu_affinity"] = {
        "requested_logical_cpu": 3,
        "observed_logical_cpus": [3],
        "logical_cpu_count": 1,
        "pinned": True,
    }
    receipt["performance"]["published"] = False
    receipt["performance"]["issues"] = [
        "missing passing profiler audit for this scale"
    ]
    receipt["performance"]["cpu_time_over_npu_time"] = None
    target = tmp_path / "affinity.json"
    target.write_text(json.dumps(receipt), encoding="utf-8")
    return target


def test_affinity_renderer_reports_verified_constraint_and_diagnostic_ratio(tmp_path):
    rendered = render_affinity_performance(_affinity_receipt(tmp_path))

    assert "CPU worker 仅允许逻辑核 3" in rendered
    assert "BLAS runtime threads=1" in rendered
    assert "诊断时延" in rendered
    assert "不发布为正式性能比值" in rendered


def test_affinity_renderer_fails_closed_on_broad_cpu_mask(tmp_path):
    target = _affinity_receipt(tmp_path)
    receipt = json.loads(target.read_text(encoding="utf-8"))
    receipt["cpu"]["cpu_affinity"]["observed_logical_cpus"] = [3, 4]
    receipt["cpu"]["cpu_affinity"]["logical_cpu_count"] = 2
    target.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(DemoEvidenceError, match="一个逻辑核"):
        render_affinity_performance(target)
