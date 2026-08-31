from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

from tools.check_release_tree import (
    PUBLIC_EVIDENCE_CASES,
    REQUIRED_FILES,
    _forbidden_path_reason,
    _validate_pptx,
    audit_release_tree,
)


ROOT = Path(__file__).resolve().parents[1]


def test_release_tree_audit_passes_current_candidate_without_clean_requirement():
    report = audit_release_tree(ROOT)

    assert report["ok"], report["errors"]
    assert report["schema"] == "numpilot.release_tree_audit.v1"
    assert report["version"] == "0.3.0a1.dev0"
    assert report["public_evidence_cases"] == sorted(PUBLIC_EVIDENCE_CASES)
    assert report["file_count"] > 200


def test_release_tree_forbidden_path_policy_covers_build_and_evidence_noise():
    forbidden = {
        "todo.md",
        "docs/AI-HANDOFF.md",
        "evidence/heat-diffusion-20260813/bench/cpu_baseline.npy",
        "evidence/poisson-ritz-p0-20260821/profiler/runtime.db",
        "build/lib/numpilot/__init__.py",
        "numpilot/__pycache__/__init__.cpython-310.pyc",
    }
    assert all(_forbidden_path_reason(path) for path in forbidden)
    assert _forbidden_path_reason("numpilot/script_migration.py") is None
    assert _forbidden_path_reason("tests/fixtures/ops-nn-recovery/gap.json") is None


def test_release_tree_cli_emits_machine_readable_report():
    completed = subprocess.run(
        [sys.executable, "tools/check_release_tree.py", "--json"],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    report = json.loads(completed.stdout)
    assert completed.returncode == 0, report["errors"]
    assert report["ok"] is True


def test_release_tree_cli_reports_missing_required_file_as_json(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("development snapshot\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", repo], check=True)
    subprocess.run(["git", "-C", repo, "add", "README.md"], check=True)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/check_release_tree.py"),
            "--repo-root",
            str(repo),
            "--json",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    report = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert report["ok"] is False
    assert "missing required release file: README.zh-CN.md" in report["errors"]
    assert "missing required release file: numpilot/__init__.py" in report["errors"]
    assert "README.zh-CN.md" in REQUIRED_FILES
    assert "numpilot/__init__.py" in REQUIRED_FILES


def test_release_tree_rejects_parseable_zip_with_invalid_pptx_xml(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    presentation = docs / "numpilot-en.pptx"
    with zipfile.ZipFile(presentation, "w") as archive:
        archive.writestr("[Content_Types].xml", "not xml")
        archive.writestr("ppt/presentation.xml", "also not xml")
        archive.writestr(
            "ppt/slides/slide1.xml", "not xml ADAPTER_CANDIDATE planned"
        )

    errors: list[str] = []
    _validate_pptx(tmp_path, errors)

    assert errors == ["invalid PPTX XML payload: docs/numpilot-en.pptx"]
