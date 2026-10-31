# This file is part of the NumPilot project at Harbin Institute of Technology (HIT).
#
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
#
# Authors (accounts):
# - Su Tonghua <@sutonghua>
# - Li Xing <@li-xingHIT>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from tools.check_release_tree import (
    ASNUMPY_COMPONENT_MANIFEST,
    ASNUMPY_COMPONENT_README,
    PRIVATE_PATH_RE,
    REQUIRED_FILES,
    _validate_asnumpy_component,
    _scan_public_text,
    _forbidden_path_reason,
    audit_release_tree,
)


ROOT = Path(__file__).resolve().parents[1]


def test_release_tree_audit_passes_current_candidate_without_clean_requirement():
    report = audit_release_tree(ROOT)

    assert report["ok"], report["errors"]
    assert report["schema"] == "numpilot.release_tree_audit.v1"
    assert report["version"] == "0.3.0a1.dev0"
    assert report["file_count"] > 200


def test_release_tree_strict_asnumpy_component_gate_verifies_current_attachment():
    report = audit_release_tree(ROOT, require_asnumpy_component=True)

    assert report["ok"], report["errors"]
    component = report["asnumpy_component"]
    assert component["manifest"] is True
    assert component["wheel_present"] is True
    assert component["wheel_verified"] is True


def test_release_tree_strict_asnumpy_component_gate_fails_closed_without_wheel(tmp_path):
    component = tmp_path / "components" / "asnumpy"
    component.mkdir(parents=True)
    source = ROOT / "components" / "asnumpy"
    for name in ("manifest.json", "README.md"):
        (component / name).write_bytes((source / name).read_bytes())
    errors: list[str] = []
    warnings: list[str] = []
    result = _validate_asnumpy_component(
        tmp_path,
        [ASNUMPY_COMPONENT_MANIFEST, ASNUMPY_COMPONENT_README],
        errors,
        warnings,
        require_artifact=True,
    )

    assert result["wheel_present"] is False
    assert any("wheel attachment is missing" in error for error in errors)


def test_release_tree_component_gate_rejects_private_paths_in_declared_files(tmp_path):
    component = tmp_path / "components" / "asnumpy"
    component.mkdir(parents=True)
    source = ROOT / "components" / "asnumpy"
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".whl":
            (component / path.name).write_bytes(path.read_bytes())
    readme = component / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8")
        + "\n"
        + "/home/" + "ma-user/private"
        + "\n",
        encoding="utf-8",
    )
    errors: list[str] = []
    warnings: list[str] = []
    _validate_asnumpy_component(
        tmp_path,
        [
            "components/asnumpy/manifest.json",
            "components/asnumpy/README.md",
            "components/asnumpy/LICENSE",
            "components/asnumpy/SBOM.spdx.json",
        ],
        errors,
        warnings,
    )

    assert any("private absolute workspace path" in error for error in errors)


def test_release_tree_component_gate_does_not_accept_untracked_attachment(tmp_path):
    """The audit must validate Git's candidate set, not an ignored local wheel."""

    component = tmp_path / "components" / "asnumpy"
    component.mkdir(parents=True)
    source = ROOT / "components" / "asnumpy"
    for path in source.iterdir():
        if path.is_file():
            (component / path.name).write_bytes(path.read_bytes())
    files = [
        f"components/asnumpy/{path.name}"
        for path in component.iterdir()
    ]
    # Simulate a Git candidate set in which the wheel is ignored/untracked,
    # while it remains physically present next to the manifest.
    wheel_name = next(path.name for path in component.glob("*.whl"))
    files.remove(f"components/asnumpy/{wheel_name}")
    errors: list[str] = []
    warnings: list[str] = []
    result = _validate_asnumpy_component(
        tmp_path, files, errors, warnings, require_artifact=True
    )

    assert result["wheel_present"] is False
    assert any("not part of the Git release candidate" in error for error in errors)


def test_release_tree_forbidden_path_policy_covers_build_and_evidence_noise():
    forbidden = {
        "todo.md",
        "TEMP-RECORDING-COMMANDS.zh-CN.md",
        "llm.json",
        "docs/AI-HANDOFF.md",
        "evidence/heat-diffusion-20260813/bench/cpu_baseline.npy",
        "evidence/poisson-ritz-p0-20260821/profiler/runtime.db",
        "build/lib/numpilot/__init__.py",
        "numpilot/__pycache__/__init__.cpython-310.pyc",
    }
    assert all(_forbidden_path_reason(path) for path in forbidden)
    assert _forbidden_path_reason("numpilot/script_migration.py") is None
    assert _forbidden_path_reason("tests/fixtures/ops-nn-recovery/gap.json") is None


def test_release_tree_private_path_regex_matches_home_boundary_without_slash():
    """The public-text scan must catch a redacted home path at end-of-string."""

    # Build the examples instead of publishing a real workspace identity in
    # the test source itself.
    ma_user = "/home/" + "ma-user"
    lixing = "/home/" + "lixing"
    assert PRIVATE_PATH_RE.search("generated in " + ma_user)
    assert PRIVATE_PATH_RE.search("generated in " + lixing)
    assert PRIVATE_PATH_RE.search("generated in " + ma_user + "/project")
    assert not PRIVATE_PATH_RE.search("/home/" + "ma-userish/project")


def test_release_tree_public_evidence_scan_rejects_private_workspace_paths(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    path = evidence / "receipt.json"
    private_path = "/home/" + "ma-user/work/numpilot"
    path.write_text(json.dumps({"path": private_path}) + "\n", encoding="utf-8")
    errors: list[str] = []

    _scan_public_text(tmp_path, ["evidence/receipt.json"], errors)

    assert errors == ["private absolute workspace path in public text: evidence/receipt.json"]


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
    (repo / "README.md").write_text("missing release label\n", encoding="utf-8")
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
    assert "llm.example.json" in REQUIRED_FILES
    assert "docs/assets/architecture/numpilot-two-routes.zh-CN.png" in REQUIRED_FILES
    assert "numpilot/__init__.py" in REQUIRED_FILES
    assert "numpilot/demo_llm_operator_generation.py" in REQUIRED_FILES
    assert "numpilot/migration_skill.py" in REQUIRED_FILES
    assert "numpilot/workflow.py" in REQUIRED_FILES
