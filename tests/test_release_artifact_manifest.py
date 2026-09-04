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

import base64
import io
import json
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools import release_artifact_manifest as release_manifest


NAME = "numpilot"
VERSION = "9.8.7.dev1"
METADATA = f"""Metadata-Version: 2.4
Name: {NAME}
Version: {VERSION}
License-Expression: Apache-2.0
Requires-Python: >=3.10
Requires-Dist: numpy>=1.26
Requires-Dist: rich>=13
License-File: LICENSE
License-File: THIRD_PARTY_NOTICES.md

test package
""".encode()
VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
VALID_SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg"><text>'
    b"ADAPTER_CANDIDATE CAPABILITY_AUDIT_REQUIRED REVIEWED_MATMUL_ONLY "
    b"SOURCE_EVIDENCE_ONLY UNVERIFIED_OPERATOR_DRAFT"
    b"</text></svg>\n"
)


REVIEWED_PAYLOADS = {
    "docs/numpilot-en-v6.svg": VALID_SVG,
    "docs/numpilot-zh-v6.svg": VALID_SVG,
    "docs/assets/architecture/numpilot-architecture-en.svg": VALID_SVG,
    "docs/assets/architecture/numpilot-architecture-zh.svg": VALID_SVG,
    "docs/assets/architecture/numpilot-two-routes.zh-CN.png": VALID_PNG,
    "docs/assets/brand/asnumpy-mark.png": VALID_PNG,
    "docs/assets/brand/ascend-wordmark.png": VALID_PNG,
    "docs/assets/brand/cann-wordmark.png": VALID_PNG,
    "docs/assets/brand/codearts-icon.png": VALID_PNG,
    "docs/assets/brand/numpilot-wordmark.png": VALID_PNG,
}

PACKAGE_MEMBERS = (
    f"{NAME}/__init__.py",
    f"{NAME}/__main__.py",
    # Public orchestration modules are part of the release surface.
    # Keep synthetic wheel/sdist fixtures aligned with the production
    # manifest's required-member contract so these tests continue to exercise
    # the intended validation (rather than failing at fixture admission).
    f"{NAME}/cli_ui.py",
    f"{NAME}/add_one_gap.py",
    f"{NAME}/demo_llm_operator_generation.py",
    f"{NAME}/interactive_migration.py",
    f"{NAME}/interactive_report.py",
    f"{NAME}/llm_check.py",
    f"{NAME}/migration_skill.py",
    f"{NAME}/repl.py",
    f"{NAME}/runtime_extensions.py",
    f"{NAME}/_script_benchmark_worker.py",
    f"{NAME}/_script_verification_worker.py",
    f"{NAME}/script_benchmark.py",
    f"{NAME}/script_migration.py",
    f"{NAME}/script_verification.py",
    f"{NAME}/workflow.py",
    f"{NAME}/agents/analysis.py",
    f"{NAME}/agents/decision.py",
    f"{NAME}/agents/migration.py",
    f"{NAME}/agents/verification.py",
)

INSTALLED_SKILL_MEMBERS = (
    "share/numpilot/codearts/ProjectSkillStatus.txt",
    "share/numpilot/codearts/agents/numpilot-analyzer.md",
    "share/numpilot/codearts/agents/numpilot-evidence-auditor.md",
    "share/numpilot/codearts/agents/numpilot-operator-engineer.md",
    "share/numpilot/codearts/agents/numpilot-orchestrator.md",
    "share/numpilot/skills/numpilot-migration-gate/SKILL.md",
    "share/numpilot/skills/numpilot-migration-gate/agents/openai.yaml",
    "share/numpilot/skills/numpilot-migration-gate/references/codearts-sidebar.md",
    "share/numpilot/skills/numpilot-migration-gate/references/terminal-contract.md",
    "share/numpilot/skill/README.md",
    "share/numpilot/skill/SKILL.md",
    "share/numpilot/skill/agents/openai.yaml",
    "share/numpilot/skill/references/codearts-sidebar.md",
    "share/numpilot/skill/references/terminal-contract.md",
    "share/numpilot/skills/asnumpy-custom-op-integration/SKILL.md",
    "share/numpilot/skills/asnumpy-custom-op-integration/agents/openai.yaml",
    "share/numpilot/skills/asnumpy-custom-op-integration/references/legacy-case-lessons.md",
    "share/numpilot/skills/asnumpy-custom-op-integration/references/manifest-workflow.md",
    "share/numpilot/skills/asnumpy-custom-op-integration/references/official-aclnn-existing-module.md",
    "share/numpilot/skills/asnumpy-custom-op-integration/references/qualification-and-debug.md",
)

SOURCE_SKILL_MEMBERS = (
    ".codeartsdoer/skills/ProjectSkillStatus.txt",
    *tuple(
        member.replace("share/numpilot/codearts/", ".codeartsdoer/").replace(
            "share/numpilot/skills/", ".codeartsdoer/skills/"
        ).replace("share/numpilot/skill/", "skill/")
        for member in INSTALLED_SKILL_MEMBERS
        if not member.endswith("ProjectSkillStatus.txt")
    ),
)


def _wheel(
    path: Path,
    *,
    extra: tuple[str, bytes] | None = None,
    omit: str | None = None,
    prefix: str = "",
) -> None:
    dist_info = f"{NAME}-{VERSION}.dist-info"
    members = {
        **{member: b"" for member in PACKAGE_MEMBERS},
        f"{dist_info}/METADATA": METADATA,
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{dist_info}/RECORD": b"",
        f"{dist_info}/entry_points.txt": b"[console_scripts]\n",
        f"{dist_info}/licenses/LICENSE": b"Apache-2.0\n",
        f"{dist_info}/licenses/THIRD_PARTY_NOTICES.md": b"notices\n",
        **{
            f"{NAME}-{VERSION}.data/data/{member}": b"fixture\n"
            for member in INSTALLED_SKILL_MEMBERS
        },
    }
    if extra is not None:
        members[extra[0]] = extra[1]
    if omit is not None:
        members.pop(omit)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, payload in members.items():
            archive.writestr(prefix + member, payload)


def _sdist(
    path: Path,
    *,
    extra: tuple[str, bytes] | None = None,
    omit: str | None = None,
    prefix: str = "",
) -> None:
    root = f"{NAME}-{VERSION}"
    members = {
        f"{root}/PKG-INFO": METADATA,
        **{f"{root}/{member}": b"" for member in PACKAGE_MEMBERS},
        f"{root}/LICENSE": b"Apache-2.0\n",
        f"{root}/MANIFEST.in": b"recursive-include tools *.py\n",
        f"{root}/README.md": b"test\n",
        f"{root}/README.zh-CN.md": b"test zh\n",
        f"{root}/THIRD_PARTY_NOTICES.md": b"notices\n",
        f"{root}/pyproject.toml": b"[build-system]\n",
        f"{root}/run_demo.sh": b"#!/bin/sh\n",
        f"{root}/docs/installation.md": b"install\n",
        f"{root}/docs/installation.zh-CN.md": b"install zh\n",
        **{f"{root}/{member}": payload for member, payload in REVIEWED_PAYLOADS.items()},
        f"{root}/docs/assets/brand/README.md": b"brand inventory\n",
        f"{root}/examples/custom_operator_add_one.py": b"pass\n",
        f"{root}/examples/add_one_migration_demo.py": b"pass\n",
        f"{root}/examples/black_scholes_portfolio.py": b"pass\n",
        **{f"{root}/{member}": b"fixture\n" for member in SOURCE_SKILL_MEMBERS},
        f"{root}/tools/check_release_tree.py": b"# tool\n",
        f"{root}/tools/release_artifact_manifest.py": b"# tool\n",
    }
    if extra is not None:
        members[extra[0]] = extra[1]
    if omit is not None:
        members.pop(omit)
    with tarfile.open(path, "w:gz") as archive:
        for member, payload in members.items():
            info = tarfile.TarInfo(member)
            info.name = prefix + info.name
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))


def _clean_repo(path: Path) -> Path:
    repo = path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "{NAME}"\nversion = "{VERSION}"\n',
        encoding="utf-8",
    )
    for relative, payload in REVIEWED_PAYLOADS.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    subprocess.run(["git", "init", "-q", repo], check=True)
    subprocess.run(["git", "-C", repo, "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            repo,
            "-c",
            "user.name=NumPilot Test",
            "-c",
            "user.email=numpilot@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return repo


def test_generate_and_verify_release_manifest(tmp_path):
    repo = _clean_repo(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / f"{NAME}-{VERSION}-py3-none-any.whl"
    sdist = dist / f"{NAME}-{VERSION}.tar.gz"
    _wheel(wheel)
    _sdist(sdist)

    manifest = release_manifest.generate_manifest(
        [wheel, sdist], repo_root=repo, require_clean=True
    )

    assert manifest["status"] == "release_candidate_ready"
    assert manifest["release_candidate_eligible"] is True
    assert manifest["source"]["dirty"] is False
    assert [item["kind"] for item in manifest["artifacts"]] == ["wheel", "sdist"]
    assert all(item["regular_files"] for item in manifest["artifacts"])
    output = dist / "release-manifest.json"
    release_manifest._write_json(output, manifest, force=False)
    result = release_manifest.verify_manifest(output)
    assert result["status"] == "passed"
    assert result["artifacts_verified"] == 2


def test_require_clean_rejects_dirty_source(tmp_path):
    repo = _clean_repo(tmp_path)
    wheel = tmp_path / f"{NAME}-{VERSION}-py3-none-any.whl"
    _wheel(wheel)
    (repo / "untracked.txt").write_text("dirty", encoding="utf-8")

    development = release_manifest.generate_manifest([wheel], repo_root=repo)
    assert development["status"] == "development_only"
    assert development["release_candidate_eligible"] is False

    with pytest.raises(release_manifest.ReleaseManifestError, match="tree is dirty"):
        release_manifest.generate_manifest(
            [wheel], repo_root=repo, require_clean=True
        )


@pytest.mark.parametrize(
    "member",
    [
        f"{NAME}-{VERSION}/todo.md",
        f"{NAME}-{VERSION}/NumPilot20260615-v1.pdf",
        f"{NAME}-{VERSION}/TEMP-RECORDING-COMMANDS.zh-CN.md",
        f"{NAME}-{VERSION}/.numpilot-config/llm.json",
    ],
)
def test_sdist_rejects_files_that_must_not_ship(tmp_path, member):
    sdist = tmp_path / f"{NAME}-{VERSION}.tar.gz"
    _sdist(sdist, extra=(member, b"private"))

    with pytest.raises(release_manifest.ReleaseManifestError, match="forbidden|private"):
        release_manifest.inspect_artifact(
            sdist, expected_name=NAME, expected_version=VERSION
        )


@pytest.mark.parametrize("suffix", ["runtime.db", "profile.log", "output.npy", "result.npz"])
def test_sdist_rejects_intermediate_artifact_suffixes(tmp_path, suffix):
    sdist = tmp_path / f"{NAME}-{VERSION}.tar.gz"
    _sdist(sdist, extra=(f"{NAME}-{VERSION}/evidence/{suffix}", b"noise"))

    with pytest.raises(release_manifest.ReleaseManifestError, match="intermediate"):
        release_manifest.inspect_artifact(
            sdist, expected_name=NAME, expected_version=VERSION
        )


def test_archive_path_traversal_is_rejected(tmp_path):
    wheel = tmp_path / f"{NAME}-{VERSION}-py3-none-any.whl"
    _wheel(wheel, extra=("../escape", b"unsafe"))

    with pytest.raises(release_manifest.ReleaseManifestError, match="unsafe path"):
        release_manifest.inspect_artifact(
            wheel, expected_name=NAME, expected_version=VERSION
        )


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_artifact_rejects_extra_noncanonical_root_prefix(tmp_path, kind):
    if kind == "wheel":
        artifact = tmp_path / f"{NAME}-{VERSION}-py3-none-any.whl"
        _wheel(artifact, prefix="junk/")
    else:
        artifact = tmp_path / f"{NAME}-{VERSION}.tar.gz"
        _sdist(artifact, prefix="junk/")

    with pytest.raises(
        release_manifest.ReleaseManifestError,
        match="top-level dist-info|sdist root mismatch",
    ):
        release_manifest.inspect_artifact(
            artifact, expected_name=NAME, expected_version=VERSION
        )


def test_verify_detects_changed_artifact(tmp_path):
    repo = _clean_repo(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / f"{NAME}-{VERSION}-py3-none-any.whl"
    _wheel(wheel)
    manifest = release_manifest.generate_manifest([wheel], repo_root=repo)
    output = dist / "release-manifest.json"
    release_manifest._write_json(output, manifest, force=False)
    with wheel.open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(release_manifest.ReleaseManifestError, match="no longer matches"):
        release_manifest.verify_manifest(output)


@pytest.mark.parametrize(
    ("relative, payload, message"),
    [
        (
            "docs/assets/architecture/numpilot-architecture-en.svg",
            b'<svg xmlns="http://www.w3.org/2000/svg"/>',
            "capability boundary",
        ),
        ("docs/assets/brand/numpilot-wordmark.png", b"not a png", "PNG"),
    ],
)
def test_sdist_rejects_invalid_reviewed_assets(tmp_path, relative, payload, message):
    sdist = tmp_path / f"{NAME}-{VERSION}.tar.gz"
    _sdist(sdist, extra=(f"{NAME}-{VERSION}/{relative}", payload))

    with pytest.raises(release_manifest.ReleaseManifestError, match=message):
        release_manifest.inspect_artifact(
            sdist, expected_name=NAME, expected_version=VERSION
        )


@pytest.mark.parametrize(
    ("kind, missing"),
    [
        ("wheel", f"{NAME}/_script_verification_worker.py"),
        (
            "wheel",
            f"{NAME}-{VERSION}.data/data/share/numpilot/codearts/ProjectSkillStatus.txt",
        ),
        ("sdist", f"{NAME}-{VERSION}/examples/add_one_migration_demo.py"),
        (
            "sdist",
            f"{NAME}-{VERSION}/.codeartsdoer/skills/numpilot-migration-gate/SKILL.md",
        ),
        ("sdist", f"{NAME}-{VERSION}/skill/SKILL.md"),
        ("sdist", f"{NAME}-{VERSION}/examples/custom_operator_add_one.py"),
        ("sdist", f"{NAME}-{VERSION}/examples/black_scholes_portfolio.py"),
    ],
)
def test_artifact_rejects_missing_runtime_or_release_member(tmp_path, kind, missing):
    if kind == "wheel":
        artifact = tmp_path / f"{NAME}-{VERSION}-py3-none-any.whl"
        _wheel(artifact, omit=missing)
    else:
        artifact = tmp_path / f"{NAME}-{VERSION}.tar.gz"
        _sdist(artifact, omit=missing)

    with pytest.raises(release_manifest.ReleaseManifestError, match="missing required"):
        release_manifest.inspect_artifact(
            artifact, expected_name=NAME, expected_version=VERSION
        )


def test_manifest_binds_sdist_architecture_to_reviewed_source(tmp_path):
    repo = _clean_repo(tmp_path)
    sdist = tmp_path / f"{NAME}-{VERSION}.tar.gz"
    changed = VALID_SVG.replace(b"</svg>", b"<!-- changed --></svg>")
    _sdist(
        sdist,
        extra=(
            f"{NAME}-{VERSION}/docs/assets/architecture/numpilot-architecture-en.svg",
            changed,
        ),
    )

    with pytest.raises(release_manifest.ReleaseManifestError, match="reviewed source"):
        release_manifest.generate_manifest([sdist], repo_root=repo)


def test_release_candidate_requires_wheel_and_sdist_pair(tmp_path):
    repo = _clean_repo(tmp_path)
    wheel = tmp_path / f"{NAME}-{VERSION}-py3-none-any.whl"
    _wheel(wheel)

    development = release_manifest.generate_manifest([wheel], repo_root=repo)
    assert development["status"] == "development_only"
    assert development["release_candidate_eligible"] is False

    with pytest.raises(release_manifest.ReleaseManifestError, match="wheel and one sdist"):
        release_manifest.generate_manifest([wheel], repo_root=repo, require_clean=True)


def _copy_asnumpy_component(root: Path) -> Path:
    source = Path(__file__).resolve().parents[1] / "components" / "asnumpy"
    target = root / "components" / "asnumpy"
    target.parent.mkdir(parents=True)
    shutil.copytree(source, target)
    return target


def _component_artifact_records(root: Path, *, kind: str) -> list[dict[str, object]]:
    component = root / "components" / "asnumpy"
    records = []
    for path in sorted(component.iterdir()):
        if not path.is_file():
            continue
        if kind == "wheel":
            relative = (
                "numpilot-0.3.0a1.dev0.data/data/share/numpilot/components/asnumpy/"
                + path.name
            )
        else:
            relative = "numpilot-0.3.0a1.dev0/components/asnumpy/" + path.name
        records.append(
            {
                "path": relative,
                "sha256": release_manifest._sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def test_asnumpy_component_binding_is_hash_bound_for_each_release_archive(tmp_path):
    _copy_asnumpy_component(tmp_path)
    loaded = release_manifest._read_component_lock(tmp_path)
    assert loaded is not None
    _raw, identity = loaded
    artifacts = [
        {"kind": "wheel", "regular_files": _component_artifact_records(tmp_path, kind="wheel")},
        {"kind": "sdist", "regular_files": _component_artifact_records(tmp_path, kind="sdist")},
    ]

    binding = release_manifest._bind_asnumpy_component(artifacts, tmp_path, required=True)

    assert binding["wheel_sha256"] == identity["wheel_sha256"]
    assert set(binding["artifacts"]) == {"wheel", "sdist"}
    release_manifest._verify_asnumpy_component_binding(binding, artifacts)


def test_asnumpy_component_binding_required_mode_fails_without_lock(tmp_path):
    artifacts = [{"kind": "wheel", "regular_files": []}]

    with pytest.raises(release_manifest.ReleaseManifestError, match="requires components/asnumpy/manifest.json"):
        release_manifest._bind_asnumpy_component(artifacts, tmp_path, required=True)


def test_sdist_manifest_excludes_historical_internal_material():
    manifest = (Path(__file__).resolve().parents[1] / "MANIFEST.in").read_text(
        encoding="utf-8"
    )
    assert "exclude todo.md" in manifest.splitlines()
    assert "exclude NumPilot20260615-v1.pdf" in manifest.splitlines()
    assert "exclude TEMP-RECORDING-COMMANDS.zh-CN.md" in manifest.splitlines()
    assert "prune tools" in manifest.splitlines()
    assert "include tools/check_release_tree.py" in manifest.splitlines()
    assert "include tools/release_artifact_manifest.py" in manifest.splitlines()
    assert "recursive-include docs" in manifest
    assert "*.svg" in manifest
    assert "*.png" in manifest
