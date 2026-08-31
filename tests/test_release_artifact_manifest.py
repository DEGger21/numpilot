import base64
import io
import json
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


def _pptx(planned_label: str) -> bytes:
    output = io.BytesIO()
    slide = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>ADAPTER_CANDIDATE CANDIDATE_READY ops-nn CANNBot {planned_label}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
</p:sld>
""".encode()
    members = {
        "[Content_Types].xml": (
            b'<?xml version="1.0"?><Types '
            b'xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>'
        ),
        "ppt/presentation.xml": (
            b'<?xml version="1.0"?><p:presentation '
            b'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>'
        ),
        "ppt/slides/slide1.xml": slide,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, payload in members.items():
            archive.writestr(member, payload)
    return output.getvalue()


REVIEWED_PAYLOADS = {
    "docs/numpilot-en.pptx": _pptx("planned"),
    "docs/numpilot-zh.pptx": _pptx("规划"),
    "docs/numpilot-en-v6.svg": VALID_SVG,
    "docs/numpilot-zh-v6.svg": VALID_SVG,
    "docs/assets/architecture/numpilot-architecture-en.svg": VALID_SVG,
    "docs/assets/architecture/numpilot-architecture-zh.svg": VALID_SVG,
    "docs/assets/brand/asnumpy-mark.png": VALID_PNG,
    "docs/assets/brand/ascend-wordmark.png": VALID_PNG,
    "docs/assets/brand/cann-wordmark.png": VALID_PNG,
    "docs/assets/brand/codearts-icon.png": VALID_PNG,
    "docs/assets/brand/numpilot-wordmark.png": VALID_PNG,
}

PACKAGE_MEMBERS = (
    f"{NAME}/__init__.py",
    f"{NAME}/__main__.py",
    f"{NAME}/_script_benchmark_worker.py",
    f"{NAME}/_script_verification_worker.py",
    f"{NAME}/script_benchmark.py",
    f"{NAME}/script_migration.py",
    f"{NAME}/script_verification.py",
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
        )
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
        f"{root}/docs/FRESH-NOTEBOOK-QUICKSTART.md": b"notebook guide\n",
        f"{root}/docs/NumPilot-Demo-v1-Quickstart.ipynb": b"{}\n",
        f"{root}/examples/competition_power_log_exp.py": b"pass\n",
        f"{root}/evidence/MANIFEST.sha256": b"",
        f"{root}/evidence/competition-ordinary-20260828/qualified-analysis.json": b"{}\n",
        f"{root}/evidence/competition-ordinary-20260828/script-benchmark-report.json": b"{}\n",
        f"{root}/evidence/competition-ordinary-20260828/script-verification-report.json": b"{}\n",
        f"{root}/evidence/competition-poisson-stage-20260828/qualified-analysis.json": b"{}\n",
        f"{root}/evidence/competition-poisson-stage-20260828/script-benchmark-report.json": b"{}\n",
        f"{root}/evidence/competition-poisson-stage-20260828/script-verification-report.json": b"{}\n",
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
        ("docs/numpilot-en.pptx", b"not a presentation", "invalid PPTX"),
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
        ("sdist", f"{NAME}-{VERSION}/examples/competition_power_log_exp.py"),
        (
            "sdist",
            f"{NAME}-{VERSION}/.codeartsdoer/skills/numpilot-migration-gate/SKILL.md",
        ),
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


def test_sdist_manifest_excludes_historical_internal_material():
    manifest = (Path(__file__).resolve().parents[1] / "MANIFEST.in").read_text(
        encoding="utf-8"
    )
    assert "exclude todo.md" in manifest.splitlines()
    assert "exclude NumPilot20260615-v1.pdf" in manifest.splitlines()
    assert "prune tools" in manifest.splitlines()
    assert "include tools/check_release_tree.py" in manifest.splitlines()
    assert "include tools/release_artifact_manifest.py" in manifest.splitlines()
    assert "recursive-include docs" in manifest
    assert "*.svg" in manifest
    assert "*.png" in manifest
