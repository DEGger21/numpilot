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


def _wheel(path: Path, *, extra: tuple[str, bytes] | None = None) -> None:
    dist_info = f"{NAME}-{VERSION}.dist-info"
    members = {
        f"{NAME}/__init__.py": b"",
        f"{NAME}/__main__.py": b"",
        f"{dist_info}/METADATA": METADATA,
        f"{dist_info}/RECORD": b"",
        f"{dist_info}/entry_points.txt": b"[console_scripts]\n",
        f"{dist_info}/licenses/LICENSE": b"Apache-2.0\n",
        f"{dist_info}/licenses/THIRD_PARTY_NOTICES.md": b"notices\n",
    }
    if extra is not None:
        members[extra[0]] = extra[1]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, payload in members.items():
            archive.writestr(member, payload)


def _sdist(path: Path, *, extra: tuple[str, bytes] | None = None) -> None:
    root = f"{NAME}-{VERSION}"
    members = {
        f"{root}/PKG-INFO": METADATA,
        f"{root}/{NAME}/__init__.py": b"",
        f"{root}/{NAME}/__main__.py": b"",
        f"{root}/LICENSE": b"Apache-2.0\n",
        f"{root}/MANIFEST.in": b"recursive-include tools *.py\n",
        f"{root}/README.md": b"test\n",
        f"{root}/THIRD_PARTY_NOTICES.md": b"notices\n",
        f"{root}/pyproject.toml": b"[build-system]\n",
        f"{root}/run_demo.sh": b"#!/bin/sh\n",
        f"{root}/docs/numpilot-architecture-paper-style-zh-readable.pptx": b"pptx\n",
        f"{root}/evidence/MANIFEST.sha256": b"",
        f"{root}/tools/check_release_tree.py": b"# tool\n",
        f"{root}/tools/release_artifact_manifest.py": b"# tool\n",
    }
    if extra is not None:
        members[extra[0]] = extra[1]
    with tarfile.open(path, "w:gz") as archive:
        for member, payload in members.items():
            info = tarfile.TarInfo(member)
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
    subprocess.run(["git", "init", "-q", repo], check=True)
    subprocess.run(["git", "-C", repo, "add", "pyproject.toml"], check=True)
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


def test_sdist_manifest_excludes_historical_internal_material():
    manifest = (Path(__file__).resolve().parents[1] / "MANIFEST.in").read_text(
        encoding="utf-8"
    )
    assert "exclude todo.md" in manifest.splitlines()
    assert "exclude NumPilot20260615-v1.pdf" in manifest.splitlines()
    assert "prune tools" in manifest.splitlines()
    assert "include tools/check_release_tree.py" in manifest.splitlines()
    assert "include tools/release_artifact_manifest.py" in manifest.splitlines()
