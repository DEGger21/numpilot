import hashlib
from pathlib import Path
import subprocess

import pytest

from numpilot.cannbot_skills import (
    CANNBOT_COMMIT,
    CANNBOT_TREE,
    CannbotSkillError,
    _BUNDLE_FILES,
    default_static_skill_root,
    load_skill_bundle,
    load_static_skill_bundle,
    resolve_skill_bundle,
)


def test_official_cannbot_pin_and_bundle_shape_are_stable():
    assert CANNBOT_COMMIT == "39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c"
    assert CANNBOT_TREE == "3a95d925b085dfb401233f26f44fa9fab3d31999"
    assert len(_BUNDLE_FILES) == 36
    assert len(set(_BUNDLE_FILES)) == 36


def test_author_owned_static_bundle_is_verified_without_git():
    bundle = load_static_skill_bundle(default_static_skill_root())
    assert bundle.commit == CANNBOT_COMMIT
    assert bundle.bundle_sha256 == "da0ee3d6a12561da0d630125d7f4ae8602848dbd4c6f1b3f7150b26a57ead79f"
    assert bundle.total_bytes == 302003
    assert len(bundle.files) == 36


def test_auto_resolution_is_static_and_fail_closed(monkeypatch):
    bundle = resolve_skill_bundle("auto", allow_fetch=True)
    assert bundle.commit == CANNBOT_COMMIT
    monkeypatch.setattr("numpilot.cannbot_skills.default_static_skill_root", lambda: Path("/missing/static"))
    with pytest.raises(CannbotSkillError, match="static CANNBot skill bundle is missing"):
        resolve_skill_bundle("auto", allow_fetch=True)


def _run(*args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def _skill_repo(tmp_path):
    root = tmp_path / "cannbot-skills"
    skill = root / "ops" / "ascendc-registry-invoke-template"
    skill.mkdir(parents=True)
    (root / "LICENSE").write_text("official license\n", encoding="utf-8")
    (skill / "SKILL.md").write_text(
        "---\nname: ascendc-registry-invoke-template\n---\n"
        "Use inputs, outputs, workspace, tiling.\n",
        encoding="utf-8",
    )
    _run("git", "init", "-q", cwd=root)
    _run(
        "git", "remote", "add", "origin",
        "https://gitcode.com/cann/cannbot-skills.git",
        cwd=root,
    )
    _run("git", "add", ".", cwd=root)
    _run(
        "git", "-c", "user.name=NumPilot Test", "-c",
        "user.email=numpilot@example.invalid", "commit", "-qm", "fixture",
        cwd=root,
    )
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return root, commit


def test_verified_skill_content_is_embedded_and_path_is_not_in_manifest(tmp_path):
    root, commit = _skill_repo(tmp_path)
    files = (
        "LICENSE",
        "ops/ascendc-registry-invoke-template/SKILL.md",
    )

    bundle = load_skill_bundle(root, expected_commit=commit, bundle_files=files)
    manifest = bundle.manifest()
    prompt = bundle.prompt_context()

    assert manifest["commit"] == commit
    assert manifest["skill_name"] == "ascendc-registry-invoke-template"
    assert str(root) not in str(manifest)
    assert "Use inputs, outputs, workspace, tiling." in prompt
    assert bundle.bundle_sha256 in prompt
    for relative in files:
        expected = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert manifest["files"][relative] == expected


def test_modified_skill_file_is_rejected(tmp_path):
    root, commit = _skill_repo(tmp_path)
    (root / "ops/ascendc-registry-invoke-template/SKILL.md").write_text(
        "locally modified\n", encoding="utf-8"
    )

    with pytest.raises(CannbotSkillError, match="local modifications"):
        load_skill_bundle(
            root,
            expected_commit=commit,
            bundle_files=(
                "ops/ascendc-registry-invoke-template/SKILL.md",
            ),
        )


def test_modified_selected_file_outside_primary_skill_prefix_is_rejected(
    tmp_path,
):
    root, _commit = _skill_repo(tmp_path)
    relative = "plugins-official/ops-registry-invoke/workflow/SKILL.md"
    workflow = root / relative
    workflow.parent.mkdir(parents=True)
    workflow.write_text("official workflow\n", encoding="utf-8")
    _run("git", "add", relative, cwd=root)
    _run(
        "git",
        "-c",
        "user.name=NumPilot Test",
        "-c",
        "user.email=numpilot@example.invalid",
        "commit",
        "-qm",
        "add workflow fixture",
        cwd=root,
    )
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    workflow.write_text("locally modified workflow\n", encoding="utf-8")

    with pytest.raises(CannbotSkillError, match="local modifications"):
        load_skill_bundle(
            root,
            expected_commit=commit,
            bundle_files=(relative,),
        )


def test_unpinned_commit_is_rejected(tmp_path):
    root, _commit = _skill_repo(tmp_path)

    with pytest.raises(CannbotSkillError, match="commit mismatch"):
        load_skill_bundle(
            root,
            expected_commit="0" * 40,
            bundle_files=(
                "ops/ascendc-registry-invoke-template/SKILL.md",
            ),
        )


def test_setup_skills_requires_explicit_license_acceptance_before_fetch(
    monkeypatch, capsys
):
    from numpilot import __main__ as cli

    def forbidden(*_args, **_kwargs):
        raise AssertionError("license gate must run before CANNBot fetch")

    monkeypatch.setattr(cli, "resolve_skill_bundle", forbidden)

    assert cli.main(["setup-skills"]) == 2
    captured = capsys.readouterr()
    assert "--accept-cannbot-license" in captured.err
