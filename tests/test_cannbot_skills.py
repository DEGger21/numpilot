import hashlib
from pathlib import Path
import subprocess

import pytest

from numpilot.cannbot_skills import CannbotSkillError, load_skill_bundle


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
