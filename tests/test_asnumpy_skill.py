import hashlib
import shutil

import pytest

import numpilot.asnumpy_skill as asnumpy_skill
from numpilot.asnumpy_skill import (
    ASNUMPY_INTEGRATION_SKILL_FILES,
    ASNUMPY_OFFICIAL_ADAPTER_SKILL_FILES,
    AsnumpyIntegrationSkillError,
    load_asnumpy_integration_skill,
)


def test_builtin_asnumpy_skill_is_complete_hash_bound_and_prompt_embedded():
    bundle = load_asnumpy_integration_skill()

    assert tuple(bundle.files) == ASNUMPY_INTEGRATION_SKILL_FILES
    assert bundle.total_bytes > 0
    assert len(bundle.bundle_sha256) == 64
    assert bundle.manifest()["source"] == "numpilot-repository"
    rendered = bundle.prompt_context()
    assert bundle.bundle_sha256 in rendered
    for relative in ASNUMPY_INTEGRATION_SKILL_FILES:
        assert bundle.files[relative] in rendered
        assert bundle.files_sha256[relative] == hashlib.sha256(
            bundle.files[relative].encode("utf-8")
        ).hexdigest()


def test_official_adapter_prompt_uses_only_route_relevant_skill_files():
    bundle = load_asnumpy_integration_skill()

    rendered = bundle.prompt_context(ASNUMPY_OFFICIAL_ADAPTER_SKILL_FILES)

    assert "official-aclnn-existing-module" in rendered
    assert "qualification-and-debug" in rendered
    assert (
        "BEGIN ASNUMPY SKILL FILE `references/manifest-workflow.md`"
        not in rendered
    )
    assert (
        "BEGIN ASNUMPY SKILL FILE `references/legacy-case-lessons.md`"
        not in rendered
    )
    assert set(ASNUMPY_OFFICIAL_ADAPTER_SKILL_FILES) < set(
        ASNUMPY_INTEGRATION_SKILL_FILES
    )


def test_asnumpy_skill_loader_fails_closed_when_a_required_file_is_missing(
    tmp_path,
):
    source = load_asnumpy_integration_skill().root
    copied = tmp_path / "asnumpy-skill"
    shutil.copytree(source, copied)
    (copied / "references/qualification-and-debug.md").unlink()

    with pytest.raises(AsnumpyIntegrationSkillError, match="missing or unsafe"):
        load_asnumpy_integration_skill(copied)


def test_asnumpy_skill_loader_rejects_symlink_root(tmp_path):
    link = tmp_path / "skill-link"
    link.symlink_to(load_asnumpy_integration_skill().root, target_is_directory=True)

    with pytest.raises(AsnumpyIntegrationSkillError, match="missing"):
        load_asnumpy_integration_skill(link)


def test_default_skill_root_supports_installed_wheel_data(tmp_path, monkeypatch):
    package_file = tmp_path / "target" / "numpilot" / "asnumpy_skill.py"
    package_file.parent.mkdir(parents=True)
    installed = (
        tmp_path
        / "target"
        / "share"
        / "numpilot"
        / "skills"
        / "asnumpy-custom-op-integration"
    )
    shutil.copytree(load_asnumpy_integration_skill().root, installed)
    monkeypatch.setattr(asnumpy_skill, "__file__", str(package_file))
    monkeypatch.setattr(
        asnumpy_skill.sysconfig,
        "get_path",
        lambda _name: str(tmp_path / "unused-prefix"),
    )

    assert asnumpy_skill._default_skill_root() == installed
    assert asnumpy_skill.load_asnumpy_integration_skill().root == installed


def test_default_skill_root_supports_user_installed_wheel_data(
    tmp_path, monkeypatch
):
    user_base = tmp_path / "user-base"
    package_file = (
        user_base
        / "lib"
        / "python3.10"
        / "site-packages"
        / "numpilot"
        / "asnumpy_skill.py"
    )
    package_file.parent.mkdir(parents=True)
    installed = (
        user_base
        / "share"
        / "numpilot"
        / "skills"
        / "asnumpy-custom-op-integration"
    )
    shutil.copytree(load_asnumpy_integration_skill().root, installed)
    monkeypatch.setattr(asnumpy_skill, "__file__", str(package_file))
    monkeypatch.setattr(
        asnumpy_skill.sysconfig,
        "get_path",
        lambda _name: str(tmp_path / "unused-prefix"),
    )
    monkeypatch.setattr(
        asnumpy_skill.site,
        "getuserbase",
        lambda: str(user_base),
    )

    assert asnumpy_skill._default_skill_root() == installed
    assert asnumpy_skill.load_asnumpy_integration_skill().root == installed


def test_default_skill_root_prefers_venv_data_over_user_data(
    tmp_path, monkeypatch
):
    source = load_asnumpy_integration_skill().root
    package_file = (
        tmp_path / "venv" / "lib" / "python3.10" / "site-packages"
        / "numpilot" / "asnumpy_skill.py"
    )
    package_file.parent.mkdir(parents=True)
    venv_data = (
        tmp_path
        / "venv"
        / "share"
        / "numpilot"
        / "skills"
        / "asnumpy-custom-op-integration"
    )
    user_base = tmp_path / "user-base"
    user_data = (
        user_base
        / "share"
        / "numpilot"
        / "skills"
        / "asnumpy-custom-op-integration"
    )
    shutil.copytree(source, venv_data)
    shutil.copytree(source, user_data)
    monkeypatch.setattr(asnumpy_skill, "__file__", str(package_file))
    monkeypatch.setattr(
        asnumpy_skill.sysconfig,
        "get_path",
        lambda _name: str(tmp_path / "venv"),
    )
    monkeypatch.setattr(
        asnumpy_skill.site,
        "getuserbase",
        lambda: str(user_base),
    )

    assert asnumpy_skill._default_skill_root() == venv_data


def test_default_skill_root_fails_closed_when_no_candidate_exists(
    tmp_path, monkeypatch
):
    package_file = tmp_path / "empty" / "numpilot" / "asnumpy_skill.py"
    package_file.parent.mkdir(parents=True)
    monkeypatch.setattr(asnumpy_skill, "__file__", str(package_file))
    monkeypatch.setattr(
        asnumpy_skill.sysconfig,
        "get_path",
        lambda _name: str(tmp_path / "unused-prefix"),
    )
    monkeypatch.setattr(
        asnumpy_skill.site,
        "getuserbase",
        lambda: str(tmp_path / "unused-user-base"),
    )

    expected_source_root = (
        package_file.parent.parent
        / ".codeartsdoer"
        / "skills"
        / "asnumpy-custom-op-integration"
    )
    assert asnumpy_skill._default_skill_root() == expected_source_root
    with pytest.raises(AsnumpyIntegrationSkillError, match="missing"):
        asnumpy_skill.load_asnumpy_integration_skill()
