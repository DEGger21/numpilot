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

"""Source-checkout and installed-wheel layout checks for the migration Skill."""

import hashlib
import shutil

import pytest

import numpilot.migration_skill as migration_skill
from numpilot.migration_skill import (
    MIGRATION_SKILL_FILES,
    MIGRATION_SKILL_MIRROR_FILES,
    MigrationSkillError,
    load_migration_skill,
)


def test_builtin_migration_skill_is_complete_hash_bound_and_mirrored():
    bundle = load_migration_skill()

    assert tuple(bundle.files) == MIGRATION_SKILL_FILES
    assert bundle.mirror_root is not None
    assert bundle.total_bytes > 0
    assert len(bundle.bundle_sha256) == 64
    assert bundle.manifest()["source"] == "numpilot-release-bundle"
    for relative in MIGRATION_SKILL_FILES:
        assert bundle.files_sha256[relative] == hashlib.sha256(
            bundle.files[relative].encode("utf-8")
        ).hexdigest()
    for relative in MIGRATION_SKILL_MIRROR_FILES:
        assert (bundle.root / relative).read_bytes() == (
            bundle.mirror_root / relative
        ).read_bytes()


def _installed_layout(tmp_path, monkeypatch):
    source = load_migration_skill()
    target = tmp_path / "target"
    package_file = target / "numpilot" / "migration_skill.py"
    package_file.parent.mkdir(parents=True)
    primary = target / "share" / "numpilot" / "skill"
    mirror = (
        target
        / "share"
        / "numpilot"
        / "skills"
        / "numpilot-migration-gate"
    )
    shutil.copytree(source.root, primary)
    assert source.mirror_root is not None
    shutil.copytree(source.mirror_root, mirror)
    monkeypatch.setattr(migration_skill, "__file__", str(package_file))
    monkeypatch.setattr(
        migration_skill.sysconfig,
        "get_path",
        lambda _name: str(tmp_path / "unused-prefix"),
    )
    monkeypatch.setattr(
        migration_skill.site,
        "getuserbase",
        lambda: str(tmp_path / "unused-user-base"),
    )
    return primary, mirror


def test_default_migration_skill_supports_installed_wheel_data(
    tmp_path, monkeypatch
):
    primary, mirror = _installed_layout(tmp_path, monkeypatch)

    bundle = migration_skill.load_migration_skill()

    assert bundle.root == primary
    assert bundle.mirror_root == mirror
    assert bundle.skill_path == primary / "SKILL.md"


def test_installed_migration_skill_fails_closed_on_mirror_drift(
    tmp_path, monkeypatch
):
    _primary, mirror = _installed_layout(tmp_path, monkeypatch)
    (mirror / "SKILL.md").write_text("changed\n", encoding="utf-8")

    with pytest.raises(MigrationSkillError, match="mirror differs"):
        migration_skill.load_migration_skill()


def test_explicit_migration_skill_requires_complete_bundle(tmp_path):
    root = tmp_path / "incomplete"
    root.mkdir()
    (root / "SKILL.md").write_text(
        "---\nname: numpilot-migration-gate\n---\n", encoding="utf-8"
    )

    with pytest.raises(MigrationSkillError, match="missing or unsafe"):
        load_migration_skill(root)
