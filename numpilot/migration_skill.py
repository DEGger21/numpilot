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

"""Locate and verify NumPilot's built-in migration Skill release bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import site
import sysconfig
from typing import Dict, Optional, Tuple


MIGRATION_SKILL_FILES: Tuple[str, ...] = (
    "SKILL.md",
    "README.md",
    "agents/openai.yaml",
    "references/codearts-sidebar.md",
    "references/terminal-contract.md",
)
MIGRATION_SKILL_MIRROR_FILES: Tuple[str, ...] = tuple(
    item for item in MIGRATION_SKILL_FILES if item != "README.md"
)
_MAX_FILE_BYTES = 128 * 1024
_MAX_TOTAL_BYTES = 512 * 1024


class MigrationSkillError(RuntimeError):
    """The release-bundled migration Skill is absent, unsafe, or inconsistent."""


@dataclass(frozen=True)
class MigrationSkillBundle:
    root: Path
    files: Dict[str, str]
    files_sha256: Dict[str, str]
    bundle_sha256: str
    total_bytes: int
    mirror_root: Optional[Path]
    skill_name: str = "numpilot-migration-gate"

    @property
    def skill_path(self) -> Path:
        return self.root / "SKILL.md"

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source": "numpilot-release-bundle",
            "skill_name": self.skill_name,
            "root": str(self.root),
            "mirror_root": (
                str(self.mirror_root) if self.mirror_root is not None else None
            ),
            "bundle_sha256": self.bundle_sha256,
            "total_bytes": self.total_bytes,
            "files": dict(self.files_sha256),
        }


def _skill_root_candidates() -> Tuple[Path, ...]:
    package_parent = Path(__file__).resolve().parent.parent
    shared_relative = Path("share") / "numpilot" / "skill"
    return (
        # Source checkout.
        package_parent / "skill",
        # ``pip --target`` places wheel data beside the package directory.
        package_parent / shared_relative,
        # Virtualenv and system installs place wheel data below their data prefix.
        Path(sysconfig.get_path("data")) / shared_relative,
        # ``pip install --user`` uses the user data prefix.
        Path(site.getuserbase()) / shared_relative,
    )


def _default_skill_root() -> Path:
    candidates = _skill_root_candidates()
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _default_mirror_root(root: Path) -> Path:
    package_parent = Path(__file__).resolve().parent.parent
    if root == package_parent / "skill":
        return (
            package_parent
            / ".codeartsdoer"
            / "skills"
            / "numpilot-migration-gate"
        )
    return root.parent / "skills" / "numpilot-migration-gate"


def _read_regular_utf8(root: Path, relative: str) -> tuple[bytes, str]:
    target = root / relative
    if target.is_symlink() or not target.is_file():
        raise MigrationSkillError(
            "NumPilot migration Skill file is missing or unsafe: %s" % relative
        )
    try:
        payload = target.read_bytes()
    except OSError as exc:
        raise MigrationSkillError(
            "cannot read NumPilot migration Skill file: %s" % relative
        ) from exc
    if not payload or len(payload) > _MAX_FILE_BYTES:
        raise MigrationSkillError(
            "NumPilot migration Skill file is empty or too large: %s" % relative
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise MigrationSkillError(
            "NumPilot migration Skill file is not UTF-8: %s" % relative
        ) from exc
    if not text.strip() or "\x00" in text:
        raise MigrationSkillError(
            "NumPilot migration Skill content is invalid: %s" % relative
        )
    return payload, text


def load_migration_skill(
    root: Optional[Path] = None,
) -> MigrationSkillBundle:
    """Load and hash the complete Skill, checking its built-in mirror by default.

    An explicit ``root`` is useful for a reviewed external bundle.  The default
    release bundle additionally has to match the CodeArts copy shipped beside
    it, both in a source checkout and after wheel installation.
    """

    built_in = root is None
    skill_root = _default_skill_root() if root is None else Path(root).absolute()
    if skill_root.is_symlink() or not skill_root.is_dir():
        raise MigrationSkillError(
            "NumPilot migration Skill bundle is missing: %s" % skill_root
        )

    files: Dict[str, str] = {}
    hashes: Dict[str, str] = {}
    payloads: Dict[str, bytes] = {}
    total_bytes = 0
    for relative in MIGRATION_SKILL_FILES:
        payload, text = _read_regular_utf8(skill_root, relative)
        total_bytes += len(payload)
        if total_bytes > _MAX_TOTAL_BYTES:
            raise MigrationSkillError("NumPilot migration Skill bundle is too large")
        payloads[relative] = payload
        files[relative] = text
        hashes[relative] = hashlib.sha256(payload).hexdigest()

    skill_text = files["SKILL.md"]
    if not skill_text.startswith("---\n") or "\n---\n" not in skill_text[4:]:
        raise MigrationSkillError("NumPilot migration Skill frontmatter is invalid")
    name = re.search(r"(?m)^name:\s*([^\s].*)$", skill_text)
    if name is None or name.group(1).strip(" '\"") != "numpilot-migration-gate":
        raise MigrationSkillError("NumPilot migration Skill name is invalid")

    mirror_root: Optional[Path] = None
    if built_in:
        mirror_root = _default_mirror_root(skill_root)
        if mirror_root.is_symlink() or not mirror_root.is_dir():
            raise MigrationSkillError(
                "NumPilot migration Skill CodeArts mirror is missing: %s"
                % mirror_root
            )
        for relative in MIGRATION_SKILL_MIRROR_FILES:
            mirror_payload, _ = _read_regular_utf8(mirror_root, relative)
            if mirror_payload != payloads[relative]:
                raise MigrationSkillError(
                    "NumPilot migration Skill CodeArts mirror differs: %s" % relative
                )

    identity = json.dumps(
        {"skill_name": "numpilot-migration-gate", "files": hashes},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return MigrationSkillBundle(
        root=skill_root,
        files=files,
        files_sha256=hashes,
        bundle_sha256=hashlib.sha256(identity).hexdigest(),
        total_bytes=total_bytes,
        mirror_root=mirror_root,
    )
