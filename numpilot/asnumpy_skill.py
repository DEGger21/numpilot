"""Load NumPilot's repository-owned AsNumpy integration skill safely."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import site
import sysconfig
from typing import Dict, Optional, Tuple


ASNUMPY_INTEGRATION_SKILL_FILES: Tuple[str, ...] = (
    "SKILL.md",
    "references/manifest-workflow.md",
    "references/legacy-case-lessons.md",
    "references/qualification-and-debug.md",
)
_MAX_FILE_BYTES = 128 * 1024
_MAX_TOTAL_BYTES = 512 * 1024


class AsnumpyIntegrationSkillError(RuntimeError):
    """The repository-owned AsNumpy integration skill is absent or invalid."""


@dataclass(frozen=True)
class AsnumpyIntegrationSkillBundle:
    root: Path
    files: Dict[str, str]
    files_sha256: Dict[str, str]
    bundle_sha256: str
    total_bytes: int
    skill_name: str = "asnumpy-custom-op-integration"

    def manifest(self):
        return {
            "schema_version": 1,
            "source": "numpilot-repository",
            "skill_name": self.skill_name,
            "root": str(self.root),
            "bundle_sha256": self.bundle_sha256,
            "total_bytes": self.total_bytes,
            "files": dict(self.files_sha256),
        }

    def prompt_context(self, selected_files=None):
        selected = tuple(selected_files or ASNUMPY_INTEGRATION_SKILL_FILES)
        if not selected or len(set(selected)) != len(selected):
            raise AsnumpyIntegrationSkillError(
                "AsNumpy integration skill selection is invalid"
            )
        sections = [
            "## Embedded NumPilot AsNumpy integration Skill",
            "skill_name: %s" % self.skill_name,
            "bundle_sha256: %s" % self.bundle_sha256,
        ]
        for relative in selected:
            content = self.files.get(relative)
            if content is None:
                raise AsnumpyIntegrationSkillError(
                    "AsNumpy integration skill file was not loaded: %s" % relative
                )
            sections.extend((
                "BEGIN ASNUMPY SKILL FILE `%s`" % relative,
                content,
                "END ASNUMPY SKILL FILE `%s`" % relative,
            ))
        return "\n\n".join(sections)


def _skill_root_candidates() -> Tuple[Path, ...]:
    package_parent = Path(__file__).resolve().parent.parent
    shared_relative = (
        Path("share")
        / "numpilot"
        / "skills"
        / "asnumpy-custom-op-integration"
    )
    return (
        package_parent
        / ".codeartsdoer"
        / "skills"
        / "asnumpy-custom-op-integration",
        # ``pip --target`` installs wheel data beside the package directory.
        package_parent / shared_relative,
        # A normal virtualenv/system install places data below sys.prefix.
        Path(sysconfig.get_path("data")) / shared_relative,
        # ``pip install --user`` places wheel data below site.USER_BASE.
        Path(site.getuserbase()) / shared_relative,
    )


def _default_skill_root() -> Path:
    candidates = _skill_root_candidates()
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    # Keep the source-layout path in the error because it is the most useful
    # remediation when running directly from a checkout.
    return candidates[0]


def load_asnumpy_integration_skill(
    root: Optional[Path] = None,
) -> AsnumpyIntegrationSkillBundle:
    """Read and hash the exact built-in integration skill files."""

    skill_root = Path(root).absolute() if root is not None else _default_skill_root()
    if skill_root.is_symlink() or not skill_root.is_dir():
        raise AsnumpyIntegrationSkillError(
            "NumPilot AsNumpy integration skill is missing: %s" % skill_root
        )
    files = {}
    hashes = {}
    total_bytes = 0
    for relative in ASNUMPY_INTEGRATION_SKILL_FILES:
        target = skill_root / relative
        if target.is_symlink() or not target.is_file():
            raise AsnumpyIntegrationSkillError(
                "AsNumpy integration skill file is missing or unsafe: %s" % relative
            )
        try:
            payload = target.read_bytes()
        except OSError as exc:
            raise AsnumpyIntegrationSkillError(
                "cannot read AsNumpy integration skill file: %s" % relative
            ) from exc
        if not payload or len(payload) > _MAX_FILE_BYTES:
            raise AsnumpyIntegrationSkillError(
                "AsNumpy integration skill file is empty or too large: %s" % relative
            )
        total_bytes += len(payload)
        if total_bytes > _MAX_TOTAL_BYTES:
            raise AsnumpyIntegrationSkillError(
                "AsNumpy integration skill bundle is too large"
            )
        try:
            content = payload.decode("utf-8")
        except UnicodeError as exc:
            raise AsnumpyIntegrationSkillError(
                "AsNumpy integration skill is not UTF-8: %s" % relative
            ) from exc
        if not content.strip() or "\x00" in content:
            raise AsnumpyIntegrationSkillError(
                "AsNumpy integration skill content is invalid: %s" % relative
            )
        files[relative] = content
        hashes[relative] = hashlib.sha256(payload).hexdigest()
    identity = json.dumps(
        {"skill_name": "asnumpy-custom-op-integration", "files": hashes},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return AsnumpyIntegrationSkillBundle(
        root=skill_root,
        files=files,
        files_sha256=hashes,
        bundle_sha256=hashlib.sha256(identity).hexdigest(),
        total_bytes=total_bytes,
    )
