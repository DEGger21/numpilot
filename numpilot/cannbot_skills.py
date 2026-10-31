"""Pinned Huawei CANNBot skill acquisition and prompt bundling.

The remote LLM cannot read a path on the NumPilot host.  This module therefore
verifies a pinned official checkout and embeds a bounded, hashed subset of the
registry-invoke skill in each operator-generation request.  Network access is
never performed at import time; it is limited to an explicit setup or gap
completion command.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sysconfig
import tempfile
from typing import Dict, Mapping, Optional, Tuple


class CannbotSkillError(RuntimeError):
    """The official skill checkout is missing, modified, or cannot be fetched."""


CANNBOT_REPOSITORY = "https://gitcode.com/cann/cannbot-skills.git"
CANNBOT_COMMIT = "39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c"
CANNBOT_TREE = "3a95d925b085dfb401233f26f44fa9fab3d31999"
CANNBOT_SKILL_NAME = "ascendc-registry-invoke-template"
CANNBOT_LICENSE = "CANN Open Software License Agreement Version 2.0"
_SKILL_PREFIX = "ops/ascendc-registry-invoke-template"
_WORKFLOW_PREFIX = "plugins-official/ops-registry-invoke"
_API_PRACTICES_PREFIX = "ops/ascendc-api-best-practices"
_PRECISION_PREFIX = "ops/ascendc-precision-debug"
_MAX_BUNDLE_BYTES = 384 * 1024
_MAX_GIT_OUTPUT = 4096
_STATIC_BUNDLE_SCHEMA = "numpilot.cannbot-static-bundle"
_STATIC_BUNDLE_VERSION = 1
_STATIC_BUNDLE_MANIFEST = "manifest.json"
_STATIC_BUNDLE_CONTENTS = "bundle.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _static_skill_root_candidates() -> Tuple[Path, ...]:
    package_parent = Path(__file__).resolve().parent.parent
    relative = Path("components") / "cannbot-skills" / CANNBOT_COMMIT
    return (
        package_parent / relative,
        Path(sysconfig.get_path("data")) / "share" / "numpilot" / relative,
    )


def default_static_skill_root() -> Path:
    """Return the repository/package-owned static CANNBot snapshot."""

    candidates = _static_skill_root_candidates()
    for candidate in candidates:
        if candidate.is_dir() and not candidate.is_symlink():
            return candidate
    return candidates[0]


def _static_file(root: Path, relative: str) -> Path:
    target = root / relative
    if target.is_symlink() or not target.is_file():
        raise CannbotSkillError(
            "static CANNBot bundle file is missing or unsafe: %s" % relative
        )
    return target


def load_static_skill_bundle(root: Optional[Path] = None) -> "SkillBundle":
    """Load the author-owned static snapshot without Git or network access."""

    snapshot = Path(root).expanduser() if root is not None else default_static_skill_root()
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise CannbotSkillError(
            "static CANNBot skill root is not a real directory: %s" % snapshot
        )
    try:
        manifest = json.loads(
            _static_file(snapshot, _STATIC_BUNDLE_MANIFEST).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CannbotSkillError("static CANNBot skill manifest is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != _STATIC_BUNDLE_SCHEMA
        or manifest.get("schema_version") != _STATIC_BUNDLE_VERSION
        or manifest.get("repository") != CANNBOT_REPOSITORY
        or manifest.get("commit") != CANNBOT_COMMIT
        or manifest.get("tree") != CANNBOT_TREE
        or manifest.get("skill_name") != CANNBOT_SKILL_NAME
        or manifest.get("license") != CANNBOT_LICENSE
    ):
        raise CannbotSkillError("static CANNBot skill manifest identity is invalid")
    declared_files = manifest.get("files")
    if (
        not isinstance(declared_files, dict)
        or set(declared_files) != set(_BUNDLE_FILES)
        or any(
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or not _SHA256_RE.fullmatch(digest)
            for relative, digest in declared_files.items()
        )
    ):
        raise CannbotSkillError("static CANNBot skill file manifest is invalid")
    try:
        contents = json.loads(
            _static_file(snapshot, _STATIC_BUNDLE_CONTENTS).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CannbotSkillError("static CANNBot skill contents are invalid") from exc
    if not isinstance(contents, dict) or set(contents) != set(_BUNDLE_FILES):
        raise CannbotSkillError("static CANNBot skill contents do not match manifest")
    files: Dict[str, str] = {}
    hashes: Dict[str, str] = {}
    total = 0
    for relative in _BUNDLE_FILES:
        content = contents[relative]
        if not isinstance(content, str) or any(ord(character) == 0 for character in content):
            raise CannbotSkillError("static CANNBot skill content is invalid: %s" % relative)
        payload = content.encode("utf-8")
        digest = _sha256(payload)
        if digest != declared_files[relative]:
            raise CannbotSkillError(
                "static CANNBot skill hash mismatch: %s" % relative
            )
        files[relative] = content
        hashes[relative] = digest
        total += len(payload)
    identity = json.dumps(
        {
            "repository": CANNBOT_REPOSITORY,
            "commit": CANNBOT_COMMIT,
            "skill_name": CANNBOT_SKILL_NAME,
            "files": dict(sorted(hashes.items())),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    bundle_sha256 = _sha256(identity)
    if (
        not isinstance(manifest.get("bundle_sha256"), str)
        or not _SHA256_RE.fullmatch(manifest["bundle_sha256"])
        or total > _MAX_BUNDLE_BYTES
        or bundle_sha256 != manifest.get("bundle_sha256")
        or total != manifest.get("total_bytes")
    ):
        raise CannbotSkillError("static CANNBot skill bundle identity is invalid")
    return SkillBundle(
        root=str(snapshot.resolve()),
        repository=CANNBOT_REPOSITORY,
        commit=CANNBOT_COMMIT,
        skill_name=CANNBOT_SKILL_NAME,
        license=CANNBOT_LICENSE,
        files=files,
        file_sha256=hashes,
        bundle_sha256=bundle_sha256,
        total_bytes=total,
    )



# These are paths in the independently installed CANNBot repository, not
# copies of Skill content.  Generation follows the official registry-invoke
# workflow; compiler-driven repair additionally reads the official Ascend C
# API practices selected for the failing source file.
CANNBOT_GENERATION_SKILL_FILES: Tuple[str, ...] = (
    _SKILL_PREFIX + "/SKILL.md",
    _WORKFLOW_PREFIX + "/workflow/SKILL.md",
    _WORKFLOW_PREFIX + "/agents/ascendc-ops-developer.md",
    _WORKFLOW_PREFIX + "/agents/ascendc-ops-tester.md",
)
CANNBOT_REPAIR_SKILL_FILES: Tuple[str, ...] = (
    *CANNBOT_GENERATION_SKILL_FILES,
    _API_PRACTICES_PREFIX + "/SKILL.md",
    _API_PRACTICES_PREFIX + "/references/api-buffer.md",
    _API_PRACTICES_PREFIX + "/references/api-arithmetic.md",
    _API_PRACTICES_PREFIX + "/references/api-datacopy.md",
    _API_PRACTICES_PREFIX + "/references/api-pipeline.md",
    _API_PRACTICES_PREFIX + "/references/api-restrictions.md",
)
CANNBOT_PRECISION_SKILL_FILES: Tuple[str, ...] = (
    *CANNBOT_GENERATION_SKILL_FILES,
    "ops/ascendc-ut-develop/SKILL.md",
    "ops/ascendc-st-design/SKILL.md",
    _PRECISION_PREFIX + "/SKILL.md",
    _PRECISION_PREFIX + "/references/diagnosis-workflow.md",
    _PRECISION_PREFIX + "/references/common-traps.md",
    _PRECISION_PREFIX + "/references/best-practices.md",
)
CANNBOT_RUNTIME_SKILL_FILES: Tuple[str, ...] = (
    *CANNBOT_GENERATION_SKILL_FILES,
    "ops/ascendc-runtime-debug/SKILL.md",
)
CANNBOT_PERFORMANCE_SKILL_FILES: Tuple[str, ...] = (
    *CANNBOT_GENERATION_SKILL_FILES,
    "ops/ascendc-perf-optimize/SKILL.md",
)

# This is intentionally a focused bundle rather than the complete 116 MB
# repository.  A 131 KB first version containing every adjacent guide was
# rejected by the configured gateway with repeated HTTP 504 responses.  The
# files below retain the official end-to-end workflow, its developer/tester
# roles, the first-development/build route, the API repair route, and the
# concrete arch22 Host/Kernel/ACLNN implementation spine.
_BUNDLE_FILES: Tuple[str, ...] = (
    "LICENSE",
    _SKILL_PREFIX + "/SKILL.md",
    _SKILL_PREFIX + "/references/basic-guide.md",
    _SKILL_PREFIX + "/references/build-deploy-guide.md",
    _SKILL_PREFIX + "/references/add_example/CMakeLists.txt",
    _SKILL_PREFIX + "/references/add_example/build.sh",
    _SKILL_PREFIX + "/references/add_example/op_host/CMakeLists.txt",
    _SKILL_PREFIX + "/references/add_example/op_host/add_example_def.cpp",
    _SKILL_PREFIX + "/references/add_example/op_host/add_example_infershape.cpp",
    _SKILL_PREFIX + "/references/add_example/op_host/arch22/add_example_tiling.cpp",
    _SKILL_PREFIX + "/references/add_example/op_kernel/CMakeLists.txt",
    _SKILL_PREFIX + "/references/add_example/op_kernel/add_example_arch22.cpp",
    _SKILL_PREFIX + "/references/add_example/op_kernel/arch22/add_example.h",
    _SKILL_PREFIX + "/references/add_example/op_kernel/arch22/add_example_tiling_data.h",
    _SKILL_PREFIX + "/references/add_example/op_kernel/arch22/add_example_tiling_key.h",
    _SKILL_PREFIX + "/references/add_example/op_api/aclnn_add_example.cpp",
    _SKILL_PREFIX + "/references/add_example/op_api/aclnn_add_example.h",
    _SKILL_PREFIX + "/references/add_example/op_api/add_example.cpp",
    _SKILL_PREFIX + "/references/add_example/op_api/add_example.h",
    _WORKFLOW_PREFIX + "/workflow/SKILL.md",
    _WORKFLOW_PREFIX + "/agents/ascendc-ops-developer.md",
    _WORKFLOW_PREFIX + "/agents/ascendc-ops-tester.md",
    _API_PRACTICES_PREFIX + "/SKILL.md",
    _API_PRACTICES_PREFIX + "/references/api-buffer.md",
    _API_PRACTICES_PREFIX + "/references/api-arithmetic.md",
    _API_PRACTICES_PREFIX + "/references/api-datacopy.md",
    _API_PRACTICES_PREFIX + "/references/api-pipeline.md",
    _API_PRACTICES_PREFIX + "/references/api-restrictions.md",
    "ops/ascendc-ut-develop/SKILL.md",
    "ops/ascendc-st-design/SKILL.md",
    _PRECISION_PREFIX + "/SKILL.md",
    _PRECISION_PREFIX + "/references/diagnosis-workflow.md",
    _PRECISION_PREFIX + "/references/common-traps.md",
    _PRECISION_PREFIX + "/references/best-practices.md",
    "ops/ascendc-runtime-debug/SKILL.md",
    "ops/ascendc-perf-optimize/SKILL.md",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_repository(value: str) -> str:
    return value.strip().rstrip("/").removesuffix(".git")


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CannbotSkillError(
            "cannot execute git for the CANNBot skill: %s" % type(exc).__name__
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[:_MAX_GIT_OUTPUT]
        raise CannbotSkillError(
            "git %s failed (exit=%d): %s"
            % (arguments[0] if arguments else "command", completed.returncode, detail)
        )
    return completed.stdout.strip()


def default_skill_checkout(environment=None) -> Path:
    """Return an explicit exact checkout or the versioned cache checkout."""

    values = os.environ if environment is None else environment
    exact = values.get("NUMPILOT_CANNBOT_SKILLS")
    if exact:
        return Path(exact).expanduser()
    from numpilot.paths import cannbot_cache_base

    return cannbot_cache_base(values) / CANNBOT_COMMIT


@dataclass(frozen=True)
class SkillBundle:
    """Content and provenance actually supplied to the remote model."""

    root: str
    repository: str
    commit: str
    skill_name: str
    license: str
    files: Mapping[str, str]
    file_sha256: Mapping[str, str]
    bundle_sha256: str
    total_bytes: int

    def manifest(self) -> Dict[str, object]:
        """Return a path-independent, JSON-safe audit record."""

        return {
            "schema_version": 1,
            "repository": self.repository,
            "commit": self.commit,
            "skill_name": self.skill_name,
            "license": self.license,
            "bundle_sha256": self.bundle_sha256,
            "total_bytes": self.total_bytes,
            "files": dict(sorted(self.file_sha256.items())),
        }

    def prompt_context(self, selected_files: Optional[Tuple[str, ...]] = None) -> str:
        """Render the verified files so a remote model can actually read them."""

        selected = selected_files or tuple(self.files)
        unknown = set(selected) - set(self.files)
        if unknown:
            raise CannbotSkillError(
                "requested prompt file is outside the verified bundle: %s"
                % sorted(unknown)[0]
            )

        header = (
            "## Embedded official Huawei CANNBot skill\n\n"
            "The following content was read from the pinned official repository. "
            "Treat it as the implementation authority, while the GapSpec remains "
            "the semantic authority for this operator.\n\n"
            "- Repository: `%s`\n"
            "- Commit: `%s`\n"
            "- Skill: `%s`\n"
            "- License: `%s`\n"
            "- Bundle SHA-256: `%s`\n\n"
            % (
                self.repository,
                self.commit,
                self.skill_name,
                self.license,
                self.bundle_sha256,
            )
        )
        parts = [header]
        for relative in selected:
            content = self.files[relative]
            parts.append(
                "--- BEGIN CANNBOT FILE: %s (sha256=%s) ---\n%s\n"
                "--- END CANNBOT FILE: %s ---\n"
                % (relative, self.file_sha256[relative], content, relative)
            )
        return "\n".join(parts)


def load_skill_bundle(
    root: Path,
    *,
    expected_commit: str = CANNBOT_COMMIT,
    expected_repository: str = CANNBOT_REPOSITORY,
    bundle_files: Tuple[str, ...] = _BUNDLE_FILES,
) -> SkillBundle:
    """Validate an immutable official checkout and load the bounded bundle."""

    checkout = Path(root).expanduser()
    if checkout.is_symlink() or not checkout.is_dir():
        raise CannbotSkillError("CANNBot skill root is not a real directory: %s" % checkout)
    commit = _git(checkout, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise CannbotSkillError(
            "CANNBot skill commit mismatch: expected %s, got %s"
            % (expected_commit, commit)
        )
    repository = _git(checkout, "remote", "get-url", "origin")
    if _canonical_repository(repository) != _canonical_repository(expected_repository):
        raise CannbotSkillError("CANNBot skill origin is not the pinned official repository")
    dirty = _git(
        checkout,
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        *bundle_files,
    )
    if dirty:
        raise CannbotSkillError("CANNBot skill files contain local modifications")

    contents: Dict[str, str] = {}
    digests: Dict[str, str] = {}
    total = 0
    for relative in bundle_files:
        candidate = checkout.joinpath(*Path(relative).parts)
        if candidate.is_symlink() or not candidate.is_file():
            raise CannbotSkillError("required CANNBot skill file is missing: %s" % relative)
        try:
            payload = candidate.read_bytes()
            content = payload.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise CannbotSkillError(
                "cannot read CANNBot skill file %s: %s"
                % (relative, type(exc).__name__)
            ) from exc
        total += len(payload)
        if total > _MAX_BUNDLE_BYTES:
            raise CannbotSkillError("CANNBot prompt bundle exceeds the size limit")
        contents[relative] = content
        digests[relative] = _sha256(payload)

    identity = json.dumps(
        {
            "repository": CANNBOT_REPOSITORY,
            "commit": commit,
            "skill_name": CANNBOT_SKILL_NAME,
            "files": dict(sorted(digests.items())),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return SkillBundle(
        root=str(checkout.resolve()),
        repository=CANNBOT_REPOSITORY,
        commit=commit,
        skill_name=CANNBOT_SKILL_NAME,
        license=CANNBOT_LICENSE,
        files=contents,
        file_sha256=digests,
        bundle_sha256=_sha256(identity),
        total_bytes=total,
    )


def fetch_pinned_skill(destination: Optional[Path] = None) -> SkillBundle:
    """Fetch the pinned official commit into a private, versioned user cache."""

    target = Path(destination).expanduser() if destination else default_skill_checkout()
    if target.exists() or target.is_symlink():
        return load_skill_bundle(target)
    try:
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            os.chmod(target.parent, 0o700)
    except OSError as exc:
        raise CannbotSkillError(
            "cannot create CANNBot skill cache: %s" % type(exc).__name__
        ) from exc

    temporary = Path(tempfile.mkdtemp(prefix=".cannbot-skills-", dir=str(target.parent)))
    try:
        _git(temporary, "init")
        _git(temporary, "remote", "add", "origin", CANNBOT_REPOSITORY)
        _git(temporary, "fetch", "--depth", "1", "origin", CANNBOT_COMMIT)
        _git(temporary, "checkout", "--detach", "FETCH_HEAD")
        bundle = load_skill_bundle(temporary)
        try:
            os.replace(temporary, target)
        except OSError:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(temporary, ignore_errors=True)
                return load_skill_bundle(target)
            raise
        return load_skill_bundle(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def resolve_skill_bundle(
    skill_root: Optional[str],
    *,
    allow_fetch: bool,
) -> SkillBundle:
    """Resolve the author-owned static bundle before any mutable checkout."""

    if skill_root and skill_root != "auto":
        explicit = Path(skill_root)
        if (explicit / _STATIC_BUNDLE_MANIFEST).is_file():
            return load_static_skill_bundle(explicit)
        return load_skill_bundle(explicit)
    static_root = default_static_skill_root()
    if static_root.is_dir() and not static_root.is_symlink():
        return load_static_skill_bundle(static_root)
    raise CannbotSkillError(
        "static CANNBot skill bundle is missing; author must update the bundled snapshot"
    )
