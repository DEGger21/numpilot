"""Immutable, local AsNumpy component lock used by the release installer.

The public NumPilot package must not resolve ``asnumpy`` as a floating Python
dependency.  This module is deliberately independent of :mod:`numpilot.bootstrap`
so it can be used by planning, installation, and environment checks without a
dependency cycle.  It only reads a small JSON lock and a neighbouring wheel;
it never invokes Git, pip, a package index, or a remote service.

The binary wheel is kept as a release-adjacent attachment because its native
extension is platform-specific.  ``components/asnumpy/manifest.json`` remains
in the source tree and binds the exact filename, digest, source identity, and
runtime constraints.  A missing attachment is an actionable error, never a
reason to fall back to ``git fetch`` or ``pip install -U``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional


COMPONENT_SCHEMA = "numpilot.asnumpy-component"
COMPONENT_SCHEMA_VERSION = 1
COMPONENT_ENV = "NUMPILOT_ASNUMPY_COMPONENT_DIR"
COMPONENT_MANIFEST_NAME = "manifest.json"
COMPONENT_README_NAME = "README.md"

_MAX_MANIFEST_BYTES = 512 * 1024
_MAX_TEXT = 4096
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+:-]{0,255}$")
_COMPONENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,255}$")
_WHEEL_FILENAME = re.compile(
    r"^asnumpy-[A-Za-z0-9][A-Za-z0-9.+!-]*-[A-Za-z0-9_]+-[A-Za-z0-9_]+-[A-Za-z0-9_]+\.whl$"
)
_ATTACHMENT_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,255}$")


class AsNumpyComponentError(RuntimeError):
    """The internal AsNumpy component lock or attachment is invalid."""


def _text(value: Any, label: str, *, pattern: Optional[re.Pattern[str]] = None) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AsNumpyComponentError("AsNumpy component %s is invalid" % label)
    if len(value) > _MAX_TEXT:
        raise AsNumpyComponentError("AsNumpy component %s is too long" % label)
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise AsNumpyComponentError("AsNumpy component %s contains control characters" % label)
    if pattern is not None and pattern.fullmatch(value) is None:
        raise AsNumpyComponentError("AsNumpy component %s has an invalid format" % label)
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise AsNumpyComponentError("AsNumpy component %s must be an object" % label)
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AsNumpyComponentError("AsNumpy component manifest has duplicate keys")
        result[key] = value
    return result


def _safe_local_path(path: Path, label: str, *, must_exist: bool = False) -> Path:
    """Normalize a local path while rejecting symlink traversal."""

    # Inspect the lexical spelling before ``abspath`` collapses ``..``.  A
    # path such as ``link/../bundle`` still traverses ``link`` at the kernel
    # level when ``link`` is a symlink; checking only the normalized spelling
    # would miss that redirect.  Relative component paths are intentionally
    # resolved from the current working directory, but every existing lexical
    # component must remain a real directory/file.
    expanded = os.path.expanduser(os.fspath(path))
    if "\x00" in expanded or "\n" in expanded or "\r" in expanded:
        raise AsNumpyComponentError("AsNumpy component %s contains unsafe characters" % label)
    lexical = Path(expanded)
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        if part in ("", "."):
            continue
        if part == "..":
            current = current.parent
            continue
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise AsNumpyComponentError("AsNumpy component %s traverses a symlink" % label)

    value = lexical
    value = Path(os.path.abspath(value))
    # Check every existing component.  A final ``O_NOFOLLOW`` check alone is
    # insufficient when a parent directory redirects outside the release.
    current = Path(value.anchor)
    for part in value.parts[1:]:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise AsNumpyComponentError("AsNumpy component %s traverses a symlink" % label)
    if must_exist and (not value.exists() or not value.is_file()):
        raise AsNumpyComponentError("AsNumpy component %s is missing: %s" % (label, value))
    return value


@dataclass(frozen=True)
class AsNumpyComponent:
    """Validated lock metadata for one immutable AsNumpy wheel attachment."""

    manifest_path: Path
    manifest_sha256: str
    component_id: str
    distribution: str
    version: str
    repository: str
    commit: str
    tree: str
    license: str
    wheel_filename: str
    wheel_sha256: str
    wheel_tags: tuple[str, ...]
    wheel_metadata_sha256: Optional[str]
    wheel_record_sha256: Optional[str]
    artifact_required: bool
    runtime: Mapping[str, Any]
    source_files_sha256: Mapping[str, str]
    installed_files: Mapping[str, Any]
    # Release metadata attachments are optional for backwards-compatible
    # development manifests, but a frozen component should bind both files.
    license_file: Mapping[str, Any] = field(default_factory=dict)
    sbom: Mapping[str, Any] = field(default_factory=dict)
    artifact_status: Optional[str] = None
    build: Mapping[str, Any] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return self.manifest_path.parent

    @property
    def wheel_path(self) -> Path:
        return self.root / self.wheel_filename

    def manifest(self) -> dict[str, Any]:
        """Return a detached JSON-safe lock representation."""

        wheel: dict[str, Any] = {
            "filename": self.wheel_filename,
            "sha256": self.wheel_sha256,
            "tags": list(self.wheel_tags),
        }
        if self.wheel_metadata_sha256 is not None:
            wheel["metadata_sha256"] = self.wheel_metadata_sha256
        if self.wheel_record_sha256 is not None:
            wheel["record_sha256"] = self.wheel_record_sha256
        if self.installed_files:
            wheel["installed_files"] = dict(self.installed_files)
        result = {
            "schema": COMPONENT_SCHEMA,
            "schema_version": COMPONENT_SCHEMA_VERSION,
            "component_id": self.component_id,
            "distribution": self.distribution,
            "version": self.version,
            "repository": self.repository,
            "commit": self.commit,
            "tree": self.tree,
            "license": self.license,
            "artifact_required": self.artifact_required,
            "wheel": wheel,
            "runtime": dict(self.runtime),
            "source_files_sha256": dict(self.source_files_sha256),
        }
        if self.artifact_status is not None:
            result["artifact_status"] = self.artifact_status
        if self.build:
            result["build"] = dict(self.build)
        if self.license_file:
            result["license_file"] = dict(self.license_file)
        if self.sbom:
            result["sbom"] = dict(self.sbom)
        return result

    def summary(self) -> dict[str, Any]:
        """Return a display-safe summary (no local paths beyond the manifest)."""

        return {
            "schema": COMPONENT_SCHEMA,
            "schema_version": COMPONENT_SCHEMA_VERSION,
            "component_id": self.component_id,
            "distribution": self.distribution,
            "version": self.version,
            "repository": self.repository,
            "commit": self.commit,
            "tree": self.tree,
            "wheel_filename": self.wheel_filename,
            "wheel_sha256": self.wheel_sha256,
            "wheel_tags": list(self.wheel_tags),
            "artifact_required": self.artifact_required,
            "artifact_status": self.artifact_status,
            "build": dict(self.build),
            "manifest_path": str(self.manifest_path),
            "wheel_path": str(self.wheel_path),
            "license": self.license,
            "runtime": dict(self.runtime),
            "license_file": dict(self.license_file),
            "sbom": dict(self.sbom),
        }


def _validate_manifest(
    raw: Mapping[str, Any],
    path: Path,
    *,
    manifest_sha256: str,
) -> AsNumpyComponent:
    schema = _text(raw.get("schema"), "schema")
    if schema != COMPONENT_SCHEMA:
        raise AsNumpyComponentError("unexpected AsNumpy component manifest schema")
    version = raw.get("schema_version")
    if type(version) is not int or version != COMPONENT_SCHEMA_VERSION:
        raise AsNumpyComponentError("unsupported AsNumpy component manifest version")

    component_id = _text(raw.get("component_id"), "component_id", pattern=_COMPONENT_ID)
    distribution = _text(raw.get("distribution"), "distribution")
    if distribution.lower().replace("-", "_") != "asnumpy":
        raise AsNumpyComponentError("component distribution must be AsNumpy")
    package_version = _text(raw.get("version"), "version", pattern=_VERSION)
    repository = _text(raw.get("repository"), "repository")
    if "://" not in repository:
        raise AsNumpyComponentError("component repository must be an explicit URL")
    commit = _text(raw.get("commit"), "commit", pattern=_HEX40)
    tree = _text(raw.get("tree"), "tree", pattern=_HEX40)
    license_name = _text(raw.get("license"), "license")
    artifact_required = raw.get("artifact_required")
    if type(artifact_required) is not bool:
        raise AsNumpyComponentError("component artifact_required must be boolean")

    artifact_status = raw.get("artifact_status")
    if artifact_status is not None:
        artifact_status = _text(artifact_status, "artifact_status")

    build_raw = raw.get("build", {})
    build = _mapping(build_raw, "build")
    build_copy: dict[str, Any] = {}
    for key, value in build.items():
        _text(str(key), "build key")
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise AsNumpyComponentError("component build value is not scalar")
        build_copy[str(key)] = value

    wheel = _mapping(raw.get("wheel"), "wheel")
    filename = _text(wheel.get("filename"), "wheel.filename")
    if _WHEEL_FILENAME.fullmatch(filename) is None or Path(filename).name != filename:
        raise AsNumpyComponentError("component wheel filename is unsafe")
    digest = _text(wheel.get("sha256"), "wheel.sha256", pattern=_SHA256).lower()
    tags_raw = wheel.get("tags")
    if not isinstance(tags_raw, list) or not tags_raw or any(
        not isinstance(tag, str) or not tag or len(tag) > 256 or tag != tag.strip()
        for tag in tags_raw
    ):
        raise AsNumpyComponentError("component wheel tags are invalid")
    tags = tuple(str(tag) for tag in tags_raw)
    metadata_digest = wheel.get("metadata_sha256")
    if metadata_digest is not None:
        metadata_digest = _text(metadata_digest, "wheel.metadata_sha256", pattern=_SHA256).lower()
    record_digest = wheel.get("record_sha256")
    if record_digest is not None:
        record_digest = _text(record_digest, "wheel.record_sha256", pattern=_SHA256).lower()
    installed_files = wheel.get("installed_files", {})
    if type(installed_files) is not dict:
        raise AsNumpyComponentError("component wheel.installed_files must be an object")

    def attachment(value: Any, label: str, *, required_format: str | None = None) -> dict[str, Any]:
        if value is None:
            return {}
        block = _mapping(value, label)
        filename = _text(block.get("filename"), label + ".filename")
        if (
            Path(filename).name != filename
            or _ATTACHMENT_FILENAME.fullmatch(filename) is None
        ):
            raise AsNumpyComponentError("AsNumpy component %s filename is unsafe" % label)
        digest = _text(block.get("sha256"), label + ".sha256", pattern=_SHA256).lower()
        result: dict[str, Any] = {"filename": filename, "sha256": digest}
        if "format" in block:
            format_name = _text(block.get("format"), label + ".format")
            if required_format is not None and format_name != required_format:
                raise AsNumpyComponentError(
                    "AsNumpy component %s format is unsupported" % label
                )
            result["format"] = format_name
        elif required_format is not None:
            raise AsNumpyComponentError("AsNumpy component %s format is missing" % label)
        return result

    license_file = attachment(raw.get("license_file"), "license_file")
    sbom = attachment(raw.get("sbom"), "sbom", required_format="SPDX-2.3")
    if artifact_required:
        # A frozen release is not complete when the binary alone is present:
        # the legal notice and machine-readable bill of materials are part of
        # the same immutable component identity.  Development manifests may
        # omit these optional blocks, but a release lock must bind both.
        if not license_file:
            raise AsNumpyComponentError(
                "frozen AsNumpy component must bind a LICENSE attachment"
            )
        if not sbom:
            raise AsNumpyComponentError(
                "frozen AsNumpy component must bind an SPDX SBOM attachment"
            )

    runtime = _mapping(raw.get("runtime", {}), "runtime")
    runtime_copy = dict(runtime)
    for key, value in runtime_copy.items():
        _text(str(key), "runtime key")
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise AsNumpyComponentError("component runtime value is not scalar")
    if artifact_required:
        required_runtime = (
            "cann_version",
            "machine",
            "loguru_version",
            "numpy_version",
            "python_implementation",
            "python_major_minor",
        )
        missing_runtime = [key for key in required_runtime if key not in runtime_copy]
        if missing_runtime:
            raise AsNumpyComponentError(
                "frozen AsNumpy component runtime constraints are missing: %s"
                % ", ".join(missing_runtime)
            )
        for key in required_runtime:
            if not isinstance(runtime_copy[key], str) or not runtime_copy[key].strip():
                raise AsNumpyComponentError(
                    "frozen AsNumpy component runtime constraint is invalid: %s" % key
                )
    source_files = raw.get("source_files_sha256", {})
    if type(source_files) is not dict:
        raise AsNumpyComponentError("component source_files_sha256 must be an object")
    source_copy: dict[str, str] = {}
    for relative, value in source_files.items():
        relative = _text(relative, "source file path")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or "\\" in relative:
            raise AsNumpyComponentError("component source file path is unsafe")
        source_copy[relative] = _text(value, "source file digest", pattern=_SHA256).lower()

    return AsNumpyComponent(
        manifest_path=path,
        manifest_sha256=manifest_sha256,
        component_id=component_id,
        distribution="asnumpy",
        version=package_version,
        repository=repository,
        commit=commit,
        tree=tree,
        license=license_name,
        wheel_filename=filename,
        wheel_sha256=digest,
        wheel_tags=tags,
        wheel_metadata_sha256=metadata_digest,
        wheel_record_sha256=record_digest,
        artifact_required=artifact_required,
        runtime=runtime_copy,
        source_files_sha256=source_copy,
        installed_files=dict(installed_files),
        license_file=license_file,
        sbom=sbom,
        artifact_status=artifact_status,
        build=build_copy,
    )


def load_component_manifest(path: Optional[Path] = None) -> AsNumpyComponent:
    """Load and validate the local component manifest without network access."""

    target = _safe_local_path(path or default_component_manifest(), "manifest")
    if target.is_dir():
        target = target / COMPONENT_MANIFEST_NAME
    target = _safe_local_path(target, "manifest")
    if target.is_symlink() or not target.is_file():
        raise AsNumpyComponentError("AsNumpy component manifest is missing: %s" % target)
    try:
        payload = target.read_bytes()
    except OSError as exc:
        raise AsNumpyComponentError("cannot read AsNumpy component manifest") from exc
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise AsNumpyComponentError("AsNumpy component manifest is too large")
    try:
        raw = json.loads(payload.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeError, json.JSONDecodeError, AsNumpyComponentError) as exc:
        raise AsNumpyComponentError("AsNumpy component manifest is invalid JSON") from exc
    return _validate_manifest(
        _mapping(raw, "manifest"),
        target,
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )


def default_component_root() -> Path:
    """Return the source/install location searched for the internal component."""

    explicit = os.getenv(COMPONENT_ENV)
    if explicit:
        return _safe_local_path(Path(explicit), "component directory")

    # Source checkout: ``numpilot/asnumpy_component.py`` is one level below
    # the repository root.  Installed wheels receive the same files under
    # ``share/numpilot/components/asnumpy`` via ``tool.setuptools.data-files``.
    source_root = Path(__file__).resolve().parents[1] / "components" / "asnumpy"
    if source_root.exists():
        return source_root
    installed_root = Path(sys.prefix) / "share" / "numpilot" / "components" / "asnumpy"
    return installed_root


def default_component_manifest() -> Path:
    return default_component_root() / COMPONENT_MANIFEST_NAME


def resolve_component_wheel(
    component: AsNumpyComponent,
    path: Optional[Path] = None,
) -> Path:
    """Resolve the exact local wheel named by ``component``.

    ``path`` is useful for a release attachment kept outside the repository,
    but its basename must still equal the lock's filename.  No alternate
    version, glob, URL, or source checkout is accepted.
    """

    candidate = (
        Path(path).expanduser()
        if path is not None
        else component.wheel_path
    )
    candidate = _safe_local_path(candidate, "wheel")
    if candidate.name != component.wheel_filename:
        raise AsNumpyComponentError(
            "AsNumpy component wheel filename mismatch: expected %s, got %s"
            % (component.wheel_filename, candidate.name)
        )
    if candidate.is_symlink() or not candidate.is_file():
        raise AsNumpyComponentError(
            "AsNumpy internal component wheel is missing; provide the release attachment "
            "%s (sha256=%s)" % (component.wheel_filename, component.wheel_sha256)
        )
    try:
        observed = hashlib.sha256(candidate.read_bytes()).hexdigest()
    except OSError as exc:
        raise AsNumpyComponentError("cannot read AsNumpy internal component wheel") from exc
    if observed != component.wheel_sha256:
        raise AsNumpyComponentError(
            "AsNumpy internal component wheel SHA-256 mismatch: expected %s, got %s"
            % (component.wheel_sha256, observed)
        )
    return candidate


def verify_component_attachments(component: AsNumpyComponent) -> dict[str, Any]:
    """Verify release-side LICENSE/SBOM attachments without executing them.

    Development manifests may omit these optional blocks.  A frozen manifest
    should include both; callers can use the returned identities in a receipt
    and fail closed when an attachment is absent or has changed.
    """

    verified: dict[str, Any] = {}
    for label, block in (("license_file", component.license_file), ("sbom", component.sbom)):
        if not block:
            continue
        filename = str(block.get("filename", ""))
        candidate = _safe_local_path(component.root / filename, label)
        if (
            candidate.name != filename
            or candidate.is_symlink()
            or not candidate.is_file()
        ):
            raise AsNumpyComponentError(
                "AsNumpy component %s is missing: %s" % (label, filename)
            )
        try:
            observed = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError as exc:
            raise AsNumpyComponentError(
                "cannot read AsNumpy component %s" % label
            ) from exc
        expected = str(block.get("sha256", "")).lower()
        if observed != expected:
            raise AsNumpyComponentError(
                "AsNumpy component %s SHA-256 mismatch: expected %s, got %s"
                % (label, expected, observed)
            )
        if label == "license_file":
            try:
                license_text = candidate.read_text(encoding="utf-8", errors="strict")
            except (OSError, UnicodeError) as exc:
                raise AsNumpyComponentError("AsNumpy component LICENSE is not UTF-8") from exc
            if "Apache License" not in license_text or "Version 2.0" not in license_text:
                raise AsNumpyComponentError(
                    "AsNumpy component LICENSE does not contain Apache-2.0 text"
                )
        else:
            try:
                sbom_payload = json.loads(
                    candidate.read_text(encoding="utf-8"),
                    object_pairs_hook=_strict_object,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError("non-standard JSON constant: %s" % value)
                    ),
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise AsNumpyComponentError("AsNumpy component SBOM is invalid JSON") from exc
            except (AsNumpyComponentError, ValueError) as exc:
                raise AsNumpyComponentError("AsNumpy component SBOM is invalid JSON") from exc
            if (
                not isinstance(sbom_payload, dict)
                or sbom_payload.get("spdxVersion") != "SPDX-2.3"
                or not isinstance(sbom_payload.get("SPDXID"), str)
                or not isinstance(sbom_payload.get("documentNamespace"), str)
                or not isinstance(sbom_payload.get("packages"), list)
            ):
                raise AsNumpyComponentError("AsNumpy component SBOM is not SPDX-2.3")
        verified[label] = {"filename": filename, "sha256": observed}
    return verified


def component_receipt_identity(component: AsNumpyComponent) -> dict[str, Any]:
    """Return the identity block persisted in a bootstrap receipt."""

    return {
        "component_id": component.component_id,
        "manifest_schema": COMPONENT_SCHEMA,
        # This digest is captured from the exact payload parsed by the loader.
        # Re-reading the path here would permit a manifest replacement between
        # validation and receipt construction.
        "manifest_sha256": component.manifest_sha256,
        "wheel_filename": component.wheel_filename,
        "wheel_sha256": component.wheel_sha256,
        "repository": component.repository,
        "commit": component.commit,
        "tree": component.tree,
        "version": component.version,
        "artifact_status": component.artifact_status,
        "build": dict(component.build),
        "runtime": dict(component.runtime),
        "license_file": dict(component.license_file),
        "sbom": dict(component.sbom),
    }


__all__ = [
    "COMPONENT_ENV",
    "COMPONENT_MANIFEST_NAME",
    "COMPONENT_SCHEMA",
    "COMPONENT_SCHEMA_VERSION",
    "AsNumpyComponent",
    "AsNumpyComponentError",
    "component_receipt_identity",
    "default_component_manifest",
    "default_component_root",
    "load_component_manifest",
    "resolve_component_wheel",
    "verify_component_attachments",
]
