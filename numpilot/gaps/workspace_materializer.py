"""Read-only source-compatible AsNumpy workspace materialization.

The materializer copies a pinned AsNumpy source tree into a new private
workspace and applies a hash-bound overlay there. It never mutates the source
tree, static component, wheel, or activation registry.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping

from numpilot.gaps.generated_adapter import (
    GeneratedAdapterContractError,
    OVERLAY_PREFLIGHT_SCHEMA,
)

MATERIALIZATION_SCHEMA = "numpilot.asnumpy.source-compatible-workspace-receipt.v1"
_SHA256 = "0123456789abcdef"


class WorkspaceMaterializationError(ValueError):
    """The source/overlay/workspace boundary is invalid."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256 for character in value)
    )


def _parse_contract(value: Mapping[str, Any] | str | bytes) -> dict[str, Any]:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceMaterializationError(
                "workspace contract is not UTF-8"
            ) from exc
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise WorkspaceMaterializationError(
                "workspace contract is not valid JSON"
            ) from exc
    if not isinstance(value, Mapping):
        raise WorkspaceMaterializationError("workspace contract is invalid")
    contract = dict(value)
    if (
        contract.get("schema")
        != "numpilot.asnumpy.source-compatible-workspace.v1"
        or contract.get("status") != "ready"
        or contract.get("materialization_performed") is not False
        or contract.get("build_performed") is not False
        or contract.get("formal_component_changed") is not False
        or contract.get("activation_performed") is not False
    ):
        raise WorkspaceMaterializationError(
            "workspace contract is not a ready read-only contract"
        )
    receipt_sha = contract.get("receipt_sha256")
    if not _is_sha256(receipt_sha):
        raise WorkspaceMaterializationError(
            "workspace contract receipt hash is invalid"
        )
    unsigned = dict(contract)
    unsigned.pop("receipt_sha256", None)
    if _canonical_sha256(unsigned) != receipt_sha:
        raise WorkspaceMaterializationError(
            "workspace contract receipt hash mismatch"
        )
    for key in ("reference_root", "workspace_root"):
        value = contract.get(key)
        if (
            not isinstance(value, str)
            or not value.startswith("/")
            or any(ord(character) < 32 for character in value)
        ):
            raise WorkspaceMaterializationError(
                "workspace contract path is invalid: %s" % key
            )
    paths = contract.get("expected_overlay_paths")
    if (
        not isinstance(paths, list)
        or not paths
        or len(set(paths)) != len(paths)
        or any(
            not isinstance(path, str)
            or str(PurePosixPath(path)) != path
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            for path in paths
        )
    ):
        raise WorkspaceMaterializationError(
            "workspace contract overlay paths are invalid"
        )
    if not _is_sha256(contract.get("preflight_receipt_sha256")):
        raise WorkspaceMaterializationError(
            "workspace contract preflight binding is invalid"
        )
    return contract


def _reject_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise WorkspaceMaterializationError(
            "%s must not contain symlinks: %s" % (label, path)
        )


def _validate_internal_symlink(path: Path, root: Path, label: str) -> str:
    """Return a safe relative symlink target, rejecting path escapes."""
    target = os.readlink(path)
    if not target or os.path.isabs(target):
        raise WorkspaceMaterializationError(
            "%s symlink must be relative: %s" % (label, path)
        )
    root_resolved = root.resolve(strict=True)
    resolved = (path.parent / target).resolve(strict=False)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise WorkspaceMaterializationError(
            "%s symlink escapes root: %s" % (label, path)
        ) from exc
    if not resolved.exists():
        raise WorkspaceMaterializationError(
            "%s symlink target does not exist: %s" % (label, path)
        )
    return target


def _tree_manifest(
    root: Path,
) -> tuple[dict[str, str], dict[str, str], str]:
    hashes: dict[str, str] = {}
    symlinks: dict[str, str] = {}
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            path = current_path / name
            if path.is_symlink():
                relative = path.relative_to(root).as_posix()
                symlinks[relative] = _validate_internal_symlink(
                    path, root, "source tree"
                )
        for name in files:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                symlinks[relative] = _validate_internal_symlink(
                    path, root, "source tree"
                )
            else:
                hashes[relative] = _sha256_bytes(path.read_bytes())
    if not hashes and not symlinks:
        raise WorkspaceMaterializationError("source tree is empty")
    identity = {
        "files_sha256": dict(sorted(hashes.items())),
        "symlinks": dict(sorted(symlinks.items())),
    }
    return (
        dict(sorted(hashes.items())),
        dict(sorted(symlinks.items())),
        _canonical_sha256(identity),
    )


def _copy_tree(
    source: Path,
    destination: Path,
    *,
    source_root: Path | None = None,
) -> None:
    if source_root is None:
        source_root = source
    _reject_symlink(source, "source root")
    if not source.is_dir():
        raise WorkspaceMaterializationError("source root is not a directory")
    if destination.exists():
        if not destination.is_dir() or destination.is_symlink():
            raise WorkspaceMaterializationError(
                "workspace copy destination is not a directory"
            )
    else:
        destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    for entry in sorted(source.iterdir(), key=lambda item: item.name):
        target = destination / entry.name
        if entry.is_symlink():
            target.symlink_to(
                _validate_internal_symlink(entry, source_root, "source tree")
            )
        elif entry.is_dir():
            _copy_tree(entry, target, source_root=source_root)
        elif entry.is_file():
            shutil.copy2(entry, target)
        else:
            raise WorkspaceMaterializationError(
                "source tree contains unsupported entry: %s" % entry
            )


def _overlay_files(root: Path, expected: list[str]) -> dict[str, Path]:
    _reject_symlink(root, "overlay root")
    if not root.is_dir():
        raise WorkspaceMaterializationError("overlay root is not a directory")
    found: dict[str, Path] = {}
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            _reject_symlink(current_path / name, "overlay")
        for name in files:
            path = current_path / name
            _reject_symlink(path, "overlay")
            relative = path.relative_to(root).as_posix()
            found[relative] = path
    expected_set = set(expected)
    if set(found) != expected_set:
        raise WorkspaceMaterializationError(
            "overlay files differ from workspace contract"
        )
    return found


def materialize_source_compatible_workspace(
    workspace_contract: Mapping[str, Any] | str | bytes,
    *,
    overlay_root: str,
) -> dict[str, Any]:
    """Copy source plus overlay into a new isolated workspace.

    The destination must not exist. The source tree and overlay are read-only
    inputs; only the new destination is written. No build, wheel installation,
    component replacement, or activation is performed.
    """
    contract = _parse_contract(workspace_contract)
    source = Path(contract["reference_root"])
    workspace = Path(contract["workspace_root"])
    overlay = Path(overlay_root)
    _reject_symlink(source, "source root")
    _reject_symlink(overlay, "overlay root")
    if not source.is_dir():
        raise WorkspaceMaterializationError("source root is not a directory")
    if not overlay.is_dir():
        raise WorkspaceMaterializationError("overlay root is not a directory")
    if workspace.exists() or workspace.is_symlink():
        raise WorkspaceMaterializationError(
            "workspace destination already exists"
        )
    if workspace == source or source in workspace.parents:
        raise WorkspaceMaterializationError(
            "workspace destination must be outside source root"
        )
    if not workspace.parent.is_dir():
        raise WorkspaceMaterializationError(
            "workspace parent directory must already exist"
        )
    expected = list(contract["expected_overlay_paths"])
    files = _overlay_files(overlay, expected)
    temporary = Path(tempfile.mkdtemp(
        prefix=".numpilot-source-compatible-",
        dir=str(workspace.parent),
    ))
    try:
        _copy_tree(source, temporary)
        overlay_hashes: dict[str, str] = {}
        for relative in expected:
            source_overlay = files[relative]
            target = temporary / PurePosixPath(relative)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copy2(source_overlay, target)
            overlay_hashes[relative] = _sha256_bytes(target.read_bytes())
        tree_hashes, tree_symlinks, tree_manifest_sha256 = _tree_manifest(temporary)
        body: dict[str, Any] = {
            "schema": MATERIALIZATION_SCHEMA,
            "status": "materialized_unverified",
            "contract_receipt_sha256": contract["receipt_sha256"],
            "preflight_receipt_sha256": contract["preflight_receipt_sha256"],
            "reference_root": str(source),
            "workspace_root": str(workspace),
            "expected_overlay_paths": expected,
            "overlay_files_sha256": dict(sorted(overlay_hashes.items())),
            "workspace_tree_manifest_sha256": tree_manifest_sha256,
            "workspace_tree_file_count": len(tree_hashes),
            "workspace_tree_symlinks": tree_symlinks,
            "target_extension_module": "asnumpy._core",
            "cmake_target": "_core",
            "object_library": "custom_ops",
            "materialization_performed": True,
            "build_performed": False,
            "formal_component_changed": False,
            "activation_performed": False,
        }
        body["receipt_sha256"] = _canonical_sha256(body)
        receipt_path = temporary / "numpilot-workspace-materialization.json"
        receipt_path.write_text(
            json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(receipt_path, 0o600)
        os.replace(temporary, workspace)
        temporary = None
        return body
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


__all__ = [
    "MATERIALIZATION_SCHEMA",
    "WorkspaceMaterializationError",
    "materialize_source_compatible_workspace",
]

