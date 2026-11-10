"""Snapshot the exact author-pinned AsNumpy bundle, including approved patches."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import stat

from numpilot.asnumpy_component import (
    default_component_root,
    load_component_manifest,
    verify_component_attachments,
)


def snapshot_bundled_source(reference: Path, target: Path, *, binding: dict | None = None):
    """Copy only manifest members, requiring every source byte to match.

    A dirty Git checkout is never accepted as an arbitrary build input. This
    path admits exactly the content already shipped in the verified component,
    and omits all untracked files, build products and unrelated local changes.
    No Git mutation, network, build, or import is performed here.
    """
    if reference.is_symlink() or not reference.is_dir():
        raise ValueError("bundled source reference must be a regular directory")
    if target.exists() or target.is_symlink():
        raise ValueError("bundled source target must not exist")
    component = load_component_manifest(default_component_root())
    verify_component_attachments(component)
    bundle = component.source_bundle
    if not bundle:
        raise ValueError("verified AsNumpy source bundle is required")
    manifest = json.loads((component.root / bundle['manifest_filename']).read_text())
    expected = manifest['files']
    contents = {}
    total = 0
    for relative, digest in expected.items():
        path = PurePosixPath(relative)
        if path.is_absolute() or '..' in path.parts or str(path) != relative:
            raise ValueError("unsafe bundled source path")
        source = reference
        for part in path.parts:
            source = source / part
            if source.is_symlink():
                # The upstream AGENTS.md is an internal alias of CLAUDE.md.
                # Materialize only a leaf that resolves inside the reference;
                # never traverse a linked directory or copy a link to staging.
                if source != reference / relative:
                    raise ValueError("bundled source contains a linked directory: " + relative)
                try:
                    source.resolve(strict=True).relative_to(reference.resolve())
                except ValueError as exc:
                    raise ValueError("bundled source link escapes reference: " + relative) from exc
        if not stat.S_ISREG(source.stat().st_mode):
            raise ValueError("bundled source is not a regular file: " + relative)
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("bundled source hash mismatch: " + relative)
        total += len(content)
        if total > 1024 * 1024 * 1024:
            raise ValueError("bundled source exceeds snapshot bound")
        contents[relative] = content
    # Validate all members before publishing any staging source file. Write the
    # bytes that were checked, rather than rereading the mutable reference.
    target.mkdir(mode=0o700)
    for relative, content in contents.items():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination.write_bytes(content)
        destination.chmod(0o600)
    if binding is not None:
        binding.update({
            "kind": "verified_component_bundle",
            "component_id": component.component_id,
            "component_manifest_sha256": component.manifest_sha256,
            "source_bundle_sha256": bundle["sha256"],
            "source_manifest_sha256": bundle["manifest_sha256"],
            "repository": component.repository,
            "commit": component.commit,
            "tree": component.tree,
        })
    return component.commit, dict(expected), total
