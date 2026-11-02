#!/usr/bin/env python3
"""Extract the author-maintained AsNumpy source snapshot for an explicit rebuild.

This tool is maintainer-facing. It verifies the bundled source archive and
source manifest through NumPilot's component lock, then extracts only regular
files into a new private directory. It never fetches Git, runs a model, or
changes the installed AsNumpy runtime.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

from numpilot.asnumpy_component import (
    AsNumpyComponentError,
    default_component_root,
    load_component_manifest,
    verify_component_attachments,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the verified bundled AsNumpy source for maintainer work."
    )
    parser.add_argument(
        "--component",
        default=None,
        help="AsNumpy component directory; defaults to the source/install bundle.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="new extraction directory; it must not already exist",
    )
    parser.add_argument("--json", action="store_true", help="print a machine-readable receipt")
    return parser


def _extract(component, output: Path) -> dict[str, object]:
    if output.exists() or output.is_symlink():
        raise ValueError("output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.mkdir(mode=0o700)
    archive_path = component.root / str(component.source_bundle["filename"])
    prefix = str(component.source_bundle["prefix"])
    extracted = 0
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isdir():
                    continue
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or not member.name.startswith(prefix)
                ):
                    raise ValueError("source archive contains an unsafe member")
                relative = member.name[len(prefix):]
                relative_path = PurePosixPath(relative)
                if (
                    not relative
                    or relative_path.is_absolute()
                    or ".." in relative_path.parts
                    or "\\" in relative
                ):
                    raise ValueError("source archive contains an unsafe path")
                destination = output.joinpath(*relative_path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("source archive member is unreadable")
                with stream, destination.open("wb") as target:
                    shutil.copyfileobj(stream, target, length=1024 * 1024)
                os.chmod(destination, member.mode & 0o777)
                extracted += 1
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return {
        "source_root": str(output),
        "source_commit": component.commit,
        "source_tree": component.tree,
        "source_version": component.version,
        "file_count": extracted,
        "archive": dict(component.source_bundle),
        "execution_performed": False,
        "network_used": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    component_path = Path(args.component).expanduser() if args.component else default_component_root()
    try:
        component = load_component_manifest(component_path)
        if not component.source_bundle:
            raise AsNumpyComponentError("component has no bundled source snapshot")
        verify_component_attachments(component)
        receipt = _extract(component, Path(args.out).expanduser())
    except (AsNumpyComponentError, OSError, tarfile.TarError, ValueError) as exc:
        print("[asnumpy-source] %s" % exc, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("[asnumpy-source] extracted %s files" % receipt["file_count"])
        print("[source] %s" % receipt["source_root"])
        print("[commit] %s" % receipt["source_commit"])
        print("[tree] %s" % receipt["source_tree"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
