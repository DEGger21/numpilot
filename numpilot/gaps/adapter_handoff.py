"""Hash-bound local handoff between NumPilot and an external coding agent."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Dict, Mapping, Tuple

from numpilot.gaps.provider import (
    _OFFICIAL_EXACT_EDIT_TARGET_WINDOWS,
    _render_official_exact_edit_target_windows,
)

HANDOFF_SCHEMA = "numpilot.codex-adapter-handoff.v1"
PROTOCOL = "NUMPILOT_WINDOW_REPLACE_V1"
_MAX_HANDOFF_BYTES = 8 * 1024 * 1024
_MAX_CONTEXT_BYTES = 256 * 1024
_MAX_PATH = 512


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_PATH:
        raise ValueError("handoff relative path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or "." in path.parts or ".." in path.parts:
        raise ValueError("handoff relative path is invalid")
    if str(path) != value:
        raise ValueError("handoff relative path is not normalized")
    return value


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        raise FileExistsError(str(path))
    encoded = value.encode("utf-8")
    if len(encoded) > _MAX_HANDOFF_BYTES:
        raise ValueError("handoff file exceeds size limit")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)


def _window_manifest(path: str, content: str):
    _rendered, intervals = _render_official_exact_edit_target_windows(
        target_path=path, base_content=content
    )
    specs = _OFFICIAL_EXACT_EDIT_TARGET_WINDOWS[path]
    return [
        {
            "index": index,
            "label": label,
            "start_character": start,
            "end_character": end,
            "sha256": _sha256(content[start:end].encode("utf-8")),
        }
        for index, ((label, _start, _end), (start, end)) in enumerate(
            zip(specs, intervals), 1
        )
    ]


def write_adapter_handoff(
    root: Path,
    *,
    operator_artifact: Mapping[str, object],
    adapter_contract: str,
    asnumpy_context: Mapping[str, str],
    adapter_context_plan: Mapping[str, object],
    skill_files: Tuple[str, ...],
    skill_bundle_sha256: str,
    gap_taskbook=None,
) -> Dict[str, object]:
    """Write a read-only prompt package for a Codex/agent response pass."""

    root = Path(root)
    if root.exists() or root.is_symlink():
        raise ValueError("handoff output must not already exist")
    paths = tuple(adapter_context_plan.get("expected_overlay_paths", ()))
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("handoff expected overlay paths are invalid")
    for path in paths:
        _safe_relative(path)
        if path not in asnumpy_context:
            raise ValueError("handoff target context is missing: %s" % path)
    context_hashes = {
        path: _sha256(content.encode("utf-8"))
        for path, content in sorted(asnumpy_context.items())
    }
    if any(
        not isinstance(path, str)
        or not isinstance(content, str)
        or not content
        or "\x00" in content
        for path, content in asnumpy_context.items()
    ):
        raise ValueError("handoff context is invalid")
    if sum(len(value.encode("utf-8")) for value in asnumpy_context.values()) > _MAX_CONTEXT_BYTES:
        raise ValueError("handoff context exceeds size limit")
    root.mkdir(parents=True, mode=0o700)
    taskbook_ref = None
    taskbook_prompt = ""
    taskbook = None
    if gap_taskbook is not None:
        from numpilot.gap_requirements import persist_gap_taskbook, render_gap_taskbook_markdown, validate_gap_taskbook
        from numpilot.gaps.taskbook_binding import validate_adapter_taskbook

        taskbook = validate_gap_taskbook(gap_taskbook)
        taskbook_ref = persist_gap_taskbook(root / "taskbook", gap_taskbook)
        taskbook = validate_adapter_taskbook(taskbook, operator_artifact=operator_artifact,
                                            adapter_contract=adapter_contract)
        taskbook_prompt = render_gap_taskbook_markdown(taskbook)
    metadata = {
        "schema": HANDOFF_SCHEMA,
        "schema_version": 1,
        "status": "waiting_for_agent",
        "protocol": PROTOCOL,
        "candidate_id": operator_artifact.get("artifact_id"),
        "operator_artifact": dict(operator_artifact),
        "adapter_contract": adapter_contract,
        "adapter_contract_sha256": _sha256(adapter_contract.encode("utf-8")),
        "adapter_context_plan": dict(adapter_context_plan),
        "context_sha256": context_hashes,
        "expected_overlay_paths": list(paths),
        "skill_files": list(skill_files),
        "skill_bundle_sha256": skill_bundle_sha256,
        "responses_suffix": ".response",
    }
    if taskbook_ref is not None:
        metadata["gap_taskbook"] = {**taskbook_ref,
            "json": {**taskbook_ref["json"], "path": "taskbook/GAP_TASKBOOK.json"},
            "markdown": {**taskbook_ref["markdown"], "path": "taskbook/GAP_TASKBOOK.md"}}
    if taskbook is not None and taskbook["generation"]["allowed"] is not True:
        metadata["status"] = "requirements_blocked"
        _write_text(root / "handoff.json", json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        _write_text(root / "README.md", "# NumPilot adapter requirements\n\nStatus: `requirements_blocked`\n\n"
                    "Generation is blocked; this package is not waiting for an agent response.\n"
                    "Read [GAP_TASKBOOK.md](taskbook/GAP_TASKBOOK.md) and its "
                    "[JSON contract](taskbook/GAP_TASKBOOK.json). New generation supports float32 only.\n"
                    "Resolve the listed dtype/scope blockers before preparing a new handoff.\n"
                    "No prompts or response targets were issued; no model was called.\n")
        return metadata
    (root / "context").mkdir(mode=0o700)
    (root / "windows").mkdir(mode=0o700)
    (root / "prompts").mkdir(mode=0o700)
    (root / "responses").mkdir(mode=0o700)
    context_root = root / "context"
    for label, content in sorted(asnumpy_context.items()):
        _safe_relative(label)
        _write_text(context_root / label, content)
    windows = {}
    for path in paths:
        windows[path] = _window_manifest(path, asnumpy_context[path])
        window_text = json.dumps(windows[path], ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _write_text(root / "windows" / (path + ".json"), window_text)
        prompt = "\n".join((
            "# NumPilot external-agent adapter task",
            "",
            "Protocol: %s" % PROTOCOL,
            "Target path: `%s`" % path,
            "Base SHA-256: %s" % context_hashes[path],
            "Response file: `responses/%s.response`" % path,
            "",
            "Return every reviewed window in order. Keep unchanged windows byte-for-byte.",
            "The local validator owns all hash, path, window, and qualification gates.",
            taskbook_prompt,
            "",
            "WINDOWS:",
            json.dumps(windows[path], ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "TARGET FILE:",
            asnumpy_context[path],
        )) + "\n"
        _write_text(root / "prompts" / (path + ".md"), prompt)
    encoded = json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_text(root / "handoff.json", encoded)
    readme = """# NumPilot Codex adapter handoff\n\nStatus: `waiting_for_agent`\n\nRead the matching file under `prompts/`, generate one raw\n`NUMPILOT_WINDOW_REPLACE_V1` response, and write it to the listed path under\n`responses/`. Do not edit the AsNumpy source tree or handoff metadata. NumPilot\nwill revalidate every hash and window before creating `generated_unverified`.\n\nThe response is an injected local agent artifact. It is never recorded as a\nremote DeepSeek call and it still must pass the later build, import, NumPy\noracle, NPU, performance, qualification, and activation gates.\n"""
    if taskbook is not None:
        readme += ("\nRead the bound [taskbook Markdown](taskbook/GAP_TASKBOOK.md) and "
                   "[JSON contract](taskbook/GAP_TASKBOOK.json) before editing.\n"
                   "The first generation scope is float32 only. Unknown shapes/layouts and "
                   "other unresolved fields remain requirements to resolve; no success is implied.\n")
    _write_text(root / "README.md", readme)
    return metadata


def load_adapter_handoff(root: Path) -> Dict[str, object]:
    root = Path(root)
    manifest = root / "handoff.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("handoff manifest is missing")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema") != HANDOFF_SCHEMA or data.get("schema_version") != 1:
        raise ValueError("handoff schema is unsupported")
    if data.get("protocol") != PROTOCOL or data.get("status") != "waiting_for_agent":
        raise ValueError("handoff status/protocol is invalid")
    paths = data.get("expected_overlay_paths")
    if not isinstance(paths, list) or not paths or len(paths) != len(set(paths)):
        raise ValueError("handoff paths are invalid")
    context_hashes = data.get("context_sha256")
    if not isinstance(context_hashes, dict):
        raise ValueError("handoff context hashes are invalid")
    context = {}
    for label, digest in sorted(context_hashes.items()):
        _safe_relative(label)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("handoff context hash is invalid")
        path = root / "context" / label
        if path.is_symlink() or not path.is_file():
            raise ValueError("handoff context file is missing: %s" % label)
        content = path.read_text(encoding="utf-8")
        if _sha256(content.encode("utf-8")) != digest:
            raise ValueError("handoff context hash mismatch: %s" % label)
        context[label] = content
    if data.get("adapter_contract_sha256") != _sha256(
        str(data.get("adapter_contract", "")).encode("utf-8")
    ):
        raise ValueError("handoff adapter contract hash mismatch")
    taskbook_ref = data.get("gap_taskbook")
    taskbook = None
    if taskbook_ref is None and (root / "taskbook").exists():
        raise ValueError("handoff taskbook reference is missing")
    if taskbook_ref is not None:
        from numpilot.gap_requirements import validate_gap_taskbook, render_gap_taskbook_markdown
        from numpilot.gaps.taskbook_binding import validate_adapter_taskbook

        if (not isinstance(taskbook_ref, dict)
                or taskbook_ref.get("schema") != "numpilot.gap-taskbook-ref"
                or type(taskbook_ref.get("schema_version")) is not int
                or taskbook_ref["schema_version"] != 1):
            raise ValueError("handoff taskbook reference is invalid")
        for key, name in (("json", "GAP_TASKBOOK.json"), ("markdown", "GAP_TASKBOOK.md")):
            ref = taskbook_ref.get(key, {})
            path = root / "taskbook" / name
            if (ref.get("path") != "taskbook/" + name or path.is_symlink()
                    or (root / "taskbook").is_symlink() or not path.is_file()
                    or path.stat().st_size > _MAX_HANDOFF_BYTES
                    or _sha256(path.read_bytes()) != ref.get("sha256")):
                raise ValueError("handoff taskbook hash/path mismatch")
        taskbook = validate_gap_taskbook(json.loads((root / "taskbook/GAP_TASKBOOK.json").read_text()))
        if (taskbook["taskbook_id"] != taskbook_ref.get("taskbook_id")
                or render_gap_taskbook_markdown(taskbook) != (root / "taskbook/GAP_TASKBOOK.md").read_text()):
            raise ValueError("handoff taskbook identity/render mismatch")
        taskbook = validate_adapter_taskbook(taskbook, operator_artifact=data.get("operator_artifact"),
                                            adapter_contract=data.get("adapter_contract"))
        if taskbook["generation"]["allowed"] is not True:
            raise ValueError("handoff taskbook blocks generation; cannot wait for an agent")
    return {
        "manifest": data,
        "operator_artifact": data.get("operator_artifact"),
        "adapter_contract": data.get("adapter_contract"),
        "adapter_context_plan": data.get("adapter_context_plan"),
        "context": context,
        "skill_files": tuple(data.get("skill_files", ())),
        "skill_bundle_sha256": data.get("skill_bundle_sha256"),
        "expected_overlay_paths": tuple(paths),
        "gap_taskbook": taskbook,
    }
