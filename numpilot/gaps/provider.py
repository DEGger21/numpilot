"""Safe providers for generating an unverified Ascend C operator draft."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

from numpilot.gaps.models import (
    ARTIFACT_SCHEMA_VERSION,
    GapSpec,
    OperatorDraft,
)


class DraftGenerationError(RuntimeError):
    """The model response is invalid or cannot be published safely."""


class DraftRepairError(DraftGenerationError):
    """A repair failed after recording whether a remote response was received."""

    def __init__(self, message: str, *, receipt_count: int = 0):
        self.receipt_count = receipt_count
        self.actual_remote_call = receipt_count > 0
        super().__init__(message)


_ALLOWED_ROOT_FILES = frozenset({
    "CMakeLists.txt",
    "README.md",
    "build.sh",
})
_ALLOWED_TOP_DIRS = frozenset({
    "op_host",
    "op_kernel",
    "op_api",
    "tests",
    "asnumpy_adapter",
    "docs",
})
_ALLOWED_SUFFIXES = frozenset({
    ".asc",
    ".cmake",
    ".cpp",
    ".h",
    ".hpp",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
})
_MAX_FILES = 64
_MAX_TOTAL_BYTES = 2 * 1024 * 1024
_MAX_RESPONSE_BYTES = _MAX_TOTAL_BYTES + 256 * 1024
_MAX_SUMMARY_CHARS = 4096
_MAX_PATH_CHARS = 240
_MAX_PATH_DEPTH = 8
_MAX_TRANSPORT_ATTEMPTS = 48
_REMOTE_RECEIPT_FIELDS = frozenset({
    "schema_version",
    "utc_start",
    "utc_end",
    "http_status",
    "requested_model",
    "response_model",
    "response_id",
    "response_created",
    "request_body_sha256",
    "response_body_sha256",
    "content_sha256",
    "request_id",
    "usage",
    "tls_verification",
})
_GENERATION_FIELDS = frozenset({
    "mode",
    "provider",
    "model",
    "configuration_source",
    "origin_action",
    "attempt_count",
    "transport_attempt_count",
    "successful_http_response_count",
    "content_response_count",
    "response_sha256",
    "remote_receipts",
    "local_validation",
    "fallback",
})
_ARTIFACT_FIELDS = frozenset({
    "schema_version",
    "artifact_id",
    "gap_id",
    "workspace",
    "summary",
    "status",
    "files",
    "generation",
    "gates",
})
_SAFE_RECEIPT_TEXT = re.compile(r"^[A-Za-z0-9._:?-]{1,512}$")
_EXPECTED_UNVERIFIED_GATES = {
    "spec": "prepared",
    "generation": "passed",
    "build": "pending",
    "unit_test": "pending",
    "numerical_validation": "pending",
    "activation": "pending",
}

_SYSTEM_PROMPT = """You generate an UNVERIFIED Ascend C registry-invoke operator draft.
Return exactly one NumPilot line bundle, without Markdown fences or commentary:
NUMPILOT_BUNDLE_V1
SUMMARY: one-line summary
FILE: relative/path
CONTENT:
raw file content (do not JSON-escape quotes or newlines)
END_FILE
END_BUNDLE

Follow these rules:
1. Target only the SoC/CANN version in the supplied GapSpec.
2. Use the registry-invoke structure: CMakeLists.txt, build.sh, op_host,
   op_kernel, op_api, tests, and asnumpy_adapter.
3. Kernel parameters are inputs, outputs, workspace, tiling in that order.
4. Include shape/dtype checks, tiling, CPU golden tests, ACLNN two-stage API,
   and an explicit AsNumpy name mapping manifest.
5. Do not invent undocumented Ascend C or ACLNN APIs. If an API cannot be
   verified from the supplied contract, leave a compile-failing
   TODO_VERIFY_CANN_API marker and explain it in README.md.
6. Do not access the network, install packages, alter system paths, or embed
   credentials. build.sh may build only this workspace.
7. This output is a draft. Never claim build, unit test, numerical validation,
   installation, or NPU execution succeeded.
8. Produce a complete, self-contained, build-ready local candidate with at most
   17 files and below 14000 characters. Include all build/host/kernel/API/test/
   AsNumpy-adapter files required by the supplied official Skill. Never put a
   line equal to END_FILE inside file content.
9. Do not emit TODO_VERIFY_CANN_API, placeholder kernels, stub success/failure
   returns, or defer missing implementation to a later model call. The local
   build and qualification gates will reject any unverified API usage.
10. When the GapSpec selects direct generation, do not fetch, clone, copy, or
    require an external operator implementation repository. Generate against
    the embedded Skill and the semantic contract, and keep all output local.
"""

_STAGED_SYSTEM_PROMPT = """You generate one bounded stage of an UNVERIFIED
Ascend C registry-invoke operator project. Return exactly one
NUMPILOT_BUNDLE_V1 line bundle and exactly the paths requested by the user.
The literal envelope is:
NUMPILOT_BUNDLE_V1
SUMMARY: one short stage summary
FILE: relative/path
CONTENT:
raw complete file content
END_FILE
(repeat FILE/CONTENT/END_FILE for every requested path)
END_BUNDLE

Use the supplied NumPy oracle as semantic truth and the selected official
Huawei CANNBot files as engineering truth. Emit complete source: no TODO,
placeholder kernel, stub return, external repository dependency, network
operation, system install, or success claim. Never put a line equal to
END_FILE inside file content.
"""

_REPAIR_SYSTEM_PROMPT = """You repair exactly one file in an UNVERIFIED
Ascend C registry-invoke operator candidate. The local CANN compiler diagnostic
is authoritative. The supplied NumPy oracle is semantic truth and the selected
official Huawei CANNBot files are engineering truth.

Return exactly one NUMPILOT_BUNDLE_V1 line bundle containing only the requested
path. The literal envelope is:
NUMPILOT_BUNDLE_V1
SUMMARY: one short repair summary
FILE: the/exact/requested/path
CONTENT:
raw complete replacement file
END_FILE
END_BUNDLE

Emit the complete replacement file, not a diff. Preserve correct behavior
outside the reported failure. Do not emit TODO, placeholders, stub returns,
external repository dependencies, network operations, system installation, or
success claims. Never put a line equal to END_FILE inside file content.
"""

_SKILL_PREFIX = "ops/ascendc-registry-invoke-template"


def _strip_json_fence(payload: str) -> str:
    text = payload.strip()
    fence = chr(96) * 3
    if not text.startswith(fence):
        return text
    first_newline = text.find("\n")
    if first_newline < 0 or not text.endswith(fence):
        return text
    return text[first_newline + 1:-len(fence)].strip()


def _safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise DraftGenerationError("draft file path must be a non-empty string")
    if len(value) > _MAX_PATH_CHARS or any(
        ord(character) < 32 for character in value
    ):
        raise DraftGenerationError("draft file path is too long or unsafe")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise DraftGenerationError("unsafe draft path: %r" % value)
    if not path.parts:
        raise DraftGenerationError("empty draft path")
    if len(path.parts) > _MAX_PATH_DEPTH:
        raise DraftGenerationError("draft path is too deep")
    if len(path.parts) == 1:
        if path.name not in _ALLOWED_ROOT_FILES:
            raise DraftGenerationError("unexpected root draft file: %s" % path)
    elif path.parts[0] not in _ALLOWED_TOP_DIRS:
        raise DraftGenerationError("unexpected draft directory: %s" % path)
    if path.name not in _ALLOWED_ROOT_FILES and path.suffix not in _ALLOWED_SUFFIXES:
        raise DraftGenerationError("unsupported draft file type: %s" % path)
    return path


def _parse_line_bundle(payload: str) -> Tuple[str, List[Dict[str, str]]]:
    text = payload.strip().replace("\r\n", "\n").replace("\r", "\n")
    prefix = "NUMPILOT_BUNDLE_V1\nSUMMARY: "
    if not text.startswith(prefix):
        raise DraftGenerationError("model output has no NumPilot bundle header")
    summary_end = text.find("\n", len(prefix))
    if summary_end < 0:
        raise DraftGenerationError("model output has no bounded summary")
    summary = text[len(prefix):summary_end]
    cursor = summary_end + 1
    files: List[Dict[str, str]] = []
    while True:
        if text[cursor:] == "END_BUNDLE":
            break
        if not text.startswith("FILE: ", cursor):
            raise DraftGenerationError("model line bundle is missing FILE marker")
        path_end = text.find("\n", cursor)
        if path_end < 0:
            raise DraftGenerationError("model line bundle has an incomplete path")
        path = text[cursor + len("FILE: "):path_end]
        content_start_marker = "CONTENT:\n"
        content_marker_start = path_end + 1
        if not text.startswith(content_start_marker, content_marker_start):
            raise DraftGenerationError("model line bundle is missing CONTENT marker")
        content_start = content_marker_start + len(content_start_marker)
        boundary = "\nEND_FILE\n"
        content_end = text.find(boundary, content_start)
        if content_end < 0:
            raise DraftGenerationError("model line bundle has an unterminated file")
        files.append({"path": path, "content": text[content_start:content_end]})
        cursor = content_end + len(boundary)
    return summary, files


def _validate_bundle_files(
    payload: str,
) -> Tuple[str, Tuple[Tuple[PurePosixPath, str], ...]]:
    if not isinstance(payload, str):
        raise DraftGenerationError("model output must be text")
    if len(payload.encode("utf-8")) > _MAX_RESPONSE_BYTES:
        raise DraftGenerationError("model output exceeds the response size limit")
    stripped = _strip_json_fence(payload)
    if stripped.startswith("NUMPILOT_BUNDLE_V1"):
        summary, raw_files = _parse_line_bundle(stripped)
        data = {"summary": summary, "files": raw_files}
    else:
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DraftGenerationError(
                "model output is neither a line bundle nor valid JSON: %s; "
                "position=%d; characters=%d; utf8_bytes=%d"
                % (exc.msg, exc.pos, len(payload), len(payload.encode("utf-8")))
            ) from exc
        if not isinstance(data, dict):
            raise DraftGenerationError("model output must be a JSON object")
        summary = data.get("summary", "")
    if not isinstance(summary, str) or len(summary) > _MAX_SUMMARY_CHARS:
        raise DraftGenerationError("summary must be a bounded string")
    raw_files = data.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise DraftGenerationError("files must be a non-empty list")
    if len(raw_files) > _MAX_FILES:
        raise DraftGenerationError("too many generated files")

    seen = set()
    files: List[Tuple[PurePosixPath, str]] = []
    total = 0
    for item in raw_files:
        if not isinstance(item, dict):
            raise DraftGenerationError("each generated file must be an object")
        path = _safe_relative_path(item.get("path"))
        content = item.get("content")
        if not isinstance(content, str):
            raise DraftGenerationError("file content must be a string: %s" % path)
        if str(path) in seen:
            raise DraftGenerationError("duplicate generated path: %s" % path)
        seen.add(str(path))
        total += len(content.encode("utf-8"))
        if total > _MAX_TOTAL_BYTES:
            raise DraftGenerationError("generated draft exceeds size limit")
        files.append((path, content))

    forbidden_markers = (
        "TODO_VERIFY_CANN_API",
        "TODO_IMPLEMENT_OPERATOR",
    )
    for path, content in files:
        if any(marker in content for marker in forbidden_markers):
            raise DraftGenerationError(
                "generated draft contains a forbidden placeholder marker: %s"
                % path
            )

    return summary.strip(), tuple(files)


def _validate_bundle(payload: str) -> Tuple[str, Tuple[Tuple[PurePosixPath, str], ...]]:
    summary, files = _validate_bundle_files(payload)
    names = {str(path) for path, _ in files}
    for required in ("CMakeLists.txt", "build.sh"):
        if required not in names:
            raise DraftGenerationError("generated draft is missing %s" % required)
    for required_dir in (
        "op_host",
        "op_kernel",
        "op_api",
        "tests",
        "asnumpy_adapter",
    ):
        if not any(path.parts[0] == required_dir for path, _ in files):
            raise DraftGenerationError(
                "generated draft is missing %s/" % required_dir
            )
    return summary, files


def _validate_stage_bundle(
    payload: str,
    required_paths: Tuple[str, ...],
) -> Tuple[str, Tuple[Tuple[PurePosixPath, str], ...]]:
    summary, files = _validate_bundle_files(payload)
    actual = {str(path) for path, _content in files}
    required = set(required_paths)
    if len(files) != len(required_paths) or actual != required:
        missing = sorted(required - actual)
        unexpected = sorted(actual - required)
        raise DraftGenerationError(
            "staged response file set mismatch; missing=%s unexpected=%s"
            % (missing, unexpected)
        )
    return summary, files


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_int(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = 2 ** 63 - 1,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise DraftGenerationError("%s is invalid" % name)
    return value


def _parse_utc_receipt_time(value: object, name: str) -> datetime:
    if (
        not isinstance(value, str)
        or not value.endswith("Z")
        or len(value) > 64
    ):
        raise DraftGenerationError(
            "remote model receipt field is invalid: %s" % name
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise DraftGenerationError(
            "remote model receipt field is invalid: %s" % name
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DraftGenerationError(
            "remote model receipt field is invalid: %s" % name
        )
    return parsed


_RESPONSE_NOT_PROVIDED = object()


def _validate_remote_receipt(
    receipt: object, response: object = _RESPONSE_NOT_PROVIDED
) -> Dict[str, object]:
    if not isinstance(receipt, dict) or frozenset(receipt) != _REMOTE_RECEIPT_FIELDS:
        raise DraftGenerationError("remote model call returned an incomplete receipt")
    if receipt.get("schema_version") != 1 or isinstance(
        receipt.get("schema_version"), bool
    ):
        raise DraftGenerationError("remote model receipt schema is invalid")
    status = receipt.get("http_status")
    if (
        isinstance(status, bool)
        or not isinstance(status, int)
        or status < 200
        or status >= 300
    ):
        raise DraftGenerationError("remote model receipt HTTP status is invalid")
    for name in ("request_body_sha256", "response_body_sha256"):
        if not _is_sha256(receipt.get(name)):
            raise DraftGenerationError("remote model receipt hash is invalid: %s" % name)
    if response is _RESPONSE_NOT_PROVIDED:
        if (
            receipt.get("content_sha256") is not None
            and not _is_sha256(receipt.get("content_sha256"))
        ):
            raise DraftGenerationError(
                "remote model receipt content hash is invalid"
            )
    elif response is None:
        if receipt.get("content_sha256") is not None:
            raise DraftGenerationError(
                "failed remote model receipt claims response content"
            )
    elif (
        not isinstance(response, str)
        or not _is_sha256(receipt.get("content_sha256"))
        or receipt["content_sha256"] != _sha256_bytes(response.encode("utf-8"))
    ):
        raise DraftGenerationError("remote model receipt content hash mismatch")
    started_at = _parse_utc_receipt_time(receipt.get("utc_start"), "utc_start")
    ended_at = _parse_utc_receipt_time(receipt.get("utc_end"), "utc_end")
    if ended_at < started_at:
        raise DraftGenerationError("remote model receipt time range is invalid")
    for name in ("requested_model", "response_model", "response_id", "request_id"):
        if name == "requested_model" and receipt.get(name) is None:
            raise DraftGenerationError(
                "remote model receipt field is invalid: %s" % name
            )
        if receipt.get(name) is not None and (
            not isinstance(receipt[name], str)
            or not _SAFE_RECEIPT_TEXT.fullmatch(receipt[name])
        ):
            raise DraftGenerationError("remote model receipt field is invalid: %s" % name)
    created = receipt.get("response_created")
    if created is not None and (
        isinstance(created, bool)
        or not isinstance(created, int)
        or created < 0
        or created > 2 ** 63 - 1
    ):
        raise DraftGenerationError("remote model receipt created value is invalid")
    usage = receipt.get("usage")
    if (
        not isinstance(usage, dict)
        or len(usage) > 32
        or any(
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", name)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > 2 ** 63 - 1
            for name, value in usage.items()
        )
    ):
        raise DraftGenerationError("remote model receipt usage is invalid")
    if receipt.get("tls_verification") not in {"enabled", "disabled"}:
        raise DraftGenerationError("remote model receipt TLS state is invalid")
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    if len(serialized.encode("utf-8")) > 64 * 1024:
        raise DraftGenerationError("remote model receipt exceeds the size limit")
    for env_name in ("NUMPILOT_LLM_AUTH_TOKEN", "NUMPILOT_LLM_BASE_URL"):
        sensitive = os.getenv(env_name, "")
        if sensitive and sensitive in serialized:
            raise DraftGenerationError("remote model receipt contains sensitive data")
    return dict(receipt)


def _validate_generation(generation: object) -> Dict[str, object]:
    if (
        not isinstance(generation, dict)
        or frozenset(generation) != _GENERATION_FIELDS
    ):
        raise DraftGenerationError(
            "existing draft generation evidence schema is invalid"
        )

    mode = generation.get("mode")
    if mode not in {"remote_llm", "injected_test_double"}:
        raise DraftGenerationError("existing draft generation mode is invalid")
    for name in ("provider", "model", "configuration_source"):
        value = generation.get(name)
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 512
            or any(ord(character) < 32 for character in value)
        ):
            raise DraftGenerationError(
                "existing draft generation field is invalid: %s" % name
            )
    if generation.get("local_validation") != "passed":
        raise DraftGenerationError(
            "existing draft must prove local validation passed"
        )
    if generation.get("fallback") != "disabled":
        raise DraftGenerationError("existing draft must prove fallback was disabled")

    attempt_count = _require_exact_int(
        generation.get("attempt_count"),
        "existing draft attempt count",
        minimum=1,
        maximum=_MAX_TRANSPORT_ATTEMPTS,
    )
    transport_count = _require_exact_int(
        generation.get("transport_attempt_count"),
        "existing draft transport attempt count",
        minimum=1,
        maximum=_MAX_TRANSPORT_ATTEMPTS,
    )
    successful_http_count = _require_exact_int(
        generation.get("successful_http_response_count"),
        "existing draft successful HTTP response count",
        maximum=_MAX_TRANSPORT_ATTEMPTS,
    )
    content_count = _require_exact_int(
        generation.get("content_response_count"),
        "existing draft content response count",
        maximum=_MAX_TRANSPORT_ATTEMPTS,
    )
    if transport_count != attempt_count:
        raise DraftGenerationError(
            "existing draft transport/attempt counts do not match"
        )

    response_hashes = generation.get("response_sha256")
    if (
        not isinstance(response_hashes, list)
        or not response_hashes
        or len(response_hashes) != content_count
        or content_count > transport_count
        or successful_http_count > transport_count
        or any(not _is_sha256(item) for item in response_hashes)
    ):
        raise DraftGenerationError(
            "existing draft response hashes/counters are inconsistent"
        )

    raw_receipts = generation.get("remote_receipts")
    if (
        not isinstance(raw_receipts, list)
        or len(raw_receipts) > _MAX_TRANSPORT_ATTEMPTS
    ):
        raise DraftGenerationError("existing draft remote receipts are invalid")
    receipts = [_validate_remote_receipt(item) for item in raw_receipts]

    normalized = dict(generation)
    normalized["response_sha256"] = list(response_hashes)
    normalized["remote_receipts"] = receipts
    if mode == "remote_llm":
        if generation.get("origin_action") != "fresh_remote":
            raise DraftGenerationError(
                "existing remote draft origin action is invalid"
            )
        if (
            successful_http_count != len(receipts)
            or content_count > successful_http_count
            or not receipts
        ):
            raise DraftGenerationError(
                "existing remote draft receipt count is inconsistent"
            )
        receipt_content_hashes = [
            item["content_sha256"]
            for item in receipts
            if item["content_sha256"] is not None
        ]
        if receipt_content_hashes != response_hashes:
            raise DraftGenerationError(
                "existing remote draft receipt/content hashes are inconsistent"
            )
        if any(
            item["requested_model"] != generation["model"]
            for item in receipts
        ):
            raise DraftGenerationError(
                "existing remote draft receipt model mismatch"
            )
    else:
        if (
            generation.get("origin_action") != "injected_test_double"
            or generation.get("configuration_source") != "injected"
            or receipts
            or successful_http_count != 0
        ):
            raise DraftGenerationError(
                "existing injected draft origin evidence is invalid"
            )
    return normalized


def _require_project_layout(names) -> None:
    names = set(names)
    for required in ("CMakeLists.txt", "build.sh"):
        if required not in names:
            raise DraftGenerationError(
                "draft manifest is missing %s" % required
            )
    for required_dir in (
        "op_host",
        "op_kernel",
        "op_api",
        "tests",
        "asnumpy_adapter",
    ):
        if not any(
            PurePosixPath(name).parts[0] == required_dir
            for name in names
        ):
            raise DraftGenerationError(
                "draft manifest is missing %s/" % required_dir
            )


def _draft_artifact_id(
    gap_id: str,
    files: Dict[str, str],
    generation: Dict[str, object],
) -> str:
    identity = json.dumps(
        {
            "gap_id": gap_id,
            "files": files,
            "generation": generation,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "draft-" + hashlib.sha256(identity).hexdigest()[:20]


def _load_existing_draft(draft_dir: Path, spec: GapSpec) -> OperatorDraft:
    manifest_path = draft_dir / "operator-artifact.json"
    if draft_dir.is_symlink() or manifest_path.is_symlink():
        raise DraftGenerationError("existing draft must not contain symlinks")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DraftGenerationError(
            "existing draft has no valid operator-artifact.json"
        ) from exc
    if not isinstance(data, dict):
        raise DraftGenerationError("existing draft manifest must be an object")
    if frozenset(data) != _ARTIFACT_FIELDS:
        raise DraftGenerationError("existing draft manifest schema is invalid")
    schema_version = data.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != ARTIFACT_SCHEMA_VERSION
    ):
        raise DraftGenerationError("existing draft has an unsupported schema")
    if data.get("gap_id") != spec.gap_id:
        raise DraftGenerationError("existing draft belongs to another gap")
    if data.get("status") != "generated_unverified":
        raise DraftGenerationError(
            "existing draft status must remain generated_unverified"
        )
    if data.get("gates") != _EXPECTED_UNVERIFIED_GATES:
        raise DraftGenerationError(
            "existing draft gates do not match the unverified lifecycle"
        )
    if data.get("workspace") != str(draft_dir):
        raise DraftGenerationError("existing draft workspace identity mismatch")
    if (
        not isinstance(data.get("summary"), str)
        or len(data["summary"]) > _MAX_SUMMARY_CHARS
    ):
        raise DraftGenerationError("existing draft summary is invalid")
    normalized_generation = _validate_generation(data.get("generation"))
    files = data.get("files")
    if not isinstance(files, dict):
        raise DraftGenerationError("existing draft file manifest is invalid")
    normalized_files: Dict[str, str] = {}
    for relative, expected in files.items():
        path_key = str(_safe_relative_path(relative))
        if not _is_sha256(expected):
            raise DraftGenerationError(
                "existing draft digest is invalid: %s" % path_key
            )
        normalized_files[path_key] = expected
    if len(normalized_files) != len(files):
        raise DraftGenerationError("existing draft has duplicate normalized paths")
    actual_names = set()
    try:
        candidates = tuple(draft_dir.rglob("*"))
    except OSError as exc:
        raise DraftGenerationError("cannot enumerate existing draft") from exc
    for candidate in candidates:
        if candidate.is_symlink():
            raise DraftGenerationError(
                "existing draft contains a symlink: %s"
                % candidate.relative_to(draft_dir).as_posix()
            )
        try:
            mode = candidate.stat().st_mode
        except OSError as exc:
            raise DraftGenerationError(
                "cannot inspect existing draft file"
            ) from exc
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise DraftGenerationError("existing draft contains a special file")
        relative = candidate.relative_to(draft_dir).as_posix()
        if relative != "operator-artifact.json":
            actual_names.add(relative)
    if actual_names != set(normalized_files):
        raise DraftGenerationError(
            "existing draft file set differs from its manifest"
        )
    _require_project_layout(normalized_files)

    for relative, expected in normalized_files.items():
        path = draft_dir / _safe_relative_path(relative)
        try:
            actual = _sha256_bytes(path.read_bytes())
        except OSError as exc:
            raise DraftGenerationError(
                "existing draft file is missing: %s" % relative
            ) from exc
        if actual != expected:
            raise DraftGenerationError(
                "existing draft hash mismatch: %s" % relative
            )
    artifact_id = _draft_artifact_id(
        spec.gap_id,
        normalized_files,
        normalized_generation,
    )
    if data.get("artifact_id") != artifact_id:
        raise DraftGenerationError("existing draft artifact_id mismatch")
    return OperatorDraft(
        schema_version=schema_version,
        artifact_id=artifact_id,
        gap_id=str(data["gap_id"]),
        workspace=str(draft_dir),
        summary=str(data.get("summary", "")),
        status="generated_unverified",
        files=normalized_files,
        generation=normalized_generation,
        gates=dict(_EXPECTED_UNVERIFIED_GATES),
    )


def _load_repair_candidate(
    candidate_dir: Path,
    spec: GapSpec,
    signed_source: OperatorDraft,
) -> Tuple[Dict[str, str], str, Dict[str, object]]:
    """Verify a repaired overlay and return its complete current file hashes.

    The signed ``operator-artifact.json`` remains immutable in every repair
    candidate.  The repair report therefore acts as a hash-bound overlay over
    that original artifact.  Newer reports carry the complete candidate file
    map; schema-v1 reports produced before repair chaining are reconstructed
    from the original manifest plus ``after_sha256``.
    """

    candidate_dir = Path(candidate_dir)
    report_path = candidate_dir / "numpilot-repair-report.json"
    artifact_path = candidate_dir / "operator-artifact.json"
    if (
        candidate_dir.is_symlink()
        or report_path.is_symlink()
        or artifact_path.is_symlink()
    ):
        raise DraftGenerationError("repair base must not contain symlinks")
    try:
        report_payload = report_path.read_bytes()
        if len(report_payload) > 2 * 1024 * 1024:
            raise DraftGenerationError("repair base report exceeds the size limit")
        report = json.loads(report_payload.decode("utf-8"))
        artifact_payload = artifact_path.read_bytes()
        signed_artifact_payload = (
            Path(signed_source.workspace) / "operator-artifact.json"
        ).read_bytes()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DraftGenerationError(
            "repair base has no valid repair report or signed artifact"
        ) from exc
    if artifact_payload != signed_artifact_payload:
        raise DraftGenerationError("repair base changed the signed artifact manifest")
    if not isinstance(report, dict):
        raise DraftGenerationError("repair base report must be an object")
    if (
        isinstance(report.get("schema_version"), bool)
        or not isinstance(report.get("schema_version"), int)
        or report.get("schema_version") != 1
        or report.get("status") != "repaired_unverified"
        or report.get("source_artifact_id") != signed_source.artifact_id
        or report.get("gap_id") != spec.gap_id
        or report.get("gates") != _EXPECTED_UNVERIFIED_GATES
        or report.get("actual_remote_call") is not True
    ):
        raise DraftGenerationError("repair base report identity is invalid")

    paths = report.get("paths")
    before = report.get("before_sha256")
    after = report.get("after_sha256")
    if (
        not isinstance(paths, list)
        or not paths
        or len(paths) > 16
        or not isinstance(before, dict)
        or not isinstance(after, dict)
    ):
        raise DraftGenerationError("repair base hash overlay is invalid")
    normalized_paths = tuple(str(_safe_relative_path(path)) for path in paths)
    if (
        len(set(normalized_paths)) != len(normalized_paths)
        or set(before) != set(normalized_paths)
        or set(after) != set(normalized_paths)
        or any(path not in signed_source.files for path in normalized_paths)
        or any(not _is_sha256(value) for value in before.values())
        or any(not _is_sha256(value) for value in after.values())
    ):
        raise DraftGenerationError("repair base path hashes are invalid")

    candidate_files = report.get("candidate_files_sha256")
    if candidate_files is None:
        normalized_files = dict(signed_source.files)
        normalized_files.update(after)
    else:
        if not isinstance(candidate_files, dict):
            raise DraftGenerationError("repair base complete file map is invalid")
        normalized_files = {}
        for relative, digest in candidate_files.items():
            path = str(_safe_relative_path(relative))
            if path in normalized_files:
                raise DraftGenerationError(
                    "repair base complete file map has duplicate paths"
                )
            if not _is_sha256(digest):
                raise DraftGenerationError(
                    "repair base digest is invalid: %s" % path
                )
            normalized_files[path] = digest
        if set(normalized_files) != set(signed_source.files):
            raise DraftGenerationError(
                "repair base complete file map differs from signed artifact"
            )
        if any(normalized_files[path] != after[path] for path in normalized_paths):
            raise DraftGenerationError(
                "repair base overlay differs from its complete file map"
            )

    actual_names = set()
    try:
        candidates = tuple(candidate_dir.rglob("*"))
    except OSError as exc:
        raise DraftGenerationError("cannot enumerate repair base") from exc
    for candidate in candidates:
        if candidate.is_symlink():
            raise DraftGenerationError(
                "repair base contains a symlink: %s"
                % candidate.relative_to(candidate_dir).as_posix()
            )
        try:
            mode = candidate.stat().st_mode
        except OSError as exc:
            raise DraftGenerationError("cannot inspect repair base file") from exc
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise DraftGenerationError("repair base contains a special file")
        relative = candidate.relative_to(candidate_dir).as_posix()
        if relative not in {
            "operator-artifact.json",
            "numpilot-repair-report.json",
        }:
            actual_names.add(relative)
    if actual_names != set(normalized_files):
        raise DraftGenerationError("repair base file set differs from its report")
    for relative, expected in normalized_files.items():
        try:
            actual = _sha256_bytes(
                (candidate_dir / _safe_relative_path(relative)).read_bytes()
            )
        except OSError as exc:
            raise DraftGenerationError(
                "repair base file is missing: %s" % relative
            ) from exc
        if actual != expected:
            raise DraftGenerationError(
                "repair base hash mismatch: %s" % relative
            )
    return normalized_files, _sha256_bytes(report_payload), dict(report)


def _load_repair_chain(
    candidate_dir: Path,
    spec: GapSpec,
    signed_source: OperatorDraft,
) -> Tuple[Dict[str, str], Tuple[Dict[str, object], ...]]:
    """Validate a complete parent-linked repair chain, oldest report first."""

    current_dir = Path(candidate_dir)
    seen = set()
    reverse_reports = []
    final_files = None
    while True:
        resolved = str(current_dir.resolve(strict=False))
        if resolved in seen:
            raise DraftGenerationError("repair chain contains a directory cycle")
        seen.add(resolved)
        current_files, current_hash, report = _load_repair_candidate(
            current_dir,
            spec,
            signed_source,
        )
        if final_files is None:
            final_files = current_files
        reverse_reports.append(dict(report, report_sha256=current_hash))

        parent_hash = report.get("parent_repair_report_sha256")
        source_workspace = report.get("source_workspace")
        if not isinstance(source_workspace, str) or not source_workspace:
            raise DraftGenerationError("repair chain source workspace is invalid")
        if parent_hash is None:
            if source_workspace != signed_source.workspace:
                raise DraftGenerationError(
                    "first repair does not descend from the signed artifact"
                )
            for path, before_hash in report["before_sha256"].items():
                if signed_source.files[path] != before_hash:
                    raise DraftGenerationError(
                        "first repair before hash differs from signed artifact"
                    )
            break
        if not _is_sha256(parent_hash):
            raise DraftGenerationError("repair chain parent hash is invalid")

        parent_dir = Path(source_workspace)
        parent_files, actual_parent_hash, _parent_report = _load_repair_candidate(
            parent_dir,
            spec,
            signed_source,
        )
        if actual_parent_hash != parent_hash:
            raise DraftGenerationError("repair chain parent report hash mismatch")
        for path, before_hash in report["before_sha256"].items():
            if parent_files[path] != before_hash:
                raise DraftGenerationError(
                    "repair chain before hash differs from its parent"
                )
        expected_current = dict(parent_files)
        expected_current.update(report["after_sha256"])
        if expected_current != current_files:
            raise DraftGenerationError(
                "repair chain current files do not derive from its parent"
            )
        current_dir = parent_dir

    return dict(final_files or {}), tuple(reversed(reverse_reports))


def _direct_generation_stages(name: str):
    """Return the fixed, auditable stage/file/reference plan."""

    add_root = _SKILL_PREFIX + "/references/add_example/"
    groups = (
        (
            "build",
            (
                "CMakeLists.txt",
                "build.sh",
                "op_host/CMakeLists.txt",
                "op_kernel/CMakeLists.txt",
            ),
            (
                _SKILL_PREFIX + "/SKILL.md",
                _SKILL_PREFIX + "/references/build-deploy-guide.md",
                add_root + "CMakeLists.txt",
                add_root + "build.sh",
                add_root + "op_host/CMakeLists.txt",
                add_root + "op_kernel/CMakeLists.txt",
            ),
        ),
        (
            "host",
            (
                "op_host/%s_def.cpp" % name,
                "op_host/%s_infershape.cpp" % name,
                "op_host/arch22/%s_tiling.cpp" % name,
            ),
            (
                _SKILL_PREFIX + "/SKILL.md",
                _SKILL_PREFIX + "/references/basic-guide.md",
                add_root + "op_host/add_example_def.cpp",
                add_root + "op_host/add_example_infershape.cpp",
                add_root + "op_host/arch22/add_example_tiling.cpp",
            ),
        ),
        (
            "kernel",
            (
                "op_kernel/%s_arch22.cpp" % name,
                "op_kernel/arch22/%s.h" % name,
                "op_kernel/arch22/%s_tiling_data.h" % name,
                "op_kernel/arch22/%s_tiling_key.h" % name,
            ),
            (
                _SKILL_PREFIX + "/SKILL.md",
                _SKILL_PREFIX + "/references/basic-guide.md",
                add_root + "op_kernel/add_example_arch22.cpp",
                add_root + "op_kernel/arch22/add_example.h",
                add_root + "op_kernel/arch22/add_example_tiling_data.h",
                add_root + "op_kernel/arch22/add_example_tiling_key.h",
            ),
        ),
        (
            "api",
            (
                "op_api/aclnn_%s.h" % name,
                "op_api/aclnn_%s.cpp" % name,
                "op_api/%s.h" % name,
                "op_api/%s.cpp" % name,
            ),
            (
                _SKILL_PREFIX + "/SKILL.md",
                add_root + "op_api/aclnn_add_example.h",
                add_root + "op_api/aclnn_add_example.cpp",
                add_root + "op_api/add_example.h",
                add_root + "op_api/add_example.cpp",
            ),
        ),
        (
            "test_adapter",
            (
                "tests/test_%s.py" % name,
                "asnumpy_adapter/%s.json" % name,
            ),
            (
                _SKILL_PREFIX + "/SKILL.md",
            ),
        ),
    )
    return tuple(
        (
            "%s_%02d" % (group_name, file_index),
            (path,),
            references,
        )
        for group_name, paths, references in groups
        for file_index, path in enumerate(paths, start=1)
    )


def _compact_generation_contract(request: str) -> str:
    marker = "\n## Embedded official Huawei CANNBot skill\n"
    return request.split(marker, 1)[0].strip()


def _stage_invariants(spec: GapSpec, path: str) -> str:
    """Render deterministic cross-file constraints for every model stage."""

    signatures = sorted({
        (
            int(item.get("positional_args", 0)),
            tuple(str(name) for name in item.get("keyword_args", ())),
        )
        for item in spec.signature_variants
    })
    signature_text = ", ".join(
        "%d positional; keywords=%s" % (
            positional,
            list(keywords),
        )
        for positional, keywords in signatures
    )
    rules = [
        "Project-wide invariants (mandatory):",
        "- C++ OpType is `%s`; file/API stem is `%s`; public API is `%s`." % (
            "".join(part.capitalize() for part in spec.api_name.split("_")),
            spec.api_name,
            spec.asnumpy_api,
        ),
        "- Source call signatures are exactly: %s." % signature_text,
        "- Admitted dtypes are exactly: %s; do not add convenient extra dtypes." % (
            ", ".join(spec.dtype_rules),
        ),
        "- Admitted shape rules are exactly: %s." % (
            ", ".join(spec.shape_rules),
        ),
        "- Preserve the NumPy oracle parameter order. Host definition, infer-shape, "
        "tiling, kernel entry, L0/L2 ACLNN declarations/definitions, tests, and adapter "
        "must agree on input/output count, order, names, dtypes, and shapes.",
    ]
    if path.startswith("tests/"):
        rules.extend((
            "- This test file must contain pure NumPy oracle tests that collect and pass "
            "when `asnumpy` and the custom operator are absent.",
            "- Do not import `asnumpy` at module scope. Runtime/public-API tests must be "
            "separately guarded or skipped and must never disguise a missing runtime as "
            "a CPU golden-test pass.",
        ))
    if path.endswith("_tiling_key.h"):
        rules.append(
            "- Declare every template parameter used by both ASCENDC_TPL_SEL_PARAM in "
            "host tiling and the kernel entry template, using the official arch22 pattern."
        )
    return "\n".join(rules)


class LlmDraftProvider:
    """Generate source files but deliberately never compile or execute them."""

    def __init__(
        self,
        chat: Optional[Callable[[str, str], str]] = None,
        *,
        max_attempts: int = 3,
        provider_label: str = "injected",
        model_name: str = "test-double",
        configuration_status=None,
        configuration_loader: Optional[Callable] = None,
        skill_bundle=None,
        staged_direct: bool = False,
    ):
        if max_attempts not in (1, 2, 3):
            raise ValueError("max_attempts must be 1, 2, or 3")
        self._chat = chat
        self.max_attempts = max_attempts
        self.provider_label = provider_label
        self.model_name = model_name
        self._configuration_status = configuration_status
        self._configuration_loader = configuration_loader
        self._skill_bundle = skill_bundle
        self._staged_direct = bool(staged_direct)
        if self._staged_direct and self._skill_bundle is None:
            raise ValueError("staged direct generation requires a verified skill bundle")
        self._run_actions: Dict[str, str] = {}

    def action_for(self, draft: OperatorDraft) -> str:
        """Return what this provider instance did during the current run."""

        return self._run_actions.get(draft.workspace, "unknown")

    def _resolve_chat(self):
        if self._chat is not None:
            return self._chat, {
                "mode": "injected_test_double",
                "provider": self.provider_label,
                "model": self.model_name,
                "configuration_source": "injected",
                "origin_action": "injected_test_double",
                "remote_receipts": [],
            }
        from numpilot.engines.llm import LlmEngine
        from numpilot.llm_config import load_llm_config

        status = self._configuration_status
        if status is None:
            loader = self._configuration_loader or load_llm_config
            status = loader()
            if status is None:
                raise DraftGenerationError(
                    "LLM is not configured; run configure-llm --check"
                )
            self._configuration_status = status
        if not status.configured:
            raise DraftGenerationError(
                "LLM is not configured; run configure-llm --check"
            )
        return LlmEngine._chat_with_receipt, {
            "mode": "remote_llm",
            "provider": status.provider or "unlabeled",
            "model": status.model or os.environ["NUMPILOT_LLM_MODEL"],
            "configuration_source": status.source,
            "origin_action": "fresh_remote",
            "remote_receipts": [],
        }

    def _generate_staged_direct(
        self,
        chat,
        generation_base,
        spec: GapSpec,
        generation_request: str,
    ):
        """Generate a direct operator as bounded per-file receipt-bound stages."""

        contract = _compact_generation_contract(generation_request)
        summaries = []
        collected = []
        response_hashes = []
        remote_receipts = []
        total_attempts = 0
        stages = _direct_generation_stages(spec.api_name)
        for stage_index, (stage, paths, references) in enumerate(stages, start=1):
            failure = ""
            failures = []
            for stage_attempt in range(1, self.max_attempts + 1):
                total_attempts += 1
                prompt = "".join((
                    "Direct local generation stage %d/%d: %s.\n" % (
                        stage_index,
                        len(stages),
                        stage,
                    ),
                    "Return exactly these paths and no others:\n",
                    "\n".join("- `%s`" % path for path in paths),
                    "\n\nEvery file must be complete and mutually consistent with files "
                    "from the other deterministic stages.\n\n",
                    _stage_invariants(spec, paths[0]),
                    "\n\n",
                    contract,
                    "\n\n",
                    self._skill_bundle.prompt_context(references),
                ))
                if failure:
                    prompt += (
                        "\nThe previous response for this stage was rejected: "
                        + failure[:500]
                        + "\nReturn a corrected NUMPILOT_BUNDLE_V1 bundle."
                    )
                try:
                    result = chat(_STAGED_SYSTEM_PROMPT, prompt)
                    if not isinstance(result, tuple) or len(result) != 2:
                        raise DraftGenerationError(
                            "remote model call returned no auditable receipt"
                        )
                    response, receipt = result
                    validated_receipt = _validate_remote_receipt(receipt, response)
                    if (
                        validated_receipt["requested_model"]
                        != generation_base["model"]
                    ):
                        raise DraftGenerationError(
                            "remote model receipt requested model mismatch"
                        )
                    remote_receipts.append(validated_receipt)
                    if isinstance(response, str):
                        response_hashes.append(
                            _sha256_bytes(response.encode("utf-8"))
                        )
                    summary, files = _validate_stage_bundle(response, paths)
                    summaries.append("%s: %s" % (stage, summary))
                    collected.extend(files)
                    break
                except DraftGenerationError as exc:
                    failure = str(exc)
                    failures.append(
                        "%s attempt %d: %s"
                        % (stage, stage_attempt, failure[:500])
                    )
                except Exception as exc:
                    from numpilot.engines.llm import LlmResponseError

                    if isinstance(exc, LlmResponseError):
                        failed_receipt = _validate_remote_receipt(exc.receipt, None)
                        if (
                            failed_receipt["requested_model"]
                            != generation_base["model"]
                        ):
                            raise DraftGenerationError(
                                "remote model receipt requested model mismatch"
                            ) from None
                        remote_receipts.append(failed_receipt)
                    status_code = getattr(exc, "status_code", None)
                    if isinstance(status_code, int) and not isinstance(
                        status_code, bool
                    ):
                        failure = (
                            "remote model call failed: HTTP status %d" % status_code
                        )
                    else:
                        failure = (
                            "remote model call failed: %s" % type(exc).__name__
                        )
                    failures.append(
                        "%s attempt %d: %s"
                        % (stage, stage_attempt, failure[:500])
                    )
            else:
                raise DraftGenerationError(
                    "staged model generation failed after %d total request(s): %s"
                    % (total_attempts, "; ".join(failures))
                )

        names = [str(path) for path, _content in collected]
        if len(names) != len(set(names)):
            raise DraftGenerationError("staged generation produced duplicate files")
        total_bytes = sum(
            len(content.encode("utf-8")) for _path, content in collected
        )
        if total_bytes > _MAX_TOTAL_BYTES:
            raise DraftGenerationError("staged generation exceeds the size limit")
        _require_project_layout({name: "" for name in names})
        return (
            "; ".join(summaries)[:_MAX_SUMMARY_CHARS],
            tuple(collected),
            response_hashes,
            remote_receipts,
            total_attempts,
        )

    def repair_to_candidate(
        self,
        spec: GapSpec,
        draft_dir: Path,
        candidate_dir: Path,
        *,
        paths: Tuple[str, ...],
        diagnostic: str,
        generation_request: str,
        base_candidate_dir: Optional[Path] = None,
    ) -> Dict[str, object]:
        """Repair selected files into a new candidate without mutating a draft."""

        if self._skill_bundle is None:
            raise DraftGenerationError("LLM repair requires a verified skill bundle")
        draft_dir = Path(draft_dir)
        source = _load_existing_draft(draft_dir, spec)
        repair_source_dir = draft_dir
        current_files = dict(source.files)
        parent_repair_sha256 = None
        if base_candidate_dir is not None:
            repair_source_dir = Path(base_candidate_dir)
            current_files, parent_repair_sha256, _parent_report = _load_repair_candidate(
                repair_source_dir,
                spec,
                source,
            )
        candidate_dir = Path(candidate_dir)
        if os.path.lexists(candidate_dir):
            raise DraftGenerationError("repair candidate path already exists")
        if not isinstance(diagnostic, str) or not diagnostic.strip():
            raise DraftGenerationError("repair diagnostic must be non-empty")
        diagnostic_bytes = diagnostic.encode("utf-8")
        if len(diagnostic_bytes) > 64 * 1024 or "\x00" in diagnostic:
            raise DraftGenerationError("repair diagnostic is invalid or too large")
        if not paths or len(paths) > 16 or len(set(paths)) != len(paths):
            raise DraftGenerationError("repair path set is empty, duplicated, or too large")

        stage_by_path = {}
        for _stage, stage_paths, references in _direct_generation_stages(
            spec.api_name
        ):
            for stage_path in stage_paths:
                stage_by_path[stage_path] = (stage_paths, references)
        normalized_paths = tuple(str(_safe_relative_path(path)) for path in paths)
        for path in normalized_paths:
            if path not in current_files or path not in stage_by_path:
                raise DraftGenerationError(
                    "repair path is not part of the signed stage plan: %s" % path
                )

        chat, generation_base = self._resolve_chat()
        contract = _compact_generation_contract(generation_request)
        collected = {}
        receipts = []
        response_hashes = []
        total_attempts = 0

        for repair_index, path in enumerate(normalized_paths, start=1):
            related_paths, references = stage_by_path[path]
            related_sections = []
            for related in related_paths:
                related_file = repair_source_dir / _safe_relative_path(related)
                try:
                    content = related_file.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    raise DraftGenerationError(
                        "cannot read repair context file: %s" % related
                    ) from exc
                related_sections.append(
                    "CURRENT FILE `%s`:\n%s" % (related, content)
                )

            failure = ""
            failures = []
            for repair_attempt in range(1, self.max_attempts + 1):
                total_attempts += 1
                prompt = "\n\n".join((
                    "Repair file %d/%d. Return exactly `%s` and no other path."
                    % (repair_index, len(normalized_paths), path),
                    _stage_invariants(spec, path),
                    "Local CANN compiler/test diagnostic (authoritative):\n"
                    + diagnostic.strip(),
                    contract,
                    "Related current project files (context only; return only the "
                    "requested path):\n\n" + "\n\n".join(related_sections),
                    self._skill_bundle.prompt_context(references),
                ))
                if failure:
                    prompt += (
                        "\n\nThe previous repair response was rejected: "
                        + failure[:500]
                    )
                try:
                    result = chat(_REPAIR_SYSTEM_PROMPT, prompt)
                    if not isinstance(result, tuple) or len(result) != 2:
                        raise DraftGenerationError(
                            "remote repair call returned no auditable receipt"
                        )
                    response, receipt = result
                    validated_receipt = _validate_remote_receipt(receipt, response)
                    if validated_receipt["requested_model"] != generation_base["model"]:
                        raise DraftGenerationError(
                            "remote repair receipt requested model mismatch"
                        )
                    receipts.append(validated_receipt)
                    response_hashes.append(
                        _sha256_bytes(response.encode("utf-8"))
                    )
                    _summary, files = _validate_stage_bundle(response, (path,))
                    collected[path] = files[0][1]
                    break
                except DraftGenerationError as exc:
                    failure = str(exc)
                    failures.append(
                        "%s attempt %d: %s"
                        % (path, repair_attempt, failure[:500])
                    )
                except Exception as exc:
                    from numpilot.engines.llm import LlmResponseError

                    if isinstance(exc, LlmResponseError):
                        failed_receipt = _validate_remote_receipt(
                            exc.receipt,
                            None,
                        )
                        if (
                            failed_receipt["requested_model"]
                            != generation_base["model"]
                        ):
                            raise DraftGenerationError(
                                "remote repair receipt requested model mismatch"
                            ) from None
                        receipts.append(failed_receipt)
                    status_code = getattr(exc, "status_code", None)
                    if isinstance(status_code, int) and not isinstance(
                        status_code, bool
                    ):
                        failure = "remote repair call failed: HTTP status %d" % status_code
                    else:
                        failure = "remote repair call failed: %s" % type(exc).__name__
                    failures.append(
                        "%s attempt %d: %s"
                        % (path, repair_attempt, failure[:500])
                    )
            else:
                raise DraftRepairError(
                    "model repair failed after %d total request(s): %s"
                    % (total_attempts, "; ".join(failures)),
                    receipt_count=len(receipts),
                )

        before_hashes = {path: current_files[path] for path in normalized_paths}
        after_hashes = {
            path: _sha256_bytes(collected[path].encode("utf-8"))
            for path in normalized_paths
        }
        candidate_files = dict(current_files)
        candidate_files.update(after_hashes)
        report = {
            "schema_version": 1,
            "status": "repaired_unverified",
            "source_artifact_id": source.artifact_id,
            "source_workspace": str(repair_source_dir),
            "signed_source_workspace": source.workspace,
            "parent_repair_report_sha256": parent_repair_sha256,
            "gap_id": spec.gap_id,
            "model": generation_base["model"],
            "provider": generation_base["provider"],
            "configuration_source": generation_base["configuration_source"],
            "attempt_count": total_attempts,
            "paths": list(normalized_paths),
            "before_sha256": before_hashes,
            "after_sha256": after_hashes,
            "candidate_files_sha256": candidate_files,
            "diagnostic_sha256": _sha256_bytes(diagnostic_bytes),
            "generation_request_sha256": _sha256_bytes(
                generation_request.encode("utf-8")
            ),
            "skill_bundle_sha256": self._skill_bundle.bundle_sha256,
            "response_sha256": response_hashes,
            "remote_receipts": receipts,
            "actual_remote_call": True,
            "gates": dict(_EXPECTED_UNVERIFIED_GATES),
        }

        candidate_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp_dir = Path(tempfile.mkdtemp(
            prefix=".repair-", dir=str(candidate_dir.parent)
        ))
        try:
            shutil.copytree(repair_source_dir, temp_dir, dirs_exist_ok=True)
            for path in normalized_paths:
                target = temp_dir / _safe_relative_path(path)
                payload = collected[path].encode("utf-8")
                temp_file = target.with_name(".%s.repair" % target.name)
                with open(temp_file, "xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temp_file, 0o600)
                os.replace(temp_file, target)
            report_payload = (
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            report_path = temp_dir / "numpilot-repair-report.json"
            report_temp = temp_dir / ".numpilot-repair-report.json.repair"
            with open(report_temp, "xb") as handle:
                handle.write(report_payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(report_temp, 0o600)
            os.replace(report_temp, report_path)
            os.replace(temp_dir, candidate_dir)
            temp_dir = None
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
        return report

    def seal_repair_candidate(
        self,
        spec: GapSpec,
        draft_dir: Path,
        candidate_dir: Path,
        artifact_dir: Path,
    ) -> OperatorDraft:
        """Publish a verified repair chain as a new immutable unverified artifact."""

        draft_dir = Path(draft_dir)
        candidate_dir = Path(candidate_dir)
        artifact_dir = Path(artifact_dir)
        source = _load_existing_draft(draft_dir, spec)
        if os.path.lexists(artifact_dir):
            raise DraftGenerationError("sealed artifact path already exists")
        candidate_files, reports = _load_repair_chain(
            candidate_dir,
            spec,
            source,
        )
        if not reports:
            raise DraftGenerationError("repair chain is empty")

        generation = dict(source.generation)
        response_hashes = list(generation["response_sha256"])
        receipts = list(generation["remote_receipts"])
        attempt_count = generation["attempt_count"]
        for report in reports:
            if (
                report.get("model") != generation["model"]
                or report.get("provider") != generation["provider"]
                or report.get("configuration_source")
                != generation["configuration_source"]
            ):
                raise DraftGenerationError(
                    "repair chain LLM identity differs from signed generation"
                )
            report_attempts = report.get("attempt_count")
            report_hashes = report.get("response_sha256")
            report_receipts = report.get("remote_receipts")
            if (
                isinstance(report_attempts, bool)
                or not isinstance(report_attempts, int)
                or report_attempts < 1
                or not isinstance(report_hashes, list)
                or not report_hashes
                or not isinstance(report_receipts, list)
                or not report_receipts
            ):
                raise DraftGenerationError("repair chain transport evidence is invalid")
            normalized_receipts = [
                _validate_remote_receipt(receipt)
                for receipt in report_receipts
            ]
            content_hashes = [
                receipt["content_sha256"]
                for receipt in normalized_receipts
                if receipt["content_sha256"] is not None
            ]
            if (
                any(not _is_sha256(digest) for digest in report_hashes)
                or content_hashes != report_hashes
                or any(
                    receipt["requested_model"] != generation["model"]
                    for receipt in normalized_receipts
                )
            ):
                raise DraftGenerationError(
                    "repair chain receipt/content hashes are inconsistent"
                )
            attempt_count += report_attempts
            response_hashes.extend(report_hashes)
            receipts.extend(normalized_receipts)
        if attempt_count > _MAX_TRANSPORT_ATTEMPTS:
            raise DraftGenerationError("sealed repair chain exceeds attempt limit")
        generation.update({
            "attempt_count": attempt_count,
            "transport_attempt_count": attempt_count,
            "successful_http_response_count": len(receipts),
            "content_response_count": len(response_hashes),
            "response_sha256": response_hashes,
            "remote_receipts": receipts,
            "local_validation": "passed",
            "fallback": "disabled",
        })
        generation = _validate_generation(generation)

        provenance = {
            "schema_version": 1,
            "source_artifact_id": source.artifact_id,
            "source_workspace": source.workspace,
            "final_candidate_workspace": str(candidate_dir),
            "final_candidate_files_sha256": candidate_files,
            "repair_reports": list(reports),
        }
        provenance_text = (
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        if len(provenance_text.encode("utf-8")) > _MAX_TOTAL_BYTES:
            raise DraftGenerationError("sealed repair provenance exceeds size limit")

        files = []
        for relative in sorted(candidate_files):
            path = candidate_dir / _safe_relative_path(relative)
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise DraftGenerationError(
                    "cannot read repaired artifact file: %s" % relative
                ) from exc
            files.append((_safe_relative_path(relative), content))
        provenance_path = _safe_relative_path(
            "docs/numpilot-repair-provenance.json"
        )
        files.append((provenance_path, provenance_text))
        total_bytes = sum(
            len(content.encode("utf-8")) for _path, content in files
        )
        if total_bytes > _MAX_TOTAL_BYTES or len(files) > _MAX_FILES:
            raise DraftGenerationError("sealed repair artifact exceeds size limits")
        _require_project_layout(str(path) for path, _content in files)

        draft = self._publish_draft(
            spec,
            artifact_dir.parent,
            artifact_dir,
            "Sealed compiler-directed repair chain for %s" % spec.qualified_api,
            tuple(files),
            generation,
        )
        self._run_actions[draft.workspace] = "sealed_repair_chain"
        return draft

    def _publish_draft(
        self,
        spec: GapSpec,
        workspace: Path,
        draft_dir: Path,
        summary: str,
        files,
        generation,
    ) -> OperatorDraft:
        workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp_dir = Path(tempfile.mkdtemp(prefix=".draft-", dir=str(workspace)))
        try:
            hashes: Dict[str, str] = {}
            for relative, content in files:
                target = temp_dir.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                payload = content.encode("utf-8")
                with open(target, "xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(target, 0o700 if relative.name == "build.sh" else 0o600)
                hashes[str(relative)] = _sha256_bytes(payload)

            draft = OperatorDraft(
                artifact_id=_draft_artifact_id(
                    spec.gap_id,
                    hashes,
                    generation,
                ),
                gap_id=spec.gap_id,
                workspace=str(draft_dir),
                summary=summary,
                files=hashes,
                generation=generation,
            )
            manifest = json.dumps(
                draft.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
            ) + "\n"
            manifest_path = temp_dir / "operator-artifact.json"
            with open(manifest_path, "x", encoding="utf-8") as handle:
                handle.write(manifest)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(manifest_path, 0o600)
            os.replace(temp_dir, draft_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        action = (
            "fresh_remote"
            if generation["mode"] == "remote_llm"
            else "injected_test_double"
        )
        self._run_actions[draft.workspace] = action
        return draft

    def generate(
        self,
        spec: GapSpec,
        workspace: Path,
        *,
        generation_request: str = "",
    ) -> OperatorDraft:
        if spec.cann_version.strip().lower() in {"", "unknown", "unset", "none"}:
            raise DraftGenerationError(
                "a concrete CANN version is required before LLM draft generation"
            )
        draft_dir = workspace / "draft"
        if os.path.lexists(draft_dir):
            if draft_dir.is_dir() and not draft_dir.is_symlink():
                draft = _load_existing_draft(draft_dir, spec)
                self._run_actions[draft.workspace] = "reused_existing"
                return draft
            raise DraftGenerationError("draft target exists and is not a directory")

        request = (
            "Generate the operator project for this GapSpec. "
            "The JSON contract is the only semantic source of truth:\n"
            + json.dumps(spec.to_dict(), ensure_ascii=False, sort_keys=True)
        )
        if generation_request:
            request += (
                "\n\nGeneration workflow contract (already extracted from the "
                "local cannbot skill and AsNumpy reference; follow it directly):\n"
                + generation_request
            )
        chat, generation_base = self._resolve_chat()
        if (
            self._staged_direct
            and generation_base["mode"] == "remote_llm"
            and spec.route == "ascendc_registry_candidate"
            and spec.semantic_source.startswith("operator-spec:")
        ):
            (
                summary,
                files,
                response_hashes,
                remote_receipts,
                attempt_count,
            ) = self._generate_staged_direct(
                chat,
                generation_base,
                spec,
                generation_request,
            )
            generation = dict(generation_base)
            generation.update({
                "attempt_count": attempt_count,
                "transport_attempt_count": attempt_count,
                "successful_http_response_count": len(remote_receipts),
                "content_response_count": len(response_hashes),
                "response_sha256": response_hashes,
                "remote_receipts": remote_receipts,
                "local_validation": "passed",
                "fallback": "disabled",
            })
            return self._publish_draft(
                spec,
                workspace,
                draft_dir,
                summary,
                files,
                generation,
            )
        failure = ""
        summary = ""
        files: Tuple[Tuple[PurePosixPath, str], ...] = ()
        response_hashes = []
        remote_receipts = []
        attempt_failures = []
        attempt_count = 0
        for attempt_count in range(1, self.max_attempts + 1):
            prompt = request
            if failure:
                prompt += (
                    "\nPrevious response was rejected by the local validator: "
                    + failure[:500]
                    + "\nReturn a corrected JSON object."
                )
            try:
                result = chat(_SYSTEM_PROMPT, prompt)
                if generation_base["mode"] == "remote_llm":
                    if (
                        not isinstance(result, tuple)
                        or len(result) != 2
                    ):
                        raise DraftGenerationError(
                            "remote model call returned no auditable receipt"
                        )
                    response, receipt = result
                    validated_receipt = _validate_remote_receipt(
                        receipt, response
                    )
                    if (
                        validated_receipt["requested_model"]
                        != generation_base["model"]
                    ):
                        raise DraftGenerationError(
                            "remote model receipt requested model mismatch"
                        )
                    remote_receipts.append(validated_receipt)
                else:
                    response = result
                if isinstance(response, str):
                    response_hashes.append(
                        _sha256_bytes(response.encode("utf-8"))
                    )
                summary, files = _validate_bundle(response)
                break
            except DraftGenerationError as exc:
                failure = str(exc)
                attempt_failures.append(
                    "attempt %d: %s" % (attempt_count, failure[:500])
                )
            except Exception as exc:
                from numpilot.engines.llm import LlmResponseError

                if (
                    generation_base["mode"] == "remote_llm"
                    and isinstance(exc, LlmResponseError)
                ):
                    failed_receipt = _validate_remote_receipt(exc.receipt, None)
                    if (
                        failed_receipt["requested_model"]
                        != generation_base["model"]
                    ):
                        raise DraftGenerationError(
                            "remote model receipt requested model mismatch"
                        ) from None
                    remote_receipts.append(failed_receipt)
                status_code = getattr(exc, "status_code", None)
                if isinstance(status_code, int) and not isinstance(status_code, bool):
                    failure = "remote model call failed: HTTP status %d" % status_code
                else:
                    failure = "remote model call failed: %s" % type(exc).__name__
                attempt_failures.append(
                    "attempt %d: %s" % (attempt_count, failure[:500])
                )
        else:
            raise DraftGenerationError(
                "model failed to produce a safe draft after %d attempt(s): %s"
                % (self.max_attempts, "; ".join(attempt_failures))
            )
        generation = dict(generation_base)
        generation.update({
            "attempt_count": attempt_count,
            "transport_attempt_count": attempt_count,
            "successful_http_response_count": len(remote_receipts),
            "content_response_count": len(response_hashes),
            "response_sha256": response_hashes,
            "remote_receipts": remote_receipts,
            "local_validation": "passed",
            "fallback": "disabled",
        })

        workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp_dir = Path(tempfile.mkdtemp(prefix=".draft-", dir=str(workspace)))
        try:
            hashes: Dict[str, str] = {}
            for relative, content in files:
                target = temp_dir.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                payload = content.encode("utf-8")
                with open(target, "xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(target, 0o700 if relative.name == "build.sh" else 0o600)
                hashes[str(relative)] = _sha256_bytes(payload)

            draft = OperatorDraft(
                artifact_id=_draft_artifact_id(
                    spec.gap_id,
                    hashes,
                    generation,
                ),
                gap_id=spec.gap_id,
                workspace=str(draft_dir),
                summary=summary,
                files=hashes,
                generation=generation,
            )
            manifest = json.dumps(
                draft.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
            ) + "\n"
            manifest_path = temp_dir / "operator-artifact.json"
            with open(manifest_path, "x", encoding="utf-8") as handle:
                handle.write(manifest)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(manifest_path, 0o600)
            os.replace(temp_dir, draft_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        action = (
            "fresh_remote"
            if generation["mode"] == "remote_llm"
            else "injected_test_double"
        )
        self._run_actions[draft.workspace] = action
        return draft
