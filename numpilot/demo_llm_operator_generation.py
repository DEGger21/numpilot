#!/usr/bin/env python3
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

"""Generate one small, receipt-bound operator draft for a terminal demo.

This is intentionally a *demo* lane, not a replacement for ``complete-gaps``.
It makes one real configured-model request with a compact contract so a local
recording can show genuine model output without asking a gateway to generate a
full multi-file Ascend C project.  The result is always marked
``generated_unverified`` and is never compiled, installed, activated, or used
as evidence of numerical or performance correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from typing import Any, Mapping

from numpilot.cli_ui import flow_print


MAX_SKILL_BYTES = 1024 * 1024
MAX_DRAFT_BYTES = 128 * 1024
MAX_RESPONSE_CHARS = 256 * 1024
DEFAULT_MAX_TOKENS = 4096

OPERATOR_NAME = "DemoAddOne"
# Keep this prompt deliberately small.  The configured GLM gateway may spend a
# bounded completion budget on hidden reasoning when given a long instruction;
# a concise, syntax-only request makes the demo responsive while the local
# validator below remains the authority for what can be written to disk.
SYSTEM_PROMPT = (
    "Return immediately with exactly one cpp fenced code block and no prose. "
    "Include // UNVERIFIED_DRAFT and a DemoAddOne function for float32 y=x+1.0f. "
    "Never claim compile, test, numerical, NPU, or activation success."
)

CANNBOT_SYSTEM_PROMPT = (
    "Use the embedded official Huawei CANNBot registry-invoke skill. "
    "Return immediately with exactly one cpp fenced code block and no prose. "
    "The block should be a tiny Ascend C device-helper fragment, contain "
    "// UNVERIFIED_DRAFT, and define only "
    "`__aicore__ inline void add_one(...)`. Use one float32 destination "
    "AscendC::LocalTensor, one float32 source AscendC::LocalTensor, a "
    "`uint32_t count` parameter, and exactly one "
    "`AscendC::Adds(dst, src, 1.0f, count);` call. This is not a standalone "
    "compilation unit: do not include headers, a host launcher, a kernel "
    "entry point, a class, or claims of build/test/NPU/activation success."
)

MAX_GAP_SOURCE_BYTES = 256 * 1024
MAX_RESOLUTION_BYTES = 256 * 1024
MAX_CANNBOT_CONTEXT_BYTES = 128 * 1024
CANNBOT_LICENSE_ACCEPTANCE_ENV = "NUMPILOT_ACCEPT_CANNBOT_LICENSE"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_text(value: object, limit: int = 256) -> str:
    if not isinstance(value, str):
        return "?"
    value = "".join(ch if ch.isprintable() else "?" for ch in value)
    return value[:limit]


def _is_regular(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


def _read_skill(path: Path, expected_sha256: str | None) -> tuple[str, str, int, int]:
    if not _is_regular(path):
        raise ValueError("skill path is not a regular file")
    payload = path.read_bytes()
    if not payload or len(payload) > MAX_SKILL_BYTES:
        raise ValueError("skill is empty or oversized")
    text = payload.decode("utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError("skill frontmatter is invalid")
    match = re.search(r"(?m)^name:\s*([^\s].*)$", text)
    if match is None or match.group(1).strip(" '\"") != "numpilot-migration-gate":
        raise ValueError("skill name is not numpilot-migration-gate")
    digest = _sha256(payload)
    if expected_sha256:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError("expected skill SHA-256 is invalid")
        if digest != expected_sha256.lower():
            raise ValueError("skill SHA-256 does not match")
    return digest, text, len(payload), text.count("\n")


def _read_bounded_text(path: Path, *, limit: int, label: str) -> tuple[str, str]:
    """Read one local route input and return text plus its byte hash."""

    if not _is_regular(path):
        raise ValueError("%s is not a regular file" % label)
    payload = path.read_bytes()
    if not payload or len(payload) > limit:
        raise ValueError("%s is empty or oversized" % label)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("%s is not UTF-8" % label) from exc
    return text, _sha256(payload)


def _validate_cannbot_route(
    *,
    gap_source: Path,
    official_resolution: Path,
) -> tuple[dict[str, Any], str, str, str]:
    """Validate the local official-first receipt before a CANNBot call.

    The mini route deliberately reuses the same resolution contract as
    ``complete-gaps``.  It cannot turn a hand-written file into a model claim:
    the source hash, custom route, and explicit model-generation allowance all
    have to be present in the immutable resolver receipt.
    """

    source_text, source_sha = _read_bounded_text(
        gap_source, limit=MAX_GAP_SOURCE_BYTES, label="gap source"
    )
    if "operator_spec" not in source_text or "add_one" not in source_text:
        raise ValueError("mini CANNBot route requires the add_one OperatorSpec gap")
    resolution_text, resolution_sha = _read_bounded_text(
        official_resolution,
        limit=MAX_RESOLUTION_BYTES,
        label="official resolution",
    )
    try:
        data = json.loads(resolution_text)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("official resolution is invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("official resolution is not an object")
    if data.get("status") != "custom_operator_generation_allowed":
        raise ValueError("official resolution did not open the custom CANNBot route")
    if data.get("model_kernel_generation_allowed") is not True:
        raise ValueError("official resolution does not allow model generation")
    rendered_source = data.get("source_path")
    if not isinstance(rendered_source, str) or not rendered_source:
        raise ValueError("official resolution source path is missing")
    if os.path.abspath(rendered_source) != os.path.abspath(str(gap_source)):
        raise ValueError("official resolution source path does not match gap source")
    if data.get("source_sha256") != source_sha:
        raise ValueError("official resolution source hash does not match gap source")
    resolution_id = data.get("resolution_id")
    if not isinstance(resolution_id, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", resolution_id):
        raise ValueError("official resolution id is invalid")

    # Re-run the repository's own preparation/receipt validator.  This checks
    # report_type, exact GapSpec hashes, target identity, route evidence, and
    # the no-network/no-system-write invariants instead of trusting a few JSON
    # flags in a hand-edited file.
    try:
        from numpilot.gaps.official_resolution import load_official_resolution_report
        from numpilot.gaps.pipeline import prepare_gap_completion

        preparation = prepare_gap_completion(
            str(gap_source),
            target_soc=str(data.get("target_soc", "")),
            cann_version=str(data.get("cann_version", "")),
        )
        loaded = load_official_resolution_report(str(official_resolution))
        loaded.validate_preparation(preparation)
        data = loaded.to_dict()
    except Exception as exc:
        raise ValueError("official resolution failed local revalidation") from exc
    return data, source_sha, resolution_sha, resolution_id


def _load_cannbot_context(root: Path) -> tuple[dict[str, Any], str]:
    """Load and hash the pinned CANNBot generation files without fetching."""

    from numpilot.cannbot_skills import (
        CANNBOT_LICENSE,
        CANNBOT_GENERATION_SKILL_FILES,
        resolve_skill_bundle,
    )

    bundle = resolve_skill_bundle(str(root), allow_fetch=False)
    manifest = bundle.manifest()
    manifest_files = manifest.get("files") or {}
    verified: dict[str, str] = {}
    for relative in CANNBOT_GENERATION_SKILL_FILES:
        content = bundle.files.get(relative)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("required CANNBot file is empty: %s" % relative)
        digest = _sha256(content.encode("utf-8"))
        if manifest_files.get(relative) != digest:
            raise ValueError("required CANNBot file hash mismatch: %s" % relative)
        verified[relative] = digest
    context = bundle.prompt_context(CANNBOT_GENERATION_SKILL_FILES)
    context_bytes = len(context.encode("utf-8"))
    if context_bytes > MAX_CANNBOT_CONTEXT_BYTES:
        raise ValueError("CANNBot prompt context is oversized")
    if bundle.bundle_sha256 not in context or any(
        bundle.files[path] not in context for path in CANNBOT_GENERATION_SKILL_FILES
    ):
        raise ValueError("CANNBot files were not embedded in prompt context")
    gate = {
        "schema_version": 1,
        "gate": "cannbot_skill_readiness",
        "status": "pass",
        "repository": manifest.get("repository"),
        "commit": bundle.commit,
        "skill_name": bundle.skill_name,
        "license": manifest.get("license"),
        "required_skill_files": list(CANNBOT_GENERATION_SKILL_FILES),
        "required_skill_files_sha256": verified,
        "bundle_sha256": bundle.bundle_sha256,
        "verified_file_count": len(bundle.files),
        "prompt_context_sha256": _sha256(context.encode("utf-8")),
        "prompt_context_bytes": context_bytes,
        "content_embedded_for_remote_model": True,
    }
    if gate["license"] != CANNBOT_LICENSE:
        raise ValueError("pinned CANNBot license identity is unexpected")
    return gate, context


def _license_acceptance(cli_value: bool) -> tuple[bool, str]:
    """Resolve the explicit CANNBot license decision without guessing.

    A cached checkout proves provenance, not that the caller accepted its
    terms.  Keep acceptance as a separate, per-invocation decision.  The
    environment form exists for non-interactive recordings; it is deliberately
    strict so a typo cannot silently opt the caller in.
    """

    raw = os.getenv(CANNBOT_LICENSE_ACCEPTANCE_ENV)
    if raw is not None and raw not in {"0", "1"}:
        raise ValueError(
            "%s must be exactly 0 or 1" % CANNBOT_LICENSE_ACCEPTANCE_ENV
        )
    if cli_value:
        return True, "cli"
    if raw == "1":
        return True, "environment"
    return False, "none"


def _print_cannbot_context(gate: Mapping[str, Any], context: str, display: str) -> None:
    """Render public CANNBot route facts, optionally including every file."""

    flow_print(
        "[CANNBot Skill] gate=pass skill=%s commit=%s bundle_sha256=%s files=%s bytes=%s"
        % (
            _safe_text(gate.get("skill_name"), 120),
            _safe_text(gate.get("commit"), 80),
            _safe_text(gate.get("bundle_sha256"), 80),
            gate.get("verified_file_count", "?"),
            gate.get("prompt_context_bytes", "?"),
        ),
        tone="success",
        flush=True,
    )
    flow_print(
        "[CANNBot Skill] license=%s accepted=%s content_embedded=%s"
        % (
            _safe_text(gate.get("license"), 120),
            "yes" if gate.get("license_accepted") is True else "no",
            "yes" if gate.get("content_embedded_for_remote_model") is True else "no",
        ),
        tone="success",
        flush=True,
    )
    for relative, digest in (gate.get("required_skill_files_sha256") or {}).items():
        flow_print(
            "[CANNBot Skill] file=%s sha256=%s" % (relative, digest),
            tone="artifact",
            flush=True,
        )
    if display == "full":
        flow_print(
            "[CANNBot Skill] embedded official context (full public text):",
            tone="warning",
            flush=True,
        )
        print(context, end="" if context.endswith("\n") else "\n", flush=True)
    elif display == "summary":
        flow_print(
            "[CANNBot Skill] context 已校验并嵌入本次请求；传 --full-cannbot-skill 可展开全部公开文本。",
            tone="success",
            flush=True,
        )


def _start_generation_heartbeat(started: float) -> tuple[threading.Event, threading.Thread]:
    """Show bounded progress while a remote generation request is in flight."""

    stop = threading.Event()

    def _run() -> None:
        while not stop.wait(5.0):
            elapsed = int(time.monotonic() - started)
            flow_print(
                "[CANNBot] 远端模型仍在生成 add_one 草稿… elapsed=%ss（等待回执，未假定成功）"
                % elapsed,
                tone="progress",
                flush=True,
            )

    thread = threading.Thread(target=_run, name="numpilot-cannbot-progress", daemon=True)
    thread.start()
    return stop, thread


def _config_path(value: str | None) -> Path | None:
    if value is None:
        return None
    if not value or "\x00" in value or len(value) > 4096:
        raise ValueError("LLM config path is invalid")
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError("LLM config is not a regular file")
    return path


def _new_output(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError("output directory must be absolute")
    if path.exists() or path.is_symlink():
        raise ValueError("output directory already exists; use a fresh private directory")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.mkdir(mode=0o700)
    os.chmod(path, 0o700)


def _atomic_write(path: Path, payload: bytes, mode: int = 0o600) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("write-once output already exists: %s" % path.name)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _extract_cpp(response: str, *, cann_bot_mini: bool = False) -> str:
    if not isinstance(response, str) or not response.strip():
        raise ValueError("model response is empty")
    if len(response) > MAX_RESPONSE_CHARS:
        raise ValueError("model response is oversized")
    matches = re.findall(
        r"```(?:cpp|c\+\+|c|ascendc)\s*\n(.*?)```", response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if len(matches) != 1 or response.count("```") != 2:
        raise ValueError("response must contain exactly one cpp code block")
    code = matches[0].strip() + "\n"
    encoded = code.encode("utf-8")
    if not encoded or len(encoded) > MAX_DRAFT_BYTES:
        raise ValueError("generated draft is empty or oversized")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in code):
        raise ValueError("generated draft contains control characters")
    if "UNVERIFIED_DRAFT" not in code:
        raise ValueError("generated draft is missing UNVERIFIED_DRAFT marker")
    if not re.search(r"\b(?:DemoAddOne|add_one|AddOne)\b", code):
        raise ValueError("generated draft is missing DemoAddOne/add_one")
    if not re.search(r"\b(float|float32|half)\b", code, re.IGNORECASE):
        raise ValueError("generated draft has no numeric type marker")
    if cann_bot_mini:
        line_count = code.count("\n")
        # Gate the demo to a genuinely tiny fragment.  These structural checks
        # bound what is persisted; they are deliberately not a compilation or
        # numerical-correctness claim.
        if line_count < 5 or line_count > 24:
            raise ValueError("CANNBot mini-gap draft is outside the 5..24 line bound")
        if re.search(r"(?m)^\s*#\s*include\b", code):
            raise ValueError("CANNBot mini-gap draft unexpectedly contains headers")
        if re.search(r"\b(?:class|__global__|main)\b", code):
            raise ValueError("CANNBot mini-gap draft is not a device-helper fragment")
        if not re.search(
            r"\b(?:__aicore__\s+inline|inline\s+__aicore__)\s+void\s+"
            r"add_one\s*\(",
            code,
        ):
            raise ValueError("CANNBot mini-gap draft lacks the add_one Ascend C helper")
        if code.count("AscendC::LocalTensor<float>") < 2:
            raise ValueError("CANNBot mini-gap draft lacks float32 LocalTensor operands")
        if re.search(r"\bfloat\s+add_one\s*\(\s*float\s+\w+\s*\)", code):
            raise ValueError("CANNBot mini-gap draft contains a scalar C++ fallback")
        if re.search(r"\buint32_t\s+count\b", code) is None:
            raise ValueError("CANNBot mini-gap draft lacks the uint32_t count parameter")
        adds_calls = re.findall(
            r"\bAscendC::Adds\s*\((.*?)\)\s*;", code, flags=re.DOTALL
        )
        if len(adds_calls) != 1:
            raise ValueError("CANNBot mini-gap draft must contain exactly one AscendC::Adds call")
        adds_arguments = [item.strip() for item in adds_calls[0].split(",")]
        if (
            len(adds_arguments) != 4
            or re.fullmatch(r"1(?:\.0*)?[fF]?", adds_arguments[2]) is None
            or adds_arguments[3] != "count"
        ):
            raise ValueError("CANNBot mini-gap AscendC::Adds arguments are outside the contract")
    else:
        direct_add = re.search(r"\+\s*1(?:\.0*)?\s*[fF]?\b", code)
        named_one = (
            re.search(r"\+\s*(?:one|ONE)\b", code)
            and re.search(r"\b(?:one|ONE)\s*=\s*1(?:\.0*)?\s*[fF]?\b", code)
        )
        if not (direct_add or named_one):
            raise ValueError("generated draft has no add-one semantic marker")
    for name in (
        "NUMPILOT_LLM_AUTH_TOKEN",
        "NUMPILOT_LLM_BASE_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        secret = os.getenv(name, "")
        if secret and secret in code:
            raise ValueError("generated draft contains sensitive environment data")
    return code


def _receipt_metadata(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the useful identity/hash fields without embedding response text."""

    fields = (
        "schema_version", "protocol", "provider", "configured_model",
        "utc_start", "utc_end", "http_status", "requested_model",
        "response_model", "response_model_aliases", "response_model_identity",
        "model_identity_sha256", "response_id", "response_created",
        "request_body_sha256", "response_body_sha256", "content_sha256",
        "request_id", "usage", "tls_verification",
    )
    return {field: receipt.get(field) for field in fields if field in receipt}


def _write_report(output: Path, report: Mapping[str, Any]) -> Path:
    target = output / "operator-generation-report.json"
    payload = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(payload) > 256 * 1024:
        raise ValueError("generation report is oversized")
    _atomic_write(target, payload)
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make one real configured-model call for a tiny unverified operator draft."
    )
    parser.add_argument("--out", required=True, help="fresh absolute private output directory")
    parser.add_argument("--llm-config", help="private llm.json path (optional when env is configured)")
    parser.add_argument(
        "--skill-path",
        help="complete NumPilot Skill bundle's SKILL.md (default: installed bundle)",
    )
    parser.add_argument("--skill-sha256", help="expected SHA-256 from the wrapper")
    parser.add_argument(
        "--skill-bundle-sha256",
        help="expected complete NumPilot Skill bundle SHA-256 from the wrapper",
    )
    parser.add_argument(
        "--gap-source",
        help="simple add_one OperatorSpec source for the CANNBot mini-gap route",
    )
    parser.add_argument(
        "--official-resolution",
        help="hash-bound resolve-gaps receipt for --gap-source",
    )
    parser.add_argument(
        "--cannbot-skill-root",
        help="pinned local cannbot-skills checkout; never fetched by this helper",
    )
    parser.add_argument(
        "--show-cannbot-skill",
        choices=("summary", "full", "none"),
        default="summary",
        help="show public CANNBot route context (default: summary)",
    )
    parser.add_argument(
        "--accept-cannbot-license",
        action="store_true",
        help=(
            "explicitly accept the pinned CANNBot/CANN Open Software License "
            "for this invocation (required by the CANNBot mini route)"
        ),
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help="single request completion budget (default: 4096)",
    )
    parser.add_argument(
        "--timeout", type=int, help="single request timeout in seconds (10..3600)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = Path(args.out).expanduser()
    report: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "numpilot_demo_operator_generation",
        "status": "failed",
        "model_lane": "configured-model",
        "generation_route": "configured_model_demo",
        "cannbot_route": False,
        "cannbot_license_accepted": False,
        "cannbot_license_acceptance_source": "none",
        "candidate_status": "not_generated",
        "actual_remote_call": False,
        "operator_name": OPERATOR_NAME,
        "claims": {
            "model_generated": False,
            "compiled": False,
            "unit_tested": False,
            "numerically_verified": False,
            "activated": False,
            "strict_npu_performance_verified": False,
        },
        "non_claims": [
            "This is a small demo draft, not a complete-gaps artifact.",
            "No compile, unit-test, numerical, activation, or performance gate was run.",
        ],
    }
    started = time.monotonic()
    try:
        license_accepted, license_source = _license_acceptance(
            args.accept_cannbot_license
        )
        _new_output(output)
        from numpilot.migration_skill import load_migration_skill

        requested_skill_path = (
            Path(args.skill_path).expanduser().absolute()
            if args.skill_path
            else None
        )
        skill_bundle = load_migration_skill(
            requested_skill_path.parent if requested_skill_path is not None else None
        )
        skill_path = skill_bundle.skill_path.absolute()
        if requested_skill_path is not None and requested_skill_path != skill_path:
            raise ValueError("skill path must name SKILL.md in a complete bundle")
        if args.skill_bundle_sha256:
            if not re.fullmatch(r"[0-9a-fA-F]{64}", args.skill_bundle_sha256):
                raise ValueError("expected Skill bundle SHA-256 is invalid")
            if skill_bundle.bundle_sha256 != args.skill_bundle_sha256.lower():
                raise ValueError("Skill bundle SHA-256 does not match")
        skill_sha, skill_text, skill_bytes, skill_lines = _read_skill(
            skill_path, args.skill_sha256
        )
        # Keep the file read above meaningful without sending the whole skill
        # to the model.  The wrapper already displays and records its digest;
        # this bounded excerpt is only a routing fact, never executable text.
        skill_name = "numpilot-migration-gate"
        if args.llm_config:
            config = _config_path(args.llm_config)
            assert config is not None
            os.environ["NUMPILOT_LLM_CONFIG_FILE"] = str(config)
        if args.timeout is not None:
            if not 10 <= args.timeout <= 3600:
                raise ValueError("timeout must be between 10 and 3600 seconds")
            os.environ["NUMPILOT_LLM_TIMEOUT"] = str(args.timeout)
        if not 1 <= args.max_tokens <= 1000000:
            raise ValueError("max-tokens must be between 1 and 1000000")

        from numpilot.engines.llm import LlmEngine, LlmResponseError, LlmHttpError, LlmTransportError, llm_proxy_route_status
        from numpilot.llm_config import load_llm_config
        from numpilot.model_contract import (
            ModelContractError,
            require_model_status,
            validate_configured_model_receipt,
        )

        cannbot_mode = any(
            value is not None
            for value in (
                args.gap_source,
                args.official_resolution,
                args.cannbot_skill_root,
            )
        )
        cannbot_context = ""
        if cannbot_mode:
            if not license_accepted:
                raise ValueError(
                    "CANNBot mini route requires explicit license acceptance; "
                    "pass --accept-cannbot-license or set "
                    "%s=1" % CANNBOT_LICENSE_ACCEPTANCE_ENV
                )
            if not all(
                value
                for value in (
                    args.gap_source,
                    args.official_resolution,
                    args.cannbot_skill_root,
                )
            ):
                raise ValueError(
                    "CANNBot mini-gap requires --gap-source, "
                    "--official-resolution, and --cannbot-skill-root"
                )
            gap_source = Path(args.gap_source).expanduser()
            resolution_path = Path(args.official_resolution).expanduser()
            resolution, gap_sha, resolution_sha, resolution_id = (
                _validate_cannbot_route(
                    gap_source=gap_source,
                    official_resolution=resolution_path,
                )
            )
            cannbot_gate, cannbot_context = _load_cannbot_context(
                Path(args.cannbot_skill_root).expanduser()
            )
            # ``load_skill_bundle`` pins the checkout and includes the license
            # identity.  Keep the acceptance decision in the report as a
            # separate fact; provenance alone must never be interpreted as
            # consent.
            cannbot_gate = dict(cannbot_gate)
            cannbot_gate["license_accepted"] = True
            report.update({
                "generation_route": "cannbot_registry_invoke_mini_gap",
                "cannbot_route": True,
                "operator_name": "add_one",
                "draft_kind": "ascendc_kernel_fragment",
                "operator_language": "ascend_c",
                "standalone_compilation_unit": False,
                "cannbot_license_accepted": True,
                "cannbot_license_acceptance_source": license_source,
                "gap_source": str(gap_source),
                "gap_source_sha256": gap_sha,
                "official_resolution_sha256": resolution_sha,
                "official_resolution_id": resolution_id,
                "official_resolution_status": resolution.get("status"),
                "official_resolution_cann_osl_accepted": resolution.get(
                    "cann_osl_accepted"
                ),
                "cannbot_skill_gate": cannbot_gate,
            })
            report["non_claims"].append(
                "The Ascend C output is a device-helper fragment, not a standalone "
                "compilation unit."
            )
            _print_cannbot_context(
                cannbot_gate,
                cannbot_context,
                args.show_cannbot_skill,
            )

        status = load_llm_config(args.llm_config)
        if not status.configured:
            raise ValueError("configured model is unavailable")
        identity = require_model_status(status)
        proxy_status = llm_proxy_route_status()
        report.update({
            "skill_name": skill_name,
            "skill_sha256": skill_sha,
            "skill_bytes": skill_bytes,
            "skill_lines": skill_lines,
            "skill_bundle_sha256": skill_bundle.bundle_sha256,
            "skill_files_sha256": dict(skill_bundle.files_sha256),
            # Keep the provenance vocabulary aligned with the terminal
            # wrapper's skill receipt.  This tiny demo is deliberately a
            # terminal/manual configured sub-run, never an agent-authored
            # candidate.
            "entry_lane": "terminal_manual",
            "model_source": "configured",
            "agent_authored_draft": False,
            "configuration_source": status.source,
            "provider": identity.provider,
            "protocol": identity.protocol,
            "configured_model": identity.model,
            "proxy_policy": proxy_status.get("policy"),
            "proxy_route": proxy_status.get("route"),
            "request_max_tokens": args.max_tokens,
        })
        # The short prompt intentionally uses the ordinary, portable non-stream
        # request shape.  For the mini route the verified CANNBot context is
        # appended verbatim, so the model follows the same official workflow
        # while the requested output remains small enough to close its fence.
        report["request_stream"] = False
        if cannbot_mode:
            system_prompt = CANNBOT_SYSTEM_PROMPT + "\n\n" + cannbot_context
            user_prompt = (
                "Generate only that tiny add_one Ascend C helper fragment now; close "
                "the cpp fence. The semantic gap is one contiguous float32 unary "
                "add-one operation. This output remains an uncompiled, unactivated "
                "fragment. NumPilot Skill sha256=%s; gap source sha256=%s; "
                "official resolution id=%s."
                % (skill_sha, gap_sha, resolution_id)
            )
        else:
            system_prompt = SYSTEM_PROMPT
            user_prompt = (
                "Generate the tiny DemoAddOne block now. Skill loaded and checked: "
                f"numpilot-migration-gate sha256={skill_sha}."
            )
        # Do not retain the full skill text in a report or pass unbounded
        # repository material to a remote endpoint.  Reading/validating it is
        # proven by the hash and wrapper receipt.
        del skill_text
        heartbeat_stop = None
        heartbeat_thread = None
        if cannbot_mode:
            flow_print(
                "[CANNBot] remote generation started · route=cannbot_registry_invoke_mini_gap "
                "· target=ascendc_kernel_fragment · budget=%s · timeout=%ss"
                % (args.max_tokens, os.getenv("NUMPILOT_LLM_TIMEOUT", "120")),
                tone="model",
                flush=True,
            )
            heartbeat_stop, heartbeat_thread = _start_generation_heartbeat(started)
        try:
            call_options: dict[str, Any] = {"max_tokens": args.max_tokens}
            # GLM gateways can consume a small bounded completion budget in
            # hidden reasoning and leave the visible code block empty.  This
            # mini CANNBot route asks for a deliberately tiny draft, so make
            # the visible completion explicit for GLM while preserving the
            # portable historical request shape for other providers.
            provider_name = str(identity.provider or "").strip().lower()
            model_name = str(identity.model or "").strip().lower()
            if (
                provider_name in {"glm", "zhipu", "zhipuai"}
                or model_name.startswith("glm-")
            ):
                call_options.update(
                    {"thinking_mode": "disabled", "reasoning_effort": "none"}
                )
            response, receipt = LlmEngine._chat_with_receipt(
                system_prompt,
                user_prompt,
                **call_options,
            )
        except (LlmResponseError, LlmHttpError, LlmTransportError) as exc:
            report["actual_remote_call"] = True
            report["error_type"] = type(exc).__name__
            if hasattr(exc, "status_code"):
                report["http_status"] = getattr(exc, "status_code")
            if isinstance(exc, LlmResponseError) and exc.receipt:
                report["remote_receipt"] = _receipt_metadata(exc.receipt)
            raise
        finally:
            if heartbeat_stop is not None:
                heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1.0)
        report["actual_remote_call"] = True
        try:
            validated = validate_configured_model_receipt(
                receipt,
                response,
                expected_model=identity.model,
                expected_provider=identity.provider,
                expected_protocol=identity.protocol,
                response_model_aliases=identity.response_model_aliases,
            )
            # Preserve the redacted remote identity even when the model
            # returns malformed C++ or a later write-once operation fails.
            # A failed draft should still retain the auditable request/response
            # hash, while never embedding the response body itself.
            report["remote_receipt"] = _receipt_metadata(validated)
        except ModelContractError:
            report["error_type"] = "ModelContractError"
            report["remote_receipt"] = _receipt_metadata(receipt)
            raise
        code = _extract_cpp(response, cann_bot_mini=cannbot_mode)
        code_bytes = code.encode("utf-8")
        code_path = output / "operator-draft.cpp"
        _atomic_write(code_path, code_bytes)
        code_sha = _sha256(code_bytes)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        report.update({
            "status": "generated_unverified",
            # Keep the generic Skill receipt vocabulary: this is a draft, not
            # an admitted/qualified artifact.  The more specific lifecycle
            # state remains in ``status=generated_unverified``.
            "candidate_status": "draft",
            "elapsed_ms": elapsed_ms,
            "generated_file": code_path.name,
            "generated_file_sha256": code_sha,
            "generated_bytes": len(code_bytes),
            "generated_lines": code.count("\n"),
            "response_content_sha256": validated.get("content_sha256"),
            "claims": {**report["claims"], "model_generated": True},
        })
        report_path = _write_report(output, report)
        flow_print(
            "[算子草稿] status=generated_unverified remote_call=yes model=%s http=%s elapsed=%sms"
            % (
                _safe_text(validated.get("response_model") or identity.model),
                validated.get("http_status", "?"),
                elapsed_ms,
            ),
            tone="warning",
        )
        flow_print(
            "[算子草稿] file=%s bytes=%d lines=%d sha256=%s"
            % (code_path, len(code_bytes), code.count("\n"), code_sha),
            tone="artifact",
        )
        preview_kind = (
            "Ascend C kernel fragment" if cannbot_mode else "generated source"
        )
        flow_print(
            "[算子草稿] bounded preview (%s; uncompiled and still unverified):"
            % preview_kind,
            tone="warning",
        )
        for line in code.splitlines()[:18]:
            flow_print("  " + _safe_text(line, 240), tone="code")
        if code.count("\n") > 18:
            flow_print("  ... (剩余源码已写入私有目录，未在终端展开)", tone="code")
        flow_print("[算子草稿] report=%s" % report_path, tone="artifact")
        flow_print(
            "[算子草稿] claims=model_generated=true compiled=false "
            "verified=false activated=false",
            tone="warning",
        )
        return 0
    except Exception as exc:
        report["elapsed_ms"] = int((time.monotonic() - started) * 1000)
        report.setdefault("error_type", type(exc).__name__)
        # Error details are deliberately reduced to a short, locally-defined
        # validation reason.  Never persist a raw provider exception or model
        # response because either could contain endpoint or prompt material.
        detail = _safe_text(str(exc), 180)
        for sensitive_name in (
            "NUMPILOT_LLM_AUTH_TOKEN",
            "NUMPILOT_LLM_BASE_URL",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
        ):
            sensitive = os.getenv(sensitive_name, "")
            if sensitive:
                detail = detail.replace(sensitive, "<redacted>")
        report["error_detail"] = detail
        try:
            report_path = _write_report(output, report)
        except Exception:
            report_path = None
        flow_print(
            "[算子草稿] status=failed remote_call=%s error=%s elapsed=%sms"
            % (
                "yes" if report.get("actual_remote_call") else "no",
                type(exc).__name__,
                report["elapsed_ms"],
            ),
            stream=sys.stderr,
            tone="error",
        )
        if report_path is not None:
            flow_print(
                "[算子草稿] report=%s" % report_path,
                stream=sys.stderr,
                tone="artifact",
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
