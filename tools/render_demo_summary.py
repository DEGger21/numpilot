#!/usr/bin/env python3
"""Render a compact, read-only summary of ordinary-script demo receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence


UNAVAILABLE = "UNAVAILABLE"
MAX_RECEIPT_BYTES = 8 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_DISPLAY_SECRET_ENV = (
    "NUMPILOT_LLM_AUTH_TOKEN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _display_text(value: Any, *, width: int = 512) -> str:
    """Render untrusted receipt metadata as one safe, bounded line.

    The four receipts are evidence inputs, not trusted presentation data.  A
    hand-edited or compromised receipt must not be able to inject terminal
    controls, a multi-line fake status, or a configured proxy/token into the
    summary that people copy into a submission log.  Keep ordinary values
    readable while replacing control/format characters and redacting secrets
    already present in the process environment.
    """

    if value is None:
        return UNAVAILABLE
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        return UNAVAILABLE
    try:
        if isinstance(value, Path):
            text = str(value)
        elif isinstance(value, (str, int, float, bool)):
            text = str(value)
        else:
            return UNAVAILABLE
    except (OverflowError, UnicodeError, ValueError, OSError, RuntimeError):
        # JSON permits very large integers and a direct API caller can still
        # provide an unusual scalar subclass.  Presentation must remain a
        # total function: a malformed receipt is unavailable, not a traceback.
        return UNAVAILABLE

    sensitive = {
        os.getenv(name, "")
        for name in _DISPLAY_SECRET_ENV
        if os.getenv(name, "")
    }
    # A bearer token can be supplied by a receipt even when the local
    # environment was scrubbed after the request.  Redact its value by shape
    # as well; this is intentionally conservative presentation-only logic.
    text = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[redacted]", text)
    for secret in sorted(sensitive, key=len, reverse=True):
        text = text.replace(secret, "[redacted]")

    # Treat a Windows CRLF pair as one line break.  Besides keeping the
    # one-line rendering compact, this avoids producing two replacement
    # glyphs for a single attacker-controlled newline sequence.
    text = text.replace("\r\n", "\n")
    safe: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if character == "\t" or character in {"\n", "\r"}:
            safe.append("�")
        elif not character.isprintable() or category.startswith("C"):
            safe.append("�")
        else:
            safe.append(character)
    rendered = "".join(safe)
    return rendered[:width] if rendered else UNAVAILABLE


def _has_symlink_component(path: Path) -> bool:
    """Return true when *path* or an existing ancestor is a symlink.

    ``O_NOFOLLOW`` protects only the final component.  Summary paths are
    caller supplied, so an otherwise regular file below a symlinked directory
    must be rejected as well; otherwise a harmless-looking receipt path could
    redirect the renderer to an unrelated file outside the intended run.
    Missing components are allowed, while inspection errors fail closed.
    """

    try:
        rendered = os.fspath(path)
        if not isinstance(rendered, str):
            return True
        # Preserve ``..`` while checking.  Normalizing first would turn
        # ``symlink/../receipt.json`` into an apparently safe path even
        # though the kernel follows the symlink before applying ``..``.
        expanded = os.path.expanduser(rendered)
        current = (
            Path(os.path.abspath(os.sep))
            if os.path.isabs(expanded)
            else Path(os.getcwd())
        )
        for component in expanded.split(os.sep):
            if not component or component == ".":
                continue
            if component == "..":
                current = current.parent
                continue
            candidate = current / component
            if os.path.lexists(candidate) and candidate.is_symlink():
                return True
            current = candidate
        return False
    except (OSError, RuntimeError, TypeError, ValueError):
        return True


def _read_receipt_bytes(path: Path) -> bytes | None:
    """Read at most one regular, non-symlink receipt into memory."""

    descriptor: int | None = None
    try:
        # O_NOFOLLOW closes the check/use race on platforms that provide it;
        # the explicit is_symlink check keeps the intent clear and supports
        # callers/tests on platforms where the flag is unavailable.
        if _has_symlink_component(path):
            return None
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        # Avoid blocking forever if a caller accidentally points at a FIFO;
        # it will be rejected by the regular-file check below.
        flags |= getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            return None
        if metadata.st_size > MAX_RECEIPT_BYTES:
            return None
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            payload = stream.read(MAX_RECEIPT_BYTES + 1)
        if len(payload) > MAX_RECEIPT_BYTES:
            return None
        return payload
    except (OSError, TypeError, ValueError):
        return None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _load_object(path: Path) -> Mapping[str, Any]:
    """Load a bounded regular receipt without following symlinks.

    Summary rendering is deliberately read-only, but receipt paths can still
    be supplied by a caller.  Refusing symlinks/directories and capping the
    read prevents an accidental or hostile path from turning the renderer
    into an unbounded file reader.  A rejected/malformed receipt remains an
    empty mapping so the existing summary keeps its ``UNAVAILABLE`` behavior.
    """

    payload = _read_receipt_bytes(path)
    if payload is None:
        return {}
    try:
        value = json.loads(payload)
    except (TypeError, ValueError, UnicodeError, RecursionError):
        return {}
    return value if isinstance(value, dict) else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _items(value: Any) -> Sequence[Any]:
    return value if isinstance(value, list) else ()


def _short(value: Any, width: int = 12) -> str:
    rendered = _display_text(value, width=width)
    return rendered if len(rendered) <= width else rendered[:width]


def _integer(value: Any) -> str:
    """Format a non-negative count without accepting arbitrary receipt text."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return UNAVAILABLE
    try:
        return str(value)
    except (OverflowError, UnicodeError, ValueError, OSError, RuntimeError):
        return UNAVAILABLE


def _number(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return UNAVAILABLE
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return UNAVAILABLE
    if not math.isfinite(number):
        return UNAVAILABLE
    return f"{number:.{digits}f}"


def _scientific(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return UNAVAILABLE
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return UNAVAILABLE
    if not math.isfinite(number):
        return UNAVAILABLE
    return f"{number:.3g}"


def _status(known: bool, passed: bool) -> str:
    if not known:
        return UNAVAILABLE
    return "PASS" if passed else "FAIL"


def _file_digest(path: Path) -> str:
    payload = _read_receipt_bytes(path)
    if payload is None:
        return UNAVAILABLE
    return hashlib.sha256(payload).hexdigest()[:12]


def _declared_sha(value: Any) -> tuple[str | None, bool]:
    """Return a normalized declared SHA and whether a value is malformed.

    ``None``/the empty string represent an absent optional field.  A present
    value that is not exactly a hexadecimal SHA-256 is malformed rather than
    absent: silently treating it as missing would make a corrupted receipt
    look merely incomplete.
    """

    if value is None or value == "":
        return None, False
    if isinstance(value, str) and _SHA256_RE.fullmatch(value):
        return value.lower(), False
    return None, True


def _binding_category(fields: Mapping[str, Any]) -> tuple[str, str, str]:
    """Evaluate one cross-receipt hash family.

    The third item is a safe short display value.  A category is publishable
    only when every expected field is present and equal.  With too few values
    to compare, the result is explicitly ``UNKNOWN``; any malformed value or
    disagreement is a hard ``FAIL``.
    """

    parsed: dict[str, str | None] = {}
    invalid: list[str] = []
    for label, value in fields.items():
        normalized, malformed = _declared_sha(value)
        parsed[label] = normalized
        if malformed:
            invalid.append(label)

    present = {label: value for label, value in parsed.items() if value is not None}
    missing = [label for label, value in parsed.items() if value is None]
    if invalid:
        return "FAIL", "invalid=" + ",".join(invalid), "invalid"

    unique = set(present.values())
    if len(unique) > 1:
        # Preserve the receipt order (A/M/V/B, M/V/B, or V/B) so the
        # one-screen diagnostic mirrors the evidence flow.
        return "FAIL", "mismatch=" + ",".join(present), "mismatch"

    if missing or len(present) < 2:
        # A lone declaration cannot establish a cross-receipt binding.  Keep
        # the reason compact because this line is intended for a one-screen
        # demo card; the raw receipts remain the detailed evidence.
        if missing:
            reason = "missing=" + ",".join(missing)
        else:
            reason = "insufficient=compare"
        return "UNKNOWN", reason, "unknown"

    # All expected declarations exist and the normalized values agree.
    value = next(iter(present.values()))
    return "PASS", "match", _short(value)


def _cross_receipt_binding(
    analysis: Mapping[str, Any],
    migration: Mapping[str, Any],
    verification: Mapping[str, Any],
    benchmark: Mapping[str, Any],
) -> tuple[str, dict[str, tuple[str, str, str]]]:
    """Compare source/migrated/report hashes carried by all four receipts."""

    verification_source = _mapping(verification.get("source"))
    verification_migration = _mapping(verification.get("migration"))
    benchmark_binding = _mapping(benchmark.get("binding"))
    benchmark_source = _mapping(benchmark_binding.get("source"))
    benchmark_migrated = _mapping(benchmark_binding.get("migrated"))
    benchmark_report = _mapping(benchmark_binding.get("migration_report"))

    categories: dict[str, Mapping[str, Any]] = {
        "source": {
            "A": analysis.get("source_sha256"),
            "M": migration.get("source_sha256"),
            "V": verification_source.get("sha256"),
            "B": benchmark_source.get("sha256"),
        },
        "migrated": {
            "M": migration.get("migrated_sha256"),
            "V": verification_migration.get("migrated_sha256"),
            "B": benchmark_migrated.get("sha256"),
        },
        "report": {
            "V": verification_migration.get("report_sha256"),
            "B": benchmark_report.get("sha256"),
        },
    }
    evaluated = {
        name: _binding_category(fields) for name, fields in categories.items()
    }
    states = [item[0] for item in evaluated.values()]
    if "FAIL" in states:
        overall = "FAIL"
    elif "UNKNOWN" in states:
        overall = "UNKNOWN"
    else:
        overall = "PASS"
    return overall, evaluated


def render_summary(
    analysis_path: Path,
    migration_path: Path,
    verification_path: Path,
    benchmark_path: Path,
) -> str:
    """Return a truthful one-screen card; missing optional fields stay visible."""

    analysis = _load_object(analysis_path)
    migration = _load_object(migration_path)
    verification = _load_object(verification_path)
    benchmark = _load_object(benchmark_path)

    calls = _items(analysis.get("calls"))
    direct = sum(
        1 for item in calls
        if _mapping(item).get("decision") == "DIRECT"
    )
    eligible = sum(
        1 for item in calls
        if _mapping(item).get("performance_eligible") is True
    )
    analysis_known = bool(analysis)
    analysis_passed = (
        analysis.get("decision") == "DIRECT"
        and bool(calls)
        and direct == len(calls)
        and eligible == len(calls)
    )

    runtime_status = analysis.get("runtime_identity_status")
    runtime_known = bool(analysis)
    runtime_passed = (
        runtime_status == "matched"
        and isinstance(analysis.get("runtime_identity_sha256"), str)
    )

    receipt = _mapping(migration.get("llm_receipt"))
    glm_known = bool(migration)
    glm_passed = (
        migration.get("actual_remote_call") is True
        and receipt.get("http_status") == 200
        and receipt.get("tls_verification") == "enabled"
        and receipt.get("response_model_identity") == "exact"
    )
    model = (
        receipt.get("response_model")
        or migration.get("response_model")
        or migration.get("configured_model")
        or UNAVAILABLE
    )
    remote_call = (
        migration.get("actual_remote_call")
        if "actual_remote_call" in migration else UNAVAILABLE
    )
    ast_known = bool(migration)
    ast_passed = (
        migration.get("ast_audit") == "passed"
        and migration.get("strict_npu") is True
        and isinstance(migration.get("source_sha256"), str)
        and isinstance(migration.get("migrated_sha256"), str)
    )

    comparisons = _mapping(verification.get("comparisons"))
    result_comparison = _mapping(comparisons.get("result"))
    numerical_known = bool(verification)
    numerical_passed = (
        verification.get("status") == "passed"
        and verification.get("numerically_consistent") is True
        and result_comparison.get("consistent") is True
    )

    gates = _mapping(benchmark.get("gates"))
    gate_count = len(gates)
    passed_gates = sum(value is True for value in gates.values())
    performance = _mapping(benchmark.get("performance"))
    npu_known = bool(benchmark)
    npu_passed = (
        benchmark.get("status") == "passed"
        and gate_count > 0
        and passed_gates == gate_count
        and performance.get("performance_valid") is True
    )
    protocol = _mapping(benchmark.get("protocol"))

    binding_status, binding_categories = _cross_receipt_binding(
        analysis, migration, verification, benchmark
    )
    binding_headline = binding_status
    if binding_status != "PASS":
        binding_headline += "（不可发布）"

    def _binding_detail(name: str) -> str:
        state, reason, display = binding_categories[name]
        if state == "PASS":
            return f"PASS({display})"
        if reason:
            return f"{state}({reason})"
        return state

    binding_text = " · ".join(
        f"{name}={_binding_detail(name)}"
        for name in ("source", "migrated", "report")
    )

    ci = _items(performance.get("median_speedup_bootstrap_95ci"))
    ci_text = (
        f"[{_number(ci[0])}, {_number(ci[1])}]"
        if len(ci) == 2 else UNAVAILABLE
    )
    headline_status = _status(npu_known, npu_passed)
    if not npu_passed:
        headline_status += "（不可发布）"

    component = _mapping(
        performance.get("warm_runtime_single_pass_component_sum")
    )
    direct_e2e = component.get("directly_sampled_end_to_end")
    if direct_e2e is True:
        component_scope = "直接采样 E2E"
    elif direct_e2e is False:
        component_scope = "诊断；非直接 E2E"
    else:
        component_scope = "口径未知"

    paths = {
        "A": analysis_path,
        "M": migration_path,
        "V": verification_path,
        "B": benchmark_path,
    }
    path_text = " · ".join(
        f"{key}={_display_text(value, width=1024)}"
        for key, value in paths.items()
    )
    digest_text = " · ".join(
        f"{key}={_file_digest(value)}" for key, value in paths.items()
    )

    lines = [
        "NumPilot · ordinary NumPy → configured GLM → strict-NPU",
        "=" * 68,
        f"[Runtime 身份] {_status(runtime_known, runtime_passed)} · "
        f"status={_display_text(runtime_status)} · "
        f"identity={_short(analysis.get('runtime_identity_sha256'))} · "
        f"audit={_display_text(analysis.get('runtime_audit_id'))}",
        f"[静态分析] {_status(analysis_known, analysis_passed)} · "
        f"DIRECT {direct}/{len(calls)} · 性能准入 {eligible}/{len(calls)}",
        f"[真实 GLM] {_status(glm_known, glm_passed)} · model={_display_text(model)} · "
        f"remote={_display_text(remote_call)} · "
        f"HTTP={_display_text(receipt.get('http_status', UNAVAILABLE))} · "
        f"TLS={_display_text(receipt.get('tls_verification', UNAVAILABLE))} · "
        f"identity={_display_text(receipt.get('response_model_identity', UNAVAILABLE))}",
        f"[跨回执绑定] {binding_headline} · {binding_text}",
        f"[AST / 哈希绑定] {_status(ast_known, ast_passed)} · "
        f"audit={_display_text(migration.get('ast_audit', UNAVAILABLE))} · "
        f"source={_short(migration.get('source_sha256'))} · "
        f"migrated={_short(migration.get('migrated_sha256'))}",
        f"[数值对拍] {_status(numerical_known, numerical_passed)} · "
        f"max_abs={_scientific(result_comparison.get('max_abs_diff'))} · "
        f"max_rel={_scientific(result_comparison.get('max_rel_diff'))} · "
        f"elements={_integer(result_comparison.get('n_elements'))}",
        f"[strict-NPU 门禁] {_status(npu_known, npu_passed)} · "
        f"{passed_gates}/{gate_count} · warmup={_integer(protocol.get('warmup'))} · "
        f"samples={_integer(protocol.get('samples'))} · "
        f"core={_integer(protocol.get('core'))} · "
        f"CPU threads={_integer(protocol.get('cpu_threads_requested'))}",
        f"[设备驻留 headline] {headline_status} · "
        f"CPU={_number(performance.get('cpu_median_ms'))} ms · "
        f"NPU={_number(performance.get('npu_median_ms'))} ms · "
        f"median={_number(performance.get('median_speedup'))}× · "
        f"conservative={_number(performance.get('conservative_cpu_p10_over_npu_p90'))}× · "
        f"bootstrap95%={ci_text}",
        f"[一次组件和（{component_scope}）] "
        f"CPU={_number(component.get('cpu_ms'))} ms · "
        f"NPU={_number(component.get('npu_ms'))} ms · "
        f"CPU/NPU={_number(component.get('diagnostic_cpu_over_npu'))}×",
        f"[证据路径] {path_text}",
        f"[回执 SHA256 前 12 位] {digest_text}",
    ]
    return "\n".join(lines)


def _summary_status(value: str) -> tuple[str, str]:
    """Return a display marker/style for a sanitized summary line."""

    upper = value.upper()
    if upper.startswith("PASS"):
        return "✓", "green"
    if upper.startswith("FAIL"):
        return "✗", "red"
    if upper.startswith(("UNKNOWN", "UNAVAILABLE")):
        return "?", "yellow"
    return "·", "cyan"


def _summary_label_detail(line: str) -> tuple[str, str]:
    """Split one already-sanitized ``[label] detail`` line safely."""

    if line.startswith("[") and "]" in line:
        label, detail = line[1:].split("]", 1)
        return label, detail.strip(" ·")
    return "", line


def _visual_stage_detail(label: str, detail: str) -> str:
    """Shorten opaque runtime identifiers while retaining their prefix."""

    if label == "Runtime 身份":
        detail = re.sub(
            r"(audit=)([^ ·]+)",
            lambda match: (
                match.group(1) + match.group(2)[:28] + "…"
                if len(match.group(2)) > 28
                else match.group(0)
            ),
            detail,
        )
    return detail


def _compact_evidence_lines(line: str, *, width: int) -> list[str]:
    """Split the combined path row into stable, wrap-friendly entries."""

    label, detail = _summary_label_detail(line)
    if label != "证据路径":
        return [line]
    parts = [part.strip() for part in detail.split(" · ") if part.strip()]
    output = []
    for part in parts:
        if "=" not in part:
            output.append(part)
            continue
        key, value = part.split("=", 1)
        if width < 140:
            # Keep the receipt type and filename visible on a small screen;
            # the full absolute path remains in the plain output and JSON.
            path = Path(value)
            components = path.parts
            keep_components = 2 if width < 100 else 3
            value = (
                ".../" + "/".join(components[-keep_components:])
                if len(components) > keep_components
                else str(path)
            )
        output.append(f"{key}={value}")
    return output or [line]


def render_summary_tty(
    analysis_path: Path,
    migration_path: Path,
    verification_path: Path,
    benchmark_path: Path,
    *,
    stream=None,
) -> bool:
    """Render a compact Rich evidence card when stdout is an interactive TTY.

    The underlying summary remains the stable plain-text contract.  This
    presentation layer only rearranges those sanitized lines into a visual
    card for a live demo; it never executes receipts, contacts the network, or
    changes the values used for a PASS/FAIL decision.
    """

    output = sys.stdout if stream is None else stream
    try:
        interactive = bool(output.isatty())
    except (AttributeError, OSError):
        interactive = False
    if (
        not interactive
        or "NO_COLOR" in os.environ
        or os.environ.get("TERM", "").lower() == "dumb"
        or os.environ.get("NUMPILOT_PLAIN") == "1"
    ):
        return False

    # Import Rich lazily so the read-only tool still has a useful plain mode
    # in minimal audit environments where optional presentation dependencies
    # are unavailable.
    try:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        return False

    lines = render_summary(
        analysis_path,
        migration_path,
        verification_path,
        benchmark_path,
    ).splitlines()
    if not lines:
        return False
    console = Console(
        file=output,
        highlight=False,
        markup=False,
        soft_wrap=True,
    )
    console.print(
        Panel.fit(
            Text.assemble(
                ("NumPilot · 证据回放", "bold cyan"),
                "\n",
                ("ordinary NumPy → configured GLM → strict-NPU", "white"),
                "\n",
                ("只读展示：不执行源码、不联网、不生成新声明", "dim"),
            ),
            border_style="cyan",
            box=box.DOUBLE,
        )
    )

    check_lines = [
        line for line in lines[2:]
        if line.startswith((
            "[Runtime 身份]",
            "[静态分析]",
            "[真实 GLM]",
            "[跨回执绑定]",
            "[AST / 哈希绑定]",
            "[数值对拍]",
            "[strict-NPU 门禁]",
        ))
    ]
    rows = []
    for line in check_lines:
        label, detail = _summary_label_detail(line)
        match = re.match(r"(PASS|FAIL|UNKNOWN|UNAVAILABLE)(?:（[^）]*）)?", detail)
        status = match.group(0) if match else "INFO"
        marker, style = _summary_status(status)
        if match:
            detail = detail[match.end():].lstrip(" ·")
        detail = _visual_stage_detail(label, detail)
        rows.append((label, status, detail, marker, style))

    if console.width < 86:
        console.print(Text("阶段门禁", style="bold cyan"))
        for label, status, detail, marker, style in rows:
            console.print(
                Text.assemble(
                    (marker + " ", "bold " + style),
                    (label, "bold"),
                    (" · " + status, style),
                )
            )
            console.print(
                Text(
                    "  " + detail,
                    style="dim",
                    no_wrap=False,
                    overflow="fold",
                )
            )
    else:
        table = Table(
            title="阶段门禁",
            box=box.ROUNDED,
            header_style="bold",
            expand=True,
        )
        table.add_column("状态", width=6, no_wrap=True)
        table.add_column("阶段", style="bold", no_wrap=True)
        table.add_column("证据摘要", overflow="fold")
        for label, status, detail, marker, style in rows:
            table.add_row(
                Text(marker + " " + status, style="bold " + style),
                Text(label),
                Text(detail, style="dim"),
            )
        console.print(table)

    metric_lines = [
        line for line in lines[2:]
        if line.startswith(("[设备驻留 headline]", "[一次组件和（"))
    ]
    if metric_lines:
        console.print(
            Panel(
                Text(
                    "\n".join(metric_lines),
                    no_wrap=False,
                    overflow="fold",
                ),
                title="性能口径（resident 与诊断组件分开）",
                border_style="magenta",
            )
        )

    evidence_lines: list[str] = []
    for line in lines[2:]:
        if line.startswith("[证据路径]"):
            evidence_lines.extend(
                _compact_evidence_lines(line, width=console.width)
            )
        elif line.startswith("[回执 SHA256 前 12 位]"):
            evidence_lines.append(line)
    if evidence_lines:
        console.print(
            Panel(
                Text(
                    "\n".join(evidence_lines),
                    no_wrap=False,
                    overflow="fold",
                ),
                title=(
                    "可追溯回执（窄屏省略路径；--plain 保留完整路径）"
                    if console.width < 140
                    else "可追溯回执"
                ),
                border_style="blue",
            )
        )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--migration", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument(
        "--plain",
        action="store_true",
        help="强制使用稳定纯文本卡片（默认：TTY 自动使用 Rich 证据回放）",
    )
    args = parser.parse_args(argv)
    paths = (
        Path(args.analysis),
        Path(args.migration),
        Path(args.verification),
        Path(args.benchmark),
    )
    if not args.plain and render_summary_tty(*paths):
        return 0
    print(render_summary(*paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
