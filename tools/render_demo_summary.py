#!/usr/bin/env python3
"""Render a compact, read-only summary of ordinary-script demo receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


UNAVAILABLE = "UNAVAILABLE"


def _load_object(path: Path) -> Mapping[str, Any]:
    """Load a report without making summary rendering a second demo gate."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _items(value: Any) -> Sequence[Any]:
    return value if isinstance(value, list) else ()


def _short(value: Any, width: int = 12) -> str:
    if not isinstance(value, str) or not value:
        return UNAVAILABLE
    return value if len(value) <= width else value[:width]


def _number(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return UNAVAILABLE
    number = float(value)
    if not math.isfinite(number):
        return UNAVAILABLE
    return f"{number:.{digits}f}"


def _scientific(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return UNAVAILABLE
    number = float(value)
    if not math.isfinite(number):
        return UNAVAILABLE
    return f"{number:.3g}"


def _status(known: bool, passed: bool) -> str:
    if not known:
        return UNAVAILABLE
    return "PASS" if passed else "FAIL"


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except OSError:
        return UNAVAILABLE


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
    path_text = " · ".join(f"{key}={value}" for key, value in paths.items())
    digest_text = " · ".join(
        f"{key}={_file_digest(value)}" for key, value in paths.items()
    )

    lines = [
        "NumPilot · ordinary NumPy → configured GLM → strict-NPU",
        "=" * 68,
        f"[Runtime 身份] {_status(runtime_known, runtime_passed)} · "
        f"status={runtime_status or UNAVAILABLE} · "
        f"identity={_short(analysis.get('runtime_identity_sha256'))} · "
        f"audit={analysis.get('runtime_audit_id') or UNAVAILABLE}",
        f"[静态分析] {_status(analysis_known, analysis_passed)} · "
        f"DIRECT {direct}/{len(calls)} · 性能准入 {eligible}/{len(calls)}",
        f"[真实 GLM] {_status(glm_known, glm_passed)} · model={model} · "
        f"remote={remote_call} · "
        f"HTTP={receipt.get('http_status', UNAVAILABLE)} · "
        f"TLS={receipt.get('tls_verification', UNAVAILABLE)} · "
        f"identity={receipt.get('response_model_identity', UNAVAILABLE)}",
        f"[AST / 哈希绑定] {_status(ast_known, ast_passed)} · "
        f"audit={migration.get('ast_audit', UNAVAILABLE)} · "
        f"source={_short(migration.get('source_sha256'))} · "
        f"migrated={_short(migration.get('migrated_sha256'))}",
        f"[数值对拍] {_status(numerical_known, numerical_passed)} · "
        f"max_abs={_scientific(result_comparison.get('max_abs_diff'))} · "
        f"max_rel={_scientific(result_comparison.get('max_rel_diff'))} · "
        f"elements={result_comparison.get('n_elements', UNAVAILABLE)}",
        f"[strict-NPU 门禁] {_status(npu_known, npu_passed)} · "
        f"{passed_gates}/{gate_count} · samples={protocol.get('samples', UNAVAILABLE)} · "
        f"core={protocol.get('core', UNAVAILABLE)} · "
        f"CPU threads={protocol.get('cpu_threads_requested', UNAVAILABLE)}",
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--migration", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--benchmark", required=True)
    args = parser.parse_args(argv)
    print(render_summary(
        Path(args.analysis),
        Path(args.migration),
        Path(args.verification),
        Path(args.benchmark),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
