#!/usr/bin/env python3
"""Render concise, fail-closed summaries for the Poisson customer demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


class DemoEvidenceError(RuntimeError):
    """Raised when a showcase receipt does not support the displayed claim."""


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DemoEvidenceError(f"无法读取证据 {path}: {error}") from error
    if not isinstance(value, dict):
        raise DemoEvidenceError(f"证据顶层必须是 JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DemoEvidenceError(message)


def _checks_passed(value) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(result is True for result in value.values())
    )


def verify_public_manifest(repo_root: Path) -> tuple[int, int]:
    """Verify every curated public artifact before displaying its claims."""

    manifest = repo_root / "evidence" / "MANIFEST.sha256"
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DemoEvidenceError(f"无法读取公开证据 manifest: {error}") from error
    evidence_root = (repo_root / "evidence").resolve()
    seen: set[str] = set()
    total_bytes = 0
    for line_number, line in enumerate(lines, start=1):
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as error:
            raise DemoEvidenceError(
                f"manifest 第 {line_number} 行格式无效"
            ) from error
        _require(relative.startswith("evidence/"), "manifest 包含越界路径")
        _require(relative not in seen, f"manifest 包含重复路径: {relative}")
        target = (repo_root / relative).resolve()
        try:
            target.relative_to(evidence_root)
        except ValueError as error:
            raise DemoEvidenceError(f"manifest 包含越界路径: {relative}") from error
        _require(target.is_file(), f"公开证据缺失: {relative}")
        payload = target.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        _require(actual == expected, f"公开证据哈希不匹配: {relative}")
        seen.add(relative)
        total_bytes += len(payload)
    _require(bool(seen), "公开证据 manifest 为空")
    return len(seen), total_bytes


def load_published_summary(repo_root: Path) -> dict:
    evidence = repo_root / "evidence" / "poisson-ritz-p0-20260821"
    qualification = _load_json(
        evidence
        / "profiler-copy-elision/qualification-v6-move/qualification-summary.json"
    )
    migration = _load_json(
        evidence / "performance-broadcast-optimized/migration/report.json"
    )
    baseline = _load_json(
        evidence / "performance-original-scale/final/evidence.json"
    )
    optimized = _load_json(
        evidence / "performance-broadcast-optimized/final/evidence.json"
    )
    copy_audit = _load_json(
        evidence / "profiler-copy-elision/audit/evidence.json"
    )

    _require(
        qualification.get("status") == "qualified_and_overlay_published",
        "Poisson 算子 qualification 未通过",
    )
    operators = qualification.get("operators")
    _require(isinstance(operators, dict) and len(operators) == 2, "qualification 算子集合无效")
    for name in ("apply_gradient_descent", "smooth_l1_loss"):
        record = operators.get(name, {})
        _require(record.get("tests_passed", 0) > 0, f"{name} 没有通过测试")
        _require(record.get("kernel_launches", 0) > 0, f"{name} 没有真实 kernel launch")

    provenance = migration.get("backend_provenance") or {}
    _require(migration.get("status") == "pass", "广播优化迁移报告未通过")
    _require(migration.get("rewrite_integrity") is True, "迁移改写完整性失败")
    _require(migration.get("numerically_consistent") is True, "迁移数值门失败")
    _require(provenance.get("ok") is True and provenance.get("strict_npu") is True,
             "迁移报告不是 strict-NPU")

    for label, receipt in (("基线", baseline), ("优化", optimized)):
        perf = receipt.get("performance") or {}
        _require(receipt.get("status") == "pass", f"{label}性能证据未通过")
        _require(perf.get("published") is True, f"{label}性能证据未准入发布")
        _require(receipt.get("comparison", {}).get("consistent") is True,
                 f"{label} CPU/NPU 数值对拍失败")
        _require(receipt.get("npu", {}).get("strict_npu") is True,
                 f"{label}不是 strict-NPU")
        _require(_checks_passed((receipt.get("profiler_audit") or {}).get("checks")),
                 f"{label} profiler 门禁失败")

    _require(copy_audit.get("status") == "pass", "复制消除审计未通过")
    _require(_checks_passed(copy_audit.get("checks")), "复制消除检查未全部通过")

    before_ms = float(baseline["performance"]["npu_median_per_iteration_ms"])
    after_ms = float(optimized["performance"]["npu_median_per_iteration_ms"])
    cpu_ms = float(optimized["performance"]["cpu_median_per_iteration_ms"])
    _require(after_ms > 0 and cpu_ms > 0, "性能证据包含非正时延")
    return {
        "operators": operators,
        "overlay_sha256": qualification["overlay_sha256"],
        "rewrites": int(migration["total_rewrites"]),
        "operator_count": len(migration["rewrites"]),
        "migration_max_abs": float(migration["max_abs_diff"]),
        "scale": optimized["scale"],
        "iterations": int(optimized["workload"]["convergence_iterations"]),
        "comparison_max_abs": float(optimized["comparison"]["max_abs_diff"]),
        "initial_loss": float(optimized["npu"]["convergence"]["initial_loss"]),
        "final_loss": float(optimized["npu"]["convergence"]["final_loss"]),
        "baseline_calls": sum(baseline["npu"]["per_iteration_calls"].values()),
        "optimized_calls": sum(optimized["npu"]["per_iteration_calls"].values()),
        "baseline_npu_ms": before_ms,
        "optimized_npu_ms": after_ms,
        "optimization_speedup": before_ms / after_ms,
        "latency_reduction_percent": (before_ms - after_ms) / before_ms * 100.0,
        "cpu_ms": cpu_ms,
        "cpu_faster": after_ms / cpu_ms,
        "copies_before": int(copy_audit["before"]["rows"]),
        "copies_after": int(copy_audit["after"]["rows"]),
    }


def render_published(summary: dict, manifest: tuple[int, int] | None = None) -> str:
    scale = summary["scale"]
    lines = [
        "NumPilot · Poisson Ritz 可审计迁移成果",
        "=" * 58,
    ]
    if manifest is not None:
        lines.append(
            f"[证据] 公开制品哈希校验通过: {manifest[0]} 个文件 / "
            f"{manifest[1] / (1024 * 1024):.2f} MiB"
        )
    names = ", ".join(sorted(summary["operators"]))
    tests = sum(item["tests_passed"] for item in summary["operators"].values())
    launches = sum(item["kernel_launches"] for item in summary["operators"].values())
    lines.extend([
        f"[算子] {names}: 六门 qualification 通过 "
        f"({tests} tests, {launches} 次真实 kernel launch)",
        f"[迁移] {summary['rewrites']}/{summary['rewrites']} 个调用点改写完整，"
        f"{summary['operator_count']} 个 API 全部 strict-NPU，"
        f"max_abs={summary['migration_max_abs']:.3g}",
        f"[训练] {scale['interior_axis']}×{scale['interior_axis']} 内点 / "
        f"{scale['boundary_points']} 边界点 / {scale['rbf_count']} RBF / "
        f"{summary['iterations']} 轮，loss {summary['initial_loss']:.6f} → "
        f"{summary['final_loss']:.6f}",
        f"[数值] CPU/NPU 最终状态对拍通过，max_abs="
        f"{summary['comparison_max_abs']:.3g}",
        f"[优化] 每轮调用 {summary['baseline_calls']} → "
        f"{summary['optimized_calls']}，NPU {summary['baseline_npu_ms']:.3f} → "
        f"{summary['optimized_npu_ms']:.3f} ms "
        f"({summary['optimization_speedup']:.3f}×，时延 -"
        f"{summary['latency_reduction_percent']:.2f}%)",
        f"[传输] aclrtMemcpy {summary['copies_before']} → "
        f"{summary['copies_after']}，profiler 账目通过",
        f"[诚信结论] 当前细粒度实现中 CPU 仍约快 "
        f"{summary['cpu_faster']:.3f}×；不宣称 NPU 超越 CPU。",
    ])
    return "\n".join(lines)


def render_live_migration(path: Path) -> str:
    report = _load_json(path)
    provenance = report.get("backend_provenance") or {}
    _require(report.get("status") == "pass", "实时迁移未通过")
    _require(report.get("rewrite_integrity") is True, "实时改写完整性失败")
    _require(report.get("numerically_consistent") is True, "实时数值校验失败")
    _require(provenance.get("ok") is True and provenance.get("strict_npu") is True,
             "实时执行不是 strict-NPU")
    _require(provenance.get("runtime_profile_ok") is True, "实时 runtime profile 失败")
    _require(provenance.get("runtime_target_integrity_ok") is True,
             "实时 runtime target identity 失败")
    calls = sum((provenance.get("runtime_call_counts") or {}).get("artifact", {}).values())
    _require(calls == report.get("total_rewrites"), "实时调用轨迹与静态改写不一致")
    return (
        f"[实时迁移] pass · {report['total_rewrites']} 处改写 · "
        f"{calls} 次公开 AsNumpy API 真实调用 · strict-NPU=yes · "
        f"max_abs={float(report['max_abs_diff']):.3g}"
    )


def render_live_performance(path: Path) -> str:
    receipt = _load_json(path)
    _require(receipt.get("status") == "pass", "实时 CPU/NPU worker 未通过")
    _require(receipt.get("comparison", {}).get("consistent") is True, "实时 CPU/NPU 对拍失败")
    npu = receipt.get("npu") or {}
    _require(npu.get("strict_npu") is True, "实时 NPU worker 不是 strict-NPU")
    _require(npu.get("runtime_profile_ok") is True, "实时 runtime profile 失败")
    _require(npu.get("runtime_target_integrity_ok") is True,
             "实时 runtime target identity 失败")
    perf = receipt.get("performance") or {}
    scale = receipt["scale"]
    line = (
        f"[实时运行] pass · {scale['interior_axis']}×{scale['interior_axis']} 内点 · "
        f"{receipt['workload']['convergence_iterations']} 轮驻留执行 · "
        f"{npu['total_runtime_calls']} 次 API · max_abs="
        f"{float(receipt['comparison']['max_abs_diff']):.3g}"
    )
    if perf.get("published"):
        ratio = float(perf["cpu_time_over_npu_time"])
        verdict = (
            f"NPU 为 CPU 的 {ratio:.3f}×"
            if ratio >= 1 else f"CPU 约快 {1.0 / ratio:.3f}×"
        )
        line += f"\n[实时性能] 全部发布门通过 · {verdict}"
    else:
        issues = "; ".join(perf.get("issues") or ["未知原因"])
        line += f"\n[实时性能] 仅体验采样，不发布比值: {issues}"
    return line


def render_affinity_performance(path: Path) -> str:
    """Verify and render a one-core, enlarged-scale diagnostic benchmark."""

    receipt = _load_json(path)
    _require(receipt.get("status") == "pass", "单核 CPU/NPU worker 未通过")
    _require(receipt.get("comparison", {}).get("consistent") is True,
             "单核 CPU/NPU 对拍失败")
    cpu = receipt.get("cpu") or {}
    npu = receipt.get("npu") or {}
    affinity = cpu.get("cpu_affinity") or {}
    requested = affinity.get("requested_logical_cpu")
    observed = affinity.get("observed_logical_cpus")
    _require(
        affinity.get("pinned") is True
        and isinstance(requested, int)
        and observed == [requested]
        and affinity.get("logical_cpu_count") == 1,
        "CPU worker 没有被验证为仅使用一个逻辑核",
    )
    _require(cpu.get("cpu_threads_requested") == 1, "CPU BLAS 请求不是单线程")
    blas_pools = [
        pool for pool in cpu.get("threadpools", [])
        if pool.get("user_api") == "blas"
    ]
    _require(bool(blas_pools), "没有观测到 CPU BLAS 运行时线程数")
    _require(all(pool.get("num_threads") == 1 for pool in blas_pools),
             "CPU BLAS 运行时不是单线程")
    for key in ("strict_npu", "runtime_profile_ok", "runtime_target_integrity_ok"):
        _require(npu.get(key) is True, f"NPU 门禁失败: {key}")
    perf = receipt.get("performance") or {}
    _require(perf.get("measured") is True, "单核性能没有完成采样")
    cpu_ms = float(perf.get("cpu_median_per_iteration_ms", 0))
    npu_ms = float(perf.get("npu_median_per_iteration_ms", 0))
    _require(cpu_ms > 0 and npu_ms > 0, "单核性能时延必须为正数")
    ratio = cpu_ms / npu_ms
    verdict = (
        f"NPU 原始中位时延约快 {ratio:.3f}×"
        if ratio > 1 else f"CPU 原始中位时延约快 {1.0 / ratio:.3f}×"
        if ratio < 1 else "CPU/NPU 原始中位时延相同"
    )
    scale = receipt.get("scale") or {}
    matrix_elements = int(scale["interior_points"]) * int(scale["rbf_count"])
    lines = [
        f"[单核约束] pass · CPU worker 仅允许逻辑核 {requested} · "
        "BLAS runtime threads=1",
        f"[放大实验] {scale['interior_axis']}×{scale['interior_axis']} 内点 · "
        f"{scale['boundary_points']} 边界点 · {scale['rbf_count']} RBF · "
        f"内点×RBF={matrix_elements:,}",
        f"[数值与执行] strict-NPU=yes · {npu['total_runtime_calls']} 次 API · "
        f"max_abs={float(receipt['comparison']['max_abs_diff']):.3g}",
        f"[诊断时延] CPU {cpu_ms:.3f} ms/轮 · NPU {npu_ms:.3f} ms/轮 · "
        f"{verdict}",
    ]
    if perf.get("published"):
        lines.append("[证据边界] profiler 与全部发布门已通过，可发布该性能比值")
    else:
        issues = "; ".join(perf.get("issues") or ["缺少匹配的 profiler audit"])
        lines.append(f"[证据边界] 诊断采样，不发布为正式性能比值: {issues}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="section", required=True)
    published = subparsers.add_parser("published", help="校验并展示公开 Poisson 证据")
    published.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    published.add_argument("--skip-manifest", action="store_true")
    migration = subparsers.add_parser("migration", help="校验并展示实时迁移报告")
    migration.add_argument("report")
    performance = subparsers.add_parser("performance", help="校验并展示实时评测")
    performance.add_argument("evidence")
    affinity = subparsers.add_parser(
        "affinity-performance", help="校验并展示单核放大规模实验"
    )
    affinity.add_argument("evidence")
    args = parser.parse_args(argv)
    try:
        if args.section == "published":
            root = Path(args.repo_root).resolve()
            manifest = None if args.skip_manifest else verify_public_manifest(root)
            print(render_published(load_published_summary(root), manifest))
        elif args.section == "migration":
            print(render_live_migration(Path(args.report).resolve()))
        elif args.section == "affinity-performance":
            print(render_affinity_performance(Path(args.evidence).resolve()))
        else:
            print(render_live_performance(Path(args.evidence).resolve()))
    except DemoEvidenceError as error:
        parser.exit(2, f"Poisson 证据失败关闭: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
