from __future__ import annotations

import json
import os
from pathlib import Path
import types
from types import SimpleNamespace

import numpy as np
import pytest

from numpilot import perf_matrix as matrix
from numpilot.capabilities import ExecutionProvenance


def _timing(values, *, warmup=1):
    return matrix.summarize_samples(
        values,
        warmup_count=warmup,
        warmup_samples_ms=[values[0]] if values else [],
    )


def _measurements():
    return {
        "cpu_warm": _timing([10.0, 10.5, 9.5, 10.2, 10.1, 9.9, 10.3]),
        "cpu_cold": _timing([20.0, 21.0, 19.5, 20.5, 20.2, 20.1, 19.9]),
        "npu_resident": _timing([4.0, 4.1, 3.9, 4.2, 4.0, 4.1, 3.8]),
        "npu_warm_e2e": _timing([8.0, 8.2, 7.8, 8.1, 8.0, 7.9, 8.3]),
        "npu_cold_e2e": _timing([20.0, 21.0, 19.8, 20.4, 20.1, 20.0, 20.2]),
    }


def _checks(consistent=True):
    return [
        {"mode": "cpu_warm", "sample": None, "consistent": consistent},
        {"mode": "npu_resident", "sample": None, "consistent": consistent},
    ]


def _provenance(native=True):
    return {
        "required_operators": ["matmul"],
        "operators": {
            "matmul": {
                "available": True,
                "provenance": "native_npu" if native else "unknown",
                "evidence": "fixture",
                "constraints": [],
            }
        },
        "native_npu": native,
        "audit_id": "fixture-audit",
        "identity": {"asnumpy_version": "fixture", "status": "ok"},
    }


def test_build_inputs_is_deterministic_and_shaped():
    first = matrix.build_inputs("matmul", 32)
    second = matrix.build_inputs("matmul", 32)
    assert first["a"].shape == (32, 32)
    assert first["a"].dtype == np.float32
    assert np.array_equal(first["a"], second["a"])
    assert np.array_equal(first["b"], second["b"])


def test_summarize_samples_reports_quantiles_and_dispersion():
    summary = matrix.summarize_samples(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        warmup_count=2,
        warmup_samples_ms=[0.5, 0.6],
    )
    assert summary["median_ms"] == 3.0
    assert summary["p10_ms"] == pytest.approx(1.4)
    assert summary["p90_ms"] == pytest.approx(4.6)
    assert summary["repeats"] == 5
    assert summary["warmup_count"] == 2
    assert summary["warmup_samples_ms"] == [0.5, 0.6]
    assert summary["stdev_ms"] > 0


def test_bootstrap_ratio_ci_is_deterministic():
    cpu = [10.0, 10.2, 9.8, 10.1, 9.9, 10.3, 10.0]
    npu = [5.0, 5.2, 4.9, 5.1, 5.0, 4.8, 5.1]
    first = matrix.bootstrap_ratio_ci(cpu, npu, seed=42)
    second = matrix.bootstrap_ratio_ci(cpu, npu, seed=42)
    assert first == second
    assert first[0] <= 2.0 <= first[1]


def test_bootstrap_requires_enough_samples():
    assert matrix.bootstrap_ratio_ci([1.0, 2.0], [1.0, 2.0], seed=1) is None


def test_provenance_summary_requires_native_npu_for_every_operator():
    def manifest(provenance):
        return SimpleNamespace(
            audit_id="fixture",
            operators={
                "sin": SimpleNamespace(
                    available=True,
                    provenance=provenance,
                    constraints=(),
                    evidence="fixture",
                ),
                "matmul": SimpleNamespace(
                    available=True,
                    provenance=provenance,
                    constraints=("float32",),
                    evidence="fixture",
                ),
            },
        )

    native = matrix.provenance_summary(manifest(ExecutionProvenance.NATIVE_NPU), "sin_matmul")
    unknown = matrix.provenance_summary(manifest(ExecutionProvenance.UNKNOWN), "sin_matmul")
    assert native["native_npu"] is True
    assert unknown["native_npu"] is False
    assert unknown["operators"]["sin"]["provenance"] == "unknown"
    assert native["operators"]["matmul"]["constraints"] == ["float32"]


def test_sanitize_identity_drops_private_absolute_paths():
    identity = SimpleNamespace(
        asnumpy_version="0.4.1",
        module_sha256="a" * 64,
        native_module_sha256="b" * 64,
        cann_home="/usr/local/Ascend/cann-9.2.0",
        cann_version="9.2.0",
        soc_model="Ascend910B3",
        python_version="3.12.14",
        audit_id="fixture",
        status="audit_required",
        issues=("fixture issue",),
        wrapper_sha256=types.MappingProxyType({"__init__.py": "c" * 64}),
        module_path="/home/example/private/asnumpy/__init__.py",
        bootstrap_receipt_path="/home/example/private/receipt.json",
    )
    sanitized = matrix.sanitize_identity(identity)
    assert sanitized["cann_home"] == "cann-9.2.0"
    assert "module_path" not in sanitized
    assert "bootstrap_receipt_path" not in sanitized
    assert "/home/example" not in json.dumps(sanitized)


def test_classify_row_labels():
    assert matrix.classify_row(
        correctness_ok=False,
        provenance_ok=False,
        ratios={},
    ) == "FAILED"
    assert matrix.classify_row(
        correctness_ok=True,
        provenance_ok=True,
        ratios={},
    ) == "STRICT_NPU_VALIDATED"
    assert matrix.classify_row(
        correctness_ok=True,
        provenance_ok=False,
        ratios={"warm_end_to_end": {"median_ratio": 0.8}},
    ) == "CPU_PREFERRED"
    assert matrix.classify_row(
        correctness_ok=True,
        provenance_ok=False,
        ratios={
            "resident": {"median_ratio": 2.0},
            "warm_end_to_end": {"median_ratio": 1.1},
        },
    ) == "RESIDENT_GAIN"
    assert matrix.classify_row(
        correctness_ok=True,
        provenance_ok=False,
        ratios={
            "resident": {"median_ratio": 2.0},
            "warm_end_to_end": {"median_ratio": 0.8},
        },
    ) == "RESIDENT_GAIN"
    assert matrix.classify_row(
        correctness_ok=True,
        provenance_ok=False,
        ratios={
            "resident": {"median_ratio": 1.2},
            "warm_end_to_end": {"median_ratio": 1.5},
        },
    ) == "DIAGNOSTIC"


def test_crossover_finds_first_size_at_or_above_threshold():
    rows = [
        {"size": 64, "ratios": {"resident": {"median_ratio": 0.5}}},
        {"size": 128, "ratios": {"resident": {"median_ratio": 0.9}}},
        {"size": 256, "ratios": {"resident": {"median_ratio": 1.3}}},
        {"size": 512, "ratios": {"resident": {"median_ratio": 2.0}}},
    ]
    result = matrix.crossover(rows, "resident")
    assert result["first_size_at_or_above"] == 256
    assert result["cpu_preferred_sizes"] == [64, 128]
    assert result["npu_preferred_sizes"] == [256, 512]


def test_assemble_row_omits_ratios_after_correctness_failure():
    row = matrix.assemble_row(
        profile="matmul",
        size=64,
        measurements=_measurements(),
        checks=_checks(consistent=False),
        provenance=_provenance(native=True),
        seed=1,
    )
    assert row["correctness"]["ok"] is False
    assert row["ratios"] == {}
    assert row["label"] == "FAILED"


def test_assemble_row_is_profile_required_without_strict_provenance():
    row = matrix.assemble_row(
        profile="matmul",
        size=64,
        measurements=_measurements(),
        checks=_checks(),
        provenance=_provenance(native=False),
        seed=1,
    )
    assert row["correctness"]["ok"] is True
    assert row["ratios"]["resident"]["median_ratio"] == pytest.approx(10.1 / 4.0, rel=0.02)
    assert row["ratios"]["resident"]["publishable"] is False
    assert row["label"] in {"CPU_PREFERRED", "RESIDENT_GAIN", "DIAGNOSTIC"}


def test_ratio_publishable_applies_all_public_claim_gates():
    base = {
        "median_ratio": 2.0,
        "conservative_p10_over_p90": 1.2,
        "bootstrap_95ci": [1.4, 2.8],
    }
    assert matrix.ratio_publishable(base, 1.5) is True
    assert matrix.ratio_publishable({**base, "median_ratio": 1.4}, 1.5) is False
    assert (
        matrix.ratio_publishable({**base, "conservative_p10_over_p90": 1.05}, 1.5)
        is False
    )
    assert (
        matrix.ratio_publishable({**base, "bootstrap_95ci": [0.95, 2.8]}, 1.5)
        is False
    )
    assert (
        matrix.ratio_publishable({**base, "bootstrap_95ci": None}, 1.5)
        is False
    )


def test_matrix_status():
    assert matrix.matrix_status(correctness_ok=False, performance_valid=False) == "failed"
    assert matrix.matrix_status(correctness_ok=True, performance_valid=False) == "profile_required"
    assert matrix.matrix_status(correctness_ok=True, performance_valid=True) == "pass"


def test_timed_call_returns_last_value_and_samples():
    counter = {"value": 0}

    def function():
        counter["value"] += 1
        return counter["value"]

    summary, last = matrix.timed_call(function, warmup=2, repeats=3)
    assert last == 5
    assert summary["warmup_count"] == 2
    assert summary["repeats"] == 3


def test_float64_forward_matches_numpy_oracle():
    inputs = matrix.build_inputs("sin_matmul", 16)
    oracle = matrix.float64_forward("sin_matmul", inputs)
    expected = np.sin(inputs["a"].astype(np.float64)) @ inputs["b"].astype(np.float64)
    assert oracle.dtype == np.float64
    assert np.array_equal(oracle, expected)


def test_gemm_check_accepts_float32_rounding_and_rejects_corruption(tmp_path):
    inputs = matrix.build_inputs("matmul", 32)
    oracle = matrix.float64_forward("matmul", inputs)
    limit = matrix.gemm_max_abs_limit("matmul", inputs)
    path = tmp_path / "output.npy"
    np.save(path, oracle.astype(np.float32))
    accepted = matrix.gemm_check(oracle, path, max_abs_limit=limit)
    assert accepted["consistent"] is True
    assert accepted["relative_frobenius"] < 1e-5

    corrupted = oracle.astype(np.float32)
    corrupted[0, 0] += 1.0
    np.save(path, corrupted)
    rejected = matrix.gemm_check(oracle, path, max_abs_limit=limit)
    assert rejected["consistent"] is False
    assert rejected["max_abs_diff"] >= 0.9


def test_gemm_max_abs_limit_scales_with_reduction_size():
    small = matrix.build_inputs("matmul", 32)
    large = matrix.build_inputs("matmul", 1024)
    assert matrix.gemm_max_abs_limit("matmul", large) > matrix.gemm_max_abs_limit(
        "matmul", small
    )


def test_compare_saved_output_detects_mismatch(tmp_path):
    reference = np.arange(8, dtype=np.float32).reshape(2, 4)
    path = tmp_path / "output.npy"
    np.save(path, reference)
    assert matrix.compare_saved_output(reference, path)["consistent"] is True
    np.save(path, reference + 1.0)
    result = matrix.compare_saved_output(reference, path)
    assert result["consistent"] is False
    assert result["max_abs_diff"] >= 1.0


@pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity"),
    reason="CPU affinity control is unavailable on this platform",
)
def test_cpu_worker_receipt_roundtrip(tmp_path):
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    tool = repo_root / "tools" / "performance_matrix.py"
    core = min(os.sched_getaffinity(0))
    completed = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--worker",
            "cpu_warm",
            "--profile",
            "matmul",
            "--size",
            "8",
            "--worker-out",
            str(tmp_path),
            "--core",
            str(core),
            "--warmup",
            "1",
            "--repeats",
            "3",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["status"] == "ok"
    assert receipt["timing"]["repeats"] == 3
    assert receipt["output"]["shape"] == [8, 8]
    assert np.load(tmp_path / "output.npy", allow_pickle=False).shape == (8, 8)
