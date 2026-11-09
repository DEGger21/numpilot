import json
from pathlib import Path

from numpilot.backends.shim import ShimBackend
from numpilot.benchmark_suite import (
    SCOPE_CATEGORIES,
    discover_cases,
    load_scope,
    resolve_scope,
    run_suite,
)
from numpilot.engines.ast_rewrite import AstRewriteEngine


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "benchmark" / "cases"
SCOPE_PATH = ROOT / "benchmark" / "scopes" / "numpy_compute_only.json"


def test_numpy_compute_scope_manifest_covers_immutable_corpus():
    scope = load_scope(SCOPE_PATH)
    discovered = discover_cases(CASES)
    assert len(discovered) == 150
    assert set(scope["include_categories"]) == set(SCOPE_CATEGORIES) - {"out_of_scope"}

    counts = {category: 0 for category in SCOPE_CATEGORIES}
    for group, name, _program in discovered:
        entry = resolve_scope(scope, group, name)
        counts[entry["category"]] += 1
    assert counts == {
        "covered_compute": 48,
        "contract_limited": 17,
        "single_gap": 35,
        "official_gap": 15,
        "multi_gap": 20,
        "out_of_scope": 15,
    }


def test_scope_filters_out_of_scope_cases_and_records_identity(tmp_path):
    summary = run_suite(
        CASES,
        tmp_path / "run",
        backend=ShimBackend(),
        engine=AstRewriteEngine(),
        groups=["A_direct_simple", "J_out_of_subset"],
        scope=SCOPE_PATH,
    )
    assert summary["total"] == 15
    assert summary["expected_total"] == 15
    assert summary["scope"]["name"] == "numpy_compute_only"
    assert summary["scope"]["category_counts"] == {"covered_compute": 15}
    assert all(case["scope"]["category"] == "covered_compute" for case in summary["cases"])
    receipt = tmp_path / "run" / "cases" / "A_direct_simple" / "A001_add_sqrt" / "receipt.json"
    assert json.loads(receipt.read_text(encoding="utf-8"))["scope"]["name"] == "numpy_compute_only"


def test_scope_can_explicitly_run_refusal_category(tmp_path):
    scope = load_scope(SCOPE_PATH)
    scope["include_categories"] = ["out_of_scope"]
    summary = run_suite(
        CASES,
        tmp_path / "run",
        backend=ShimBackend(),
        engine=AstRewriteEngine(),
        groups=["J_out_of_subset"],
        scope=scope,
    )
    assert summary["total"] == 15
    assert summary["status_counts"] == {"refused": 15}
    assert summary["scope"]["category_counts"] == {"out_of_scope": 15}


def test_scope_categories_selects_one_phase_and_records_actual_include(tmp_path):
    summary = run_suite(
        CASES,
        tmp_path / "run",
        backend=ShimBackend(),
        engine=AstRewriteEngine(),
        groups=["B_direct_semantics"],
        scope=SCOPE_PATH,
        scope_categories=["contract_limited"],
    )
    assert summary["total"] == 7
    assert summary["scope"]["include_categories"] == ["contract_limited"]
    assert summary["scope"]["category_counts"] == {"contract_limited": 7}
    assert {case["name"] for case in summary["cases"]} == {
        "B001_out_buffer",
        "B002_where_preserves",
        "B003_unsafe_cast",
        "B011_nan_inf_extrema",
        "B014_overlapping_alias",
        "B016_reverse_strides",
        "B018_signed_zero",
    }
