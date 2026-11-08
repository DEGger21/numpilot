from pathlib import Path

from numpilot.backends.shim import ShimBackend
from numpilot.benchmark_suite import run_suite
from numpilot.engines.ast_rewrite import AstRewriteEngine


def test_run_suite_writes_runtime_qualification_receipt(tmp_path):
    root = Path(__file__).resolve().parents[1] / "benchmark" / "cases"
    summary = run_suite(
        root,
        tmp_path / "run",
        backend=ShimBackend(),
        engine=AstRewriteEngine(),
        groups=["A_direct_simple"],
        limit=1,
    )
    assert summary["total"] == 1
    assert summary["operator_artifacts"]["runtime_qualifications"] == 1
    case = summary["cases"][0]
    assert case["runtime_identity_status"] == "not_applicable"
    assert case["runtime_qualification_id"]
    case_root = tmp_path / "run" / "cases" / "A_direct_simple" / case["name"]
    assert (case_root / "runtime-qualification.json").is_file()
