"""Tests for agent contracts layer."""
from numpilot.contracts import (
    OpUse, AnalysisResult, OffloadPlan, MigrationResult, VerificationReport,
)


def test_opuse_fields():
    op = OpUse(name="matmul", lineno=3, supported=True, est_elements=None)
    assert op.name == "matmul" and op.supported


def test_analysis_result_migratable():
    r = AnalysisResult(source_path="a.py", op_uses=[], unsupported=[], migratable=True)
    assert r.migratable and r.unsupported == []


def test_report_status():
    rep = VerificationReport(numerically_consistent=True, max_rel_diff=0.0,
                             tol=1e-4, speedup=2.0, per_op=[], status="pass")
    assert rep.status == "pass"
