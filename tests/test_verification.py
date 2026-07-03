# tests/test_verification.py
from numpilot.agents.analysis import AnalysisAgent
from numpilot.agents.decision import DecisionAgent
from numpilot.agents.migration import MigrationAgent
from numpilot.agents.verification import VerificationAgent
from numpilot.backends.shim import ShimBackend

ORIG = open("examples/mlp_forward.py").read()


def _migrate(src: str) -> str:
    analysis = AnalysisAgent().analyze(src)
    plan = DecisionAgent().plan(analysis)
    return MigrationAgent().migrate(src, plan).migrated_source


def test_migrated_matches_original_under_shim():
    migrated = _migrate(ORIG)
    assert "ap.matmul(" in migrated          # 确认确实发生了 np->ap 迁移
    rep = VerificationAgent(ShimBackend()).verify(ORIG, migrated)
    assert rep.status == "pass"
    assert rep.numerically_consistent
    assert rep.max_rel_diff < rep.tol
