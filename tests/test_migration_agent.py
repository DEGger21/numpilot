from numpilot.agents.migration import MigrationAgent
from numpilot.engines.ast_rewrite import AstRewriteEngine
from numpilot.contracts import OffloadPlan

SRC = "import numpy as np\ndef forward(x, w):\n    return np.matmul(x, w)\n"

def test_migrate_returns_result_with_engine_name():
    plan = OffloadPlan(placement={"matmul": "NPU"}, threshold_elements=1_000_000)
    res = MigrationAgent(AstRewriteEngine()).migrate(SRC, plan)
    assert res.engine == "ast"
    assert "ap.matmul(" in res.migrated_source
    assert res.expected_rewrites == {"matmul": 1}
    assert res.rewrite_integrity
