"""有界修复回路（架构图：Bounded Repair, up to 3 iterations）。"""
from numpilot.backends.shim import ShimBackend
from numpilot.orchestrator import MAX_REPAIR_ITERATIONS, Orchestrator

SOURCE = (
    "import numpy as np\n"
    "def build_inputs(): return (np.ones(3, dtype=np.float32),)\n"
    "def forward(x): return np.sqrt(x)\n"
)

TEMPLATE = (
    "import numpy as np\n"
    "import asnumpy as ap\n"
    "def build_inputs(): return (np.ones(3, dtype=np.float32),)\n"
    "def forward(x):\n"
    "    if isinstance(x, np.ndarray):\n"
    "        x = ap.ndarray.from_numpy(x)\n"
    "    return ap.%s(x)\n"
)


class _Engine:
    name = "repair-fake"

    def __init__(self, repairs):
        self.repairs = repairs
        self.repair_calls = 0

    def rewrite(self, source, plan):
        return TEMPLATE % "square"

    def repair(self, source, plan, previous_source, diagnosis):
        self.repair_calls += 1
        if self.repairs <= 0:
            return TEMPLATE % "square"
        self.repairs -= 1
        return TEMPLATE % "sqrt"


def _run(tmp_path, engine):
    result = Orchestrator(
        backend=ShimBackend(), engine=engine
    ).run_source(SOURCE, str(tmp_path / "case.py"), out_dir=str(tmp_path / "out"))
    return result


def test_repair_loop_recovers_and_records_attempt(tmp_path):
    engine = _Engine(repairs=1)
    result = _run(tmp_path, engine)
    assert result["status"] == "pass"
    assert result["repair_attempts"] == 1
    assert engine.repair_calls == 1
    assert result["rewrites"] == {"sqrt": 1}


def test_repair_loop_is_bounded_at_three_iterations(tmp_path):
    engine = _Engine(repairs=0)
    result = _run(tmp_path, engine)
    assert result["status"] == "fail"
    assert result["repair_attempts"] == MAX_REPAIR_ITERATIONS == 3
    assert engine.repair_calls == 3


def test_ast_engine_never_enters_the_repair_loop(tmp_path):
    result = Orchestrator(backend=ShimBackend()).run_source(
        SOURCE, str(tmp_path / "case.py")
    )
    assert result["status"] == "pass"
    assert result["repair_attempts"] == 0
    assert result["route"] == "direct_migration"
