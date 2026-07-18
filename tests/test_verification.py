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
    assert rep.artifact_inputs_executed


def test_verifier_rejects_forward_error_compensated_by_changed_inputs():
    original = """
import numpy as np
def build_inputs():
    return (np.zeros(3, dtype=np.float32),)
def forward(x):
    return x
"""
    # artifact 结果看似一致，但同一规范输入上的 forward 明显错误，必须失败。
    migrated = """
import numpy as np
def build_inputs():
    return (np.ones(3, dtype=np.float32),)
def forward(x):
    return x - np.float32(1.0)
"""
    report = VerificationAgent(ShimBackend()).verify(original, migrated, measure=False)
    assert report.status == "fail"
    assert not report.kernel_consistent
    assert not report.input_consistent
    assert report.artifact_inputs_executed


def test_migrated_artifact_runs_its_own_build_inputs():
    original = """
import numpy as np
def build_inputs():
    return (np.zeros(3, dtype=np.float32),)
def forward(x):
    return x
"""
    migrated = """
import numpy as np
BUILT = False
def build_inputs():
    global BUILT
    BUILT = True
    return (np.zeros(3, dtype=np.float32),)
def forward(x):
    if not BUILT:
        raise RuntimeError('artifact build_inputs was skipped')
    return x
"""
    report = VerificationAgent(ShimBackend()).verify(original, migrated, measure=False)
    assert report.status == "pass"
    assert report.artifact_inputs_executed
    assert report.input_consistent
    assert report.kernel_consistent
    assert report.artifact_consistent


def test_migrated_runtime_exception_is_reported():
    import json
    from dataclasses import asdict

    original = """
import numpy as np
def build_inputs():
    return (np.zeros(1, dtype=np.float32),)
def forward(x):
    return x
"""
    migrated = """
import numpy as np
def build_inputs():
    return (np.zeros(1, dtype=np.float32),)
def forward(x):
    raise RuntimeError('fault injection')
"""
    report = VerificationAgent(ShimBackend()).verify(original, migrated, measure=False)
    assert report.status == "fail"
    assert report.exception_phase == "migrated.forward"
    assert report.exception_type == "RuntimeError"
    assert "fault injection" in report.exception_message
    assert "RuntimeError: fault injection" in report.traceback
    json.dumps(asdict(report), allow_nan=False)


def test_npu_provenance_fails_closed_when_no_direct_ap_call_is_traceable():
    from types import SimpleNamespace

    from numpilot.backends.npu import NpuBackend

    original = """
import numpy as np
def build_inputs(): return (np.ones(2, dtype=np.float32),)
def forward(x): return np.matmul(x, x)
"""
    migrated = """
import asnumpy as ap
import numpy as np
op = getattr(ap, 'matmul')
def build_inputs(): return (np.ones(2, dtype=np.float32),)
def forward(x): return op(x, x)
"""
    module = SimpleNamespace(
        __version__="0.3.1.dev4+g1929a3ba4",
        __file__="/fake/asnumpy/__init__.py",
    )
    report = VerificationAgent(NpuBackend(module=module)).verify(
        original, migrated, measure=False
    )
    assert report.status == "fail"
    assert report.exception_phase in {"migrated.surface", "migrated.provenance"}
    assert "未审计" in report.exception_message or "fail closed" in report.exception_message


def _strict_sqrt_backend():
    import types

    import numpy as np

    from numpilot.backends.npu import AUDITED_ASNUMPY_VERSION, NpuBackend

    module = types.ModuleType("asnumpy")
    module.__version__ = AUDITED_ASNUMPY_VERSION
    module.__file__ = "/fake/asnumpy/__init__.py"
    module.sqrt = np.sqrt
    return NpuBackend(module=module)


def test_verifier_rejects_dead_ap_decoy_and_host_numpy_escape():
    original = """
import numpy as np
def build_inputs(): return (np.ones(3, dtype=np.float32),)
def forward(x): return np.sqrt(x)
"""
    migrated = """
import numpy as np
import asnumpy as ap
def build_inputs(): return (np.ones(3, dtype=np.float32),)
def forward(x):
    if False:
        return ap.sqrt(x)
    host = x.to_numpy() if hasattr(x, 'to_numpy') else x
    return np.sqrt(host)
"""
    report = VerificationAgent(_strict_sqrt_backend()).verify(
        original, migrated, measure=False, required_ops={"sqrt": 1}
    )
    assert report.status == "fail"
    assert report.exception_phase == "migrated.surface"
    assert "host" in report.exception_message.lower()


def test_verifier_rejects_statically_present_but_unexecuted_ap_call():
    original = """
import numpy as np
def build_inputs(): return (np.ones(3, dtype=np.float32),)
def forward(x): return np.sqrt(x)
"""
    migrated = """
import numpy as np
import asnumpy as ap
def build_inputs(): return (np.ones(3, dtype=np.float32),)
def forward(x):
    if False:
        return ap.sqrt(x)
    return x
"""
    report = VerificationAgent(_strict_sqrt_backend()).verify(
        original, migrated, measure=False, required_ops={"sqrt": 1}
    )
    assert report.status == "fail"
    assert report.exception_phase == "migrated.surface"
    assert "控制流" in report.exception_message or "数据流" in report.exception_message


def test_verifier_rejects_executed_ap_decoy_with_global_host_result():
    original = """
import numpy as np
FIXED = np.ones(3, dtype=np.float32)
def build_inputs(): return (FIXED.copy(),)
def forward(x): return np.sqrt(x)
"""
    migrated = """
import numpy as np
import asnumpy as ap
PRECOMPUTED = np.sqrt(np.ones(3, dtype=np.float32))
def build_inputs(): return (np.ones(3, dtype=np.float32),)
def forward(x):
    ignored = ap.sqrt(x)
    return PRECOMPUTED
"""
    report = VerificationAgent(_strict_sqrt_backend()).verify(
        original, migrated, measure=False, required_ops={"sqrt": 1}
    )
    assert report.status == "fail"
    assert report.exception_phase == "migrated.surface"
    assert "PRECOMPUTED" in report.exception_message
    assert "不依赖" in report.exception_message
