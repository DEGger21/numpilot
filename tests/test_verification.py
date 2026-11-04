# tests/test_verification.py
import numpy as np

from numpilot.agents.analysis import AnalysisAgent
from numpilot.agents.decision import DecisionAgent
from numpilot.agents.migration import MigrationAgent
from numpilot.agents.verification import VerificationAgent, _clone_tree
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
    assert "既不来自直接 ap.* 结果" in report.exception_message


def _scoped_cbrt_backend(target_calls):
    import types

    import numpy as np

    from numpilot.backends.npu import NpuBackend
    from numpilot.capabilities import (
        CapabilityManifest,
        ExecutionProvenance,
        OperatorCapability,
        OperatorSignature,
    )

    class ArrayFactory:
        @staticmethod
        def from_numpy(value):
            return value

    module = types.ModuleType("asnumpy")
    module.__version__ = "private-cbrt"
    module.__file__ = "/private/asnumpy/__init__.py"
    module.ndarray = ArrayFactory

    def cbrt(value):
        target_calls.append(value.dtype.name)
        return np.cbrt(value)

    module.cbrt = cbrt

    class ScopedBackend(NpuBackend):
        def capability_manifest(self, required_ops=None):
            return CapabilityManifest(
                backend="npu",
                backend_version=module.__version__,
                audit_id="qualified-cbrt-test",
                operators={
                    "cbrt": OperatorCapability(
                        name="cbrt",
                        available=True,
                        provenance=ExecutionProvenance.CUSTOM_NPU,
                        evidence="qualification:qual-0123456789abcdefabcd",
                        constraints=(
                            "dtype=float32",
                            "rank>=1",
                            "contiguous=true",
                        ),
                        signature_variants=(
                            OperatorSignature(positional_args=1),
                        ),
                    )
                },
            )

    return ScopedBackend(module=module)


def _cbrt_sources(dtype):
    original = f"""
import numpy as np
def build_inputs(): return (np.array([1.0, 8.0], dtype=np.{dtype}),)
def forward(x): return np.cbrt(x)
"""
    migrated = f"""
import numpy as np
import asnumpy as ap
def build_inputs(): return (np.array([1.0, 8.0], dtype=np.{dtype}),)
def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    return ap.cbrt(x)
"""
    return original, migrated


def test_verifier_executes_scoped_custom_op_only_after_profile_match():
    target_calls = []
    original, migrated = _cbrt_sources("float32")

    report = VerificationAgent(_scoped_cbrt_backend(target_calls)).verify(
        original,
        migrated,
        measure=False,
        required_ops={"cbrt": 1},
    )

    assert report.status == "pass"
    assert target_calls == ["float32", "float32"]
    assert report.backend_provenance["profile_scoped"] == ["cbrt"]
    assert report.backend_provenance["runtime_profile_ok"] is True
    assert len(report.backend_provenance["runtime_profile_checks"]["cbrt"]) == 2


def test_verifier_rejects_scoped_dtype_before_custom_op_dispatch():
    target_calls = []
    original, migrated = _cbrt_sources("float64")

    report = VerificationAgent(_scoped_cbrt_backend(target_calls)).verify(
        original,
        migrated,
        measure=False,
        required_ops={"cbrt": 1},
    )

    assert report.status == "fail"
    assert report.exception_phase == "migrated.forward"
    assert report.exception_type == "RuntimeProfileError"
    assert "dtype=float64" in report.exception_message
    assert target_calls == []
    checks = report.backend_provenance["runtime_profile_checks"]["cbrt"]
    assert checks and checks[0]["matched"] is False


def test_verifier_rejects_original_source_target_monkeypatch_before_dispatch():
    target_calls = []
    backend = _scoped_cbrt_backend(target_calls)
    original, migrated = _cbrt_sources("float32")
    original = original.replace(
        "def build_inputs(): return",
        "def build_inputs():\n    backend.cbrt = np.cbrt\n    return",
        1,
    )
    original = "import sys\nbackend = sys.modules['poison_backend']\n" + original

    import sys
    sys.modules["poison_backend"] = backend.get_module()

    try:
        report = VerificationAgent(backend).verify(
            original,
            migrated,
            measure=False,
            required_ops={"cbrt": 1},
        )
    finally:
        sys.modules.pop("poison_backend", None)

    assert report.status == "fail"
    assert report.exception_phase == "migrated.forward"
    assert report.exception_type == "RuntimeError"
    assert "target integrity rejected" in report.exception_message
    assert target_calls == []
    assert report.backend_provenance["ok"] is False
    assert report.backend_provenance["strict_npu"] is False


def test_clone_tree_preserves_overlapping_aliases():
    base = np.array([1, 2, 4, 8, 16, 32], dtype=np.float32)
    clone = _clone_tree({"x": base[1:], "y": base[:-1]})

    assert np.shares_memory(clone["x"], clone["y"])
    clone["x"] += clone["y"]

    assert clone["x"].tolist() == [3.0, 6.0, 12.0, 24.0, 48.0]
    assert clone["y"].tolist() == [1.0, 3.0, 6.0, 12.0, 24.0]
    assert base.tolist() == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def test_clone_tree_preserves_layout_and_object_identity():
    fortran = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    reversed_array = np.linspace(-1, 1, 19, dtype=np.float32)[::-1]
    shared = np.arange(6, dtype=np.float32)

    clone = _clone_tree({
        "fortran": fortran,
        "reversed": reversed_array,
        "a": shared,
        "b": shared,
    })

    assert clone["fortran"].flags["F_CONTIGUOUS"]
    assert clone["fortran"].strides == fortran.strides
    assert np.array_equal(clone["fortran"], fortran)
    assert clone["reversed"].strides == (-4,)
    assert np.array_equal(clone["reversed"], reversed_array)
    assert clone["a"] is clone["b"]
    assert not np.shares_memory(clone["a"], shared)


def test_clone_tree_preserves_read_only_inputs():
    original = np.arange(4, dtype=np.float32)
    original.setflags(write=False)

    clone = _clone_tree({"x": original})["x"]

    assert clone.flags.writeable is False
    assert np.array_equal(clone, original)


def test_verifier_rejects_migration_that_drops_in_place_alias_semantics():
    original = """
import numpy as np
def build_inputs():
    base = np.array([1, 2, 4, 8, 16, 32], dtype=np.float32)
    return {'x': base[1:], 'y': base[:-1]}
def forward(x, y):
    x += y
    return {'out': x, 'alias': y}
"""
    migrated = """
import numpy as np
def build_inputs():
    base = np.array([1, 2, 4, 8, 16, 32], dtype=np.float32)
    return {'x': base[1:], 'y': base[:-1]}
def forward(x, y):
    x = x + y
    return {'out': x, 'alias': y}
"""
    report = VerificationAgent(ShimBackend()).verify(
        original, migrated, measure=False
    )

    assert report.status == "fail"
    assert report.numerically_consistent is False
    assert report.kernel_consistent is False


def test_verifier_accepts_non_mutating_forward_with_aliased_inputs():
    original = """
import numpy as np
def build_inputs():
    base = np.array([1, 2, 4, 8, 16, 32], dtype=np.float32)
    return {'x': base[1:], 'y': base[:-1]}
def forward(x, y):
    return {'out': np.add(x, y)}
"""
    report = VerificationAgent(ShimBackend()).verify(
        original, original, measure=False
    )

    assert report.status == "pass"
    assert report.kernel_consistent
    assert report.numerically_consistent
