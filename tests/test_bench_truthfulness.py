"""真实性能评测：数值、线程观测与 strict-NPU provenance 门禁。"""
import json
import hashlib
import sys
import types

import numpy as np

from numpilot.bench import (
    _audit_execution_sources,
    _blas_runtime_info,
    _comparison_data,
    _speedup_verdict,
    _timing_samples_ms,
    _to_host,
    _write_benchmark_artifacts,
)
from numpilot.comparison import compare_outputs
from numpilot.contracts import OffloadPlan


def test_benchmark_comparison_rejects_zero_reference_mismatch():
    result = compare_outputs(
        np.zeros((2,), dtype=np.float32),
        np.ones((2,), dtype=np.float32),
    )
    assert not result.consistent
    assert result.max_abs_diff == 1.0
    assert np.isfinite(result.max_rel_diff)
    assert result.max_rel_diff > 1.0
    data = _comparison_data(result)
    assert not data["consistent"]
    assert np.isfinite(data["max_rel_diff"])


def test_speedup_verdict_says_which_side_is_faster():
    assert _speedup_verdict(2.0) == "NPU 快 2.00x"
    assert _speedup_verdict(0.25) == "CPU 快 4.00x"


def test_timing_evidence_keeps_raw_samples_and_warmup_counts():
    calls = []

    def measured(value):
        calls.append(value)
        return value

    timing = _timing_samples_ms(measured, (1,), warmup=2, iters=4)
    assert calls == [1] * 6
    assert timing["clock"] == "time.perf_counter"
    assert timing["warmup_iterations"] == 2
    assert timing["measurement_iterations"] == 4
    assert len(timing["samples_ms"]) == 4
    assert timing["median_ms"] == sorted(timing["samples_ms"])[2]


def test_blas_runtime_info_prefers_observed_threadpool(monkeypatch):
    fake = types.ModuleType("threadpoolctl")
    fake.threadpool_info = lambda: [{
        "user_api": "blas", "internal_api": "openblas", "num_threads": 7,
        "prefix": "libopenblas", "version": "x", "threading_layer": "pthreads",
        "architecture": "armv8",
    }]
    monkeypatch.setitem(sys.modules, "threadpoolctl", fake)
    info = _blas_runtime_info()
    assert info["source"] == "threadpoolctl"
    assert info["effective_threads"] == 7
    assert info["pools"][0]["internal_api"] == "openblas"


def test_to_host_leaves_plain_numpy_and_scalar_values_alone():
    class Backend:
        def to_host(self, value):
            raise AssertionError("plain host values must not be sent to backend.to_host")

    array = np.ones((2,), dtype=np.float32)
    result = _to_host(Backend(), {"array": array, "scalar": 3.0})
    assert result["array"] is array
    assert result["scalar"] == 3.0


class _Validation:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return self.data


class _AuditedBackend:
    def __init__(self, data):
        self.data = data

    def validate_required_ops(self, required_ops, strict_npu=True, raise_on_error=True):
        assert strict_npu is True
        assert raise_on_error is False
        assert set(required_ops) == {"add", "matmul"}
        data = dict(self.data)
        data.setdefault("strict_npu", True)
        return _Validation(data)


def test_provenance_gate_rejects_known_cpu_fallback():
    backend = _AuditedBackend({
        "ok": False,
        "missing": [],
        "non_strict": ["matmul"],
        "capabilities": {
            "add": {"provenance": "aclnn", "strict_npu": True},
            "matmul": {"provenance": "cpu_fallback", "strict_npu": False},
        },
    })
    plan = OffloadPlan({"add": "NPU", "matmul": "NPU"}, 1)
    audit = _audit_execution_sources(backend, plan, ["add", "matmul"])
    assert not audit["ok"]
    assert audit["cpu_fallback_ops"] == ["matmul"]
    assert any("CPU fallback" in issue for issue in audit["issues"])


def test_provenance_gate_distinguishes_unknown_from_known_fallback():
    backend = _AuditedBackend({
        "ok": False,
        "missing": [],
        "non_strict": ["add"],
        "capabilities": {
            "add": {"provenance": "unknown", "strict_npu": False},
            "matmul": {"provenance": "native_npu", "strict_npu": True},
        },
    })
    plan = OffloadPlan({"add": "NPU", "matmul": "NPU"}, 1)
    audit = _audit_execution_sources(backend, plan, ["add", "matmul"])
    assert audit["unknown_ops"] == ["add"]
    assert audit["cpu_fallback_ops"] == []


def test_provenance_gate_fails_closed_without_audit_interface():
    plan = OffloadPlan({"add": "NPU"}, 1)
    audit = _audit_execution_sources(object(), plan, ["add"])
    assert not audit["ok"]
    assert audit["unknown_ops"] == ["add"]


def test_provenance_gate_rejects_cpu_placement_even_when_ops_are_native():
    backend = _AuditedBackend({"ok": True, "missing": [], "non_strict": []})
    plan = OffloadPlan({"add": "NPU", "matmul": "CPU"}, 1)
    audit = _audit_execution_sources(backend, plan, ["add", "matmul"])
    assert not audit["ok"]
    assert audit["planned_cpu_ops"] == ["matmul"]


def test_provenance_gate_requires_explicit_strict_contract():
    class Backend:
        def validate_required_ops(self, *args, **kwargs):
            return _Validation({
                "ok": True, "missing": [], "non_strict": [],
                "strict_npu": False,
            })

    audit = _audit_execution_sources(
        Backend(), OffloadPlan({"add": "NPU"}, 1), ["add"]
    )
    assert not audit["ok"]
    assert audit["strict_npu"] is False
    assert any("strict_npu=true" in issue for issue in audit["issues"])


def test_rejected_benchmark_artifacts_never_publish_speedup(tmp_path):
    result = {
        "net": "x", "file": "x.py", "status": "rejected",
        "allow_cpu_fallback": False, "fallback_override_used": False,
        "performance_valid": False, "speedup": None,
        "provenance": {
            "ok": False, "required_ops": ["matmul"],
            "issues": ["执行来源未知: matmul"],
        },
    }
    json_path, md_path = _write_benchmark_artifacts(str(tmp_path), result)
    saved = json.loads(open(json_path, encoding="utf-8").read())
    report = open(md_path, encoding="utf-8").read()
    assert saved["speedup"] is None
    assert saved["performance_valid"] is False
    assert "未发布 NPU 加速比" in report


def test_run_npu_compare_rejects_audited_matmul_before_execution(monkeypatch, tmp_path):
    import types

    from numpilot import bench
    from numpilot.backends import npu as npu_module
    from numpilot.backends.npu import AUDITED_ASNUMPY_VERSION, NpuBackend

    source = tmp_path / "matmul_case.py"
    source.write_text(
        "import numpy as np\n"
        "def build_inputs():\n"
        "    return (np.ones((2, 2), dtype=np.float32),)\n"
        "def forward(x):\n"
        "    return np.matmul(x, x)\n",
        encoding="utf-8",
    )
    module = types.ModuleType("asnumpy")
    module.__version__ = AUDITED_ASNUMPY_VERSION
    module.__file__ = "/fake/asnumpy/__init__.py"
    module.matmul = lambda x, y: None
    monkeypatch.setattr(
        npu_module, "_audited_runtime_identity_matches", lambda _module: True
    )
    backend = NpuBackend(module)
    monkeypatch.setattr(bench, "select_backend", lambda name: backend)

    rc = bench.run_npu_compare(str(source), str(tmp_path / "out"))

    assert rc == 3
    result = json.loads((tmp_path / "out" / "npu_benchmark.json").read_text())
    assert result["status"] == "rejected"
    assert "matmul" in result["provenance"]["cpu_fallback_ops"]
    assert result["comparison"] is None
    assert result["speedup"] is None


def test_run_npu_compare_executes_migrated_artifact_build_inputs(monkeypatch, tmp_path):
    from numpilot import bench

    source = tmp_path / "add_case.py"
    source.write_text(
        "import numpy as np\n"
        "def build_inputs():\n"
        "    return (np.ones((2,), dtype=np.float32),)\n"
        "def forward(x):\n"
        "    return np.add(x, x)\n",
        encoding="utf-8",
    )

    class Engine:
        name = "test"

        def rewrite(self, source, plan):
            return (
                "import asnumpy as ap\n"
                "def build_inputs():\n"
                "    raise RuntimeError('artifact build_inputs executed')\n"
                "def forward(x):\n"
                "    return ap.add(x, x)\n"
            )

    module = types.ModuleType("asnumpy")
    module.__version__ = "test"
    module.add = lambda x, y: x + y

    class Backend:
        def get_module(self):
            return module

        def validate_required_ops(self, required_ops, **kwargs):
            return _Validation({
                "ok": True, "missing": [], "non_strict": [],
                "strict_npu": True,
                "required_ops": ["add"],
                "capabilities": {
                    "add": {
                        "name": "add",
                        "available": True,
                        "provenance": "native_npu",
                        "strict_npu": True,
                        "evidence": "test source audit",
                        "constraints": [],
                        "signature_variants": [],
                    },
                },
            })

    monkeypatch.setattr(bench, "select_backend", lambda name: Backend())
    rc = bench.run_npu_compare(
        str(source), str(tmp_path / "artifact-out"), engine=Engine())

    assert rc == 4
    result = json.loads(
        (tmp_path / "artifact-out" / "npu_benchmark.json").read_text())
    assert result["status"] == "runtime_error"
    assert "artifact build_inputs executed" in result["runtime_error"]
    assert result["artifact_inputs_executed"] is False


def test_run_npu_compare_rejects_dead_ap_decoy_with_host_escape(monkeypatch, tmp_path):
    from numpilot import bench

    source = tmp_path / "sqrt_case.py"
    source.write_text(
        "import numpy as np\n"
        "def build_inputs(): return (np.ones(3, dtype=np.float32),)\n"
        "def forward(x): return np.sqrt(x)\n",
        encoding="utf-8",
    )

    class Engine:
        name = "host-escape"

        def rewrite(self, source, plan):
            return (
                "import numpy as np\nimport asnumpy as ap\n"
                "def build_inputs(): return (np.ones(3, dtype=np.float32),)\n"
                "def forward(x):\n"
                "    if False: return ap.sqrt(x)\n"
                "    host = x.to_numpy() if hasattr(x, 'to_numpy') else x\n"
                "    return np.sqrt(host)\n"
            )

    module = types.ModuleType("asnumpy")
    module.__version__ = "test"
    module.__file__ = "/fake/asnumpy/__init__.py"
    module.sqrt = np.sqrt

    class Backend:
        def get_module(self):
            return module

        def validate_required_ops(self, required_ops, **kwargs):
            return _Validation({
                "ok": True,
                "missing": [],
                "non_strict": [],
                "strict_npu": True,
                "capabilities": {
                    name: {"provenance": "native_npu", "strict_npu": True}
                    for name in required_ops
                },
            })

    monkeypatch.setattr(bench, "select_backend", lambda name: Backend())
    out = tmp_path / "host-escape-out"
    rc = bench.run_npu_compare(str(source), str(out), engine=Engine())

    assert rc == 3
    result = json.loads((out / "npu_benchmark.json").read_text())
    assert result["performance_valid"] is False
    assert result["speedup"] is None
    assert any("执行表面" in issue for issue in result["performance_issues"])


def test_successful_run_without_cpu_baseline_is_not_performance_valid(
        monkeypatch, tmp_path):
    from numpilot import bench

    source = tmp_path / "add_case.py"
    source.write_text(
        "import numpy as np\n"
        "def build_inputs(): return (np.ones(3, dtype=np.float32),)\n"
        "def forward(x): return np.add(x, x)\n",
        encoding="utf-8",
    )

    class ArrayFactory:
        @staticmethod
        def from_numpy(value):
            return value

    module = types.ModuleType("asnumpy")
    module.__version__ = "test"
    module.__file__ = "/fake/asnumpy/__init__.py"
    module.ndarray = ArrayFactory
    module.add = np.add

    class Backend:
        def get_module(self):
            return module

        def validate_required_ops(self, required_ops, **kwargs):
            return _Validation({
                "ok": True, "missing": [], "non_strict": [],
                "strict_npu": True,
                "required_ops": ["add"],
                "capabilities": {
                    "add": {
                        "name": "add",
                        "available": True,
                        "provenance": "native_npu",
                        "strict_npu": True,
                        "evidence": "test source audit",
                        "constraints": [],
                        "signature_variants": [],
                    },
                },
            })

        def to_device(self, value):
            return value

        def to_host(self, value):
            return value

    monkeypatch.setattr(bench, "select_backend", lambda name: Backend())
    out = tmp_path / "no-baseline"
    rc = bench.run_npu_compare(str(source), str(out))

    assert rc == 0
    result = json.loads((out / "npu_benchmark.json").read_text())
    assert result["status"] == "pass_unqualified"
    assert result["baseline_compatible"] is None
    assert result["performance_valid"] is False
    assert result["speedup"] is None


def test_cli_forwards_allow_cpu_fallback(monkeypatch, tmp_path):
    from numpilot import __main__ as cli

    seen = {}
    monkeypatch.setattr(cli, "_preflight", lambda *args: None)
    backend = object()
    profiles = object()

    def fake_runtime(args, backend_name):
        seen["overlay"] = args.capability_overlay
        seen["root"] = args.qualification_evidence_root
        seen["profiles_path"] = args.analysis_profiles
        assert backend_name == "npu"
        return backend, profiles

    monkeypatch.setattr(cli, "_qualified_runtime_inputs", fake_runtime)

    def fake_run(
        file,
        out_dir,
        engine,
        threshold,
        allow_cpu_fallback,
        *,
        backend,
        analysis_profiles,
    ):
        seen["allow"] = allow_cpu_fallback
        seen["backend"] = backend
        seen["profiles"] = analysis_profiles
        return 0

    monkeypatch.setattr(cli, "run_npu_compare", fake_run)
    rc = cli.main([
        "bench-npu", "x.py", "--out", str(tmp_path),
        "--allow-cpu-fallback",
        "--capability-overlay", "/private/overlay.json",
        "--qualification-evidence-root", "/private/evidence",
        "--analysis-profiles", "/private/profiles.json",
    ])
    assert rc == 0
    assert seen["allow"] is True
    assert seen["backend"] is backend
    assert seen["profiles"] is profiles
    assert seen["overlay"] == "/private/overlay.json"
    assert seen["root"] == "/private/evidence"
    assert seen["profiles_path"] == "/private/profiles.json"


def test_cli_rejects_incomplete_overlay_pair_before_preflight(monkeypatch, capsys):
    from numpilot import __main__ as cli

    monkeypatch.setattr(
        cli,
        "_preflight",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("invalid activation must fail before preflight")
        ),
    )

    rc = cli.main([
        "bench-npu",
        "x.py",
        "--capability-overlay",
        "/private/overlay.json",
    ])

    assert rc == 2
    assert "必须成对提供" in capsys.readouterr().err


def _scoped_benchmark_case(tmp_path, dtype):
    from numpilot.agents.analysis import AnalysisAgent
    from numpilot.capabilities import (
        AnalysisCallsiteProfile,
        AnalysisProfileSet,
        OperatorCallProfile,
        TensorProfile,
    )
    from numpilot.backends.npu import NpuBackend
    from numpilot.capabilities import (
        AnalysisCallsiteProfile,
        AnalysisProfileSet,
        CapabilityManifest,
        ExecutionProvenance,
        OperatorCallProfile,
        OperatorCapability,
        OperatorSignature,
        TensorProfile,
    )

    source_text = (
        "import numpy as np\n"
        "def build_inputs():\n"
        f"    return (np.array([1.0, 8.0], dtype=np.{dtype}),)\n"
        "def forward(x):\n"
        "    return np.cbrt(x)\n"
    )
    source = tmp_path / ("cbrt_" + dtype + ".py")
    source.write_text(source_text, encoding="utf-8")
    op = AnalysisAgent().analyze(source_text, str(source)).op_uses[0]
    profiles = AnalysisProfileSet(
        source_sha256=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        callsites=(AnalysisCallsiteProfile(
            qualified_name="numpy.cbrt",
            lineno=op.lineno,
            col_offset=op.col_offset,
            profile=OperatorCallProfile(
                inputs=(TensorProfile("float32", (2,), True),),
                positional_args=1,
                keyword_args=(),
            ),
        ),),
    )
    calls = []

    class ArrayFactory:
        @staticmethod
        def from_numpy(value):
            return value.copy()

    module = types.ModuleType("asnumpy")
    module.__version__ = "private-cbrt"
    module.__file__ = "/private/asnumpy/__init__.py"
    module.ndarray = ArrayFactory

    def cbrt(value):
        calls.append(value.dtype.name)
        return np.cbrt(value)

    module.cbrt = cbrt

    class Backend(NpuBackend):
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
                        evidence="qualification:qual-test",
                        constraints=(
                            "dtype=float32",
                            "rank>=1",
                            "contiguous=true",
                        ),
                        signature_variants=(
                            OperatorSignature(positional_args=1),
                        ),
                    ),
                },
            )

        def to_device(self, value):
            return value.copy()

        def to_host(self, value):
            return value

    return source, Backend(module=module), profiles, calls


def test_benchmark_scoped_profile_matches_before_performance_path(tmp_path):
    from numpilot import bench

    source, backend, profiles, calls = _scoped_benchmark_case(
        tmp_path,
        "float32",
    )
    out = tmp_path / "valid-scoped"

    rc = bench.run_npu_compare(
        str(source),
        str(out),
        backend=backend,
        analysis_profiles=profiles,
    )

    assert rc == 0
    assert calls
    result = json.loads((out / "npu_benchmark.json").read_text())
    assert result["provenance"]["profile_status"] == "matched"
    assert result["provenance"]["runtime_profile_ok"] is True


def test_benchmark_runtime_profile_rechecks_stale_static_receipt_before_target(
    tmp_path,
):
    from numpilot import bench

    source, backend, profiles, calls = _scoped_benchmark_case(
        tmp_path,
        "float64",
    )
    out = tmp_path / "stale-scoped"

    rc = bench.run_npu_compare(
        str(source),
        str(out),
        backend=backend,
        analysis_profiles=profiles,
    )

    assert rc == 4
    assert calls == []
    result = json.loads((out / "npu_benchmark.json").read_text())
    assert result["status"] == "runtime_error"
    assert "dtype=float64" in result["runtime_error"]
    assert result["provenance"]["profile_status"] == "rejected"
    assert result["provenance"]["ok"] is False
    assert result["provenance"]["strict_npu"] is False


def test_benchmark_rejects_original_source_target_monkeypatch(tmp_path):
    from numpilot import bench

    source, backend, profiles, calls = _scoped_benchmark_case(
        tmp_path,
        "float32",
    )
    source_text = source.read_text(encoding="utf-8")
    poisoned = source_text.replace(
        "def build_inputs():",
        "def build_inputs():\n"
        "    __import__('sys').modules['poison_backend'].cbrt = np.cbrt",
        1,
    )
    source.write_text(poisoned, encoding="utf-8")
    from numpilot.agents.analysis import AnalysisAgent
    from numpilot.capabilities import (
        AnalysisCallsiteProfile,
        AnalysisProfileSet,
        OperatorCallProfile,
        TensorProfile,
    )

    op = AnalysisAgent().analyze(poisoned, str(source)).op_uses[0]
    profiles = AnalysisProfileSet(
        source_sha256=hashlib.sha256(poisoned.encode("utf-8")).hexdigest(),
        callsites=(AnalysisCallsiteProfile(
            qualified_name="numpy.cbrt",
            lineno=op.lineno,
            col_offset=op.col_offset,
            profile=OperatorCallProfile(
                inputs=(TensorProfile("float32", (2,), True),),
                positional_args=1,
                keyword_args=(),
            ),
        ),),
    )
    out = tmp_path / "poisoned-scoped"

    sys.modules["poison_backend"] = backend.get_module()
    try:
        rc = bench.run_npu_compare(
            str(source),
            str(out),
            backend=backend,
            analysis_profiles=profiles,
        )
    finally:
        sys.modules.pop("poison_backend", None)

    assert rc == 4
    assert calls == []
    result = json.loads((out / "npu_benchmark.json").read_text())
    assert "target integrity rejected" in result["runtime_error"]
    assert result["provenance"]["ok"] is False
    assert result["provenance"]["strict_npu"] is False
