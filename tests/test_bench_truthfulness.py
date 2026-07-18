"""真实性能评测：数值、线程观测与 strict-NPU provenance 门禁。"""
import json
import sys
import types

import numpy as np

from numpilot.bench import (
    _audit_execution_sources,
    _blas_runtime_info,
    _comparison_data,
    _speedup_verdict,
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
                "capabilities": {
                    "add": {"provenance": "native_npu", "strict_npu": True},
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
                "capabilities": {
                    "add": {"provenance": "native_npu", "strict_npu": True},
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

    def fake_run(file, out_dir, engine, threshold, allow_cpu_fallback):
        seen["allow"] = allow_cpu_fallback
        return 0

    monkeypatch.setattr(cli, "run_npu_compare", fake_run)
    rc = cli.main([
        "bench-npu", "x.py", "--out", str(tmp_path),
        "--allow-cpu-fallback",
    ])
    assert rc == 0
    assert seen["allow"] is True
