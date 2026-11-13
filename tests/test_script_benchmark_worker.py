import hashlib
import json
from pathlib import Path
import types

import numpy as np
import pytest

from numpilot import _script_benchmark_worker as worker


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _contract(backend="cpu"):
    value = {
        "schema_version": 1,
        "role": backend,
        "backend": backend,
        "script_sha256": SHA_A,
        "source_sha256": SHA_A if backend == "cpu" else SHA_B,
        "migration_report_sha256": None,
        "migration_report": None,
        "reference_source": None,
        "marker": worker.MARKER,
        "output_names": ["result"],
        "warmup": 2,
        "samples": 7,
        "cpu_threads": 1,
        "max_output_bytes": 1024 * 1024,
    }
    if backend == "npu":
        value.update({
            "migration_report_sha256": SHA_C,
            "migration_report": "/snapshots/migration-report.json",
            "reference_source": "/snapshots/source.py",
        })
    return value


def _report():
    return {
        "schema_version": 2,
        "mode": "script",
        "status": "migrated_strict_npu",
        "strict_npu": True,
        "actual_remote_call": True,
        "ast_audit": "passed",
        "execution_performed": False,
        "numerical_verification": "not_run",
        "performance_valid": False,
        "published_files": ["migrated.py"],
        "source_sha256": SHA_B,
        "migrated_sha256": SHA_A,
        "qualified_analysis": {
            "source_sha256": SHA_B,
            "decision": "DIRECT",
            "runtime_identity_status": "matched",
            "runtime_identity_sha256": SHA_C,
            "calls": [{
                "canonical_key": "exp",
                "decision": "DIRECT",
                "performance_eligible": True,
            }],
        },
        "llm_receipt": {
            "http_status": 200,
            "tls_verification": "enabled",
            "response_model_identity": "exact",
        },
    }


def test_parse_args_exposes_stable_worker_cli():
    args = worker._parse_args([
        "--backend", "npu",
        "--script", "/tmp/migrated.py",
        "--contract", "/tmp/contract.json",
        "--out", "/tmp/out",
        "--core", "3",
    ])

    assert args.backend == "npu"
    assert args.script == "/tmp/migrated.py"
    assert args.contract == "/tmp/contract.json"
    assert args.out == "/tmp/out"
    assert args.core == 3


@pytest.mark.parametrize(
    ("backend", "role"),
    [("cpu", "cpu"), ("cpu", "reference"), ("npu", "npu"),
     ("npu", "candidate")],
)
def test_validate_contract_accepts_backend_bound_roles(backend, role):
    contract = _contract(backend)
    contract["role"] = role

    result = worker._validate_contract(contract, backend)

    assert result["backend"] == backend


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("marker", "# another marker"),
        ("cpu_threads", 2),
        ("warmup", -1),
        ("samples", 0),
        ("output_names", ["not-valid!"]),
        ("script_sha256", "short"),
    ],
)
def test_validate_contract_rejects_invalid_performance_contract(field, value):
    contract = _contract()
    contract[field] = value

    with pytest.raises(worker._WorkerFailure) as caught:
        worker._validate_contract(contract, "cpu")

    assert caught.value.code == "invalid_contract"


def test_validate_contract_rejects_npu_without_hash_bound_snapshots():
    contract = _contract("npu")
    contract["migration_report"] = None

    with pytest.raises(worker._WorkerFailure) as caught:
        worker._validate_contract(contract, "npu")

    assert caught.value.code == "invalid_contract"


def test_strict_json_rejects_duplicate_keys():
    with pytest.raises(worker._WorkerFailure) as caught:
        worker._strict_json(
            b'{"schema_version":1,"schema_version":1}',
            code="invalid_contract",
            phase="worker_input",
        )

    assert caught.value.code == "invalid_contract"


def test_cpu_script_plan_runs_resident_suffix_for_each_sample(tmp_path):
    path = tmp_path / "source.py"
    source = (
        "import numpy as np\n"
        "x = np.arange(4, dtype=np.float32)\n"
        f"{worker.MARKER}\n"
        "result = np.add(x, x)\n"
    )

    plan = worker._build_script_plan(source.encode(), path, "cpu")
    namespace, timing = worker._run_resident(
        plan, warmup=1, samples=2, filename=str(path)
    )

    np.testing.assert_array_equal(namespace["result"], np.arange(4) * 2)
    assert len(timing["warmup_ns"]) == 1
    assert len(timing["samples_ns"]) == 2
    assert timing["prefix_input_preparation_ns"] >= 0


def test_npu_script_plan_records_prefix_and_suffix_operator_counts(tmp_path):
    path = tmp_path / "migrated.py"
    source = (
        "import asnumpy as ap\n"
        "x = ap.full((4,), 1.0, dtype=ap.float32)\n"
        f"{worker.MARKER}\n"
        "result = ap.linalg.solve(x, x)\n"
        "result = ap.exp(result)\n"
    )

    plan = worker._build_script_plan(source.encode(), path, "npu")

    assert plan["prefix_call_counts"] == {"full": 1}
    assert plan["suffix_call_counts"] == {"linalg.solve": 1, "exp": 1}


@pytest.mark.parametrize(
    ("suffix", "expected_code"),
    [
        ("import numpy as np\nresult = x\n", "forbidden_suffix_node"),
        ("for item in x:\n    result = item\n", "forbidden_suffix_node"),
        ("result = np.asarray(x)\n", "forbidden_suffix_transfer"),
        ("result = x.to_numpy()\n", "forbidden_suffix_transfer"),
        ("result = x.item()\n", "forbidden_suffix_transfer"),
        ("result = [item for item in x]\n", "forbidden_suffix_node"),
    ],
)
def test_suffix_audit_rejects_control_flow_and_host_transfers(
    tmp_path, suffix, expected_code
):
    source = (
        "import numpy as np\n"
        "x = np.arange(1)\n"
        + worker.MARKER + "\n" + suffix
    )

    with pytest.raises(worker._WorkerFailure) as caught:
        worker._build_script_plan(source.encode(), tmp_path / "source.py", "cpu")

    assert caught.value.code == expected_code


def test_script_plan_requires_exactly_one_marker(tmp_path):
    with pytest.raises(worker._WorkerFailure) as caught:
        worker._build_script_plan(
            b"result = 1\n", tmp_path / "source.py", "cpu"
        )

    assert caught.value.code == "invalid_timing_marker"


def test_npu_script_plan_requires_reserved_alias_and_resident_operation(tmp_path):
    wrong_alias = (
        "import asnumpy as xp\n"
        f"{worker.MARKER}\n"
        "result = xp.exp(x)\n"
    )
    with pytest.raises(worker._WorkerFailure) as caught:
        worker._build_script_plan(
            wrong_alias.encode(), tmp_path / "migrated.py", "npu"
        )
    assert caught.value.code == "invalid_asnumpy_import"

    no_operation = (
        "import asnumpy as ap\n"
        "x = ap.full((1,), 1.0, dtype=ap.float32)\n"
        f"{worker.MARKER}\n"
        "result = x\n"
    )
    with pytest.raises(worker._WorkerFailure) as caught:
        worker._build_script_plan(
            no_operation.encode(), tmp_path / "migrated.py", "npu"
        )
    assert caught.value.code == "no_resident_npu_operations"


def test_guarded_module_validates_nested_calls_and_counts_exactly():
    calls = []
    module = types.SimpleNamespace(
        __name__="asnumpy",
        exp=lambda value: value + 1,
        linalg=types.SimpleNamespace(solve=lambda left, right: left - right),
    )

    class Guard:
        runtime_targets = {
            "exp": module.exp,
            "linalg.solve": module.linalg.solve,
        }

        def validate_call(self, name, args, kwargs):
            calls.append((name, args, kwargs))

    proxy, counts = worker._guarded_module(
        module, Guard(), ("exp", "linalg.solve")
    )

    assert proxy.exp(3) == 4
    assert proxy.linalg.solve(5, 2) == 3
    assert counts == {"exp": 1, "linalg.solve": 1}
    assert [entry[0] for entry in calls] == ["exp", "linalg.solve"]


def test_guarded_module_rejects_target_set_drift():
    module = types.SimpleNamespace(__name__="asnumpy", exp=lambda value: value)
    guard = types.SimpleNamespace(runtime_targets={"exp": module.exp})

    with pytest.raises(worker._WorkerFailure) as caught:
        worker._guarded_module(module, guard, ("exp", "log"))

    assert caught.value.code == "profile_target_mismatch"


def test_validate_migration_report_returns_all_true_gates():
    gates = worker._validate_migration_report(
        _report(),
        source_sha256=SHA_B,
        script_sha256=SHA_A,
        runtime_identity_sha256=SHA_C,
    )

    assert gates
    assert all(gates.values())


def test_validate_migration_report_accepts_scoped_profile_bound_artifact():
    report = _report()
    report["status"] = "migrated_scoped_npu"
    report["strict_npu"] = False

    gates = worker._validate_migration_report(
        report,
        source_sha256=SHA_B,
        script_sha256=SHA_A,
        runtime_identity_sha256=SHA_C,
    )
    assert all(gates.values())


def test_validate_migration_report_rejects_scoped_or_non_llm_artifact():
    report = _report()
    report["status"] = "migrated_scoped_npu"
    report["llm_receipt"]["http_status"] = None

    with pytest.raises(worker._WorkerFailure) as caught:
        worker._validate_migration_report(
            report,
            source_sha256=SHA_B,
            script_sha256=SHA_A,
            runtime_identity_sha256=SHA_C,
        )

    assert caught.value.code == "migration_report_gate_failed"


def test_qualification_receipt_is_fail_closed():
    direct = {
        "source_sha256": SHA_B,
        "decision": "DIRECT",
        "runtime_identity_status": "matched",
        "runtime_identity_sha256": SHA_C,
        "calls": [{
            "canonical_key": "exp",
            "decision": "DIRECT",
            "effective_route": "NATIVE_NPU",
            "performance_eligible": True,
        }],
        "issues": [],
    }
    assert worker._qualification_receipt(direct)[
        "all_calls_performance_eligible"
    ] is True

    direct["calls"][0]["performance_eligible"] = False
    with pytest.raises(worker._WorkerFailure) as caught:
        worker._qualification_receipt(direct)
    assert caught.value.code == "qualification_failed"


def test_npu_output_is_transferred_exactly_once():
    class DeviceValue:
        def __init__(self):
            self.calls = 0

        def to_numpy(self):
            self.calls += 1
            return np.array([1.0, 2.0], dtype=np.float32)

    value = DeviceValue()
    result, calls, elapsed = worker._normalise_output(
        value, "result", np, "npu"
    )

    assert value.calls == 1
    assert calls == 1
    assert elapsed >= 0
    assert result.dtype == np.float32


def test_npu_output_must_remain_device_resident():
    with pytest.raises(worker._WorkerFailure) as caught:
        worker._normalise_output(
            np.array([1.0], dtype=np.float32), "result", np, "npu"
        )

    assert caught.value.code == "npu_output_not_device_resident"


def test_save_outputs_uses_write_once_numbered_npy_files(tmp_path):
    entries, transfer = worker._save_outputs(
        tmp_path,
        {
            "left": np.array([1, 2], dtype=np.int32),
            "right": np.array(3.0, dtype=np.float32),
        },
        ["left", "right"],
        np,
        "cpu",
        1024 * 1024,
    )

    assert [entry["filename"] for entry in entries] == [
        "output-0000.npy", "output-0001.npy"
    ]
    np.testing.assert_array_equal(
        np.load(tmp_path / "output-0000.npy", allow_pickle=False), [1, 2]
    )
    assert transfer["device_to_host_calls"] == 0

    with pytest.raises(worker._WorkerFailure) as caught:
        worker._save_outputs(
            tmp_path,
            {"left": np.array([1]), "right": np.array([2])},
            ["left", "right"],
            np,
            "cpu",
            1024 * 1024,
        )
    assert caught.value.code == "unexpected_output_file"


def test_prepare_process_controls_pins_before_runtime_import(monkeypatch):
    state = {"affinity": {1, 3}}
    monkeypatch.setattr(
        worker.os, "sched_getaffinity", lambda pid: set(state["affinity"])
    )

    def set_affinity(pid, cores):
        state["affinity"] = set(cores)

    monkeypatch.setattr(worker.os, "sched_setaffinity", set_affinity)
    for name in worker.THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")

    affinity, environment = worker._prepare_process_controls(3)

    assert affinity == {
        "requested_core": 3,
        "before": [1, 3],
        "after_binding": [3],
        "after_execution": None,
    }
    assert set(environment.values()) == {"1"}


def test_prepare_process_controls_rejects_thread_environment(monkeypatch):
    monkeypatch.setattr(worker.os, "sched_getaffinity", lambda pid: {0})
    monkeypatch.setattr(worker.os, "sched_setaffinity", lambda pid, cores: None)
    for name in worker.THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")

    with pytest.raises(worker._WorkerFailure) as caught:
        worker._prepare_process_controls(0)

    assert caught.value.code == "thread_environment_mismatch"


def test_load_contract_binds_raw_contract_hash(tmp_path):
    path = tmp_path / "contract.json"
    payload = (json.dumps(_contract(), sort_keys=True) + "\n").encode()
    path.write_bytes(payload)

    value, observed = worker._load_contract(path, "cpu")

    assert value["script_sha256"] == SHA_A
    assert observed == payload
    assert hashlib.sha256(observed).hexdigest() == hashlib.sha256(payload).hexdigest()


def test_cpu_execute_publishes_hash_bound_manifest_and_outputs(
    tmp_path, monkeypatch
):
    script = tmp_path / "source.py"
    source = (
        "import numpy as np\n"
        "x = np.arange(4, dtype=np.float32)\n"
        f"{worker.MARKER}\n"
        "result = np.add(x, x)\n"
    ).encode()
    script.write_bytes(source)
    digest = hashlib.sha256(source).hexdigest()
    contract_value = _contract()
    contract_value.update({
        "script_sha256": digest,
        "source_sha256": digest,
        "warmup": 1,
        "samples": 2,
    })
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(contract_value), encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    monkeypatch.setattr(worker.os, "sched_getaffinity", lambda pid: {3})
    monkeypatch.setattr(
        worker,
        "_threadpool_info",
        lambda: [{"user_api": "blas", "num_threads": 1}],
    )

    worker._execute(
        "cpu",
        script,
        contract,
        output_dir,
        {
            "requested_core": 3,
            "before": [3],
            "after_binding": [3],
            "after_execution": None,
        },
        {name: "1" for name in worker.THREAD_ENVIRONMENT},
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "pass"
    assert manifest["backend"] == "cpu"
    assert manifest["script_sha256"] == digest
    assert manifest["contract_sha256"] == hashlib.sha256(
        contract.read_bytes()
    ).hexdigest()
    assert len(manifest["timing"]["warmup_ns"]) == 1
    assert len(manifest["timing"]["samples_ns"]) == 2
    assert manifest["timing"]["final_output_transfer"][
        "device_to_host_calls"
    ] == 0
    np.testing.assert_array_equal(
        np.load(output_dir / "output-0000.npy", allow_pickle=False),
        np.arange(4, dtype=np.float32) * 2,
    )
