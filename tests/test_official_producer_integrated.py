"""CPU-only checks for producing truthful integrated concatenate evidence.

Synthetic files exercise trust boundaries; none of these tests import AsNumpy,
contact a model, or execute a device kernel.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

from numpilot.gaps import official_producer as producer


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    path.write_bytes(payload)
    path.chmod(0o600)
    return {"path": path.name, "sha256": digest(payload)}


def write_json(path, value):
    return write(path, (json.dumps(value, sort_keys=True) + "\n").encode())


@pytest.mark.parametrize("expression", [
    "np.concatenate(arrays)", "numpy.asarray(arrays)", "np.linalg.norm(arrays)",
    "arrays[0].to_numpy()", "_concatenate_legacy(arrays, axis)", "eval('arrays')",
])
def test_public_wrapper_audit_rejects_compute_conversion_and_legacy(expression):
    assert producer._public_concat_host_markers(
        "def concatenate(arrays, axis=0):\n    return " + expression + "\n"
    ) > 0


def test_public_wrapper_audit_accepts_dtype_metadata_and_ignores_unrelated_functions():
    source = '''
import numpy as np
def concatenate(arrays, axis=0):
    if np.dtype(arrays[0].dtype) != np.dtype(np.float32):
        raise TypeError("unsupported")
    return ndarray(_concatenate(list(arrays), axis))
def unrelated(arrays):
    return np.asarray(arrays)
'''
    assert producer._public_concat_host_markers(source) == 0


def test_new_kernel_receipts_cannot_infer_launches_from_native_calls():
    with pytest.raises(producer.OfficialProducerError, match="independently measured"):
        producer._run_npu_execution_gate(SimpleNamespace(runtime_site=None, entries=("concatenate",)))
    with pytest.raises(producer.OfficialProducerError, match="only.*concatenate"):
        producer._run_npu_execution_gate(SimpleNamespace(runtime_site=Path("site"), entries=("ones",)))


def test_rank_claims_require_explicit_runtime_profile():
    legacy = producer._entry_spec(SimpleNamespace(runtime_site=None), "concatenate")
    current = producer._entry_spec(SimpleNamespace(runtime_site=Path("site")), "concatenate")
    assert "rank>=2" in legacy.constraints and "rank<=2" in legacy.constraints
    assert "rank>=1" in current.constraints and "rank<=3" in current.constraints
    assert legacy.signatures == current.signatures


def test_official_handoff_keeps_schema_and_validates_original_bytes(tmp_path, monkeypatch):
    from numpilot.gaps import official_qualification

    source = tmp_path / "source.py"
    resolution = tmp_path / "resolution.json"
    request_path = tmp_path / "request.json"
    write(source, b"import numpy as np\n")
    write(resolution, b'{"typed": true}\n')
    request = {"request_type": "official_candidate_gap", "source_sha256": digest(source.read_bytes()), "gap_specs": []}
    write_json(request_path, request)
    observed = []
    monkeypatch.setattr(official_qualification, "validate_official_handoff_evidence",
                        lambda *args: observed.append(args))
    actual, actual_digest = producer._load_gap_request(request_path, source, resolution)
    assert actual == request and actual_digest == digest(request_path.read_bytes())
    assert observed == [(request, source.read_bytes(), resolution.read_bytes())]
    with pytest.raises(producer.OfficialProducerError, match="explicit source and resolution"):
        producer._load_gap_request(request_path, source)


@pytest.fixture
def runtime_fixture(tmp_path, monkeypatch):
    attempt = tmp_path / "attempt"
    site = attempt / "runtime/site"
    names = {"asnumpy/__init__.py": b"version = 1\n", "asnumpy/array.py": b"wrapper = 1\n",
             "asnumpy/_core.fake.so": b"fake native bytes"}
    for name, payload in names.items():
        write(site / name, payload)
    wheel = attempt / "dist/candidate.whl"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in names.items():
            archive.writestr(name, payload)
    write(wheel, buffer.getvalue())
    identity = SimpleNamespace(module_path=str(site / "asnumpy/__init__.py"),
                               module_sha256=digest(names["asnumpy/__init__.py"]),
                               native_module_sha256=digest(names["asnumpy/_core.fake.so"]),
                               wrapper_sha256={"__init__.py": digest(names["asnumpy/__init__.py"]),
                                               "array.py": digest(names["asnumpy/array.py"])})
    txn = {"runtime": {"site_relative": "runtime/site", "core_relative": "asnumpy/_core.fake.so",
                       "python_sha256": digest(Path(sys.executable).read_bytes()),
                       "core_sha256": identity.native_module_sha256},
           "build": {"wheel_sha256": digest(wheel.read_bytes())}}
    verified_source = []
    monkeypatch.setattr(producer, "_verify_integrated_source", lambda *args: verified_source.append(args))
    return attempt, txn, wheel, site, identity, verified_source


def test_transaction_runtime_binds_wheel_core_wrappers_and_source(runtime_fixture):
    attempt, txn, wheel, site, identity, verified = runtime_fixture
    producer._validate_transaction_runtime_site(attempt, txn, wheel, site, identity)
    assert verified == [(attempt, txn)]


@pytest.mark.parametrize("mutation,match", [
    ("core", "core hash"), ("wrapper", "bound wheel"), ("wheel", "wheel hash"),
    ("python", "interpreter hash"), ("site", "transaction-bound installation"),
])
def test_runtime_site_does_not_relax_identity(runtime_fixture, tmp_path, mutation, match):
    attempt, txn, wheel, site, identity, verified = runtime_fixture
    if mutation == "core":
        write(site / "asnumpy/_core.fake.so", b"tampered")
    elif mutation == "wrapper":
        write(site / "asnumpy/array.py", b"tampered")
    elif mutation == "wheel":
        write(wheel, b"tampered")
    elif mutation == "python":
        txn["runtime"]["python_sha256"] = "0" * 64
    else:
        site = tmp_path / "unrelated-site"
        site.mkdir(mode=0o700)
    with pytest.raises(producer.OfficialProducerError, match=match):
        producer._validate_transaction_runtime_site(attempt, txn, wheel, site, identity)
    assert verified == []


def profile_fixture(tmp_path, *, op_type="ConcatD", duration="2.34", mutation=None):
    root = tmp_path / "profile"
    bundle = tmp_path / "bundle"
    gates = bundle / "gates/r1"
    gates.mkdir(parents=True, mode=0o700)
    core, module, array, txn, txnfile = (str(i) * 64 for i in range(1, 6))
    worker_ref = write(root / "worker.py", b"# synthetic profiler worker\n")
    cases = [{"rank": rank, "shape": [2] * rank, "axis": 1, "dtype": "float32", "exact_equal": True}
             for rank in (2, 2, 3, 3)]
    execution = {"status": "passed", "worker_sha256": worker_ref["sha256"],
                 "transaction_sha256": txnfile, "dispatch": {"native_calls": 4, "legacy_calls": 0},
                 "cases": cases, "runtime": {name: {"sha256": value}
                 for name, value in (("core", core), ("module", module), ("array", array))}}
    execution_ref = write_json(root / "worker.json", execution)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["OP Type", "Op Name", "Task Type", "Device_id", "Task Duration(us)"])
    writer.writeheader()
    for _ in cases:
        writer.writerow({"OP Type": op_type, "Op Name": "aclnnCat_ConcatD_ConcatD",
                         "Task Type": "AI_VECTOR_CORE", "Device_id": "0", "Task Duration(us)": duration})
    trace_ref = write(root / "raw/trace.csv", stream.getvalue().encode())
    trace_ref.update(path="raw/trace.csv", rows=4)
    profile = {"schema": "numpilot.concatenate-kernel-profile.v1", "status": "passed",
               "api": "numpy.concatenate", "performance_claim": False,
               "transaction_sha256": txn, "transaction_file_sha256": txnfile,
               "runtime_core_sha256": core, "runtime_module_sha256": module, "runtime_array_sha256": array,
               "worker": worker_ref, "execution": execution_ref, "traces": [trace_ref],
               "native_calls": 4, "tested_ranks": [2, 3], "cat_kernel_rows": 4,
               "op_type_counts": {op_type: 4}}
    if mutation:
        mutation(profile, execution, root)
    write_json(root / "kernel-profile.json", profile)
    context = SimpleNamespace(kernel_profile_root=root, bundle=bundle, gates_dir=gates,
                              transaction={"transaction_sha256": txn}, transaction_file_sha256=txnfile,
                              runtime_core_sha256=core,
                              identity=SimpleNamespace(module_sha256=module, wrapper_sha256={"array.py": array}))
    return context, profile


def test_kernel_launches_are_copied_from_real_csv_not_python_native_counter(tmp_path):
    context, profile = profile_fixture(tmp_path)
    launches, refs = producer._validate_kernel_profile(context)
    assert launches == 4 and len(refs) == 4
    for ref in refs:
        assert digest((context.bundle / ref["path"]).read_bytes()) == ref["sha256"]
    assert (context.gates_dir / "kernel-profile/raw/trace.csv").read_bytes() == (context.kernel_profile_root / "raw/trace.csv").read_bytes()


@pytest.mark.parametrize("duration", ["", "NaN", "Inf", "0", "-1"])
def test_kernel_trace_requires_observed_positive_finite_duration(tmp_path, duration):
    context, _ = profile_fixture(tmp_path, duration=duration)
    with pytest.raises(producer.OfficialProducerError, match="duration"):
        producer._validate_kernel_profile(context)


def test_kernel_trace_does_not_confuse_scatter_with_cat(tmp_path):
    context, _ = profile_fixture(tmp_path, op_type="Scatter")
    with pytest.raises(producer.OfficialProducerError, match="Cat count"):
        producer._validate_kernel_profile(context)


@pytest.mark.parametrize("mutation,match", [
    (lambda p, e, r: p.update(runtime_core_sha256="0" * 64), "binding mismatch"),
    (lambda p, e, r: p.update(cat_kernel_rows=99), "Cat count"),
    (lambda p, e, r: p["traces"].append(p["traces"][0]), "repeats a trace"),
    (lambda p, e, r: write(r / "raw/trace.csv", b"tampered"), "hash mismatch"),
    (lambda p, e, r: p.update(op_type_counts={"ConcatD": 99}), "operator counts"),
])
def test_kernel_profile_rejects_forged_or_stale_evidence(tmp_path, mutation, match):
    context, _ = profile_fixture(tmp_path, mutation=mutation)
    with pytest.raises(producer.OfficialProducerError, match=match):
        producer._validate_kernel_profile(context)


def test_rank_three_numerical_cases_produce_actual_oracle_records(tmp_path, monkeypatch):
    class Array:
        def __init__(self, value):
            self.value = value

        @classmethod
        def from_numpy(cls, value):
            return cls(value.copy())

        def to_numpy(self):
            return self.value.copy()

    fake = SimpleNamespace(ndarray=Array,
                           full=lambda *args, **kwargs: Array(numpy.full(*args, **kwargs)),
                           concatenate=lambda values, **kwargs: Array(numpy.concatenate([v.value for v in values], **kwargs)))
    monkeypatch.setitem(sys.modules, "asnumpy", fake)
    bundle = tmp_path / "bundle"
    gates = bundle / "gates/r1"
    gates.mkdir(parents=True, mode=0o700)
    context = SimpleNamespace(runtime_site=Path("site"), entries=("concatenate",), gates_dir=gates, bundle=bundle)
    result = producer._run_numerical_gate(context)["concatenate"]
    raw = json.loads((gates / "numerical_validation.concatenate.json").read_text())
    assert result["status"] == "passed"
    assert result["claims"]["tested_ranks"] == raw["totals"]["tested_ranks"] == [1, 2, 3]
    assert result["claims"]["cases"] == len(raw["cases"]) == 8
    assert result["claims"]["oracle_comparisons"] == 24
    assert all(case["ok"] and case["comparisons"] == 3 for case in raw["cases"])
