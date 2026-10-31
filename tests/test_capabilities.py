import copy
import hashlib
import json
from pathlib import Path
import platform
import sys
import types
from zipfile import ZipFile

import numpy as np
import pytest

from numpilot.backends import npu as npu_module
from numpilot.backends.npu import (
    AUDITED_ASNUMPY_VERSION,
    ASNUMPY_INTERNAL_COMPONENT_AUDIT_ID,
    ASNUMPY_SOURCE_BUILD_AUDIT_ID,
    CBRT_ASNUMPY_AUDIT_ID,
    CBRT_ASNUMPY_VERSION,
    NpuBackend,
    OPS_NN_MATMUL_ASNUMPY_VERSION,
    OverlayRuntimeConfig,
    P0_ASNUMPY_AUDIT_ID,
    P0_ASNUMPY_VERSION,
    P0_LINALG_ASNUMPY_AUDIT_ID,
    P0_MOVE_ASNUMPY_AUDIT_ID,
    P0_MOVE_LEGACY_OVERLAY_AUDIT_ID,
    P0_MOVE_ASNUMPY_VERSION,
)
from numpilot.capability_profile import OPS_NN_MATMUL_ASNUMPY_AUDIT_ID
from numpilot.capabilities import (
    AnalysisCallsiteProfile,
    AnalysisProfileSet,
    BackendCapabilityError,
    CapabilityManifest,
    ExecutionProvenance,
    OperatorCallProfile,
    OperatorSignature,
    OperatorCapability,
    RuntimeIdentity,
    RuntimeProfileError,
    RuntimeProfileGuard,
    TensorProfile,
    load_analysis_profile_set,
    match_capability_profile,
    profile_operator_call,
    strict_receipt_issues,
    tensor_profile_from_value,
)
from numpilot.contracts import OffloadPlan


_REAL_AUDITED_RUNTIME_IDENTITY_MATCHES = (
    npu_module._audited_runtime_identity_matches
)


@pytest.fixture(autouse=True)
def _trust_generic_fake_runtime_identity(monkeypatch):
    """Unit fakes have no wheel tree; identity matching has dedicated tests."""

    monkeypatch.setattr(
        npu_module, "_audited_runtime_identity_matches", lambda _module: True
    )


def _fake_asnumpy(version=AUDITED_ASNUMPY_VERSION, **apis):
    module = types.ModuleType("asnumpy")
    module.__version__ = version
    for name, api in apis.items():
        setattr(module, name, api)
    return module


def _callable(*args, **kwargs):
    return None


def _valid_runtime_identity(**overrides):
    values = {
        "asnumpy_repository": "https://gitcode.com/cann/asnumpy.git",
        "asnumpy_commit": "1" * 40,
        "asnumpy_tree": "2" * 40,
        "asnumpy_version": AUDITED_ASNUMPY_VERSION,
        "wrapper_sha256": {
            "math.py": "3" * 64,
            "random/_random.py": "4" * 64,
        },
        "module_path": "/opt/asnumpy/asnumpy/__init__.py",
        "module_sha256": "5" * 64,
        "native_module_path": "/opt/asnumpy/asnumpy/_core.so",
        "native_module_sha256": "6" * 64,
        "cann_home": "/usr/local/Ascend/cann-9.2.0-beta.2",
        "cann_version": "9.2.0-beta.2",
        "soc_model": "Ascend910B3",
        "python_executable": "/opt/numpilot/.venv/bin/python",
        "python_version": "3.12.14",
        "bootstrap_receipt_path": "/opt/numpilot/state/bootstrap/receipt.json",
        "bootstrap_receipt_sha256": "7" * 64,
        "audit_id": "asnumpy-audit-v1",
        "status": "matched",
        "issues": (),
    }
    values.update(overrides)
    return RuntimeIdentity(**values)


def test_runtime_identity_strict_json_round_trip_and_canonical_hash():
    identity = _valid_runtime_identity()
    restored = RuntimeIdentity.from_json(identity.to_json())
    reordered = _valid_runtime_identity(wrapper_sha256={
        "random/_random.py": "4" * 64,
        "math.py": "3" * 64,
    })

    assert restored == identity
    assert restored.to_dict() == identity.to_dict()
    assert identity.identity_sha256 == reordered.identity_sha256
    assert json.loads(identity.to_json())["identity_sha256"] == (
        identity.identity_sha256
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("asnumpy_repository", "unknown"),
        ("asnumpy_commit", "unknown"),
        ("asnumpy_tree", "unknown"),
        ("asnumpy_version", "unknown"),
        ("wrapper_sha256", {"math.py": "unknown"}),
        ("module_path", "unknown"),
        ("module_sha256", "unknown"),
        ("native_module_path", "unknown"),
        ("native_module_sha256", "unknown"),
        ("cann_home", "unknown"),
        ("cann_version", "unknown"),
        ("soc_model", "unknown"),
        ("python_executable", "unknown"),
        ("python_version", "unknown"),
        ("bootstrap_receipt_path", "unknown"),
        ("bootstrap_receipt_sha256", "unknown"),
        ("audit_id", "unknown"),
    ),
)
def test_runtime_identity_never_matches_with_unknown_critical_field(field, value):
    with pytest.raises(ValueError, match="matched runtime identity"):
        _valid_runtime_identity(**{field: value})


def test_runtime_identity_allows_unknown_only_as_explicit_audit_required():
    identity = _valid_runtime_identity(
        soc_model="unknown",
        status="audit_required",
        issues=("SoC model is unavailable",),
    )

    assert RuntimeIdentity.from_json(identity.to_json()) == identity
    assert identity.status == "audit_required"


def test_runtime_identity_hash_binds_audit_status_and_issues():
    identity = _valid_runtime_identity(
        status="audit_required",
        issues=("SoC model requires a renewed audit",),
    )
    payload = identity.to_dict()
    payload["issues"] = ["CANN version requires a renewed audit"]

    with pytest.raises(ValueError, match="identity_sha256 mismatch"):
        RuntimeIdentity.from_dict(payload)


@pytest.mark.parametrize(
    "field",
    (
        "asnumpy_repository",
        "asnumpy_commit",
        "asnumpy_tree",
        "asnumpy_version",
        "wrapper_sha256",
        "module_path",
        "module_sha256",
        "native_module_path",
        "native_module_sha256",
        "cann_home",
        "cann_version",
        "soc_model",
        "python_executable",
        "python_version",
        "bootstrap_receipt_path",
        "bootstrap_receipt_sha256",
        "audit_id",
    ),
)
def test_runtime_identity_rejects_single_field_drift_against_canonical_hash(field):
    identity = _valid_runtime_identity()
    payload = identity.to_dict()
    if field == "wrapper_sha256":
        payload[field]["math.py"] = "8" * 64
    elif field in {"asnumpy_commit", "asnumpy_tree"}:
        payload[field] = "8" * 40
    elif field.endswith("sha256"):
        payload[field] = "8" * 64
    else:
        payload[field] = payload[field] + "-drift"

    with pytest.raises(ValueError, match="identity_sha256 mismatch"):
        RuntimeIdentity.from_dict(payload)


def test_p0_audit_ids_bind_wrapper_dependencies():
    assert "init7061af13" in npu_module.ASNUMPY_AUDIT_ID
    assert "cannac907b3a" in npu_module.ASNUMPY_AUDIT_ID
    assert "linalg68ba1971" in npu_module.ASNUMPY_AUDIT_ID
    assert "random6c9c7891" in npu_module.ASNUMPY_AUDIT_ID
    assert "core02f38dfa" in npu_module.ASNUMPY_AUDIT_ID
    assert npu_module.ASNUMPY_AUDIT_ID.endswith("-v5")
    assert "cannac907b3a" in ASNUMPY_SOURCE_BUILD_AUDIT_ID
    assert "linalg68ba1971" in ASNUMPY_SOURCE_BUILD_AUDIT_ID
    assert "random6c9c7891" in ASNUMPY_SOURCE_BUILD_AUDIT_ID
    assert ASNUMPY_SOURCE_BUILD_AUDIT_ID.endswith("-receipt-v3")
    assert "internal-component" in ASNUMPY_INTERNAL_COMPONENT_AUDIT_ID
    assert "manifest8bd7893a" in ASNUMPY_INTERNAL_COMPONENT_AUDIT_ID
    assert "wheel-a8c47cfe" in ASNUMPY_INTERNAL_COMPONENT_AUDIT_ID
    assert "core02f38dfa" in ASNUMPY_INTERNAL_COMPONENT_AUDIT_ID
    assert ASNUMPY_INTERNAL_COMPONENT_AUDIT_ID.endswith("-v2")
    assert "utilsfd94c025" in P0_ASNUMPY_AUDIT_ID
    assert P0_ASNUMPY_AUDIT_ID.endswith("-v2")
    assert "array16c901f5" in P0_LINALG_ASNUMPY_AUDIT_ID
    assert "utilsfd94c025" in P0_LINALG_ASNUMPY_AUDIT_ID
    assert P0_LINALG_ASNUMPY_AUDIT_ID.endswith("-v4")
    assert P0_LINALG_ASNUMPY_AUDIT_ID != (
        "asnumpy-0.1.dev106-g21368abf9-poisson-linalg-"
        "init4f4e8083-math62e0ba09-direct4479c588-cored3ccb39f-v1"
    )
    assert P0_LINALG_ASNUMPY_AUDIT_ID != (
        "asnumpy-0.1.dev106-g21368abf9-poisson-linalg-"
        "init4f4e8083-array16c901f5-math62e0ba09-direct4479c588-"
        "utilsfd94c025-cored3ccb39f-v3"
    )
    assert "array16c901f5" in P0_MOVE_ASNUMPY_AUDIT_ID
    assert P0_MOVE_ASNUMPY_AUDIT_ID.endswith("-v3")
    assert P0_MOVE_ASNUMPY_AUDIT_ID != (
        "asnumpy-0.1.dev106-g21368abf9-poisson-move-"
        "init4f4e8083-math62e0ba09-direct4479c588-"
        "utils1b298145-core4465ce6e-v1"
    )
    assert P0_MOVE_ASNUMPY_AUDIT_ID != (
        "asnumpy-0.1.dev106-g21368abf9-poisson-move-"
        "init4f4e8083-array16c901f5-math62e0ba09-direct4479c588-"
        "utils1b298145-core4465ce6e-v2"
    )
    assert P0_MOVE_LEGACY_OVERLAY_AUDIT_ID.endswith("-v1")
    assert "utils1b298145" in CBRT_ASNUMPY_AUDIT_ID
    assert CBRT_ASNUMPY_AUDIT_ID.endswith("-v2")


def test_p0_core_identity_rejects_utils_drift(tmp_path, monkeypatch):
    from numpilot.backends import npu

    module, root = _write_runtime_fixture(
        tmp_path,
        {
            "__init__.py": b"init",
            "math.py": b"math",
            "utils.py": b"utils",
            "_core.so": b"core",
        },
        monkeypatch,
        "_P0_RUNTIME_FILE_SHA256",
    )

    assert npu._p0_runtime_identity_matches(module)
    (root / "utils.py").write_bytes(b"drift")
    assert not npu._p0_runtime_identity_matches(module)


def _write_runtime_fixture(tmp_path, relative_payloads, monkeypatch, digest_name):
    from numpilot.backends import npu

    root = tmp_path / "asnumpy"
    for relative, payload in relative_payloads.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    digests = {
        relative: hashlib.sha256(payload).hexdigest()
        for relative, payload in relative_payloads.items()
    }
    core = digests.pop("_core.so")
    digests["_core"] = core
    monkeypatch.setattr(npu, digest_name, digests)
    module = types.ModuleType("asnumpy")
    module.__file__ = str(root / "__init__.py")
    return module, root


def test_generic_audited_identity_rejects_wrapper_drift(tmp_path, monkeypatch):
    module, root = _write_runtime_fixture(
        tmp_path,
        {
            "__init__.py": b"init",
            "_types.py": b"types",
            "_version.py": b"version",
            "array.py": b"array",
            "cann.py": b"cann",
            "linalg/__init__.py": b"linalg-init",
            "linalg/_linalg.py": b"linalg-nested",
            "linalg/direct.py": b"direct",
            "logic.py": b"logic",
            "math.py": b"math",
            "nn.py": b"nn",
            "sorting.py": b"sorting",
            "statistics.py": b"statistics",
            "random/__init__.py": b"random-init",
            "random/_random.py": b"random-nested",
            "utils.py": b"utils",
            "_core.so": b"core",
        },
        monkeypatch,
        "_AUDITED_RUNTIME_FILE_SHA256",
    )

    assert _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)
    (root / "logic.py").write_bytes(b"drift")
    assert not _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)


@pytest.mark.parametrize(
    "relative",
    ("linalg/_linalg.py", "random/__init__.py", "random/_random.py"),
)
def test_generic_audited_identity_rejects_nested_wrapper_drift(
    tmp_path, monkeypatch, relative
):
    module, root = _write_runtime_fixture(
        tmp_path,
        {
            "__init__.py": b"init",
            "linalg/__init__.py": b"linalg-init",
            "linalg/_linalg.py": b"linalg-nested",
            "linalg/direct.py": b"direct",
            "random/__init__.py": b"random-init",
            "random/_random.py": b"random-nested",
            "_core.so": b"core",
        },
        monkeypatch,
        "_AUDITED_RUNTIME_FILE_SHA256",
    )

    assert _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)
    (root / relative).write_bytes(b"drift")
    assert not _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)


def test_generic_version_without_exact_runtime_hashes_is_unaudited(monkeypatch):
    monkeypatch.setattr(
        npu_module, "_audited_runtime_identity_matches", lambda _module: False
    )
    manifest = NpuBackend(
        _fake_asnumpy(add=_callable)
    ).capability_manifest(["add"])

    assert manifest.audit_id == "unaudited-asnumpy-version"
    assert not manifest.operators["add"].strict_npu


def _source_receipt_runtime(tmp_path, monkeypatch):
    from numpilot import __version__ as numpilot_version
    from numpilot.backends import npu
    from numpilot.bootstrap import (
        ASNUMPY_COMMIT,
        ASNUMPY_REPOSITORY,
        ASNUMPY_TREE,
        ASNUMPY_VERSION,
    )

    module, root = _write_runtime_fixture(
        tmp_path,
        {
            "__init__.py": b"init",
            "_types.py": b"types",
            "_version.py": b"version",
            "array.py": b"array",
            "cann.py": b"cann",
            "linalg/__init__.py": b"linalg-init",
            "linalg/_linalg.py": b"linalg-nested",
            "linalg/direct.py": b"direct",
            "logic.py": b"logic",
            "math.py": b"math",
            "nn.py": b"nn",
            "sorting.py": b"sorting",
            "statistics.py": b"statistics",
            "random/__init__.py": b"random-init",
            "random/_random.py": b"random-nested",
            "utils.py": b"utils",
            "_core.so": b"locally-built-core",
        },
        monkeypatch,
        "_AUDITED_RUNTIME_FILE_SHA256",
    )
    expected = dict(npu._AUDITED_RUNTIME_FILE_SHA256)
    expected["_core"] = hashlib.sha256(b"fixed-release-core").hexdigest()
    monkeypatch.setattr(npu, "_AUDITED_RUNTIME_FILE_SHA256", expected)

    module.__version__ = ASNUMPY_VERSION
    module.add = _callable
    module.matmul = _callable
    module_hash = hashlib.sha256((root / "__init__.py").read_bytes()).hexdigest()
    core_path = root / "_core.so"
    core_hash = hashlib.sha256(core_path.read_bytes()).hexdigest()
    valid_hash = "a" * 64
    versions = {
        "asnumpy": ASNUMPY_VERSION,
        "numpy": "2.2.6",
        "loguru": "0.7.3",
    }
    monkeypatch.setattr(
        npu.importlib_metadata,
        "version",
        lambda distribution: versions[distribution],
    )
    receipt = {
        "schema_version": 1,
        "producer": "numpilot-bootstrap-v1",
        "numpilot_version": numpilot_version,
        "status": "ready",
        "ready": True,
        "plan_sha256": valid_hash,
        "actions": sorted(npu._ASNUMPY_SOURCE_RECEIPT_ACTIONS),
        "repositories": {
            "asnumpy": {
                "repository": ASNUMPY_REPOSITORY,
                "commit": ASNUMPY_COMMIT,
                "tree": ASNUMPY_TREE,
                "mode": "pinned-source-build",
                "files_sha256": {
                    name: valid_hash for name in npu._ASNUMPY_SOURCE_KEY_FILES
                },
            }
        },
        "wheel": {
            "kind": "built-from-pinned-source",
            "distribution": "asnumpy",
            "version": ASNUMPY_VERSION,
            "sha256": valid_hash,
            "metadata_sha256": valid_hash,
            "record_sha256": valid_hash,
            "trust": {
                "kind": "built-from-pinned-source",
                "repository": ASNUMPY_REPOSITORY,
                "commit": ASNUMPY_COMMIT,
                "tree": ASNUMPY_TREE,
            },
            "installed_files": {
                "module": {
                    "path": "asnumpy/__init__.py",
                    "sha256": module_hash,
                },
                "native_modules": [
                    {"path": "asnumpy/_core.so", "sha256": core_hash}
                ],
            },
        },
        "verification": {
            "asnumpy": {
                "status": "passed",
                "source": "official AsNumpy README.md installation verification",
                "interpreter": str(Path(sys.executable).resolve()),
                "asnumpy_version": ASNUMPY_VERSION,
                "numpy_version": versions["numpy"],
                "loguru_version": versions["loguru"],
                "module_path": str((root / "__init__.py").resolve()),
                "module_sha256": module_hash,
                "native_modules": [
                    {"path": "_core.so", "sha256": core_hash}
                ],
                "wheel_file_identity_verified": True,
                "shape": [1000, 1000],
                "dtype": "float32",
            }
        },
    }
    receipt_path = tmp_path / "bootstrap.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)
    monkeypatch.setenv("NUMPILOT_BOOTSTRAP_RECEIPT", str(receipt_path))
    return module, root, receipt_path, receipt


def _component_receipt_runtime(tmp_path, monkeypatch):
    from numpilot import __version__ as numpilot_version
    from numpilot.asnumpy_component import (
        component_receipt_identity,
        load_component_manifest,
    )
    from numpilot.backends import npu
    from numpilot.bootstrap import _verify_internal_component

    component_root = Path(__file__).resolve().parents[1] / "components/asnumpy"
    monkeypatch.setenv("NUMPILOT_ASNUMPY_COMPONENT_DIR", str(component_root))
    component = load_component_manifest(component_root)
    wheel_identity = _verify_internal_component(component)

    root = tmp_path / "installed" / "asnumpy"
    root.mkdir(parents=True)
    with ZipFile(component.wheel_path) as archive:
        for relative in npu._AUDITED_RUNTIME_FILE_SHA256:
            if relative == "_core":
                continue
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read("asnumpy/" + relative))
        native_relative = component.installed_files["native_modules"][0]["path"]
        core_path = root / Path(native_relative).name
        core_path.write_bytes(archive.read(native_relative))

    module = types.ModuleType("asnumpy")
    module.__file__ = str(root / "__init__.py")
    module.__version__ = component.version
    module.add = _callable
    module.matmul = _callable

    versions = {
        "asnumpy": component.version,
        "numpy": component.runtime["numpy_version"],
        "loguru": component.runtime["loguru_version"],
    }
    monkeypatch.setattr(
        npu.importlib_metadata,
        "version",
        lambda distribution: versions[distribution],
    )
    monkeypatch.setattr(
        npu.platform, "machine", lambda: component.runtime["machine"]
    )
    monkeypatch.setattr(
        npu.platform,
        "python_implementation",
        lambda: component.runtime["python_implementation"],
    )

    module_hash = hashlib.sha256((root / "__init__.py").read_bytes()).hexdigest()
    core_hash = hashlib.sha256(core_path.read_bytes()).hexdigest()
    current_python_version = platform.python_version()
    current_python_major_minor = "%d.%d" % sys.version_info[:2]
    component_identity = component_receipt_identity(component)
    receipt = {
        "schema_version": 1,
        "producer": "numpilot-bootstrap-v1",
        "numpilot_version": numpilot_version,
        "status": "ready",
        "ready": True,
        "plan_sha256": "a" * 64,
        "actions": list(npu._ASNUMPY_COMPONENT_RECEIPT_ACTIONS),
        "environment": {
            "python": {
                "executable": str(Path(sys.executable).resolve()),
                "version": current_python_version,
                "implementation": component.runtime["python_implementation"],
                "major_minor": current_python_major_minor,
            },
            "machine": {"machine": component.runtime["machine"]},
            "cann": {
                "version": component.runtime["cann_version"],
                "version_conflict": False,
            },
        },
        "repositories": {
            "asnumpy": {
                "mode": "internal-component",
                "source_repository_claimed": True,
                "component": component_identity,
                "repository": component.repository,
                "commit": component.commit,
                "tree": component.tree,
                "license": component.license,
            }
        },
        "wheel": {"kind": "internal-component", **wheel_identity},
        "verification": {
            "asnumpy": {
                "status": "passed",
                "source": "official AsNumpy README.md installation verification",
                "interpreter": str(Path(sys.executable).resolve()),
                "asnumpy_version": component.version,
                "numpy_version": versions["numpy"],
                "loguru_version": versions["loguru"],
                "module_path": str((root / "__init__.py").resolve()),
                "module_sha256": module_hash,
                "native_modules": [
                    {"path": core_path.name, "sha256": core_hash}
                ],
                "wheel_file_identity_verified": True,
                "shape": [1000, 1000],
                "dtype": "float32",
            }
        },
    }
    receipt_path = tmp_path / "state" / "bootstrap" / "receipt.json"
    receipt_path.parent.mkdir(parents=True, mode=0o700)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)
    monkeypatch.setenv("NUMPILOT_BOOTSTRAP_RECEIPT", str(receipt_path))
    return module, root, core_path, receipt_path, receipt


def _write_source_receipt(receipt_path, receipt):
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)


def test_pinned_source_receipt_admits_dynamic_core_and_qualifies_matmul_profile(
    tmp_path, monkeypatch
):
    from numpilot.backends import npu

    module, _, _, _ = _source_receipt_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(
        npu, "_audited_runtime_identity_matches",
        _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES,
    )

    manifest = NpuBackend(module).capability_manifest(["add", "matmul"])

    assert manifest.audit_id == ASNUMPY_SOURCE_BUILD_AUDIT_ID
    assert manifest.operators["add"].provenance is ExecutionProvenance.NATIVE_NPU
    matmul = manifest.operators["matmul"]
    assert matmul.provenance is ExecutionProvenance.NATIVE_NPU
    assert matmul.signature_variants == (OperatorSignature(positional_args=2),)
    assert set(matmul.constraints) == {
        "dtype=float32",
        "rank>=2",
        "rank<=2",
        "contiguous=true",
        "numel>=1",
    }


def test_pinned_source_receipt_rejects_native_and_wrapper_drift(
    tmp_path, monkeypatch
):
    module, root, _, _ = _source_receipt_runtime(tmp_path, monkeypatch)

    assert _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)
    (root / "_core.so").write_bytes(b"native-drift")
    assert not _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)
    (root / "_core.so").write_bytes(b"locally-built-core")
    assert _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)
    (root / "cann.py").write_bytes(b"wrapper-drift")
    assert not _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)


@pytest.mark.parametrize(
    ("keys", "value"),
    [
        (("status",), "failed"),
        (("repositories", "asnumpy", "commit"), "0" * 40),
        (("repositories", "asnumpy", "tree"), "0" * 40),
        (("wheel", "trust", "tree"), "0" * 40),
        (("verification", "asnumpy", "module_path"), "/wrong/asnumpy"),
        (("verification", "asnumpy", "wheel_file_identity_verified"), False),
        (
            ("verification", "asnumpy", "native_modules"),
            [{"path": "_core.so", "sha256": "0" * 64}],
        ),
    ],
)
def test_pinned_source_receipt_rejects_identity_field_drift(
    tmp_path, monkeypatch, keys, value
):
    module, _, receipt_path, receipt = _source_receipt_runtime(
        tmp_path, monkeypatch
    )
    changed = copy.deepcopy(receipt)
    target = changed
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    _write_source_receipt(receipt_path, changed)

    assert not _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)


def test_pinned_source_receipt_rejects_unsafe_file_and_duplicate_keys(
    tmp_path, monkeypatch
):
    module, _, receipt_path, receipt = _source_receipt_runtime(
        tmp_path, monkeypatch
    )
    receipt_path.chmod(0o644)
    assert not _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)

    duplicate = json.dumps(receipt)
    duplicate = duplicate.replace(
        '"schema_version": 1,',
        '"schema_version": 1, "schema_version": 1,',
        1,
    )
    receipt_path.write_text(duplicate, encoding="utf-8")
    receipt_path.chmod(0o600)
    assert not _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)

    real_receipt = tmp_path / "real-bootstrap.json"
    _write_source_receipt(real_receipt, receipt)
    receipt_path.unlink()
    receipt_path.symlink_to(real_receipt)
    assert not _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)


def test_internal_component_receipt_has_distinct_hash_bound_runtime_identity(
    tmp_path, monkeypatch
):
    module, root, core_path, receipt_path, _ = _component_receipt_runtime(
        tmp_path, monkeypatch
    )
    _configure_runtime_identity_environment(tmp_path, monkeypatch)
    backend = NpuBackend(module)

    assert _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)
    identity = backend.runtime_identity()
    manifest = backend.capability_manifest(["add", "matmul"])

    assert identity.status == "matched"
    assert identity.issues == ()
    assert identity.audit_id == ASNUMPY_INTERNAL_COMPONENT_AUDIT_ID
    assert identity.bootstrap_receipt_path == str(receipt_path.resolve())
    assert identity.module_sha256 == hashlib.sha256(
        (root / "__init__.py").read_bytes()
    ).hexdigest()
    assert identity.native_module_sha256 == hashlib.sha256(
        core_path.read_bytes()
    ).hexdigest()
    assert manifest.audit_id == ASNUMPY_INTERNAL_COMPONENT_AUDIT_ID
    assert manifest.operators["add"].provenance is ExecutionProvenance.NATIVE_NPU
    assert manifest.operators["matmul"].provenance is ExecutionProvenance.NATIVE_NPU


@pytest.mark.parametrize(
    ("keys", "value"),
    (
        (("numpilot_version",), "0.0.0-forged"),
        (("plan_sha256",), "not-a-sha256"),
        (
            ("actions",),
            [*npu_module._ASNUMPY_COMPONENT_RECEIPT_ACTIONS, "unexpected"],
        ),
        (("repositories", "asnumpy", "commit"), "0" * 40),
        (
            ("repositories", "asnumpy", "component", "manifest_sha256"),
            "0" * 64,
        ),
        (("wheel", "sha256"), "0" * 64),
        (("wheel", "metadata_sha256"), "0" * 64),
        (("wheel", "record_sha256"), "0" * 64),
        (("wheel", "installed_files", "module", "sha256"), "0" * 64),
        (
            ("wheel", "installed_files", "native_modules"),
            [{"path": "asnumpy/_core.so", "sha256": "0" * 64}],
        ),
        (
            (
                "wheel",
                "component",
                "verified_attachments",
                "license_file",
                "sha256",
            ),
            "0" * 64,
        ),
        (("verification", "asnumpy", "module_sha256"), "0" * 64),
        (("verification", "asnumpy", "interpreter"), "/wrong/python"),
        (("verification", "asnumpy", "numpy_version"), "0.0.0"),
        (("environment", "cann", "version"), "8.0.0"),
    ),
)
def test_internal_component_receipt_rejects_every_bound_identity_drift(
    tmp_path, monkeypatch, keys, value
):
    module, _, _, receipt_path, receipt = _component_receipt_runtime(
        tmp_path, monkeypatch
    )
    _configure_runtime_identity_environment(tmp_path, monkeypatch)
    changed = copy.deepcopy(receipt)
    target = changed
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    _write_source_receipt(receipt_path, changed)

    assert not _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)
    identity = NpuBackend(module).runtime_identity()
    assert identity.status == "audit_required"
    assert identity.issues


def test_internal_component_receipt_rejects_unsafe_or_ambiguous_json(
    tmp_path, monkeypatch
):
    module, _, _, receipt_path, receipt = _component_receipt_runtime(
        tmp_path, monkeypatch
    )
    _configure_runtime_identity_environment(tmp_path, monkeypatch)

    receipt_path.chmod(0o644)
    assert not _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)

    duplicate = json.dumps(receipt).replace(
        '"schema_version": 1,',
        '"schema_version": 1, "schema_version": 1,',
        1,
    )
    receipt_path.write_text(duplicate, encoding="utf-8")
    receipt_path.chmod(0o600)
    assert not _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)


def test_internal_component_runtime_file_drift_is_audit_required(
    tmp_path, monkeypatch
):
    module, _, core_path, _, _ = _component_receipt_runtime(
        tmp_path, monkeypatch
    )
    _configure_runtime_identity_environment(tmp_path, monkeypatch)
    assert NpuBackend(module).runtime_identity().status == "matched"

    core_path.write_bytes(b"component-native-drift")

    identity = NpuBackend(module).runtime_identity()
    assert identity.status == "audit_required"
    assert identity.native_module_sha256 == hashlib.sha256(
        b"component-native-drift"
    ).hexdigest()


def test_pinned_source_receipt_uses_absolute_default_state_path(
    tmp_path, monkeypatch
):
    module, _, receipt_path, receipt = _source_receipt_runtime(
        tmp_path, monkeypatch
    )
    state_home = tmp_path / "state" / "numpilot"
    default_receipt = state_home / "bootstrap" / "receipt.json"
    default_receipt.parent.mkdir(parents=True, mode=0o700)
    _write_source_receipt(default_receipt, receipt)
    receipt_path.unlink()
    monkeypatch.delenv("NUMPILOT_BOOTSTRAP_RECEIPT")
    monkeypatch.setenv("NUMPILOT_STATE_HOME", str(state_home))

    assert _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)

    monkeypatch.setenv("NUMPILOT_STATE_HOME", "relative-state")
    assert not _REAL_AUDITED_RUNTIME_IDENTITY_MATCHES(module)


def _configure_runtime_identity_environment(tmp_path, monkeypatch):
    import numpilot.env_check as env_check
    from numpilot.backends import npu

    cann_home = tmp_path / "Ascend" / "cann-9.2.0-beta.2"
    opp = cann_home / "opp"
    opp.mkdir(parents=True)
    (opp / "version.info").write_text("Version=9.2.0-beta.2\n", encoding="utf-8")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann_home))
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(cann_home))
    monkeypatch.setenv("ASCEND_OPP_PATH", str(opp))
    monkeypatch.setattr(npu, "AUDITED_CANN_HOME", str(cann_home.resolve()))
    monkeypatch.setattr(env_check, "_probe_npu_smi_soc_model", lambda: "Ascend910B3")
    return cann_home


def test_pinned_source_runtime_identity_is_complete_hash_bound_and_matched(
    tmp_path, monkeypatch
):
    module, root, receipt_path, _ = _source_receipt_runtime(
        tmp_path, monkeypatch
    )
    _configure_runtime_identity_environment(tmp_path, monkeypatch)

    identity = NpuBackend(module).runtime_identity()
    restored = RuntimeIdentity.from_json(identity.to_json())

    assert identity.status == "matched"
    assert identity.issues == ()
    assert identity.audit_id == ASNUMPY_SOURCE_BUILD_AUDIT_ID
    assert identity.asnumpy_commit == "1e7177f12039b2d87e3cdff6693459d895744f68"
    assert identity.asnumpy_tree == "a5fb1f3a70be9fc17ecf0dd928776924d4c78e1e"
    assert identity.soc_model == "Ascend910B3"
    assert identity.cann_version == "9.2.0-beta.2"
    assert identity.bootstrap_receipt_path == str(receipt_path.resolve())
    assert identity.module_sha256 == hashlib.sha256(
        (root / "__init__.py").read_bytes()
    ).hexdigest()
    assert identity.native_module_sha256 == hashlib.sha256(
        (root / "_core.so").read_bytes()
    ).hexdigest()
    assert {
        "linalg/_linalg.py",
        "random/__init__.py",
        "random/_random.py",
    }.issubset(identity.wrapper_sha256)
    assert restored.identity_sha256 == identity.identity_sha256
    assert "plan_sha256" not in identity.to_json()


def test_capability_and_runtime_identity_share_read_only_soc_probe(
    tmp_path, monkeypatch
):
    module, _, _, _ = _source_receipt_runtime(tmp_path, monkeypatch)
    _configure_runtime_identity_environment(tmp_path, monkeypatch)
    for name in ("ASCEND_SOC_VERSION", "NPU_SOC_VERSION", "SOC_VERSION"):
        monkeypatch.delenv(name, raising=False)
    backend = NpuBackend(module)

    identity = backend.runtime_identity()
    manifest = backend.capability_manifest(["add"])

    assert identity.status == "matched"
    assert identity.soc_model == "Ascend910B3"
    assert manifest.environment["soc_version"] == identity.soc_model


def test_runtime_identity_does_not_trust_forged_soc_environment(
    tmp_path, monkeypatch
):
    import numpilot.env_check as env_check

    module, _, _, _ = _source_receipt_runtime(tmp_path, monkeypatch)
    _configure_runtime_identity_environment(tmp_path, monkeypatch)
    monkeypatch.setenv("ASCEND_SOC_VERSION", "Ascend910B3")
    monkeypatch.setattr(
        env_check, "_probe_npu_smi_soc_model", lambda: "Ascend310P"
    )
    backend = NpuBackend(module)

    identity = backend.runtime_identity()
    manifest = backend.capability_manifest(["add"])

    assert manifest.environment["soc_version"] == "Ascend910B3"
    assert identity.soc_model == "Ascend310P"
    assert identity.status == "audit_required"
    assert "SoC model does not match the audited runtime" in identity.issues


def test_runtime_identity_locates_metadata_without_importing_asnumpy(
    tmp_path, monkeypatch
):
    from numpilot.backends import npu

    module, _, _, _ = _source_receipt_runtime(tmp_path, monkeypatch)
    _configure_runtime_identity_environment(tmp_path, monkeypatch)
    specification = types.SimpleNamespace(origin=module.__file__)
    monkeypatch.setattr(
        npu.importlib_util, "find_spec", lambda name: specification
    )
    backend = NpuBackend()

    def forbidden_import():
        raise AssertionError("runtime identity must not import AsNumpy")

    monkeypatch.setattr(backend, "get_module", forbidden_import)

    assert backend.runtime_identity().status == "matched"


@pytest.mark.parametrize(
    "drift",
    (
        "module_hash",
        "native_hash",
        "nested_wrapper",
        "receipt",
        "cann",
        "soc",
        "python",
    ),
)
def test_runtime_identity_turns_single_observed_drift_into_audit_required(
    tmp_path, monkeypatch, drift
):
    import numpilot.env_check as env_check
    from numpilot.backends import npu

    module, root, receipt_path, receipt = _source_receipt_runtime(
        tmp_path, monkeypatch
    )
    cann_home = _configure_runtime_identity_environment(tmp_path, monkeypatch)

    if drift == "module_hash":
        (root / "__init__.py").write_bytes(b"module-drift")
    elif drift == "native_hash":
        (root / "_core.so").write_bytes(b"native-drift")
    elif drift == "nested_wrapper":
        (root / "random" / "_random.py").write_bytes(b"nested-drift")
    elif drift == "receipt":
        changed = copy.deepcopy(receipt)
        changed["repositories"]["asnumpy"]["commit"] = "0" * 40
        _write_source_receipt(receipt_path, changed)
    elif drift == "cann":
        (cann_home / "opp" / "version.info").write_text(
            "Version=8.5.0\n", encoding="utf-8"
        )
    elif drift == "soc":
        monkeypatch.setattr(env_check, "_probe_npu_smi_soc_model", lambda: None)
    elif drift == "python":
        monkeypatch.setattr(npu.platform, "python_version", lambda: "3.10.19")

    identity = NpuBackend(module).runtime_identity()

    assert identity.status == "audit_required"
    assert identity.issues
    assert RuntimeIdentity.from_json(identity.to_json()) == identity


def test_p0_linalg_identity_rejects_utils_drift(tmp_path, monkeypatch):
    from numpilot.backends import npu

    module, root = _write_runtime_fixture(
        tmp_path,
        {
            "__init__.py": b"init",
            "array.py": b"array",
            "math.py": b"math",
            "linalg/direct.py": b"direct",
            "utils.py": b"utils",
            "_core.so": b"core",
        },
        monkeypatch,
        "_P0_LINALG_RUNTIME_FILE_SHA256",
    )

    assert npu._p0_linalg_runtime_identity_matches(module)
    (root / "utils.py").write_bytes(b"drift")
    assert not npu._p0_linalg_runtime_identity_matches(module)


def test_cbrt_identity_rejects_utils_drift(tmp_path, monkeypatch):
    from numpilot.backends import npu

    module, root = _write_runtime_fixture(
        tmp_path,
        {
            "__init__.py": b"init",
            "math.py": b"math",
            "custom_ops.py": b"custom",
            "_custom_ops_registry.py": b"registry",
            "utils.py": b"utils",
            "_core.so": b"core",
        },
        monkeypatch,
        "_CBRT_ASNUMPY_RUNTIME_FILE_SHA256",
    )

    assert npu._cbrt_runtime_identity_matches(module)
    (root / "utils.py").write_bytes(b"drift")
    assert not npu._cbrt_runtime_identity_matches(module)


def test_capability_manifest_is_versioned_and_json_round_trips():
    backend = NpuBackend(
        _fake_asnumpy(matmul=_callable, add=_callable, not_callable=42)
    )

    manifest = backend.capability_manifest(["matmul", "add", "not_callable"])
    payload = manifest.to_json()
    restored = CapabilityManifest.from_json(payload)

    assert restored.to_dict() == manifest.to_dict()
    assert json.loads(payload)["schema_version"] == "1.1"
    assert restored.backend_version == AUDITED_ASNUMPY_VERSION
    assert set(restored.environment) == {
        "module_path",
        "cann_home",
        "cann_version",
        "python_version",
        "soc_version",
    }
    assert restored.operators["matmul"].provenance is ExecutionProvenance.CPU_FALLBACK
    assert restored.operators["add"].strict_npu
    assert not restored.operators["not_callable"].available


def test_audited_matmul_is_available_but_rejected_as_strict_npu():
    backend = NpuBackend(_fake_asnumpy(matmul=_callable))

    validation = backend.validate_required_ops(
        ["matmul"], raise_on_error=False
    )

    assert not validation.ok
    assert validation.missing == ()
    assert validation.non_strict == ("matmul",)
    assert validation.capabilities["matmul"].available
    assert not validation.capabilities["matmul"].strict_npu
    with pytest.raises(BackendCapabilityError, match="non_strict_npu=matmul"):
        backend.validate_required_ops(["matmul"])


def test_p0_wheel_audit_is_hash_bound_and_limited_to_poisson_core(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(npu, "_p0_runtime_identity_matches", lambda module: True)
    backend = NpuBackend(_fake_asnumpy(
        version=P0_ASNUMPY_VERSION,
        add=_callable,
        multiply=_callable,
        clip=_callable,
        full=_callable,
        subtract=_callable,
        apply_gradient_descent=_callable,
    ))

    manifest = backend.capability_manifest(
        [
            "add",
            "multiply",
            "clip",
            "full",
            "subtract",
            "apply_gradient_descent",
        ]
    )

    assert manifest.audit_id == P0_ASNUMPY_AUDIT_ID
    assert all(
        manifest.operators[name].strict_npu
        for name in ("add", "multiply", "clip")
    )
    assert manifest.operators["full"].provenance is ExecutionProvenance.UNKNOWN
    assert manifest.operators["subtract"].provenance is ExecutionProvenance.UNKNOWN
    assert (
        manifest.operators["apply_gradient_descent"].provenance
        is ExecutionProvenance.UNKNOWN
    )


def test_p0_version_without_exact_runtime_hashes_remains_unaudited(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(npu, "_p0_runtime_identity_matches", lambda module: False)
    manifest = NpuBackend(_fake_asnumpy(
        version=P0_ASNUMPY_VERSION,
        add=_callable,
    )).capability_manifest(["add"])

    assert manifest.audit_id == "unaudited-asnumpy-version"
    assert manifest.operators["add"].provenance is ExecutionProvenance.UNKNOWN


def test_cbrt_wheel_audit_admits_only_inherited_base_ops(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(npu, "_cbrt_runtime_identity_matches", lambda module: True)
    manifest = NpuBackend(_fake_asnumpy(
        version=CBRT_ASNUMPY_VERSION,
        add=_callable,
        multiply=_callable,
        cbrt=_callable,
    )).capability_manifest(["add", "multiply", "cbrt"])

    assert manifest.audit_id == CBRT_ASNUMPY_AUDIT_ID
    assert manifest.operators["add"].strict_npu
    assert manifest.operators["multiply"].strict_npu
    assert manifest.operators["cbrt"].provenance is ExecutionProvenance.UNKNOWN
    assert not manifest.operators["cbrt"].strict_npu


def test_cbrt_wheel_without_exact_runtime_hashes_remains_unaudited(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(npu, "_cbrt_runtime_identity_matches", lambda module: False)
    manifest = NpuBackend(_fake_asnumpy(
        version=CBRT_ASNUMPY_VERSION,
        add=_callable,
    )).capability_manifest(["add"])

    assert manifest.audit_id == "unaudited-asnumpy-version"
    assert manifest.operators["add"].provenance is ExecutionProvenance.UNKNOWN


def test_p0_linalg_wheel_is_hash_bound_and_profile_scoped(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(
        npu, "_p0_linalg_runtime_identity_matches", lambda module: True
    )
    backend = NpuBackend(_fake_asnumpy(
        version=P0_ASNUMPY_VERSION,
        add=_callable,
        outer=_callable,
        matmul=_callable,
        subtract=_callable,
    ))

    manifest = backend.capability_manifest(["add", "outer", "matmul", "subtract"])

    assert manifest.audit_id == P0_LINALG_ASNUMPY_AUDIT_ID
    assert manifest.operators["add"].strict_npu
    for name in ("outer", "matmul"):
        capability = manifest.operators[name]
        assert capability.provenance is ExecutionProvenance.NATIVE_NPU
        assert not capability.strict_npu
        assert capability.signature_variants == (OperatorSignature(positional_args=2),)
        assert "rank<=2" in capability.constraints
        assert "numel>=1" in capability.constraints
    subtract = manifest.operators["subtract"]
    assert subtract.provenance is ExecutionProvenance.NATIVE_NPU
    assert not subtract.strict_npu
    assert subtract.signature_variants == (OperatorSignature(positional_args=2),)
    assert "numel>=1" in subtract.constraints


def test_p0_linalg_wheel_admits_hash_bound_inherited_demo_apis(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(
        npu, "_p0_linalg_runtime_identity_matches", lambda module: True
    )
    backend = NpuBackend(_fake_asnumpy(
        version=P0_ASNUMPY_VERSION,
        absolute=_callable,
        amax=_callable,
        cos=_callable,
        full=_callable,
        linspace=_callable,
        sin=_callable,
        sum=_callable,
        angle=_callable,
    ))

    manifest = backend.capability_manifest(
        [
            "absolute",
            "amax",
            "cos",
            "full",
            "linspace",
            "sin",
            "sum",
            "angle",
        ]
    )

    for name in ("absolute", "amax", "cos", "full", "linspace", "sin", "sum"):
        capability = manifest.operators[name]
        assert capability.provenance is ExecutionProvenance.NATIVE_NPU
        assert capability.strict_npu
    angle = manifest.operators["angle"]
    assert angle.provenance is ExecutionProvenance.UNKNOWN
    assert not angle.strict_npu


def test_p0_move_wheel_is_hash_bound_and_keeps_linalg_scope(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(
        npu, "_p0_move_runtime_identity_matches", lambda module: True
    )
    backend = NpuBackend(_fake_asnumpy(
        version=P0_MOVE_ASNUMPY_VERSION,
        add=_callable,
        outer=_callable,
        matmul=_callable,
        subtract=_callable,
    ))

    manifest = backend.capability_manifest(["add", "outer", "matmul", "subtract"])

    assert manifest.audit_id == P0_MOVE_ASNUMPY_AUDIT_ID
    assert manifest.operators["add"].strict_npu
    for name in ("outer", "matmul", "subtract"):
        capability = manifest.operators[name]
        assert capability.provenance is ExecutionProvenance.NATIVE_NPU
        assert not capability.strict_npu
        assert capability.signature_variants == (OperatorSignature(positional_args=2),)


def test_p0_move_wheel_admits_hash_bound_inherited_demo_apis(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(
        npu, "_p0_move_runtime_identity_matches", lambda module: True
    )
    backend = NpuBackend(_fake_asnumpy(
        version=P0_MOVE_ASNUMPY_VERSION,
        absolute=_callable,
        amax=_callable,
        cos=_callable,
        full=_callable,
        linspace=_callable,
        sin=_callable,
        sum=_callable,
        angle=_callable,
    ))

    manifest = backend.capability_manifest(
        [
            "absolute",
            "amax",
            "cos",
            "full",
            "linspace",
            "sin",
            "sum",
            "angle",
        ]
    )

    for name in ("absolute", "amax", "cos", "full", "linspace", "sin", "sum"):
        capability = manifest.operators[name]
        assert capability.provenance is ExecutionProvenance.NATIVE_NPU
        assert capability.strict_npu
    angle = manifest.operators["angle"]
    assert angle.provenance is ExecutionProvenance.UNKNOWN
    assert not angle.strict_npu


def test_p0_full_audit_does_not_expand_legacy_overlay_identity():
    from numpilot.backends import npu

    current, _, _ = npu._provenance_profile(
        "full",
        P0_MOVE_ASNUMPY_VERSION,
        P0_MOVE_ASNUMPY_AUDIT_ID,
    )
    legacy, evidence, constraints = npu._provenance_profile(
        "full",
        P0_MOVE_ASNUMPY_VERSION,
        P0_MOVE_LEGACY_OVERLAY_AUDIT_ID,
    )

    assert current is ExecutionProvenance.NATIVE_NPU
    assert legacy is ExecutionProvenance.UNKNOWN
    assert "outside the limited audited set" in evidence
    assert constraints == ("requires an operator-specific source audit",)


def test_p0_move_version_without_exact_runtime_hashes_remains_unaudited(monkeypatch):
    from numpilot.backends import npu

    monkeypatch.setattr(
        npu, "_p0_move_runtime_identity_matches", lambda module: False
    )
    manifest = NpuBackend(_fake_asnumpy(
        version=P0_MOVE_ASNUMPY_VERSION,
        add=_callable,
    )).capability_manifest(["add"])

    assert manifest.audit_id == "unaudited-asnumpy-version"
    assert manifest.operators["add"].provenance is ExecutionProvenance.UNKNOWN


def test_ops_nn_matmul_runtime_is_hash_bound_and_two_operator_scoped(monkeypatch):
    monkeypatch.setattr(
        npu_module, "_ops_nn_matmul_runtime_identity_matches", lambda _module: True
    )
    backend = NpuBackend(_fake_asnumpy(
        version=OPS_NN_MATMUL_ASNUMPY_VERSION,
        full=_callable,
        matmul=_callable,
        add=_callable,
    ))

    manifest = backend.capability_manifest(["full", "matmul", "add"])

    assert manifest.audit_id == OPS_NN_MATMUL_ASNUMPY_AUDIT_ID
    assert manifest.operators["full"].strict_npu
    matmul = manifest.operators["matmul"]
    assert matmul.provenance is ExecutionProvenance.NATIVE_NPU
    assert not matmul.strict_npu
    assert matmul.signature_variants == (OperatorSignature(positional_args=2),)
    assert set(matmul.constraints) == {
        "dtype=float32",
        "rank>=2",
        "rank<=2",
        "contiguous=true",
        "numel>=1",
    }
    assert manifest.operators["add"].provenance is ExecutionProvenance.UNKNOWN


def test_ops_nn_matmul_version_without_exact_hashes_remains_unaudited(monkeypatch):
    monkeypatch.setattr(
        npu_module, "_ops_nn_matmul_runtime_identity_matches", lambda _module: False
    )
    manifest = NpuBackend(_fake_asnumpy(
        version=OPS_NN_MATMUL_ASNUMPY_VERSION,
        matmul=_callable,
    )).capability_manifest(["matmul"])

    assert manifest.audit_id == "unaudited-asnumpy-version"
    assert manifest.operators["matmul"].provenance is ExecutionProvenance.UNKNOWN


def test_matmul_fallback_profile_survives_runtime_lookup_failure():
    backend = NpuBackend(_fake_asnumpy())

    capability = backend.capability_manifest(["matmul"]).operators["matmul"]

    assert not capability.available
    assert capability.provenance is ExecutionProvenance.CPU_FALLBACK
    assert "runtime lookup failed" in capability.evidence


def test_availability_only_mode_keeps_legacy_execution_compatible():
    backend = NpuBackend(_fake_asnumpy(matmul=_callable))

    validation = backend.validate_required_ops(
        ["matmul"], strict_npu=False
    )

    assert validation.ok
    assert validation.non_strict == ("matmul",)


def test_runtime_missing_api_fails_even_without_strict_provenance():
    backend = NpuBackend(_fake_asnumpy(add=_callable))

    validation = backend.validate_required_ops(
        ["add", "missing"], strict_npu=False, raise_on_error=False
    )

    assert not validation.ok
    assert validation.missing == ("missing",)
    assert validation.capabilities["add"].strict_npu
    assert not validation.capabilities["missing"].available


def test_plan_validation_checks_only_npu_placements():
    backend = NpuBackend(_fake_asnumpy(add=_callable, matmul=_callable))
    plan = OffloadPlan(
        placement={"add": "NPU", "missing_cpu_op": "CPU"},
        threshold_elements=1,
    )

    validation = backend.validate_plan(plan)

    assert validation.ok
    assert validation.required_ops == ("add",)
    assert "missing_cpu_op" not in validation.capabilities


def test_plan_validation_rejects_matmul_cpu_fallback():
    backend = NpuBackend(_fake_asnumpy(matmul=_callable))
    plan = OffloadPlan(placement={"matmul": "NPU"}, threshold_elements=1)

    validation = backend.validate_plan(plan, raise_on_error=False)

    assert not validation.ok
    assert validation.missing == ()
    assert validation.non_strict == ("matmul",)


def test_unaudited_asnumpy_version_defaults_to_unknown_and_non_strict():
    backend = NpuBackend(_fake_asnumpy(version="0.3.2", add=_callable))

    validation = backend.validate_required_ops(["add"], raise_on_error=False)

    capability = validation.capabilities["add"]
    assert capability.available
    assert capability.provenance is ExecutionProvenance.UNKNOWN
    assert not validation.ok
    assert validation.non_strict == ("add",)


def test_callable_outside_audited_coverage_is_not_promoted_to_npu():
    backend = NpuBackend(_fake_asnumpy(mystery=_callable))

    validation = backend.validate_required_ops(["mystery"], raise_on_error=False)

    assert not validation.ok
    assert validation.capabilities["mystery"].provenance is ExecutionProvenance.UNKNOWN


@pytest.mark.parametrize(
    "provenance",
    [
        ExecutionProvenance.NATIVE_NPU,
        ExecutionProvenance.CUSTOM_NPU,
        ExecutionProvenance.COMPOSITE_NPU,
    ],
)
def test_name_only_strict_npu_rejects_all_scoped_provenance(provenance):
    capability = OperatorCapability(
        name="scoped_op",
        available=True,
        provenance=provenance,
        evidence="profile-scoped evidence",
        constraints=("dtype=float32",),
    )

    assert not capability.strict_npu


def test_static_demo_coverage_alone_does_not_prove_angle_provenance():
    backend = NpuBackend(_fake_asnumpy(angle=_callable))

    validation = backend.validate_required_ops(["angle"], raise_on_error=False)

    assert not validation.ok
    assert validation.capabilities["angle"].available
    assert validation.capabilities["angle"].provenance is ExecutionProvenance.UNKNOWN


def test_manifest_rejects_unknown_schema_version():
    with pytest.raises(ValueError, match="unsupported capability schema"):
        CapabilityManifest.from_dict(
            {
                "schema_version": "2.0",
                "audit_id": "runtime",
                "backend": "npu",
                "backend_version": AUDITED_ASNUMPY_VERSION,
                "environment": {},
                "operators": {},
            }
        )


def test_manifest_detaches_caller_owned_mappings():
    capability = _scoped_cbrt_capability()
    operators = {"cbrt": capability}
    environment = {"module_path": "/private/asnumpy"}
    manifest = CapabilityManifest(
        backend="npu",
        backend_version="private",
        operators=operators,
        environment=environment,
    )

    operators["cbrt"] = OperatorCapability(
        name="cbrt",
        available=True,
        provenance=ExecutionProvenance.NATIVE_NPU,
    )
    environment["module_path"] = "/poisoned"

    assert manifest.operators["cbrt"] is capability
    assert manifest.environment["module_path"] == "/private/asnumpy"
    with pytest.raises(TypeError):
        manifest.operators["other"] = capability


def test_capability_json_rejects_duplicate_keys_and_inconsistent_derived_value():
    manifest = NpuBackend(_fake_asnumpy(add=_callable)).capability_manifest(["add"])
    data = manifest.to_dict()
    data["operators"]["add"]["strict_npu"] = False

    with pytest.raises(ValueError, match="strict_npu is inconsistent"):
        CapabilityManifest.from_dict(data)
    with pytest.raises(ValueError, match="duplicate key"):
        CapabilityManifest.from_json(
            '{"schema_version":"1.1","schema_version":"1.1"}'
        )


def test_fake_module_environment_probe_degrades_to_unknown(monkeypatch):
    import numpilot.env_check as env_check

    for name in (
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_OPP_PATH",
        "CANN_HOME",
        "CANN_VERSION",
        "ASCEND_SOC_VERSION",
        "NPU_SOC_VERSION",
        "SOC_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env_check, "_probe_npu_smi_soc_model", lambda: None)
    backend = NpuBackend(_fake_asnumpy(add=_callable))

    environment = backend.capability_manifest(["add"]).environment

    assert environment == {
        "module_path": "unknown",
        "cann_home": "unknown",
        "cann_version": "unknown",
        "soc_version": "unknown",
        "python_version": platform.python_version(),
    }


def test_npu_environment_uses_read_only_npu_smi_soc_fallback(monkeypatch):
    import numpilot.env_check as env_check

    for name in ("ASCEND_SOC_VERSION", "NPU_SOC_VERSION", "SOC_VERSION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        env_check, "_probe_npu_smi_soc_model", lambda: "Ascend910B3"
    )

    environment = NpuBackend(
        _fake_asnumpy(add=_callable)
    ).capability_manifest(["add"]).environment

    assert environment["soc_version"] == "Ascend910B3"


def test_npu_environment_soc_fallback_fails_closed(monkeypatch):
    import numpilot.env_check as env_check

    for name in ("ASCEND_SOC_VERSION", "NPU_SOC_VERSION", "SOC_VERSION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        env_check, "_probe_npu_smi_soc_model", lambda: "invalid soc\nvalue"
    )

    environment = NpuBackend(
        _fake_asnumpy(add=_callable)
    ).capability_manifest(["add"]).environment

    assert environment["soc_version"] == "unknown"


def test_runtime_environment_can_be_bound_without_initialising_acl(monkeypatch):
    module = _fake_asnumpy(add=_callable)
    module.__file__ = "/opt/asnumpy/asnumpy/__init__.py"
    monkeypatch.setenv("ASCEND_HOME_PATH", "/opt/ascend/cann")
    monkeypatch.setenv("CANN_VERSION", "9.1.0")
    monkeypatch.setenv("ASCEND_SOC_VERSION", "Ascend910B3")

    environment = NpuBackend(module).capability_manifest(["add"]).environment

    assert environment == {
        "module_path": "/opt/asnumpy/asnumpy/__init__.py",
        "cann_home": "/opt/ascend/cann",
        "cann_version": "9.1.0",
        "soc_version": "Ascend910B3",
        "python_version": platform.python_version(),
    }


def _scoped_cbrt_capability(*, constraints=None, signatures=None):
    return OperatorCapability(
        name="cbrt",
        available=True,
        provenance=ExecutionProvenance.CUSTOM_NPU,
        evidence="qualification:qual-0123456789abcdefabcd",
        constraints=tuple(constraints or (
            "dtype=float32",
            "rank>=1",
            "contiguous=true",
        )),
        signature_variants=tuple(signatures or (
            OperatorSignature(positional_args=1),
        )),
    )


def test_scoped_capability_round_trip_preserves_qualified_signatures():
    capability = _scoped_cbrt_capability()
    restored = OperatorCapability.from_dict(capability.to_dict())

    assert restored == capability
    assert not restored.strict_npu
    assert restored.signature_variants == (
        OperatorSignature(positional_args=1),
    )


def test_analysis_profile_set_round_trip_binds_source_callsite_and_signature():
    profiles = AnalysisProfileSet(
        source_sha256="a" * 64,
        callsites=(AnalysisCallsiteProfile(
            qualified_name="numpy.cbrt",
            lineno=3,
            col_offset=11,
            profile=profile_operator_call(
                (np.ones((4,), dtype=np.float32),),
                {},
            ),
        ),),
    )

    restored = AnalysisProfileSet.from_json(json.dumps(profiles.to_dict()))

    assert restored == profiles
    assert restored.find("numpy.cbrt", 3, 11, 1, ()) == profiles.callsites[0].profile
    assert restored.find("numpy.cbrt", 4, 11, 1, ()) is None


def test_analysis_profile_set_rejects_duplicate_callsite_identity():
    callsite = AnalysisCallsiteProfile(
        qualified_name="numpy.cbrt",
        lineno=3,
        col_offset=11,
        profile=profile_operator_call(
            (np.ones((4,), dtype=np.float32),),
            {},
        ),
    )

    with pytest.raises(ValueError, match="duplicate callsites"):
        AnalysisProfileSet(
            source_sha256="a" * 64,
            callsites=(callsite, callsite),
        )


def test_analysis_profile_loader_rejects_symlink_and_reads_private_file(tmp_path):
    profiles = AnalysisProfileSet(
        source_sha256="a" * 64,
        callsites=(AnalysisCallsiteProfile(
            qualified_name="numpy.cbrt",
            lineno=3,
            col_offset=11,
            profile=profile_operator_call(
                (np.ones((4,), dtype=np.float32),),
                {},
            ),
        ),),
    )
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(profiles.to_dict()), encoding="utf-8")
    linked = tmp_path / "linked.json"
    linked.symlink_to(path)

    assert load_analysis_profile_set(str(path)) == profiles
    with pytest.raises(ValueError, match="cannot open"):
        load_analysis_profile_set(str(linked))


@pytest.mark.parametrize(
    "profile",
    [
        lambda: TensorProfile("float32", (True,), True),
        lambda: TensorProfile("float32", tuple(1 for _ in range(65)), True),
        lambda: OperatorCallProfile(
            inputs=(TensorProfile("float32", (1,), True),),
            positional_args=2,
            keyword_args=(),
        ),
    ],
)
def test_profile_contracts_reject_malformed_or_inconsistent_values(profile):
    with pytest.raises(ValueError):
        profile()


def test_tensor_profile_uses_shape_dtype_and_real_c_contiguity():
    contiguous = np.ones((2, 3), dtype=np.float32)
    transposed = contiguous.T

    assert tensor_profile_from_value(contiguous).to_dict() == {
        "dtype": "float32",
        "shape": [2, 3],
        "rank": 2,
        "c_contiguous": True,
    }
    assert not tensor_profile_from_value(transposed).c_contiguous


def test_generic_tensor_profile_rejects_forged_duck_metadata():
    class ForgedTensor:
        shape = (4,)
        dtype = np.dtype("float32")

        class Flags:
            c_contiguous = True

        flags = Flags()

    assert tensor_profile_from_value(ForgedTensor()) is None


def test_generic_tensor_profile_rejects_subclasses_and_non_native_byte_order():
    class ForgedArray(np.ndarray):
        @property
        def dtype(self):
            return np.dtype("float32")

    backing = np.ones((4,), dtype=np.float64)
    forged = backing.view(ForgedArray)
    big_endian = np.ones((4,), dtype=">f4")

    assert tensor_profile_from_value(forged) is None
    assert tensor_profile_from_value(big_endian) is None


def test_profile_match_accepts_only_the_exact_qualified_cbrt_scope():
    capability = _scoped_cbrt_capability()
    valid = profile_operator_call(
        (np.ones((4,), dtype=np.float32),),
        {},
    )

    match = match_capability_profile(capability, valid)

    assert match.matched
    assert match.reasons == ()


@pytest.mark.parametrize(
    "args,kwargs,reason",
    [
        ((np.ones((4,), dtype=np.float64),), {}, "dtype=float64"),
        ((np.array(1.0, dtype=np.float32),), {}, "rank=0"),
        ((np.ones((2, 3), dtype=np.float32).T,), {}, "contiguous=false"),
        ((1.0,), {}, "unprofiled operands"),
        (
            (np.ones((4,), dtype=np.float32), np.ones((4,), dtype=np.float32)),
            {},
            "call signature",
        ),
        ((np.ones((4,), dtype=np.float32),), {"out": None}, "call signature"),
    ],
)
def test_profile_match_fails_closed_outside_scope(args, kwargs, reason):
    match = match_capability_profile(
        _scoped_cbrt_capability(),
        profile_operator_call(args, kwargs),
    )

    assert not match.matched
    assert reason in "; ".join(match.reasons)


@pytest.mark.parametrize(
    "constraints,reason",
    [
        (("dtype=float16", "rank>=1", "contiguous=true"), "unsupported dtype"),
        (("dtype=float32", "layout=ND"), "unsupported profile"),
    ],
)
def test_profile_constraint_grammar_rejects_unknown_or_duplicate_rules(
    constraints, reason
):
    match = match_capability_profile(
        _scoped_cbrt_capability(constraints=constraints),
        profile_operator_call((np.ones((2,), dtype=np.float32),), {}),
    )

    assert not match.matched
    assert reason in "; ".join(match.reasons)


@pytest.mark.parametrize(
    "value,reason",
    [
        (np.ones((1, 2, 3), dtype=np.float32), "rank<=2"),
        (np.ones((0, 3), dtype=np.float32), "numel>=1"),
    ],
)
def test_profile_constraint_upper_rank_and_nonempty_bounds(value, reason):
    capability = _scoped_cbrt_capability(constraints=(
        "dtype=float32",
        "rank>=1",
        "rank<=2",
        "contiguous=true",
        "numel>=1",
    ))

    match = match_capability_profile(
        capability,
        profile_operator_call((value,), {}),
    )

    assert not match.matched
    assert reason in "; ".join(match.reasons)


def test_matcher_rejects_signature_only_half_scoped_capability():
    capability = OperatorCapability(
        name="cbrt",
        available=True,
        provenance=ExecutionProvenance.CUSTOM_NPU,
        evidence="incomplete qualification",
        signature_variants=(OperatorSignature(positional_args=1),),
    )

    match = match_capability_profile(
        capability,
        profile_operator_call((object(),), {}),
    )

    assert not match.matched
    assert "no profile constraints" in "; ".join(match.reasons)


def test_profile_capability_rejects_duplicate_constraints_at_construction():
    with pytest.raises(ValueError, match="constraints are invalid"):
        _scoped_cbrt_capability(
            constraints=("dtype=float32", "dtype=float32")
        )


def test_runtime_guard_rejects_before_target_dispatch():
    calls = []
    manifest = CapabilityManifest(
        backend="npu",
        backend_version="private",
        operators={"cbrt": _scoped_cbrt_capability()},
    )
    guard = RuntimeProfileGuard(manifest, ["cbrt"])

    with pytest.raises(RuntimeProfileError, match="runtime profile rejected"):
        guard.validate_call(
            "cbrt",
            (np.ones((2,), dtype=np.float64),),
            {},
        )
        calls.append("dispatched")

    assert calls == []
    assert not guard.runtime_profile_ok
    assert not guard.checks["cbrt"][0]["matched"]
    assert guard.to_dict()["profile_status"] == "rejected"
    assert guard.to_dict()["ok"] is False
    assert guard.to_dict()["strict_npu"] is False


def test_runtime_guard_records_a_valid_scoped_call():
    manifest = CapabilityManifest(
        backend="npu",
        backend_version="private",
        operators={"cbrt": _scoped_cbrt_capability()},
    )
    guard = RuntimeProfileGuard(manifest, ["cbrt"])

    guard.validate_call("cbrt", (np.ones((2,), dtype=np.float32),), {})

    assert guard.runtime_profile_ok
    assert guard.to_dict()["profile_scoped"] == ["cbrt"]
    assert guard.to_dict()["profile_status"] == "matched"
    assert guard.to_dict()["ok"] is True


def test_runtime_guard_reports_profile_pending_before_real_dispatch():
    manifest = CapabilityManifest(
        backend="npu",
        backend_version="private",
        operators={"cbrt": _scoped_cbrt_capability()},
    )

    data = RuntimeProfileGuard(manifest, ["cbrt"]).to_dict()

    assert data["preflight_ok"] is True
    assert data["profile_status"] == "pending"
    assert data["ok"] is False
    assert data["strict_npu"] is False


def test_runtime_guard_receipt_is_a_detached_snapshot():
    manifest = CapabilityManifest(
        backend="npu",
        backend_version="private",
        operators={"cbrt": _scoped_cbrt_capability()},
    )
    guard = RuntimeProfileGuard(manifest, ["cbrt"])
    receipt = guard.to_dict()

    receipt["runtime_profile_checks"]["cbrt"].append({"matched": True})

    assert guard.profile_status == "pending"
    assert guard.ok is False
    assert guard.checks == {"cbrt": []}


def test_native_npu_guard_does_not_require_a_runtime_profile():
    backend = NpuBackend(_fake_asnumpy(add=_callable))

    guard = backend.prepare_profile_guard(["add"])

    assert guard.profile_scoped == ()
    assert guard.runtime_profile_ok


def test_runtime_guard_rejects_callable_identity_drift_before_dispatch():
    module = _fake_asnumpy(add=_callable)
    backend = NpuBackend(module)
    guard = backend.prepare_profile_guard(["add"])
    module.add = lambda *args: "host replacement"

    with pytest.raises(RuntimeError, match="callable identity drifted"):
        guard.validate_call("add", (object(), object()), {})

    receipt = guard.to_dict()
    assert receipt["profile_status"] == "rejected"
    assert receipt["runtime_target_integrity_ok"] is False
    assert receipt["ok"] is False


def test_profile_guard_binds_target_from_same_capability_lookup():
    first = lambda *args: "first"
    second = lambda *args: "second"

    class RotatingModule(types.ModuleType):
        def __init__(self):
            super().__init__("asnumpy")
            self.__version__ = AUDITED_ASNUMPY_VERSION
            self.lookups = 0

        def __getattribute__(self, name):
            if name == "add":
                lookups = object.__getattribute__(self, "lookups")
                object.__setattr__(self, "lookups", lookups + 1)
                return first if lookups == 0 else second
            return super().__getattribute__(name)

    module = RotatingModule()
    guard = NpuBackend(module).prepare_profile_guard(["add"])

    assert module.lookups == 1
    assert guard.runtime_targets["add"] is first
    with pytest.raises(RuntimeError, match="callable identity drifted"):
        guard.validate_call("add", (object(), object()), {})


def test_strict_receipt_rejects_scoped_capability_without_real_guard():
    capability = _scoped_cbrt_capability()
    receipt = {
        "ok": True,
        "strict_npu": True,
        "required_ops": ["cbrt"],
        "missing": [],
        "non_strict": [],
        "profile_status": "matched",
        "capabilities": {"cbrt": capability.to_dict()},
    }

    issues = strict_receipt_issues(receipt, ["cbrt"])

    assert any("concrete runtime profile guard" in issue for issue in issues)


def test_strict_receipt_rejects_malformed_capability_schema():
    receipt = {
        "ok": True,
        "strict_npu": True,
        "required_ops": ["add"],
        "missing": [],
        "non_strict": [],
        "capabilities": {
            "add": {
                "name": "add",
                "available": True,
                "provenance": "native_npu",
                "strict_npu": True,
                "evidence": "fake",
                "constraints": ["layout=anything"],
                "signature_variants": [],
            }
        },
    }

    issues = strict_receipt_issues(receipt, ["add"])

    assert any("schema rejected" in issue for issue in issues)


def test_overlay_configuration_is_explicit_and_fail_closed(monkeypatch):
    from numpilot.gaps import qualification
    from numpilot.gaps.qualification import (
        CapabilityOverlay,
        QualificationRef,
        ZERO_SHA256,
    )

    module = _fake_asnumpy(cbrt=_callable)
    overlay = CapabilityOverlay(
        revision=1,
        parent_sha256=ZERO_SHA256,
        base_manifest_sha256="1" * 64,
        entries=(QualificationRef("cbrt", "cbrt/qualification.json", "2" * 64),),
    )
    seen = {}

    monkeypatch.setattr(qualification, "load_overlay", lambda path: overlay)

    def fake_apply(base, supplied, **kwargs):
        seen["names"] = tuple(base.operators)
        seen["root"] = kwargs["evidence_root"]
        return CapabilityManifest(
            backend=base.backend,
            backend_version=base.backend_version,
            environment=base.environment,
            audit_id=base.audit_id + "+qualification-test",
            operators={"cbrt": _scoped_cbrt_capability()},
        )

    monkeypatch.setattr(qualification, "apply_overlay", fake_apply)
    backend = NpuBackend(
        module,
        overlay_config=OverlayRuntimeConfig(
            "/private/overlay.json",
            "/private/evidence",
        ),
    )

    manifest = backend.capability_manifest(["cbrt"])

    assert seen == {"names": ("cbrt",), "root": "/private/evidence"}
    assert manifest.operators["cbrt"].provenance is ExecutionProvenance.CUSTOM_NPU


def test_overlay_failure_never_falls_back_to_unqualified_base(monkeypatch):
    from numpilot.gaps import qualification

    monkeypatch.setattr(
        qualification,
        "load_overlay",
        lambda path: (_ for _ in ()).throw(RuntimeError("tampered overlay")),
    )
    backend = NpuBackend(
        _fake_asnumpy(cbrt=_callable),
        overlay_config=OverlayRuntimeConfig("/private/overlay.json", "/private/evidence"),
    )

    with pytest.raises(RuntimeError, match="tampered overlay"):
        backend.capability_manifest(["cbrt"])


def test_stricter_move_identity_can_activate_exact_legacy_overlay_base(
    monkeypatch,
):
    from numpilot.backends import npu
    from numpilot.gaps import qualification
    from numpilot.gaps.qualification import (
        CapabilityOverlay,
        QualificationRef,
        ZERO_SHA256,
        capability_manifest_sha256,
    )

    monkeypatch.setattr(
        npu, "_p0_move_runtime_identity_matches", lambda module: True
    )
    module = _fake_asnumpy(
        version=P0_MOVE_ASNUMPY_VERSION,
        apply_gradient_descent=_callable,
    )
    plain = NpuBackend(module)._raw_capability_manifest(
        ("apply_gradient_descent",)
    )
    assert plain.audit_id == P0_MOVE_ASNUMPY_AUDIT_ID
    legacy = npu._overlay_base_with_audit_id(
        plain,
        ("apply_gradient_descent",),
        P0_MOVE_LEGACY_OVERLAY_AUDIT_ID,
    )
    overlay = CapabilityOverlay(
        revision=1,
        parent_sha256=ZERO_SHA256,
        base_manifest_sha256=capability_manifest_sha256(legacy),
        entries=(QualificationRef(
            "apply_gradient_descent",
            "apply_gradient_descent/qualification.json",
            "2" * 64,
        ),),
    )
    monkeypatch.setattr(qualification, "load_overlay", lambda _path: overlay)
    seen = {}

    def fake_apply(base, _overlay, **_kwargs):
        seen["audit_id"] = base.audit_id
        return CapabilityManifest(
            backend=base.backend,
            backend_version=base.backend_version,
            environment=base.environment,
            audit_id=base.audit_id + "+qualification-test",
            operators={
                "apply_gradient_descent": OperatorCapability(
                    name="apply_gradient_descent",
                    available=True,
                    provenance=ExecutionProvenance.CUSTOM_NPU,
                    evidence="qualification:qual-" + "1" * 20,
                )
            },
        )

    monkeypatch.setattr(qualification, "apply_overlay", fake_apply)
    manifest = NpuBackend(
        module,
        overlay_config=OverlayRuntimeConfig(
            "/private/overlay.json", "/private/evidence"
        ),
    ).capability_manifest(("apply_gradient_descent",))

    assert seen["audit_id"] == P0_MOVE_LEGACY_OVERLAY_AUDIT_ID
    assert manifest.audit_id.endswith("+qualification-test")


def test_overlay_runtime_uses_one_atomic_raw_snapshot(monkeypatch):
    from numpilot.gaps import qualification
    from numpilot.gaps.qualification import (
        CapabilityOverlay,
        QualificationRef,
        ZERO_SHA256,
    )

    overlay = CapabilityOverlay(
        revision=1,
        parent_sha256=ZERO_SHA256,
        base_manifest_sha256="1" * 64,
        entries=(QualificationRef("cbrt", "cbrt/qualification.json", "2" * 64),),
    )
    monkeypatch.setattr(qualification, "load_overlay", lambda path: overlay)

    module = _fake_asnumpy(cbrt=_callable, add=_callable)
    backend = NpuBackend(
        module,
        overlay_config=OverlayRuntimeConfig("/private/overlay.json", "/private/evidence"),
    )
    calls = []
    original = backend._raw_capability_manifest

    def counted(required):
        calls.append(tuple(sorted(required)))
        return original(required)

    backend._raw_capability_manifest = counted

    def fake_apply(base, supplied, **kwargs):
        assert tuple(base.operators) == ("cbrt",)
        return CapabilityManifest(
            backend=base.backend,
            backend_version=base.backend_version,
            environment=base.environment,
            audit_id=base.audit_id + "+qualification-test",
            operators={"cbrt": _scoped_cbrt_capability()},
        )

    monkeypatch.setattr(qualification, "apply_overlay", fake_apply)

    manifest = backend.capability_manifest(["add"])

    assert calls == [("add", "cbrt")]
    assert tuple(manifest.operators) == ("add",)
    assert manifest.operators["add"].strict_npu
