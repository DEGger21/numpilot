"""Contracts for run-scoped custom-operator admission.

The admission file is the only way a verified custom AsNumpy operator can enter
a contract migration run.  It binds the public API identity and call contract to
the adapter manifest and a passing adapter build receipt, by SHA-256.
"""

import hashlib
import json
from pathlib import Path

import pytest

from numpilot.agents.analysis import AnalysisAgent, _ASNUMPY_CALL_CONTRACTS
from numpilot.coverage import is_supported
from numpilot.custom_operator_admission import (
    CustomOperatorAdmissionError,
    load_custom_operator_admission,
)

WHEEL = "a" * 64

STD_CONTRACT = {
    "required": ["a"],
    "positional_limit": 1,
    "allowed_keywords": ["axis"],
    "positional_names": ["a"],
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(name: str = "std", schema: str = "numpilot.asnumpy_adapter") -> dict:
    if schema == "asnumpy.adapter.manifest.v1":
        return {
            "schema": schema,
            "operator": {
                "source_api": "numpy.%s" % name,
                "public_api": "asnumpy.%s" % name,
            },
        }
    if schema == "numpilot.asnumpy.adapter":
        return {
            "schema": schema,
            "operator": {
                "numpy_api": "numpy.%s" % name,
                "asnumpy_api": "asnumpy.%s" % name,
            },
        }
    if schema == "numpilot.asnumpy_adapter.operator":
        return {
            "schema": schema,
            "api": {"numpy": "numpy.%s" % name, "asnumpy": "asnumpy.%s" % name},
        }
    if schema == "numpilot.asnumpy-adapter.v1":
        return {
            "schema": schema,
            "qualified_api": "numpy.%s" % name,
            "public_api": "asnumpy.%s" % name,
        }
    if schema == "numpilot.asnumpy_adapter.test_manifest":
        return {
            "schema": schema,
            "operator": {
                "api_name": name,
                "semantic_source": "numpy.%s" % name,
                "public_api": "asnumpy.%s" % name,
            },
        }
    if schema == "numpilot.asnumpy_adapter.operator_nested":
        return {
            "schema": "numpilot.asnumpy_adapter",
            "operator": {
                "numpy_api": "numpy.%s" % name,
                "asnumpy_api": "asnumpy.%s" % name,
            },
        }
    if schema == "numpilot.asnumpy_adapter.api_name":
        return {
            "schema": "numpilot.asnumpy_adapter",
            "api_name": name,
            "asnumpy_api": "asnumpy.%s" % name,
        }
    return {
        "schema": schema,
        "numpy_api": "numpy.%s" % name,
        "asnumpy_api": "asnumpy.%s" % name,
    }


def _receipt(status: str = "pass", passed: int = 3, failed: int = 0, wheel: str = WHEEL) -> dict:
    return {
        "schema": "numpilot.adapter_build_receipt",
        "status": status,
        "wheel_sha256": wheel,
        "tests": {"passed": passed, "failed": failed},
    }


def _write(root: Path, payload: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / "admission.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def _fixture(
    tmp_path: Path,
    *,
    name: str = "std",
    schema: str = "numpilot.asnumpy_adapter",
    receipt: dict = None,
    manifest: dict = None,
    contract: dict = None,
    wheel_sha256: str = None,
) -> Path:
    root = tmp_path / "admission"
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    manifest_path = evidence / "operator.json"
    manifest_path.write_text(json.dumps(manifest or _manifest(name, schema)), encoding="utf-8")
    receipt_path = evidence / "receipt.json"
    receipt_path.write_text(json.dumps(receipt or _receipt()), encoding="utf-8")
    payload = {
        "schema": "numpilot.custom-operator-admission",
        "schema_version": 1,
        "target_soc": "Ascend910B3",
        "cann_version": "9.2.0-beta.2",
        "entries": [
            {
                "name": name,
                "source_api": "numpy." + name,
                "asnumpy_api": "asnumpy." + name,
                "contract": contract or STD_CONTRACT,
                "evidence": {
                    "adapter_manifest": {
                        "path": "evidence/operator.json",
                        "sha256": _sha(manifest_path),
                    },
                    "adapter_build_receipt": {
                        "path": "evidence/receipt.json",
                        "sha256": _sha(receipt_path),
                    },
                    "wheel_sha256": wheel_sha256 or WHEEL,
                },
            }
        ],
    }
    return _write(root, payload)


def test_loads_admitted_operator_with_bound_contract(tmp_path):
    path = _fixture(tmp_path)
    admission = load_custom_operator_admission(path)

    assert admission.admission_id == _sha(path)
    assert admission.names == ("std",)
    assert admission.target_soc == "Ascend910B3"
    required, positional_limit, allowed, names = admission.contracts["std"]
    assert required == ("a",)
    assert positional_limit == 1
    assert allowed == frozenset({"axis"})
    assert names == ("a",)


@pytest.mark.parametrize(
    "schema",
    [
        "asnumpy.adapter.manifest.v1",
        "numpilot.asnumpy_adapter.operator",
        "numpilot.asnumpy_adapter",
        "numpilot.asnumpy_adapter.test_manifest",
        "numpilot.asnumpy.adapter",
        "numpilot.asnumpy-adapter.v1",
        "numpilot.asnumpy-adapter.v1",
        "numpilot.asnumpy_adapter.api_name",
        "numpilot.asnumpy_adapter.api_name",
        "numpilot.asnumpy_adapter.operator_nested",
    ],
)
def test_accepts_every_known_manifest_identity_shape(tmp_path, schema):
    admission = load_custom_operator_admission(_fixture(tmp_path, schema=schema))
    assert admission.names == ("std",)


def test_rejects_unknown_admission_field(tmp_path):
    payload = json.loads(_fixture(tmp_path).read_text(encoding="utf-8"))
    payload["extra"] = 1
    with pytest.raises(CustomOperatorAdmissionError, match="unknown fields"):
        load_custom_operator_admission(_write(tmp_path / "admission", payload))


def test_rejects_unknown_entry_field(tmp_path):
    path = _fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["model_identity"] = "private"
    with pytest.raises(CustomOperatorAdmissionError, match="unknown fields"):
        load_custom_operator_admission(_write(path.parent, payload))


def test_rejects_absolute_evidence_path(tmp_path):
    root = tmp_path / "admission"
    path = _fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["evidence"]["adapter_manifest"]["path"] = "/etc/hostname"
    with pytest.raises(CustomOperatorAdmissionError, match="relative path"):
        load_custom_operator_admission(_write(root, payload))


def test_rejects_evidence_sha_mismatch(tmp_path):
    root = tmp_path / "admission"
    path = _fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["evidence"]["adapter_manifest"]["sha256"] = "b" * 64
    with pytest.raises(CustomOperatorAdmissionError, match="SHA-256 mismatch"):
        load_custom_operator_admission(_write(root, payload))


def test_rejects_missing_evidence_file(tmp_path):
    root = tmp_path / "admission"
    path = _fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["evidence"]["adapter_build_receipt"]["path"] = "evidence/absent.json"
    payload["entries"][0]["evidence"]["adapter_build_receipt"]["sha256"] = WHEEL
    with pytest.raises(CustomOperatorAdmissionError, match="does not exist"):
        load_custom_operator_admission(_write(root, payload))


@pytest.mark.parametrize(
    "receipt",
    [
        _receipt(status="fail"),
        _receipt(passed=0, failed=0),
        _receipt(passed=2, failed=1),
    ],
)
def test_rejects_receipt_without_passing_tests(tmp_path, receipt):
    path = _fixture(tmp_path, receipt=receipt)
    with pytest.raises(CustomOperatorAdmissionError):
        load_custom_operator_admission(path)


def test_rejects_wheel_sha_not_matching_receipt(tmp_path):
    path = _fixture(tmp_path, wheel_sha256="c" * 64)
    with pytest.raises(CustomOperatorAdmissionError, match="does not match the build receipt"):
        load_custom_operator_admission(path)


def test_rejects_manifest_identity_mismatch(tmp_path):
    path = _fixture(tmp_path, manifest=_manifest("flip"))
    with pytest.raises(CustomOperatorAdmissionError, match="manifest identity mismatch"):
        load_custom_operator_admission(path)


def test_rejects_unrecognized_manifest_schema(tmp_path):
    manifest = _manifest()
    manifest["schema"] = "unknown.schema"
    path = _fixture(tmp_path, manifest=manifest)
    with pytest.raises(CustomOperatorAdmissionError, match="recognized API identity"):
        load_custom_operator_admission(path)


def test_rejects_symlinked_admission_file(tmp_path):
    real = _fixture(tmp_path)
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(CustomOperatorAdmissionError, match="regular non-symlink"):
        load_custom_operator_admission(link)


def test_rejects_duplicate_api_names(tmp_path):
    path = _fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"].append(dict(payload["entries"][0]))
    with pytest.raises(CustomOperatorAdmissionError, match="duplicate"):
        load_custom_operator_admission(_write(path.parent, payload))


def test_admission_contract_is_not_in_the_pinned_catalog(tmp_path):
    admission = load_custom_operator_admission(_fixture(tmp_path))
    assert admission.names == ("std",)
    assert not is_supported("std")
    assert "std" not in _ASNUMPY_CALL_CONTRACTS


def test_analysis_accepts_admitted_contract_without_mutating_catalog(tmp_path):
    admission = load_custom_operator_admission(_fixture(tmp_path))
    source = (
        "import numpy as np\n"
        "def forward(x):\n"
        "    return np.std(x, axis=1)\n"
    )

    baseline = AnalysisAgent().analyze(source)
    assert baseline.unsupported == ["std"]

    admitted = AnalysisAgent().analyze(
        source, custom_contracts=admission.contracts
    )
    assert admitted.unsupported == []
    assert admitted.migratable
    assert admitted.op_uses[0].name == "std"

    assert "std" not in _ASNUMPY_CALL_CONTRACTS
    assert AnalysisAgent().analyze(source).unsupported == ["std"]


def test_admitted_contract_still_rejects_unverified_keywords(tmp_path):
    admission = load_custom_operator_admission(_fixture(tmp_path))
    source = (
        "import numpy as np\n"
        "def forward(x):\n"
        "    return np.std(x, axis=1, ddof=1)\n"
    )
    result = AnalysisAgent().analyze(source, custom_contracts=admission.contracts)
    assert result.unsupported == ["std"]
    assert "ddof" in result.op_uses[0].reason
