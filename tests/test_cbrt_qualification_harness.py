"""Safety regressions for the external cbrt qualification harness."""

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from numpilot.capabilities import (
    ExecutionProvenance,
    OperatorCapability,
    OperatorSignature,
)


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "cbrt_qualification_harness.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "cbrt_qualification_harness", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
harness = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(harness)


def test_subprocess_environment_is_private_allowlisted_and_device_zero(
    tmp_path, monkeypatch
):
    sentinel = "NUMPILOT_PARENT_SECRET_MUST_NOT_CROSS"
    inherited_names = (
        "NUMPILOT_LLM_AUTH_TOKEN",
        "NUMPILOT_LLM_BASE_URL",
        "JPY_AUTH_TOKEN",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "SSH_AUTH_SOCK",
        "GIT_ASKPASS",
        "PYTHONSTARTUP",
        "LD_PRELOAD",
        "PATH",
        "PYTHONPATH",
        "HOME",
        "TMPDIR",
        "XDG_CONFIG_HOME",
    )
    for name in inherited_names:
        monkeypatch.setenv(name, sentinel)
    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", "7,8")

    sandbox = tmp_path / "sandbox"
    vendor = tmp_path / "private-opp" / "vendors" / "cbrt_custom"
    env = harness._command_env(
        sandbox_root=sandbox,
        python=Path(sys.executable),
        custom_opp_path=vendor,
    )

    assert env["ASCEND_VISIBLE_DEVICES"] == "0"
    assert env["ASCEND_CUSTOM_OPP_PATH"] == str(vendor)
    assert sentinel not in env.values()
    for name in inherited_names[:11]:
        assert name not in env
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(_SCRIPT.parents[1]),
        "/usr/local/Ascend/cann-9.0.0/python/site-packages",
        "/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe",
    ]
    assert env["PATH"].split(os.pathsep)[0] == str(
        Path(sys.executable).absolute().parent
    )
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert env["TMP"] == env["TEMP"] == env["TMPDIR"]
    resolved_sandbox = sandbox.resolve()
    for name in (
        "HOME",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_RUNTIME_DIR",
        "ASNUMPY_LOG_DIR",
    ):
        path = Path(env[name])
        path.relative_to(resolved_sandbox)
        assert path.is_dir()
        assert path.stat().st_mode & 0o777 == 0o700

    build_env = harness._command_env(
        sandbox_root=tmp_path / "build-sandbox",
        python=Path(sys.executable),
    )
    assert "ASCEND_CUSTOM_OPP_PATH" not in build_env
    assert build_env["ASCEND_VISIBLE_DEVICES"] == "0"


def test_virtualenv_python_symlink_keeps_invoked_path(tmp_path):
    executable = tmp_path / "venv-python"
    executable.symlink_to(Path(sys.executable).resolve())

    selected = harness._require_python(str(executable))

    assert selected == executable.absolute()
    assert selected.is_symlink()


def test_ordinary_input_file_rejects_a_symlink(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "input.json"
    link.symlink_to(target)

    with pytest.raises(RuntimeError, match="ordinary file"):
        harness._require_file(str(link), "test input")


def test_qualification_spec_changes_only_the_three_scope_fields():
    # Keep this regression self-contained: a fresh NumPilot clone must not
    # require the separately maintained numpilot-showcases repository.
    original = {
        "api_name": "cbrt",
        "asnumpy_api": "asnumpy.cbrt",
        "callsites": [{
            "call_expr": "numpy.cbrt(<arg0>)",
            "col_offset": 17,
            "lineno": 18,
            "qualified_api": "numpy.cbrt",
            "reason": "API 不在静态覆盖集",
        }],
        "cann_version": "9.0.0",
        "dtype_rules": ["runtime_profile_required"],
        "gap_id": "cbrt-a48e7704b679eb02",
        "qualified_api": "numpy.cbrt",
        "reference_cases": [],
        "route": "ascendc_registry_candidate",
        "route_decisions": [
            {
                "reason": "requires a semantic composition resolver",
                "route": "asnumpy_composite",
                "status": "not_evaluated",
            },
            {
                "reason": "check a pinned CANN operator catalog before new kernel work",
                "route": "cann_existing",
                "status": "review_required",
            },
            {
                "reason": "explicit draft generation is allowed; execution remains gated",
                "route": "ascendc_registry",
                "status": "candidate_for_draft",
            },
        ],
        "schema_version": 1,
        "semantic_source": "numpy.cbrt",
        "shape_rules": ["runtime_profile_required"],
        "signature_variants": [{"keyword_args": [], "positional_args": 1}],
        "source_path": (
            "/REDACTED/showcases/cbrt_sensor_calibration/"
            "source/cube_root_sensor_calibration.py"
        ),
        "source_sha256": (
            "9022b2174d5e1dde99eb110728edb6708b42b7cbc86374accc2fe3a2c94f1453"
        ),
        "target_soc": "Ascend910B",
    }

    qualified = harness._qualification_spec(original)

    assert qualified["source_path"] == "source/cube_root_sensor_calibration.py"
    assert qualified["dtype_rules"] == ["float32"]
    assert qualified["shape_rules"] == [
        "rank>=1",
        "contiguous=true",
        "numel>=1",
    ]
    for key in set(original) - {"source_path", "dtype_rules", "shape_rules"}:
        assert qualified[key] == original[key]
    assert original["dtype_rules"] == ["runtime_profile_required"]
    assert original["shape_rules"] == ["runtime_profile_required"]


def test_private_opp_internal_symlink_is_materialized(tmp_path):
    target = tmp_path / "vendor" / "lib" / "libtarget.so"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"bound-runtime-library")
    link = tmp_path / "vendor" / "liboptiling.so"
    link.symlink_to("lib/libtarget.so")

    harness._materialize_private_opp_symlinks(tmp_path)

    assert link.is_file()
    assert not link.is_symlink()
    assert link.read_bytes() == target.read_bytes()


def test_private_opp_symlink_may_not_escape_the_evidence_root(tmp_path):
    root = tmp_path / "opp"
    root.mkdir()
    outside = tmp_path / "outside.so"
    outside.write_bytes(b"outside")
    link = root / "liboptiling.so"
    link.symlink_to(outside)

    with pytest.raises(RuntimeError, match="escapes"):
        harness._materialize_private_opp_symlinks(root)

    assert link.is_symlink()


def test_junit_counts_preserve_skip_failure_and_error_totals(tmp_path):
    report = tmp_path / "junit.xml"
    report.write_text(
        '<testsuites><testsuite tests="4" failures="1" errors="1" '
        'skipped="1" /></testsuites>',
        encoding="utf-8",
    )

    assert harness._junit_counts(report) == {
        "tests": 4,
        "failures": 1,
        "errors": 1,
        "skipped": 1,
    }


def test_non_signature_gates_cannot_claim_signature_coverage():
    common = {
        "artifact_files": 18,
        "build": {"output_artifacts": 2},
        "tests_passed": 25,
        "runtime_result": {
            "numerical_cases": 7,
            "cpu_fallback_detected": False,
            "public_callable": True,
            "kernel_launches": 9,
        },
        "signature_count": 1,
        "signature_sha256": "1" * 64,
    }

    artifact = harness._gate_claims("artifact_integrity", **common)
    build = harness._gate_claims("build", **common)
    unit = harness._gate_claims("unit_test", **common)

    assert set(artifact) == {"verified_files", "mismatches"}
    assert set(build) == {"exit_code", "output_artifacts"}
    assert unit["covered_signature_variants"] == 1
    assert unit["signature_variants_sha256"] == "1" * 64


def test_scoped_cbrt_is_admitted_by_profile_not_name_only():
    capability = OperatorCapability(
        name="cbrt",
        available=True,
        provenance=ExecutionProvenance.CUSTOM_NPU,
        evidence="qualification:qual-" + "1" * 20,
        constraints=harness.CONSTRAINTS,
        signature_variants=(OperatorSignature(positional_args=1),),
    )

    assert not capability.strict_npu
    harness._assert_profile_admitted(capability)


def test_showcase_chain_uploads_all_numpy_forward_parameters():
    uploads = []

    class FakeNdarray:
        @staticmethod
        def from_numpy(value):
            uploads.append(value)
            return ("device", value.shape)

    class FakeAsnumpy:
        ndarray = FakeNdarray

    signal = np.ones((8193,), dtype=np.float32)
    gain = np.ones((1,), dtype=np.float32)
    offset = np.zeros((1,), dtype=np.float32)
    existing_device = object()

    converted = harness._upload_showcase_inputs(
        FakeAsnumpy, signal, gain, offset, existing_device
    )

    assert len(uploads) == 3
    assert all(
        observed is expected
        for observed, expected in zip(uploads, (signal, gain, offset))
    )
    assert converted[:3] == (
        ("device", (8193,)),
        ("device", (1,)),
        ("device", (1,)),
    )
    assert converted[3] is existing_device


def test_publication_rollback_does_not_hide_unlink_failures(tmp_path, monkeypatch):
    publication = tmp_path / "qualification.json"
    publication.write_text("{}", encoding="utf-8")

    def fail_unlink(_self):
        raise PermissionError("injected rollback failure")

    monkeypatch.setattr(type(publication), "unlink", fail_unlink)
    with pytest.raises(RuntimeError, match="failed to roll back"):
        harness._remove_final_publications((publication,))
