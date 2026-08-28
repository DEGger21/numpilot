import hashlib
import json
import sys
import types

import numpy as np

import numpilot.__main__ as cli
from numpilot.capabilities import (
    CapabilityManifest,
    ExecutionProvenance,
    OperatorCapability,
    RuntimeIdentity,
)
from numpilot.engines.llm import LlmEngine


SOURCE = """\
import numpy as np
x = np.full((2,), 1.0, dtype=np.float32)
y = np.exp(x)
print(y)
"""

MISSING_SOURCE = """\
import numpy as np
x = np.full((2,), 1.0, dtype=np.float32)
y = np.definitely_missing(x)
print(y)
"""

CONTRACT_SHAPED_SOURCE = """\
import numpy as np

def build_inputs():
    return np.full((2,), 1.0, dtype=np.float32)

def forward(x):
    return np.exp(x)
"""

CONTRACT_SHAPED_MIGRATED = """\
import asnumpy as ap

def build_inputs():
    return ap.full((2,), 1.0, dtype=ap.float32)

def forward(x):
    return ap.exp(x)
"""

MIGRATED = """\
import asnumpy as ap
x = ap.full((2,), 1.0, dtype=ap.float32)
y = ap.exp(x)
print(y)
"""


class _Backend:
    name = "npu"

    def __init__(self, exp_provenance=ExecutionProvenance.NATIVE_NPU):
        self.exp_provenance = exp_provenance
        self.module = types.SimpleNamespace(
            full=lambda *args, **kwargs: None,
            exp=lambda *args, **kwargs: None,
            float32=np.float32,
        )

    def get_module(self):
        return self.module

    def capability_manifest(self, required_ops=None):
        operators = {}
        for name in required_ops or ():
            provenance = (
                self.exp_provenance
                if name == "exp"
                else ExecutionProvenance.NATIVE_NPU
            )
            operators[name] = OperatorCapability(
                name=name,
                available=True,
                provenance=provenance,
                evidence="fake audited " + name,
            )
        return CapabilityManifest(
            backend="npu",
            backend_version="test-wheel",
            audit_id="test-audit-v1",
            environment={},
            operators=operators,
        )


def _runtime_identity(native_hash="6" * 64):
    return RuntimeIdentity(
        asnumpy_repository="https://gitcode.com/cann/asnumpy.git",
        asnumpy_commit="1" * 40,
        asnumpy_tree="2" * 40,
        asnumpy_version="0.3.1.dev4+g1929a3ba4",
        wrapper_sha256={"math.py": "3" * 64},
        module_path="/opt/asnumpy/__init__.py",
        module_sha256="4" * 64,
        native_module_path="/opt/asnumpy/_core.so",
        native_module_sha256=native_hash,
        cann_home="/usr/local/Ascend/cann-9.0.0",
        cann_version="9.0.0",
        soc_model="Ascend910B4",
        python_executable="/opt/numpilot/bin/python",
        python_version="3.10.20",
        bootstrap_receipt_path="/opt/numpilot/receipt.json",
        bootstrap_receipt_sha256="7" * 64,
        audit_id="test-runtime-audit-v1",
        status="matched",
        issues=(),
    )


class _QualifiedBackend(_Backend):
    def __init__(self, *args, identities=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.identities = list(identities or [_runtime_identity()])

    def runtime_identity(self):
        if len(self.identities) > 1:
            return self.identities.pop(0)
        return self.identities[0]


class _MissingQualifiedBackend(_QualifiedBackend):
    def capability_manifest(self, required_ops=None):
        manifest = super().capability_manifest(required_ops)
        operators = dict(manifest.operators)
        operators["definitely_missing"] = OperatorCapability(
            name="definitely_missing",
            available=False,
            provenance=ExecutionProvenance.UNKNOWN,
            evidence="fake audited missing operator",
        )
        return CapabilityManifest(
            backend=manifest.backend,
            backend_version=manifest.backend_version,
            audit_id=manifest.audit_id,
            environment=manifest.environment,
            operators=operators,
        )


class _NonInteractiveInput:
    def isatty(self):
        return False


def _receipt(content, model="glm-5.3"):
    return {
        "schema_version": 1,
        "utc_start": "2026-08-25T12:00:00Z",
        "utc_end": "2026-08-25T12:00:01Z",
        "http_status": 200,
        "requested_model": model,
        "response_model": model,
        "response_id": "response-cli-test",
        "response_created": 1787659200,
        "request_body_sha256": "1" * 64,
        "response_body_sha256": "2" * 64,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "request_id": "request-cli-test",
        "usage": {"total_tokens": 101},
        "tls_verification": "enabled",
    }


def _setup(monkeypatch, backend):
    monkeypatch.setattr(
        cli, "_qualified_runtime_inputs", lambda _args, _name: (backend, None)
    )
    monkeypatch.setattr(cli, "_preflight", lambda _backend, _skip: None)
    monkeypatch.setattr(
        cli,
        "_load_llm_or_report",
        lambda: types.SimpleNamespace(model="glm-5.3", provider="glm"),
    )

    raw = "```python\n" + MIGRATED + "```"

    def chat(_system, _user):
        return raw, _receipt(raw)

    monkeypatch.setattr(LlmEngine, "_chat_with_receipt", staticmethod(chat))


def test_cli_auto_routes_plain_script_to_glm_lane(tmp_path, monkeypatch):
    source = tmp_path / "case.py"
    source.write_text(SOURCE)
    out = tmp_path / "out"
    _setup(monkeypatch, _Backend())

    rc = cli.main([
        "migrate", str(source), "--backend", "npu", "--out", str(out)
    ])

    assert rc == 0
    assert (out / "migrated.py").read_text() == MIGRATED
    report = json.loads((out / "script-migration-report.json").read_text())
    assert report["status"] == "migrated_strict_npu"
    assert report["generated_by"] == "configured_model"
    assert report["model"] == "glm-5.3"
    assert report["requested_model"] == "glm-5.3"
    assert report["response_model"] == "glm-5.3"
    assert report["execution_performed"] is False


def test_cli_production_lane_embeds_and_revalidates_qualified_analysis(
    tmp_path, monkeypatch
):
    source = tmp_path / "case.py"
    source.write_text(SOURCE)
    out = tmp_path / "out"
    _setup(monkeypatch, _QualifiedBackend())

    rc = cli.main([
        "migrate", str(source), "--backend", "npu", "--out", str(out)
    ])

    assert rc == 0
    report = json.loads((out / "script-migration-report.json").read_text())
    assert report["qualified_analysis"]["decision"] == "DIRECT"
    assert report["qualified_analysis"]["actual_remote_call"] is False
    assert report["qualified_analysis"]["execution_performed"] is False


def test_cli_runtime_identity_drift_stops_before_the_model(
    tmp_path, monkeypatch
):
    source = tmp_path / "case.py"
    source.write_text(SOURCE)
    out = tmp_path / "out"
    backend = _QualifiedBackend(identities=[
        _runtime_identity("6" * 64),
        _runtime_identity("8" * 64),
    ])
    _setup(monkeypatch, backend)

    def forbidden_chat(*_args, **_kwargs):
        raise AssertionError("model must not be called after runtime drift")

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(forbidden_chat)
    )

    rc = cli.main([
        "migrate", str(source), "--backend", "npu", "--out", str(out)
    ])

    assert rc == 2
    assert not out.exists() or not any(out.iterdir())


def test_cli_default_keeps_contract_shaped_file_in_ordinary_script_lane(
    tmp_path, monkeypatch
):
    source = tmp_path / "contract_shaped.py"
    source.write_text(CONTRACT_SHAPED_SOURCE)
    out = tmp_path / "out"
    _setup(monkeypatch, _Backend())
    raw = "```python\n" + CONTRACT_SHAPED_MIGRATED + "```"

    def chat(_system, _user):
        return raw, _receipt(raw)

    monkeypatch.setattr(LlmEngine, "_chat_with_receipt", staticmethod(chat))

    rc = cli.main([
        "migrate", str(source), "--backend", "npu", "--out", str(out)
    ])

    assert rc == 0
    assert (out / "migrated.py").read_text() == CONTRACT_SHAPED_MIGRATED
    report = json.loads((out / "script-migration-report.json").read_text())
    assert report["mode"] == "script"
    assert report["generated_by"] == "configured_model"


def test_cli_noninteractive_fallback_requires_explicit_consent(
    tmp_path, monkeypatch
):
    source = tmp_path / "case.py"
    source.write_text(SOURCE)
    out = tmp_path / "out"
    _setup(monkeypatch, _Backend(ExecutionProvenance.CPU_FALLBACK))
    monkeypatch.setattr(sys, "stdin", _NonInteractiveInput())

    def forbidden_chat(*_args, **_kwargs):
        raise AssertionError("GLM must not be called before fallback consent")

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(forbidden_chat)
    )
    rc = cli.main([
        "migrate", str(source), "--backend", "npu", "--out", str(out)
    ])

    assert rc == 1
    assert not (out / "migrated.py").exists()
    report = json.loads((out / "script-migration-report.json").read_text())
    assert report["status"] == "confirmation_required"


def test_cli_explicitly_accepted_fallback_is_never_strict_or_performance_valid(
    tmp_path, monkeypatch
):
    source = tmp_path / "case.py"
    source.write_text(SOURCE)
    out = tmp_path / "out"
    _setup(monkeypatch, _Backend(ExecutionProvenance.CPU_FALLBACK))

    rc = cli.main([
        "migrate", str(source), "--backend", "npu", "--out", str(out),
        "--fallback-policy", "accept",
    ])

    assert rc == 0
    report = json.loads((out / "script-migration-report.json").read_text())
    assert report["status"] == "migrated_with_fallback"
    assert report["fallback_accepted"] is True
    assert report["strict_npu"] is False
    assert report["performance_valid"] is False


def test_cli_available_unknown_api_requires_audit_without_glm_or_gap(
    tmp_path, monkeypatch
):
    source = tmp_path / "case.py"
    source.write_text(SOURCE)
    out = tmp_path / "out"
    _setup(monkeypatch, _Backend(ExecutionProvenance.UNKNOWN))

    def forbidden_chat(*_args, **_kwargs):
        raise AssertionError("an unaudited API must not enter GLM migration")

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(forbidden_chat)
    )

    rc = cli.main([
        "migrate", str(source), "--backend", "npu", "--out", str(out)
    ])

    assert rc == 1
    assert not (out / "migrated.py").exists()
    assert not (out / "gap-request.json").exists()
    report = json.loads((out / "script-migration-report.json").read_text())
    assert report["status"] == "capability_audit_required"
    assert report["analysis"]["counts"]["capability_audit_required"] == 1


def test_cli_rejected_fallback_prepares_gap_handoff_without_calling_glm(
    tmp_path, monkeypatch
):
    source = tmp_path / "case.py"
    source.write_text(SOURCE)
    out = tmp_path / "out"
    _setup(monkeypatch, _Backend(ExecutionProvenance.CPU_FALLBACK))

    def forbidden_chat(*_args, **_kwargs):
        raise AssertionError("migration GLM must not run after fallback rejection")

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(forbidden_chat)
    )
    monkeypatch.setattr(
        cli,
        "_load_llm_or_report",
        lambda: (_ for _ in ()).throw(
            AssertionError("fallback rejection must not load LLM configuration")
        ),
    )
    rc = cli.main([
        "migrate", str(source), "--backend", "npu", "--out", str(out),
        "--engine", "llm",
        "--fallback-policy", "reject", "--target-soc", "ascend910b",
        "--cann-version", "9.0.0",
    ])

    assert rc == 1
    assert not (out / "migrated.py").exists()
    gap = json.loads((out / "gap-request.json").read_text())
    assert gap["status"] == "gap_generation_required"
    assert gap["gaps"][0]["qualified_api"] == "numpy.exp"
    assert gap["gaps"][0]["current_provenance"] == "cpu_fallback"


def test_cli_qualified_missing_api_writes_canonical_gap_request_without_model(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "missing.py"
    source.write_text(MISSING_SOURCE)
    out = tmp_path / "out"
    _setup(monkeypatch, _MissingQualifiedBackend())

    def forbidden_model_access(*_args, **_kwargs):
        raise AssertionError("qualified GAP_REQUIRED must not load or call the model")

    monkeypatch.setattr(cli, "_load_llm_or_report", forbidden_model_access)
    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(forbidden_model_access)
    )

    rc = cli.main([
        "migrate", str(source), "--backend", "npu", "--out", str(out),
        "--target-soc", "ascend910b", "--cann-version", "9.0.0",
    ])

    assert rc == 1
    assert not (out / "migrated.py").exists()
    gap = json.loads((out / "gap-request.json").read_text())
    report = json.loads((out / "script-migration-report.json").read_text())
    assert gap["status"] == "gap_generation_required"
    assert report["status"] == "gap_generation_required"
    assert [item["qualified_api"] for item in gap["gap_specs"]] == [
        "numpy.definitely_missing"
    ]
    spec = gap["gap_specs"][0]
    assert spec["api_name"] == "definitely_missing"
    assert spec["asnumpy_api"] == "asnumpy.definitely_missing"
    assert spec["source_sha256"] == gap["source_sha256"]
    assert "[下一步]" in capsys.readouterr().out


def test_cli_script_mode_rejects_ast_engine_before_model_call(tmp_path, monkeypatch):
    source = tmp_path / "case.py"
    source.write_text(SOURCE)
    out = tmp_path / "out"
    _setup(monkeypatch, _Backend())

    rc = cli.main([
        "migrate", str(source), "--backend", "npu", "--out", str(out),
        "--engine", "ast",
    ])

    assert rc == 2
    assert not out.exists()


def test_cli_plain_script_defaults_to_real_npu_backend(tmp_path, monkeypatch):
    source = tmp_path / "case.py"
    source.write_text(SOURCE)
    out = tmp_path / "out"
    seen = []
    backend = _Backend()
    monkeypatch.delenv("NUMPILOT_BACKEND", raising=False)
    monkeypatch.setattr(
        cli,
        "_qualified_runtime_inputs",
        lambda _args, name: (seen.append(name) or backend, None),
    )
    monkeypatch.setattr(cli, "_preflight", lambda _backend, _skip: None)
    monkeypatch.setattr(
        cli,
        "_load_llm_or_report",
        lambda: types.SimpleNamespace(model="glm-5.3", provider="glm"),
    )
    raw = "```python\n" + MIGRATED + "```"
    monkeypatch.setattr(
        LlmEngine,
        "_chat_with_receipt",
        staticmethod(lambda _s, _u: (
            raw,
            _receipt(raw),
        )),
    )

    rc = cli.main(["migrate", str(source), "--out", str(out)])

    assert rc == 0
    assert seen == ["npu"]


class _GapSkillBundle:
    root = "/opt/cannbot-skills"
    skill_name = "ascendc-registry-invoke-template"
    commit = "a" * 40
    bundle_sha256 = "b" * 64
    files = {}

    def manifest(self):
        return {
            "schema_version": 1,
            "skill_name": self.skill_name,
            "commit": self.commit,
            "bundle_sha256": self.bundle_sha256,
            "files": {},
        }

    def prompt_context(self, selected_files=None):
        del selected_files
        return "## Embedded test Skill\n"


def test_cli_rejected_fallback_request_can_prepare_gap_workspace(
    tmp_path, monkeypatch
):
    source = tmp_path / "case.py"
    source.write_text(SOURCE)
    migration_out = tmp_path / "migration"
    completion_out = tmp_path / "completion"
    _setup(monkeypatch, _Backend(ExecutionProvenance.CPU_FALLBACK))

    migrate_rc = cli.main([
        "migrate", str(source), "--backend", "npu", "--out", str(migration_out),
        "--fallback-policy", "reject", "--target-soc", "ascend910b",
        "--cann-version", "9.0.0",
    ])
    assert migrate_rc == 1

    def forbidden_skill_access(*_args, **_kwargs):
        raise AssertionError("prepare must not read or fetch CANNBot")

    monkeypatch.setattr(cli, "resolve_skill_bundle", forbidden_skill_access)
    monkeypatch.setattr(cli, "_run_cannbot_skill_gate", forbidden_skill_access)
    complete_rc = cli.main([
        "complete-gaps", str(source),
        "--gap-request", str(migration_out / "gap-request.json"),
        "--out", str(completion_out),
        "--target-soc", "ascend910b", "--cann-version", "9.0.0",
        "--provider", "prepare",
    ])

    assert complete_rc == 0
    report = json.loads((completion_out / "gap-report.json").read_text())
    assert report["status"] == "prepared"
    assert [item["qualified_api"] for item in report["gap_specs"]] == [
        "numpy.exp"
    ]
    assert report["generated_count"] == 0


def test_invalid_gap_request_stops_before_llm_and_skill_access(
    tmp_path, monkeypatch
):
    source = tmp_path / "case.py"
    source.write_text(SOURCE)
    migration_out = tmp_path / "migration"
    _setup(monkeypatch, _Backend(ExecutionProvenance.CPU_FALLBACK))
    assert cli.main([
        "migrate", str(source), "--backend", "npu", "--out", str(migration_out),
        "--fallback-policy", "reject", "--target-soc", "ascend910b",
        "--cann-version", "9.0.0",
    ]) == 1
    request_path = migration_out / "gap-request.json"
    request = json.loads(request_path.read_text())
    request["generated_code"] = True
    request_path.write_text(json.dumps(request))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid local request crossed a remote/Skill boundary")

    monkeypatch.setattr(cli, "_run_llm_readiness_gate", forbidden)
    monkeypatch.setattr(cli, "resolve_skill_bundle", forbidden)
    rc = cli.main([
        "complete-gaps", str(source), "--gap-request", str(request_path),
        "--out", str(tmp_path / "must-not-exist"),
        "--target-soc", "ascend910b", "--cann-version", "9.0.0",
        "--provider", "llm", "--operator-goal", "accuracy",
    ])

    assert rc == 2
    assert not (tmp_path / "must-not-exist").exists()


def test_unknown_cann_with_no_gap_skips_readiness_and_skill(tmp_path, monkeypatch):
    source = tmp_path / "case.py"
    source.write_text(SOURCE)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unknown CANN crossed a remote/Skill boundary")

    monkeypatch.setattr(cli, "_run_llm_readiness_gate", forbidden)
    monkeypatch.setattr(cli, "resolve_skill_bundle", forbidden)
    rc = cli.main([
        "complete-gaps", str(source), "--out", str(tmp_path / "out"),
        "--provider", "llm", "--operator-goal", "accuracy",
    ])

    assert rc == 0
    assert (tmp_path / "out" / "gap-report.json").is_file()
