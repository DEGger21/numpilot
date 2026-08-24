import hashlib
import json
from pathlib import Path
import re
import shutil

import pytest

from numpilot.agents.analysis import AnalysisAgent
from numpilot.asnumpy_skill import ASNUMPY_INTEGRATION_SKILL_FILES
from numpilot.cannbot_skills import (
    CANNBOT_GENERATION_SKILL_FILES,
    CANNBOT_REPAIR_SKILL_FILES,
)
from numpilot.engines.llm import LlmEngine, LlmHttpError, LlmResponseError
from numpilot.gaps.models import GapSpec
from numpilot.gaps.pipeline import GapCompletionError, GapCompletionPipeline
from numpilot.gaps.provider import (
    DraftGenerationError,
    DraftRepairError,
    LlmDraftProvider,
    _MAX_ADAPTER_FILES,
    _MAX_ADAPTER_FILE_BYTES,
    _MAX_ADAPTER_TOTAL_BYTES,
    _MAX_ADAPTER_TREE_ENTRIES,
    _MAX_FILES,
    _MAX_METADATA_BYTES,
    _MAX_OPERATOR_TREE_ENTRIES,
    _MAX_TOTAL_BYTES,
    _adapter_artifact_id,
    _load_asnumpy_adapter_report,
    _load_existing_draft,
    _load_repair_candidate,
    _should_use_staged_direct,
)
from numpilot.gaps.spec import build_gap_specs
from numpilot.llm_config import LlmConfigStatus


GAP_SOURCE = """\
import numpy as np

def build_inputs():
    return (np.ones(8, dtype=np.float32),)

def forward(x):
    return np.cbrt(x)
"""


def _write_gap_source(tmp_path):
    path = tmp_path / "cbrt_case.py"
    path.write_text(GAP_SOURCE, encoding="utf-8")
    return path


def _valid_bundle():
    return json.dumps({
        "summary": "Unverified cbrt registry-invoke draft.",
        "files": [
            {"path": "CMakeLists.txt", "content": "cmake_minimum_required(VERSION 3.16)\n"},
            {"path": "build.sh", "content": "#!/usr/bin/env bash\nset -euo pipefail\n"},
            {"path": "op_host/cbrt_def.cpp", "content": "// host registration\n"},
            {"path": "op_kernel/cbrt.cpp", "content": "// __aicore__ kernel\n"},
            {"path": "op_api/aclnn_cbrt.cpp", "content": "// ACLNN wrapper\n"},
            {"path": "tests/test_cbrt.py", "content": "# NumPy golden test\n"},
            {
                "path": "asnumpy_adapter/operator.json",
                "content": '{"python_name":"cbrt"}\n',
            },
        ],
    })


def _valid_line_bundle():
    return """NUMPILOT_BUNDLE_V1
SUMMARY: Unverified cbrt registry-invoke draft.
FILE: CMakeLists.txt
CONTENT:
cmake_minimum_required(VERSION 3.16)
END_FILE
FILE: build.sh
CONTENT:
#!/usr/bin/env bash
set -euo pipefail
END_FILE
FILE: op_host/cbrt_def.cpp
CONTENT:
// host registration
END_FILE
FILE: op_kernel/cbrt.cpp
CONTENT:
// __aicore__ kernel with \"quotes\"
END_FILE
FILE: op_api/aclnn_cbrt.cpp
CONTENT:
// ACLNN wrapper
END_FILE
FILE: tests/test_cbrt.py
CONTENT:
# NumPy golden test
END_FILE
FILE: asnumpy_adapter/operator.json
CONTENT:
{\"python_name\":\"cbrt\"}
END_FILE
END_BUNDLE"""


def test_analysis_preserves_qualified_api_and_call_shape():
    source = """\
import numpy as np
def forward(x):
    return np.fft.fft(x, axis=-1)
"""
    result = AnalysisAgent().analyze(source)
    op = result.op_uses[0]

    assert op.name == "fft"
    assert op.qualified_name == "numpy.fft.fft"
    assert op.call_expr == "numpy.fft.fft(<arg0>, axis=<redacted>)"
    assert op.positional_args == 1
    assert op.keyword_args == ("axis",)


def test_nested_numpy_namespace_is_reported_but_not_generated():
    source = """\
import numpy as np
def forward(x):
    return np.fft.fft(x)
"""
    analysis = AnalysisAgent().analyze(source)

    assert build_gap_specs(
        source,
        analysis,
        target_soc="ascend910b",
        cann_version="9.0.0",
    ) == ()


def test_gap_spec_is_deterministic_and_records_callsites():
    analysis = AnalysisAgent().analyze(GAP_SOURCE, "case.py")
    first = build_gap_specs(
        GAP_SOURCE,
        analysis,
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    second = build_gap_specs(
        GAP_SOURCE,
        analysis,
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    assert len(first) == 1
    assert first[0].to_dict() == second[0].to_dict()
    assert first[0].qualified_api == "numpy.cbrt"
    assert first[0].asnumpy_api == "asnumpy.cbrt"
    assert first[0].callsites[0].lineno == 7
    assert first[0].signature_variants == ({
        "positional_args": 1,
        "keyword_args": [],
    },)
    restored = GapSpec.from_dict(first[0].to_dict())
    assert restored.to_dict() == first[0].to_dict()
    assert _should_use_staged_direct(first[0]) is True


def test_gap_schema_rejects_unknown_version():
    data = build_gap_specs(
        GAP_SOURCE,
        AnalysisAgent().analyze(GAP_SOURCE),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )[0].to_dict()
    data["schema_version"] = 2

    with pytest.raises(ValueError, match="unsupported gap schema"):
        GapSpec.from_dict(data)


def test_gap_schema_defaults_runtime_profile_requirements():
    data = build_gap_specs(
        GAP_SOURCE,
        AnalysisAgent().analyze(GAP_SOURCE),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )[0].to_dict()
    data.pop("shape_rules")
    data.pop("dtype_rules")

    restored = GapSpec.from_dict(data)

    assert restored.shape_rules == ("runtime_profile_required",)
    assert restored.dtype_rules == ("runtime_profile_required",)


def test_signature_gap_is_blocked_not_sent_to_operator_generation(tmp_path):
    source = tmp_path / "scalar_add.py"
    source.write_text(
        "import numpy as np\n"
        "def forward(x): return np.add(x, 1.0)\n",
        encoding="utf-8",
    )

    result = GapCompletionPipeline().run(
        str(source),
        str(tmp_path / "out"),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    assert result.status == "blocked"
    assert result.specs == ()
    assert result.blocked[0]["name"] == "add"
    assert "标量" in result.blocked[0]["reason"]


def test_prepare_creates_private_specs_without_executing_source(tmp_path):
    source = _write_gap_source(tmp_path)
    out = tmp_path / "out"

    result = GapCompletionPipeline(
        skill_root="/opt/cannbot-skills",
        asnumpy_reference="/opt/numpyexample",
    ).run(
        str(source),
        str(out),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    assert result.status == "prepared"
    assert result.generated_count == 0
    workspace = out / "gaps" / result.specs[0].gap_id
    spec = json.loads((workspace / "gap-spec.json").read_text(encoding="utf-8"))
    request = (workspace / "GENERATION_REQUEST.md").read_text(encoding="utf-8")
    report = json.loads((out / "gap-report.json").read_text(encoding="utf-8"))
    assert spec["qualified_api"] == "numpy.cbrt"
    assert "/opt/cannbot-skills/ops/ascendc-registry-invoke-template/SKILL.md" in request
    assert "/opt/numpyexample" in request
    assert report["qualified_count"] == 0
    assert not (workspace / "draft").exists()
    assert workspace.stat().st_mode & 0o777 == 0o700
    assert (workspace / "gap-spec.json").stat().st_mode & 0o777 == 0o600


def test_prepare_rejects_symlink_output_directory(tmp_path):
    source = _write_gap_source(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(GapCompletionError, match="not a real directory"):
        GapCompletionPipeline().run(
            str(source),
            str(linked),
            target_soc="ascend910b",
            cann_version="9.0.0",
        )


def test_llm_provider_writes_hashed_unverified_draft_idempotently(tmp_path):
    source = _write_gap_source(tmp_path)
    out = tmp_path / "out"
    calls = []

    prompts = []

    def chat(_system, user):
        calls.append(1)
        prompts.append(user)
        return _valid_bundle()

    provider = LlmDraftProvider(chat=chat, max_attempts=1)
    pipeline = GapCompletionPipeline(
        provider=provider,
        skill_root="/opt/cannbot-skills",
        asnumpy_reference="/opt/numpyexample",
    )
    first = pipeline.run(
        str(source),
        str(out),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    assert provider.action_for(first.drafts[0]) == "injected_test_double"
    second = pipeline.run(
        str(source),
        str(out),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    assert first.status == second.status == "generated_unverified"
    assert len(calls) == 1
    assert provider.action_for(second.drafts[0]) == "reused_existing"
    draft = Path(first.drafts[0].workspace)
    artifact = json.loads(
        (draft / "operator-artifact.json").read_text(encoding="utf-8")
    )
    assert artifact["status"] == "generated_unverified"
    assert artifact["gates"]["build"] == "pending"
    assert artifact["gates"]["activation"] == "pending"
    assert artifact["generation"]["mode"] == "injected_test_double"
    assert artifact["generation"]["origin_action"] == "injected_test_double"
    assert artifact["generation"]["remote_receipts"] == []
    assert artifact["generation"]["fallback"] == "disabled"
    assert artifact["generation"]["attempt_count"] == 1
    assert len(artifact["generation"]["response_sha256"][0]) == 64
    assert len(artifact["files"]) == 7
    assert (draft / "build.sh").stat().st_mode & 0o777 == 0o700
    assert "/opt/cannbot-skills" in prompts[0]
    assert "/opt/numpyexample" in prompts[0]
    assert "GetWorkspaceSize" in prompts[0]


def test_llm_provider_accepts_bounded_line_bundle_without_json_escaping(tmp_path):
    source = _write_gap_source(tmp_path)
    result = GapCompletionPipeline(provider=LlmDraftProvider(
        chat=lambda _system, _user: _valid_line_bundle(), max_attempts=1
    )).run(
        str(source),
        str(tmp_path / "line-bundle"),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    draft = Path(result.drafts[0].workspace)
    kernel = (draft / "op_kernel/cbrt.cpp").read_text(encoding="utf-8")
    assert 'with "quotes"' in kernel
    assert result.status == "generated_unverified"


def _configured_status(tmp_path, source="user-config"):
    return LlmConfigStatus(
        path=tmp_path / "llm.json",
        configured=True,
        missing=(),
        source=source,
        model="glm-audited",
        provider="glm",
        token_present=True,
    )


def test_provider_rejects_unimplemented_performance_goal_before_model_access():
    calls = []

    with pytest.raises(ValueError, match="performance generation is not implemented"):
        LlmDraftProvider(
            chat=lambda *_args: calls.append(1),
            operator_goal="performance",
        )

    assert calls == []


def test_complete_gaps_performance_goal_fails_before_network_skill_or_output(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli

    source = _write_gap_source(tmp_path)
    out = tmp_path / "must-not-exist"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unimplemented performance lane crossed a side effect")

    monkeypatch.setattr(cli, "_load_llm_or_report", forbidden)
    monkeypatch.setattr(cli, "_run_llm_readiness_gate", forbidden)
    monkeypatch.setattr(cli, "resolve_skill_bundle", forbidden)
    monkeypatch.setattr(cli, "_write_generation_preflight", forbidden)

    rc = cli.main([
        "complete-gaps",
        str(source),
        "--out", str(out),
        "--provider", "llm",
        "--operator-goal", "performance",
    ])

    assert rc == 2
    assert not out.exists()
    terminal = capsys.readouterr()
    assert "performance 尚未开放" in terminal.err


class _FakeSkillBundle:
    root = "/opt/cannbot-skills"
    skill_name = "ascendc-registry-invoke-template"
    commit = "a" * 40
    bundle_sha256 = "b" * 64
    files = {
        path: "fixture for %s" % path
        for path in dict.fromkeys(
            (
                *CANNBOT_GENERATION_SKILL_FILES,
                *CANNBOT_REPAIR_SKILL_FILES,
                *ASNUMPY_INTEGRATION_SKILL_FILES,
            )
        )
    }
    total_bytes = sum(len(value.encode("utf-8")) for value in files.values())

    def prompt_context(self, selected_files=None):
        selected = selected_files or tuple(self.files)
        return (
            "## Embedded test Skill\n"
            + self.bundle_sha256
            + "\n"
            + "\n".join(
                self.files.get(path, "fixture for %s" % path)
                for path in selected
            )
        )

    def manifest(self):
        return {
            "schema_version": 1,
            "repository": "https://gitcode.com/cann/cannbot-skills.git",
            "commit": self.commit,
            "skill_name": self.skill_name,
            "license": "test fixture",
            "bundle_sha256": self.bundle_sha256,
            "total_bytes": self.total_bytes,
            "files": {
                path: hashlib.sha256(content.encode("utf-8")).hexdigest()
                for path, content in self.files.items()
            },
        }


def _remote_receipt(content):
    content_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "utc_start": "2026-08-12T12:00:00Z",
        "utc_end": "2026-08-12T12:00:01Z",
        "http_status": 200,
        "requested_model": "glm-audited",
        "response_model": "glm-audited",
        "response_id": "response-123",
        "response_created": 1786555200,
        "request_body_sha256": "1" * 64,
        "response_body_sha256": "2" * 64,
        "content_sha256": content_digest,
        "request_id": "request-456",
        "usage": {"total_tokens": 101},
        "tls_verification": "enabled",
    }


def _write_adapter_base(tmp_path):
    root = tmp_path / "adapter-base"
    candidate = root / "candidate"
    manifest = candidate / "csrc/custom_ops/registry/cbrt/op.yaml"
    test_file = candidate / "tests/asnumpy_tests/custom_ops_tests/test_cbrt.py"
    manifest.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    manifest.write_text("opp_vendor: custom_math\n", encoding="utf-8")
    test_file.write_text("def test_cbrt():\n    assert True\n", encoding="utf-8")
    hashes = {
        path.relative_to(candidate).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in (manifest, test_file)
    }
    response = "audited adapter fixture response"
    report = {
        "schema_version": 2,
        "status": "generated_unverified",
        "source_artifact_id": "draft-fixture",
        "model": "glm-audited",
        "provider": "glm",
        "configuration_source": "user-config",
        "attempt_count": 1,
        "files_sha256": hashes,
        "adapter_contract_sha256": "c" * 64,
        "asnumpy_context_sha256": {"CMakeLists.txt": "d" * 64},
        "skill_bundle_sha256": _FakeSkillBundle.bundle_sha256,
        "integration_skill_files": list(ASNUMPY_INTEGRATION_SKILL_FILES),
        "response_sha256": [
            hashlib.sha256(response.encode("utf-8")).hexdigest()
        ],
        "remote_receipts": [_remote_receipt(response)],
        "actual_remote_call": True,
        "gates": {
            "spec": "prepared",
            "generation": "passed",
            "build": "pending",
            "unit_test": "pending",
            "numerical_validation": "pending",
            "activation": "pending",
        },
    }
    report["adapter_artifact_id"] = _adapter_artifact_id(report)
    (root / "asnumpy-adapter-report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return root


def test_schema_one_generated_adapter_report_remains_reloadable(tmp_path):
    base = _write_adapter_base(tmp_path)
    report_path = base / "asnumpy-adapter-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema_version"] = 1
    report["adapter_artifact_id"] = _adapter_artifact_id(report)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    reloaded, _payload, hashes = _load_asnumpy_adapter_report(base)
    assert reloaded["schema_version"] == 1
    assert hashes == report["files_sha256"]


def test_schema_one_repair_report_fails_closed_with_migration_guidance(tmp_path):
    base = _write_adapter_base(tmp_path)
    report_path = base / "asnumpy-adapter-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema_version"] = 1
    report["status"] = "repaired_unverified"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(
        DraftGenerationError,
        match="legacy.*cannot prove.*regenerate.*fail-closed",
    ):
        _load_asnumpy_adapter_report(base)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("file_count", "file-count limit"),
        ("entry_count", "entry limit"),
        ("single_size", "file exceeds the size limit"),
        ("total_size", "total-size limit"),
        ("report_size", "report exceeds the size limit"),
    ),
)
def test_asnumpy_adapter_base_enforces_bounded_candidate_tree(
    tmp_path, mutation, message
):
    base = _write_adapter_base(tmp_path)
    candidate = base / "candidate"
    report_path = base / "asnumpy-adapter-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if mutation == "file_count":
        extra_count = _MAX_ADAPTER_FILES - len(report["files_sha256"]) + 1
        for index in range(extra_count):
            path = candidate / ("docs/extra-%02d.md" % index)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("extra\n", encoding="utf-8")
    elif mutation == "entry_count":
        for index in range(_MAX_ADAPTER_TREE_ENTRIES):
            (candidate / ("padding/dir-%03d" % index)).mkdir(
                parents=True, exist_ok=True
            )
    elif mutation == "single_size":
        path = candidate / "csrc/custom_ops/registry/cbrt/op.yaml"
        with open(path, "wb") as handle:
            handle.truncate(_MAX_ADAPTER_FILE_BYTES + 1)
    elif mutation == "total_size":
        payload = b"x" * _MAX_ADAPTER_FILE_BYTES
        extra_count = _MAX_ADAPTER_TOTAL_BYTES // _MAX_ADAPTER_FILE_BYTES
        for index in range(extra_count):
            path = candidate / ("docs/large-%02d.md" % index)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            relative = path.relative_to(candidate).as_posix()
            report["files_sha256"][relative] = hashlib.sha256(payload).hexdigest()
        report["adapter_artifact_id"] = _adapter_artifact_id(report)
        report_path.write_text(json.dumps(report), encoding="utf-8")
    else:
        with open(report_path, "wb") as handle:
            handle.truncate(_MAX_METADATA_BYTES + 1)

    with pytest.raises(DraftGenerationError, match=message):
        _load_asnumpy_adapter_report(base)


def test_generated_asnumpy_adapter_is_schema_two_and_reloadable(
    tmp_path, monkeypatch
):
    def remote_chat(_system, user, **_kwargs):
        path = re.search(r"Return exactly `([^`]+)`", user).group(1)
        response = "\n".join((
            "NUMPILOT_BUNDLE_V1",
            "SUMMARY: generated adapter fixture",
            "FILE: " + path,
            "CONTENT:",
            "{}" if path.endswith(".yaml") else "def test_adapter():\n    pass",
            "END_FILE",
            "END_BUNDLE",
        ))
        return response, _remote_receipt(response)

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(remote_chat)
    )
    out = tmp_path / "generated-adapter"
    out.mkdir()
    provider = LlmDraftProvider(
        max_attempts=1,
        configuration_status=_configured_status(tmp_path),
        skill_bundle=_FakeSkillBundle(),
        staged_direct=True,
    )
    report = provider.generate_asnumpy_adapter_candidate(
        out,
        operator_artifact={"artifact_id": "draft-fixture"},
        adapter_contract='{"api_name":"cbrt"}',
        asnumpy_context={"CMakeLists.txt": "project(asnumpy)\n"},
        expected_paths=(
            "csrc/custom_ops/registry/cbrt/op.yaml",
            "tests/asnumpy_tests/custom_ops_tests/test_cbrt.py",
        ),
    )

    assert report["schema_version"] == 2
    assert report["integration_skill_files"] == list(
        ASNUMPY_INTEGRATION_SKILL_FILES
    )
    assert report["adapter_artifact_id"] == _adapter_artifact_id(report)
    reloaded, _payload, hashes = _load_asnumpy_adapter_report(out)
    assert reloaded == report
    assert hashes == report["files_sha256"]


@pytest.mark.parametrize("repair_profile", ("compile", "precision", "runtime"))
def test_remote_model_repairs_hash_bound_asnumpy_adapter_from_raw_log(
    tmp_path, monkeypatch, repair_profile
):
    base = _write_adapter_base(tmp_path)
    prompts = []

    def remote_chat(system, user, **_kwargs):
        prompts.append((system, user))
        response = """NUMPILOT_BUNDLE_V1
SUMMARY: use installed private OPP vendor
FILE: csrc/custom_ops/registry/cbrt/op.yaml
CONTENT:
opp_vendor: cbrt_custom
END_FILE
END_BUNDLE"""
        return response, _remote_receipt(response)

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(remote_chat)
    )
    provider = LlmDraftProvider(
        max_attempts=1,
        configuration_status=_configured_status(tmp_path),
        skill_bundle=_FakeSkillBundle(),
        staged_direct=True,
    )
    out = tmp_path / "adapter-repair"
    out.mkdir()
    diagnostic = (
        "CMake Error: ASNUMPY_CUSTOM_OPS=ON but these vendors are missing: "
        "custom_math; installed vendor cbrt_custom"
    )
    report = provider.repair_asnumpy_adapter_candidate(
        out,
        base_adapter_root=base,
        diagnostic=diagnostic,
        path="csrc/custom_ops/registry/cbrt/op.yaml",
        asnumpy_context={
            "CMakeLists.txt": "include(cmake/CustomOps.cmake)\n",
            "cmake/CustomOps.cmake": "message(FATAL_ERROR missing vendor)\n",
        },
        repair_profile=repair_profile,
    )

    assert len(prompts) == 1
    assert diagnostic in prompts[0][1]
    assert "opp_vendor: custom_math" in prompts[0][1]
    assert "fixture for" in prompts[0][1]
    assert (
        "Repair qualification profile (audit binding): `%s`."
        % repair_profile
    ) in prompts[0][1]
    assert (base / "candidate/csrc/custom_ops/registry/cbrt/op.yaml").read_text(
        encoding="utf-8"
    ) == "opp_vendor: custom_math\n"
    assert (out / "candidate/csrc/custom_ops/registry/cbrt/op.yaml").read_text(
        encoding="utf-8"
    ) == "opp_vendor: cbrt_custom"
    assert report["status"] == "repaired_unverified"
    assert report["repair_profile"] == repair_profile
    assert report["actual_remote_call"] is True
    assert report["integration_skill_files"] == list(
        ASNUMPY_INTEGRATION_SKILL_FILES
    )
    assert report["diagnostic_sha256"] == hashlib.sha256(
        diagnostic.encode("utf-8")
    ).hexdigest()
    assert report["before_sha256"] != report["after_sha256"]
    assert len(report["remote_receipts"]) == 1
    base_report = json.loads(
        (base / "asnumpy-adapter-report.json").read_text(encoding="utf-8")
    )
    assert report["adapter_artifact_id"] != base_report["adapter_artifact_id"]
    assert report["parent_adapter_artifact_id"] == (
        base_report["adapter_artifact_id"]
    )
    assert report["adapter_artifact_id"] == _adapter_artifact_id(report)
    reloaded, _payload, reloaded_hashes = _load_asnumpy_adapter_report(out)
    assert reloaded["adapter_artifact_id"] == report["adapter_artifact_id"]
    assert reloaded_hashes == report["files_sha256"]


@pytest.mark.parametrize(
    "mutation",
    ("schema", "artifact_id", "receipt", "content", "skill", "context"),
)
def test_asnumpy_adapter_base_revalidates_all_audit_bindings(
    tmp_path, monkeypatch, mutation
):
    base = _write_adapter_base(tmp_path)
    report_path = base / "asnumpy-adapter-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if mutation == "schema":
        report["unexpected"] = True
    elif mutation == "artifact_id":
        report["adapter_artifact_id"] = "adapter-" + "0" * 20
    elif mutation == "receipt":
        report["remote_receipts"][0]["requested_model"] = "other-model"
        report["adapter_artifact_id"] = _adapter_artifact_id(report)
    elif mutation == "content":
        (
            base / "candidate/csrc/custom_ops/registry/cbrt/op.yaml"
        ).write_text("tampered\n", encoding="utf-8")
    elif mutation == "skill":
        report["skill_bundle_sha256"] = "e" * 64
        report["adapter_artifact_id"] = _adapter_artifact_id(report)
    else:
        report["asnumpy_context_sha256"]["CMakeLists.txt"] = "not-a-hash"
        report["adapter_artifact_id"] = _adapter_artifact_id(report)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(
        LlmEngine,
        "_chat_with_receipt",
        staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid base must be rejected before model access")
        )),
    )
    provider = LlmDraftProvider(
        max_attempts=1,
        configuration_status=_configured_status(tmp_path),
        skill_bundle=_FakeSkillBundle(),
        staged_direct=True,
    )
    out = tmp_path / "invalid-base-output"
    out.mkdir()
    with pytest.raises(DraftGenerationError) as caught:
        provider.repair_asnumpy_adapter_candidate(
            out,
            base_adapter_root=base,
            diagnostic="raw compiler diagnostic",
            path="csrc/custom_ops/registry/cbrt/op.yaml",
            asnumpy_context={"CMakeLists.txt": "project(asnumpy)\n"},
            skill_files=CANNBOT_REPAIR_SKILL_FILES,
        )
    assert not isinstance(caught.value, AssertionError)
    assert not (out / "candidate").exists()


def test_asnumpy_adapter_repair_revalidates_exact_parent_report_bytes(
    tmp_path, monkeypatch
):
    base = _write_adapter_base(tmp_path)

    def remote_chat(_system, _user, **_kwargs):
        response = """NUMPILOT_BUNDLE_V1
SUMMARY: parent-chain fixture
FILE: csrc/custom_ops/registry/cbrt/op.yaml
CONTENT:
opp_vendor: repaired
END_FILE
END_BUNDLE"""
        return response, _remote_receipt(response)

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(remote_chat)
    )
    provider = LlmDraftProvider(
        max_attempts=1,
        configuration_status=_configured_status(tmp_path),
        skill_bundle=_FakeSkillBundle(),
        staged_direct=True,
    )
    child = tmp_path / "adapter-child"
    child.mkdir()
    provider.repair_asnumpy_adapter_candidate(
        child,
        base_adapter_root=base,
        diagnostic="raw compiler diagnostic",
        path="csrc/custom_ops/registry/cbrt/op.yaml",
        asnumpy_context={"CMakeLists.txt": "project(asnumpy)\n"},
        skill_files=CANNBOT_REPAIR_SKILL_FILES,
    )
    parent_report_path = base / "asnumpy-adapter-report.json"
    parent_report_path.write_bytes(parent_report_path.read_bytes() + b"\n")

    with pytest.raises(
        DraftGenerationError,
        match="parent report hash|derive from its parent",
    ):
        _load_asnumpy_adapter_report(child)


def _write_simple_signed_draft(tmp_path, monkeypatch=None):
    source = _write_gap_source(tmp_path)
    if monkeypatch is None:
        provider = LlmDraftProvider(
            chat=lambda _system, _user: _valid_bundle(),
            max_attempts=1,
        )
    else:
        response = _valid_bundle()
        monkeypatch.setattr(
            LlmEngine,
            "_chat_with_receipt",
            staticmethod(
                lambda _system, _user, **_kwargs: (
                    response,
                    _remote_receipt(response),
                )
            ),
        )
        provider = LlmDraftProvider(
            max_attempts=1,
            configuration_status=_configured_status(tmp_path),
        )
    result = GapCompletionPipeline(
        provider=provider
    ).run(
        str(source),
        str(tmp_path / "simple-draft-output"),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    draft_path = Path(result.drafts[0].workspace)
    signed = _load_existing_draft(draft_path, result.specs[0])
    return result.specs[0], draft_path, signed


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("manifest_size", "manifest exceeds the size limit"),
        ("manifest_file_count", "file manifest is invalid"),
        ("entry_count", "entry limit"),
        ("single_size", "file exceeds the size limit"),
        ("total_size", "total-size limit"),
    ),
)
def test_existing_draft_loader_bounds_metadata_and_tree_before_hashing(
    tmp_path, mutation, message
):
    spec, draft_path, _signed = _write_simple_signed_draft(tmp_path)
    manifest_path = draft_path / "operator-artifact.json"
    if mutation == "manifest_size":
        with open(manifest_path, "wb") as handle:
            handle.truncate(_MAX_METADATA_BYTES + 1)
    elif mutation == "manifest_file_count":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for index in range(_MAX_FILES + 1):
            manifest["files"]["docs/fake-%03d.md" % index] = "f" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "entry_count":
        for index in range(_MAX_OPERATOR_TREE_ENTRIES):
            (draft_path / ("padding/dir-%03d" % index)).mkdir(
                parents=True, exist_ok=True
            )
    elif mutation == "single_size":
        with open(draft_path / "CMakeLists.txt", "wb") as handle:
            handle.truncate(_MAX_TOTAL_BYTES + 1)
    else:
        with open(draft_path / "CMakeLists.txt", "wb") as handle:
            handle.truncate(_MAX_TOTAL_BYTES)

    with pytest.raises(DraftGenerationError, match=message):
        _load_existing_draft(draft_path, spec)


def _write_operator_repair_base(tmp_path, monkeypatch=None):
    spec, draft_path, signed = _write_simple_signed_draft(tmp_path, monkeypatch)
    candidate = tmp_path / "operator-repair-base"
    shutil.copytree(draft_path, candidate)
    response = "audited operator repair response"
    receipt = _remote_receipt(response)
    receipt["requested_model"] = signed.generation["model"]
    receipt["response_model"] = signed.generation["model"]
    path = "op_host/cbrt_def.cpp"
    report = {
        "schema_version": 1,
        "status": "repaired_unverified",
        "source_artifact_id": signed.artifact_id,
        "source_workspace": str(draft_path),
        "signed_source_workspace": signed.workspace,
        "parent_repair_report_sha256": None,
        "gap_id": spec.gap_id,
        "model": signed.generation["model"],
        "provider": signed.generation["provider"],
        "configuration_source": signed.generation["configuration_source"],
        "attempt_count": 1,
        "paths": [path],
        "before_sha256": {path: signed.files[path]},
        "after_sha256": {path: signed.files[path]},
        "candidate_files_sha256": dict(signed.files),
        "diagnostic_sha256": "3" * 64,
        "generation_request_sha256": "4" * 64,
        "skill_bundle_sha256": _FakeSkillBundle.bundle_sha256,
        "cannbot_repair_skill_files": list(CANNBOT_REPAIR_SKILL_FILES),
        "response_sha256": [
            hashlib.sha256(response.encode("utf-8")).hexdigest()
        ],
        "remote_receipts": [receipt],
        "actual_remote_call": True,
        "gates": {
            "spec": "prepared",
            "generation": "passed",
            "build": "pending",
            "unit_test": "pending",
            "numerical_validation": "pending",
            "activation": "pending",
        },
    }
    report_path = candidate / "numpilot-repair-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return spec, signed, candidate, report, report_path


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("report_size", "report exceeds the size limit"),
        ("file_map_count", "complete file map is invalid"),
        ("entry_count", "entry limit"),
        ("single_size", "file exceeds the size limit"),
        ("total_size", "total-size limit"),
    ),
)
def test_operator_repair_loader_bounds_report_and_candidate_tree(
    tmp_path, mutation, message
):
    spec, signed, candidate, report, report_path = _write_operator_repair_base(
        tmp_path
    )
    if mutation == "report_size":
        with open(report_path, "wb") as handle:
            handle.truncate(_MAX_METADATA_BYTES + 1)
    elif mutation == "file_map_count":
        for index in range(_MAX_FILES + 1):
            report["candidate_files_sha256"]["docs/fake-%03d.md" % index] = (
                "f" * 64
            )
        report_path.write_text(json.dumps(report), encoding="utf-8")
    elif mutation == "entry_count":
        for index in range(_MAX_OPERATOR_TREE_ENTRIES):
            (candidate / ("padding/dir-%03d" % index)).mkdir(
                parents=True, exist_ok=True
            )
    elif mutation == "single_size":
        with open(candidate / "CMakeLists.txt", "wb") as handle:
            handle.truncate(_MAX_TOTAL_BYTES + 1)
    else:
        with open(candidate / "CMakeLists.txt", "wb") as handle:
            handle.truncate(_MAX_TOTAL_BYTES)

    with pytest.raises(DraftGenerationError, match=message):
        _load_repair_candidate(candidate, spec, signed, _FakeSkillBundle())


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_field",
        "signed_workspace",
        "diagnostic_hash",
        "request_hash",
        "skill_hash",
        "skill_files",
    ),
)
def test_operator_repair_loader_revalidates_exact_audit_bindings(
    tmp_path, mutation
):
    spec, signed, candidate, report, report_path = _write_operator_repair_base(
        tmp_path
    )
    if mutation == "extra_field":
        report["unexpected"] = True
    elif mutation == "signed_workspace":
        report["signed_source_workspace"] = "/different/signed/source"
    elif mutation == "diagnostic_hash":
        report["diagnostic_sha256"] = "not-a-hash"
    elif mutation == "request_hash":
        report["generation_request_sha256"] = "not-a-hash"
    elif mutation == "skill_hash":
        report["skill_bundle_sha256"] = "e" * 64
    else:
        report["cannbot_repair_skill_files"] = ["missing/skill-file.md"]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(DraftGenerationError):
        _load_repair_candidate(candidate, spec, signed, _FakeSkillBundle())


def test_operator_repair_rejects_duplicate_normalized_paths_before_model(
    tmp_path
):
    spec, draft_path, _signed = _write_simple_signed_draft(tmp_path)
    calls = []
    provider = LlmDraftProvider(
        chat=lambda *_args: calls.append(1),
        max_attempts=1,
        skill_bundle=_FakeSkillBundle(),
    )

    with pytest.raises(DraftGenerationError, match="unique normalized"):
        provider.repair_to_candidate(
            spec,
            draft_path,
            tmp_path / "unpublishable-repair",
            paths=("op_host//cbrt_def.cpp", "op_host/cbrt_def.cpp"),
            diagnostic="local compiler failure",
            generation_request="bounded request",
        )

    assert calls == []


def test_seal_repair_cli_verifies_fixed_skill_offline_without_model(
    tmp_path, monkeypatch
):
    from numpilot import __main__ as cli

    spec, signed, candidate, _report, _report_path = _write_operator_repair_base(
        tmp_path, monkeypatch
    )
    spec_path = tmp_path / "seal-gap-spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    resolutions = []

    def fixed_bundle(skill_root, *, allow_fetch):
        resolutions.append((skill_root, allow_fetch))
        return _FakeSkillBundle()

    monkeypatch.setattr(cli, "resolve_skill_bundle", fixed_bundle)
    monkeypatch.setattr(
        LlmEngine,
        "_chat_with_receipt",
        staticmethod(
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("seal-repair must not call a model")
            )
        ),
    )
    out = tmp_path / "sealed-by-cli"

    result = cli.main([
        "seal-repair",
        "--spec", str(spec_path),
        "--draft", signed.workspace,
        "--candidate", str(candidate),
        "--out", str(out),
        "--skill-root", "/fixed/offline/cannbot",
        "--offline-skills",
    ])

    assert result == 0
    assert resolutions == [("/fixed/offline/cannbot", False)]
    sealed = _load_existing_draft(out, spec)
    assert sealed.generation["attempt_count"] == 2
    assert "docs/numpilot-repair-provenance.json" in sealed.files


def test_remote_domain_operator_generation_uses_bounded_per_file_stages(
    tmp_path, monkeypatch
):
    source = tmp_path / "direct_domain.py"
    source.write_text(
        '''\
import numpy as np
from numpilot import operator_spec

@operator_spec(
    name="scale_delta",
    version="1",
    numpy_oracle="scale_delta_oracle",
    target_api="asnumpy.scale_delta",
    provenance="custom_generation_candidate",
    route="ascendc_registry_candidate",
    signature_variants=({"positional_args": 2, "keyword_args": ()},),
    dtype_rules=("float32",),
    shape_rules=("rank>=1", "contiguous"),
    source="numpilot://local-llm-generation/operator/scale_delta",
    license="CANN-Open-Software-License-Agreement-2.0",
    target_soc="ascend910b",
    cann_version="9.0.0",
)
def scale_delta_oracle(x, delta):
    return np.multiply(x, delta)

def forward(x, delta):
    return scale_delta_oracle(x, delta)
''',
        encoding="utf-8",
    )
    prompts = []

    def remote_chat(system, user, **_kwargs):
        prompts.append(user)
        if "repair exactly one file" in system:
            path = re.search(r"Return exactly `([^`]+)`", user).group(1)
            response = "\n".join((
                "NUMPILOT_BUNDLE_V1",
                "SUMMARY: compiler-directed repair fixture",
                "FILE: " + path,
                "CONTENT:",
                "// repaired from compiler diagnostic",
                "END_FILE",
                "END_BUNDLE",
            ))
            return response, _remote_receipt(response)
        path_section = user.split(
            "Return exactly these paths and no others:\n", 1
        )[1].split("\n\nEvery file", 1)[0]
        paths = re.findall(r"^- `([^`]+)`$", path_section, flags=re.MULTILINE)
        parts = ["NUMPILOT_BUNDLE_V1", "SUMMARY: complete stage fixture"]
        for path in paths:
            content = "{}" if path.endswith(".json") else "// complete fixture"
            if path.endswith(".sh"):
                content = "#!/usr/bin/env bash\nset -euo pipefail"
            elif path.endswith(".py"):
                content = "def test_generated_contract():\n    assert True"
            parts.extend((
                "FILE: " + path,
                "CONTENT:",
                content,
                "END_FILE",
            ))
        parts.append("END_BUNDLE")
        response = "\n".join(parts)
        return response, _remote_receipt(response)

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(remote_chat)
    )
    bundle = _FakeSkillBundle()
    provider = LlmDraftProvider(
        max_attempts=1,
        configuration_status=_configured_status(tmp_path),
        skill_bundle=bundle,
        staged_direct=True,
    )

    result = GapCompletionPipeline(
        provider=provider,
        skill_bundle=bundle,
    ).run(
        str(source),
        str(tmp_path / "staged"),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    draft = result.drafts[0]
    assert len(prompts) == 17
    assert len(draft.files) == 17
    assert draft.generation["attempt_count"] == 17
    assert draft.generation["transport_attempt_count"] == 17
    assert draft.generation["successful_http_response_count"] == 17
    assert len(draft.generation["remote_receipts"]) == 17
    assert all("return np.multiply(x, delta)" in prompt for prompt in prompts)
    assert all("external operator implementation repository" in prompt for prompt in prompts)
    assert all("Project-wide invariants (mandatory)" in prompt for prompt in prompts)
    assert all("Admitted dtypes are exactly: float32" in prompt for prompt in prompts)
    test_prompt = next(prompt for prompt in prompts if "tests/test_scale_delta.py" in prompt)
    assert "Do not import `asnumpy` at module scope" in test_prompt

    draft_path = Path(draft.workspace)
    original = (
        draft_path / "op_host/scale_delta_def.cpp"
    ).read_text(encoding="utf-8")
    candidate = tmp_path / "repair-candidate"
    repair = provider.repair_to_candidate(
        result.specs[0],
        draft_path,
        candidate,
        paths=("op_host/scale_delta_def.cpp",),
        diagnostic="CANN compiler: generated Host API does not match CANN 9.0",
        generation_request=(
            Path(result.out_dir)
            / "gaps"
            / result.specs[0].gap_id
            / "GENERATION_REQUEST.md"
        ).read_text(encoding="utf-8"),
    )
    assert repair["actual_remote_call"] is True
    assert repair["source_artifact_id"] == draft.artifact_id
    assert repair["attempt_count"] == 1
    assert len(repair["remote_receipts"]) == 1
    assert (
        candidate / "op_host/scale_delta_def.cpp"
    ).read_text(encoding="utf-8") == "// repaired from compiler diagnostic"
    assert json.loads(
        (candidate / "numpilot-repair-report.json").read_text(encoding="utf-8")
    )["status"] == "repaired_unverified"
    assert (
        draft_path / "op_host/scale_delta_def.cpp"
    ).read_text(encoding="utf-8") == original

    first_report_payload = (
        candidate / "numpilot-repair-report.json"
    ).read_bytes()
    chained = tmp_path / "repair-candidate-2"
    chained_repair = provider.repair_to_candidate(
        result.specs[0],
        draft_path,
        chained,
        paths=("op_host/scale_delta_infershape.cpp",),
        diagnostic="CANN compiler: infer-shape API needs a second repair",
        generation_request=(
            Path(result.out_dir)
            / "gaps"
            / result.specs[0].gap_id
            / "GENERATION_REQUEST.md"
        ).read_text(encoding="utf-8"),
        base_candidate_dir=candidate,
    )
    assert chained_repair["attempt_count"] == 1
    assert chained_repair["parent_repair_report_sha256"] == hashlib.sha256(
        first_report_payload
    ).hexdigest()
    assert chained_repair["source_workspace"] == str(candidate)
    assert (
        chained / "op_host/scale_delta_def.cpp"
    ).read_text(encoding="utf-8") == "// repaired from compiler diagnostic"
    assert (
        chained / "op_host/scale_delta_infershape.cpp"
    ).read_text(encoding="utf-8") == "// repaired from compiler diagnostic"
    assert set(chained_repair["candidate_files_sha256"]) == set(draft.files)

    sealed_dir = tmp_path / "sealed-repair-artifact"
    sealed = provider.seal_repair_candidate(
        result.specs[0],
        draft_path,
        chained,
        sealed_dir,
    )
    assert sealed.status == "generated_unverified"
    assert sealed.artifact_id != draft.artifact_id
    assert len(sealed.files) == len(draft.files) + 1
    assert provider.action_for(sealed) == "sealed_repair_chain"
    assert sealed.generation["attempt_count"] == 19
    assert len(sealed.generation["remote_receipts"]) == 19
    sealed_manifest = json.loads(
        (sealed_dir / "operator-artifact.json").read_text(encoding="utf-8")
    )
    assert sealed_manifest["workspace"] == str(sealed_dir)
    provenance = json.loads(
        (sealed_dir / "docs/numpilot-repair-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["source_artifact_id"] == draft.artifact_id
    assert len(provenance["repair_reports"]) == 2
    assert provenance["final_candidate_files_sha256"] == (
        chained_repair["candidate_files_sha256"]
    )

    reseal_candidate = tmp_path / "repair-after-seal"
    reseal_repair = provider.repair_to_candidate(
        result.specs[0],
        sealed_dir,
        reseal_candidate,
        paths=("op_kernel/scale_delta_arch22.cpp",),
        diagnostic="CANN runtime: sealed artifact needs a kernel repair",
        generation_request=(
            Path(result.out_dir)
            / "gaps"
            / result.specs[0].gap_id
            / "GENERATION_REQUEST.md"
        ).read_text(encoding="utf-8"),
    )
    resealed_dir = tmp_path / "resealed-repair-artifact"
    resealed = provider.seal_repair_candidate(
        result.specs[0],
        sealed_dir,
        reseal_candidate,
        resealed_dir,
    )
    assert len(resealed.files) == len(sealed.files)
    assert resealed.generation["attempt_count"] == 20
    assert resealed.files["docs/numpilot-repair-provenance.json"] != (
        sealed.files["docs/numpilot-repair-provenance.json"]
    )
    resealed_provenance = json.loads(
        (resealed_dir / "docs/numpilot-repair-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert resealed_provenance["source_artifact_id"] == sealed.artifact_id
    assert len(resealed_provenance["repair_reports"]) == 1
    assert resealed_provenance["final_candidate_files_sha256"] == (
        reseal_repair["candidate_files_sha256"]
    )

    invalid_receipt = _remote_receipt("invalid-repair-response")
    invalid_receipt["content_sha256"] = None

    def invalid_repair_response(_system, _user, **_kwargs):
        raise LlmResponseError("missing_content", invalid_receipt)

    monkeypatch.setattr(
        LlmEngine,
        "_chat_with_receipt",
        staticmethod(invalid_repair_response),
    )
    failing_provider = LlmDraftProvider(
        max_attempts=1,
        configuration_status=_configured_status(tmp_path),
        skill_bundle=bundle,
        staged_direct=True,
    )
    failed_candidate = tmp_path / "failed-repair-candidate"
    with pytest.raises(DraftRepairError) as caught:
        failing_provider.repair_to_candidate(
            result.specs[0],
            draft_path,
            failed_candidate,
            paths=("op_host/scale_delta_def.cpp",),
            diagnostic="CANN compiler rejected the response body",
            generation_request=(
                Path(result.out_dir)
                / "gaps"
                / result.specs[0].gap_id
                / "GENERATION_REQUEST.md"
            ).read_text(encoding="utf-8"),
        )
    assert caught.value.actual_remote_call is True
    assert caught.value.receipt_count == 1
    assert not failed_candidate.exists()


@pytest.mark.parametrize("expected_attempts", (49, 50, 51))
def test_staged_direct_attempt_limit_is_publish_and_reload_consistent(
    tmp_path, monkeypatch, expected_attempts
):
    source = _write_gap_source(tmp_path)
    extra_failures = expected_attempts - 17
    stage_order = {}
    per_stage_calls = {}
    repair_calls = {}

    def remote_chat(system, user, **_kwargs):
        if "repair exactly one file" in system.lower():
            path = re.search(r"Return exactly `([^`]+)`", user).group(1)
            repair_calls[path] = repair_calls.get(path, 0) + 1
            if repair_calls[path] <= 2:
                response = "locally rejected repair response"
                return response, _remote_receipt(response)
            response = "\n".join((
                "NUMPILOT_BUNDLE_V1",
                "SUMMARY: post-limit repair fixture",
                "FILE: " + path,
                "CONTENT:",
                "// repaired after maximum fresh generation",
                "END_FILE",
                "END_BUNDLE",
            ))
            return response, _remote_receipt(response)
        path_section = user.split(
            "Return exactly these paths and no others:\n", 1
        )[1].split("\n\nEvery file", 1)[0]
        path = re.findall(r"^- `([^`]+)`$", path_section, flags=re.MULTILINE)[0]
        stage_index = stage_order.setdefault(path, len(stage_order))
        per_stage_calls[path] = per_stage_calls.get(path, 0) + 1
        failures_for_stage = min(2, max(0, extra_failures - stage_index * 2))
        if per_stage_calls[path] <= failures_for_stage:
            response = "locally rejected response"
        else:
            content = "{}" if path.endswith(".json") else "// complete fixture"
            if path.endswith(".sh"):
                content = "#!/usr/bin/env bash\nset -euo pipefail"
            elif path.endswith(".py"):
                content = "def test_generated_contract():\n    assert True"
            response = "\n".join((
                "NUMPILOT_BUNDLE_V1",
                "SUMMARY: bounded retry fixture",
                "FILE: " + path,
                "CONTENT:",
                content,
                "END_FILE",
                "END_BUNDLE",
            ))
        return response, _remote_receipt(response)

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(remote_chat)
    )
    bundle = _FakeSkillBundle()
    provider = LlmDraftProvider(
        max_attempts=3,
        configuration_status=_configured_status(tmp_path),
        skill_bundle=bundle,
        staged_direct=True,
    )
    result = GapCompletionPipeline(
        provider=provider,
        skill_bundle=bundle,
    ).run(
        str(source),
        str(tmp_path / ("attempts-%d" % expected_attempts)),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    draft = result.drafts[0]
    assert draft.generation["attempt_count"] == expected_attempts
    assert draft.generation["transport_attempt_count"] == expected_attempts
    assert len(draft.generation["remote_receipts"]) == expected_attempts
    reloaded = _load_existing_draft(Path(draft.workspace), result.specs[0])
    assert reloaded.artifact_id == draft.artifact_id
    assert reloaded.generation["attempt_count"] == expected_attempts
    if expected_attempts == 51:
        candidate = tmp_path / "repair-after-51"
        generation_request = (
            Path(result.out_dir)
            / "gaps"
            / result.specs[0].gap_id
            / "GENERATION_REQUEST.md"
        ).read_text(encoding="utf-8")
        repaired_paths = tuple(sorted(draft.files))[:16]
        repair = provider.repair_to_candidate(
            result.specs[0],
            Path(draft.workspace),
            candidate,
            paths=repaired_paths,
            diagnostic="compiler requested one full bounded repair round",
            generation_request=generation_request,
        )
        assert repair["attempt_count"] == 48
        assert set(repair_calls) == set(repaired_paths)
        assert set(repair_calls.values()) == {3}
        sealed_dir = tmp_path / "sealed-after-51"
        sealed = provider.seal_repair_candidate(
            result.specs[0],
            Path(draft.workspace),
            candidate,
            sealed_dir,
        )
        assert sealed.generation["attempt_count"] == 99
        sealed_reloaded = _load_existing_draft(sealed_dir, result.specs[0])
        assert sealed_reloaded.artifact_id == sealed.artifact_id
        assert sealed_reloaded.generation["attempt_count"] == 99

        from numpilot.gaps import provider as provider_module

        def fake_chain(candidate_dir, _spec, signed, _skill_bundle):
            count = 8 if Path(candidate_dir).name == "depth-base" else 7
            reports = tuple(
                {
                    "attempt_count": 48,
                    "report_sha256": "%064x" % (index + 1),
                }
                for index in range(count)
            )
            return dict(signed.files), reports

        monkeypatch.setattr(provider_module, "_load_repair_chain", fake_chain)
        provider._resolve_chat = lambda: (_ for _ in ()).throw(
            AssertionError("repair budget failures must precede model access")
        )
        with pytest.raises(DraftGenerationError, match="remaining sealable"):
            provider.repair_to_candidate(
                result.specs[0],
                sealed_dir,
                tmp_path / "over-budget-repair",
                paths=("op_host/cbrt_def.cpp",),
                diagnostic="another local failure",
                generation_request=generation_request,
                base_candidate_dir=tmp_path / "budget-base",
            )
        with pytest.raises(DraftGenerationError, match="remaining depth"):
            provider.repair_to_candidate(
                result.specs[0],
                sealed_dir,
                tmp_path / "over-depth-repair",
                paths=("op_host/cbrt_def.cpp",),
                diagnostic="another local failure",
                generation_request=generation_request,
                base_candidate_dir=tmp_path / "depth-base",
            )


def _refresh_artifact_id(manifest):
    identity = json.dumps(
        {
            "gap_id": manifest["gap_id"],
            "files": manifest["files"],
            "generation": manifest["generation"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest["artifact_id"] = (
        "draft-" + hashlib.sha256(identity).hexdigest()[:20]
    )


def test_remote_generation_persists_receipt_and_distinguishes_reuse(
    tmp_path, monkeypatch
):
    source = _write_gap_source(tmp_path)
    out = tmp_path / "out"
    bundle = _valid_bundle()
    receipt = _remote_receipt(bundle)
    calls = []

    def remote_chat(_system, _user, **_kwargs):
        calls.append(1)
        return bundle, receipt

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(remote_chat)
    )
    provider = LlmDraftProvider(
        max_attempts=1,
        configuration_status=_configured_status(tmp_path),
    )
    pipeline = GapCompletionPipeline(provider=provider)

    first = pipeline.run(
        str(source), str(out), target_soc="ascend910b", cann_version="9.0.0"
    )
    first_draft = first.drafts[0]
    assert provider.action_for(first_draft) == "fresh_remote"
    manifest = json.loads(
        (Path(first_draft.workspace) / "operator-artifact.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["generation"]["configuration_source"] == "user-config"
    assert manifest["generation"]["origin_action"] == "fresh_remote"
    assert manifest["generation"]["remote_receipts"] == [receipt]
    assert manifest["generation"]["transport_attempt_count"] == 1
    assert manifest["generation"]["successful_http_response_count"] == 1
    assert manifest["generation"]["content_response_count"] == 1

    second = pipeline.run(
        str(source), str(out), target_soc="ascend910b", cann_version="9.0.0"
    )

    assert len(calls) == 1
    assert provider.action_for(second.drafts[0]) == "reused_existing"
    assert second.drafts[0].generation["origin_action"] == "fresh_remote"


@pytest.mark.parametrize(
    "mutation",
    (
        "bool_attempt_count",
        "counter_mismatch",
        "receipt_model_mismatch",
        "response_hash_mismatch",
        "legacy_without_receipts",
    ),
)
def test_existing_remote_draft_revalidates_complete_generation_evidence(
    tmp_path, monkeypatch, mutation
):
    source = _write_gap_source(tmp_path)
    out = tmp_path / mutation
    bundle = _valid_bundle()
    monkeypatch.setattr(
        LlmEngine,
        "_chat_with_receipt",
        staticmethod(lambda _system, _user, **_kwargs: (bundle, _remote_receipt(bundle))),
    )
    first = GapCompletionPipeline(provider=LlmDraftProvider(
        max_attempts=1,
        configuration_status=_configured_status(tmp_path),
    )).run(
        str(source), str(out), target_soc="ascend910b", cann_version="9.0.0"
    )
    manifest_path = Path(first.drafts[0].workspace) / "operator-artifact.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generation = manifest["generation"]
    if mutation == "bool_attempt_count":
        generation["attempt_count"] = True
    elif mutation == "counter_mismatch":
        generation["successful_http_response_count"] = 0
    elif mutation == "receipt_model_mismatch":
        generation["remote_receipts"][0]["requested_model"] = "other-model"
    elif mutation == "response_hash_mismatch":
        generation["response_sha256"][0] = "f" * 64
    else:
        for name in (
            "remote_receipts",
            "transport_attempt_count",
            "successful_http_response_count",
            "content_response_count",
        ):
            generation.pop(name)
    _refresh_artifact_id(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    offline_provider = LlmDraftProvider(
        configuration_loader=lambda: (_ for _ in ()).throw(
            AssertionError("existing-draft validation must precede credential load")
        )
    )
    with pytest.raises(GapCompletionError, match="existing .*draft|existing draft"):
        GapCompletionPipeline(provider=offline_provider).run(
            str(source),
            str(out),
            target_soc="ascend910b",
            cann_version="9.0.0",
        )


def test_valid_existing_remote_draft_reuses_without_credentials(
    tmp_path, monkeypatch
):
    source = _write_gap_source(tmp_path)
    out = tmp_path / "remote-offline-reuse"
    bundle = _valid_bundle()
    monkeypatch.setattr(
        LlmEngine,
        "_chat_with_receipt",
        staticmethod(lambda _system, _user, **_kwargs: (bundle, _remote_receipt(bundle))),
    )
    GapCompletionPipeline(provider=LlmDraftProvider(
        max_attempts=1,
        configuration_status=_configured_status(tmp_path),
    )).run(
        str(source), str(out), target_soc="ascend910b", cann_version="9.0.0"
    )
    offline_provider = LlmDraftProvider(
        configuration_loader=lambda: (_ for _ in ()).throw(
            AssertionError("reuse must not load credentials")
        )
    )

    reused = GapCompletionPipeline(provider=offline_provider).run(
        str(source), str(out), target_soc="ascend910b", cann_version="9.0.0"
    )

    assert offline_provider.action_for(reused.drafts[0]) == "reused_existing"
    assert reused.drafts[0].generation["origin_action"] == "fresh_remote"


def test_remote_generation_persists_every_successful_http_receipt(
    tmp_path, monkeypatch
):
    source = _write_gap_source(tmp_path)
    invalid = "not-a-bundle"
    valid = _valid_bundle()
    responses = iter((
        (invalid, _remote_receipt(invalid)),
        (valid, _remote_receipt(valid)),
    ))

    monkeypatch.setattr(
        LlmEngine,
        "_chat_with_receipt",
        staticmethod(lambda _system, _user, **_kwargs: next(responses)),
    )
    provider = LlmDraftProvider(
        max_attempts=2,
        configuration_status=_configured_status(tmp_path),
    )

    result = GapCompletionPipeline(provider=provider).run(
        str(source),
        str(tmp_path / "retry"),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    evidence = result.drafts[0].generation
    assert evidence["attempt_count"] == 2
    assert evidence["transport_attempt_count"] == 2
    assert evidence["successful_http_response_count"] == 2
    assert evidence["content_response_count"] == 2
    assert len(evidence["remote_receipts"]) == 2
    assert [item["content_sha256"] for item in evidence["remote_receipts"]] == [
        _remote_receipt(invalid)["content_sha256"],
        _remote_receipt(valid)["content_sha256"],
    ]
    assert evidence["response_sha256"] == [
        _remote_receipt(invalid)["content_sha256"],
        _remote_receipt(valid)["content_sha256"],
    ]


def test_remote_generation_keeps_receipt_when_http_body_is_invalid(
    tmp_path, monkeypatch
):
    source = _write_gap_source(tmp_path)
    valid = _valid_bundle()
    invalid_receipt = _remote_receipt("invalid-envelope")
    invalid_receipt["content_sha256"] = None
    responses = iter((
        LlmResponseError("missing_content", invalid_receipt),
        (valid, _remote_receipt(valid)),
    ))

    def remote_chat(_system, _user, **_kwargs):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(remote_chat)
    )
    provider = LlmDraftProvider(
        max_attempts=2,
        configuration_status=_configured_status(tmp_path),
    )

    result = GapCompletionPipeline(provider=provider).run(
        str(source),
        str(tmp_path / "invalid-envelope"),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    receipts = result.drafts[0].generation["remote_receipts"]
    assert receipts == [invalid_receipt, _remote_receipt(valid)]
    assert result.drafts[0].generation["transport_attempt_count"] == 2
    assert result.drafts[0].generation["successful_http_response_count"] == 2
    assert result.drafts[0].generation["content_response_count"] == 1


def test_remote_http_failures_fail_closed_with_only_safe_status(
    tmp_path, monkeypatch
):
    source = _write_gap_source(tmp_path)

    def remote_failure(_system, _user, **_kwargs):
        raise LlmHttpError(503)

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(remote_failure)
    )
    provider = LlmDraftProvider(
        max_attempts=2,
        configuration_status=_configured_status(tmp_path),
    )

    with pytest.raises(GapCompletionError) as caught:
        GapCompletionPipeline(provider=provider).run(
            str(source),
            str(tmp_path / "failed"),
            target_soc="ascend910b",
            cann_version="9.0.0",
        )

    message = str(caught.value)
    assert "HTTP status 503" in message
    assert "token" not in message.lower()
    assert not list((tmp_path / "failed").rglob("draft"))


def test_remote_receipt_with_unknown_fields_or_sensitive_data_is_rejected(
    tmp_path, monkeypatch
):
    source = _write_gap_source(tmp_path)
    monkeypatch.setenv("NUMPILOT_LLM_AUTH_TOKEN", "do-not-persist")
    monkeypatch.setenv("NUMPILOT_LLM_BASE_URL", "https://secret.example/v1")
    bundle = _valid_bundle()
    receipt = _remote_receipt(bundle)
    receipt["unexpected"] = "do-not-persist https://secret.example/v1"
    monkeypatch.setattr(
        LlmEngine,
        "_chat_with_receipt",
        staticmethod(lambda _system, _user, **_kwargs: (bundle, receipt)),
    )

    with pytest.raises(GapCompletionError, match="incomplete receipt"):
        GapCompletionPipeline(provider=LlmDraftProvider(
            max_attempts=1,
            configuration_status=_configured_status(tmp_path),
        )).run(
            str(source),
            str(tmp_path / "unsafe-receipt"),
            target_soc="ascend910b",
            cann_version="9.0.0",
        )

    assert not list((tmp_path / "unsafe-receipt").rglob("draft"))


def test_injected_callable_cannot_supply_remote_receipts_or_claim_fresh_remote(
    tmp_path
):
    source = _write_gap_source(tmp_path)
    provider = LlmDraftProvider(
        chat=lambda _system, _user: _valid_bundle(), max_attempts=1
    )

    result = GapCompletionPipeline(provider=provider).run(
        str(source),
        str(tmp_path / "injected"),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    evidence = result.drafts[0].generation
    assert provider.action_for(result.drafts[0]) == "injected_test_double"
    assert evidence["origin_action"] == "injected_test_double"
    assert evidence["remote_receipts"] == []


def test_complete_gaps_cli_reports_fresh_and_reused_actions_without_reloading_config(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli

    source = _write_gap_source(tmp_path)
    out = tmp_path / "cli-out"
    status = _configured_status(tmp_path, source="environment+user-config")
    monkeypatch.setattr(
        cli, "resolve_skill_bundle", lambda *_args, **_kwargs: _FakeSkillBundle()
    )
    calls = []

    def remote_chat(system, user, **_kwargs):
        calls.append(system)
        if "connectivity probe" in system:
            response = "NUMPILOT_LLM_READY."
            return response, _remote_receipt(response)
        path_section = user.split(
            "Return exactly these paths and no others:\n", 1
        )[1].split("\n\nEvery file", 1)[0]
        paths = re.findall(r"^- `([^`]+)`$", path_section, flags=re.MULTILINE)
        parts = ["NUMPILOT_BUNDLE_V1", "SUMMARY: direct NumPy gap stage"]
        for path in paths:
            content = "{}" if path.endswith(".json") else "// complete fixture"
            if path.endswith(".sh"):
                content = "#!/usr/bin/env bash\nset -euo pipefail"
            elif path.endswith(".py"):
                content = "def test_generated_contract():\n    assert True"
            parts.extend(("FILE: " + path, "CONTENT:", content, "END_FILE"))
        parts.append("END_BUNDLE")
        response = "\n".join(parts)
        return response, _remote_receipt(response)

    monkeypatch.setattr(cli, "_load_llm_or_report", lambda: status)
    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(remote_chat)
    )
    args = [
        "complete-gaps",
        str(source),
        "--out",
        str(out),
        "--target-soc",
        "ascend910b",
        "--cann-version",
        "9.0.0",
        "--provider",
        "llm",
        "--max-attempts",
        "1",
    ]

    assert cli.main(args) == 0
    first_output = capsys.readouterr().out
    assert "action=fresh_remote" in first_output
    assert "actual_remote_generation_call=yes" in first_output
    assert "verified_new_remote_artifact=yes" in first_output
    assert "生成门禁 1/2] pass" in first_output
    assert "生成门禁 2/2] pass" in first_output
    preflight = json.loads(
        (out / "generation-preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["status"] == "pass"
    assert preflight["required_order"] == [
        "llm_readiness", "cannbot_skill_readiness",
    ]
    assert preflight["gates"]["llm_readiness"]["actual_remote_call"] is True
    assert preflight["gates"]["cannbot_skill_readiness"]["status"] == "pass"
    assert preflight["operator_goal"] == "accuracy"
    assert preflight["performance_optimization"] == "skipped_accuracy_first"
    report = json.loads((out / "gap-report.json").read_text(encoding="utf-8"))
    assert report["current_run_actions"] == {
        report["drafts"][0]["gap_id"]: "fresh_remote"
    }
    assert report["drafts"][0]["current_run_action"] == "fresh_remote"
    assert report["drafts"][0]["current_run_actual_remote_call"] is True
    assert report["current_run_actual_remote_call_count"] == 1
    assert report["drafts"][0]["generation"]["configuration_source"] == (
        "environment+user-config"
    )

    assert cli.main(args) == 0
    second_output = capsys.readouterr().out
    assert "action=reused_existing" in second_output
    assert "action=fresh_remote" not in second_output
    assert "actual_remote_generation_call=no" in second_output
    assert len(calls) == 19
    report = json.loads((out / "gap-report.json").read_text(encoding="utf-8"))
    assert report["current_run_actions"] == {
        report["drafts"][0]["gap_id"]: "reused_existing"
    }
    assert report["drafts"][0]["current_run_action"] == "reused_existing"
    assert report["drafts"][0]["current_run_actual_remote_call"] is False
    assert report["current_run_actual_remote_call_count"] == 0
    assert report["drafts"][0]["generation"]["origin_action"] == "fresh_remote"


def test_complete_gaps_cli_explicitly_reports_no_verified_remote_artifact_on_failure(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli

    source = _write_gap_source(tmp_path)
    status = _configured_status(tmp_path)
    monkeypatch.setattr(
        cli, "resolve_skill_bundle", lambda *_args, **_kwargs: _FakeSkillBundle()
    )
    monkeypatch.setattr(cli, "_load_llm_or_report", lambda: status)
    calls = []

    def probe_then_fail(system, _user, **_kwargs):
        calls.append(system)
        if "connectivity probe" in system:
            response = "NUMPILOT_LLM_READY."
            return response, _remote_receipt(response)
        raise LlmHttpError(503)

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(probe_then_fail)
    )

    result = cli.main([
        "complete-gaps",
        str(source),
        "--out",
        str(tmp_path / "failed-cli"),
        "--target-soc",
        "ascend910b",
        "--cann-version",
        "9.0.0",
        "--provider",
        "llm",
        "--max-attempts",
        "1",
    ])

    captured = capsys.readouterr()
    assert result == 2
    assert "verified_new_remote_artifact=no" in captured.err
    assert "未回退到 AST" in captured.err
    assert "生成门禁 1/2] pass" in captured.out
    assert "生成门禁 2/2] pass" in captured.out
    assert len(calls) == 2


def test_complete_gaps_llm_gate_failure_prevents_skill_read_and_generation(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli

    source = _write_gap_source(tmp_path)
    monkeypatch.setattr(cli, "_load_llm_or_report", lambda: _configured_status(tmp_path))
    monkeypatch.setattr(
        cli,
        "resolve_skill_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Skill must not be read before LLM gate passes")
        ),
    )
    monkeypatch.setattr(
        LlmEngine,
        "_chat_with_receipt",
        staticmethod(
            lambda _system, _user, **_kwargs: (_ for _ in ()).throw(
                LlmHttpError(503)
            )
        ),
    )

    result = cli.main([
        "complete-gaps", str(source), "--out", str(tmp_path / "gate-one-fail"),
        "--target-soc", "ascend910b", "--cann-version", "9.0.0",
        "--provider", "llm",
    ])

    captured = capsys.readouterr()
    assert result == 4
    assert "生成门禁 1/2] fail" in captured.err
    assert "未读取 Skill" in captured.err
    assert not (tmp_path / "gate-one-fail").exists()


def test_complete_gaps_skill_gate_failure_prevents_operator_generation(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli

    source = _write_gap_source(tmp_path)
    calls = []

    def ready(_system, _user, **_kwargs):
        calls.append(1)
        response = "NUMPILOT_LLM_READY."
        return response, _remote_receipt(response)

    monkeypatch.setattr(cli, "_load_llm_or_report", lambda: _configured_status(tmp_path))
    monkeypatch.setattr(LlmEngine, "_chat_with_receipt", staticmethod(ready))
    monkeypatch.setattr(
        cli,
        "resolve_skill_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            cli.CannbotSkillError("missing external Skill")
        ),
    )

    result = cli.main([
        "complete-gaps", str(source), "--out", str(tmp_path / "gate-two-fail"),
        "--target-soc", "ascend910b", "--cann-version", "9.0.0",
        "--provider", "llm",
    ])

    captured = capsys.readouterr()
    assert result == 2
    assert calls == [1]
    assert "生成门禁 2/2] fail" in captured.err
    assert "actual_remote_generation_call=no" in captured.err
    assert not (tmp_path / "gate-two-fail").exists()


def test_complete_gaps_cli_runs_gates_but_no_generation_when_there_is_no_gap(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli

    source = tmp_path / "covered.py"
    source.write_text(
        "import numpy as np\n\n"
        "def build_inputs():\n    return (np.ones(4),)\n\n"
        "def forward(x):\n    return np.add(x, x)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli, "resolve_skill_bundle", lambda *_args, **_kwargs: _FakeSkillBundle()
    )
    monkeypatch.setattr(cli, "_load_llm_or_report", lambda: _configured_status(tmp_path))
    calls = []

    def ready(_system, _user, **_kwargs):
        calls.append(1)
        response = "NUMPILOT_LLM_READY."
        return response, _remote_receipt(response)

    monkeypatch.setattr(LlmEngine, "_chat_with_receipt", staticmethod(ready))

    result = cli.main([
        "complete-gaps",
        str(source),
        "--out",
        str(tmp_path / "no-gap"),
        "--target-soc",
        "ascend910b",
        "--cann-version",
        "9.0.0",
        "--provider",
        "llm",
    ])

    captured = capsys.readouterr()
    assert result == 0
    assert calls == [1]
    assert "actual_remote_generation_call=no" in captured.out
    assert "没有可生成的合格缺口" in captured.out


def test_complete_gaps_cli_runs_gates_then_reuses_existing_draft(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli

    source = _write_gap_source(tmp_path)
    out = tmp_path / "offline-reuse"
    monkeypatch.setattr(
        cli, "resolve_skill_bundle", lambda *_args, **_kwargs: _FakeSkillBundle()
    )
    GapCompletionPipeline(
        provider=LlmDraftProvider(
            chat=lambda _system, _user: _valid_bundle(), max_attempts=1
        )
    ).run(
        str(source),
        str(out),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    monkeypatch.setattr(cli, "_load_llm_or_report", lambda: _configured_status(tmp_path))
    calls = []

    def ready(_system, _user, **_kwargs):
        calls.append(1)
        response = "NUMPILOT_LLM_READY."
        return response, _remote_receipt(response)

    monkeypatch.setattr(LlmEngine, "_chat_with_receipt", staticmethod(ready))

    result = cli.main([
        "complete-gaps",
        str(source),
        "--out",
        str(out),
        "--target-soc",
        "ascend910b",
        "--cann-version",
        "9.0.0",
        "--provider",
        "llm",
    ])

    assert result == 0
    output = capsys.readouterr().out
    assert calls == [1]
    assert "生成门禁 1/2] pass" in output
    assert "生成门禁 2/2] pass" in output
    assert "action=reused_existing" in output
    assert "actual_remote_generation_call=no" in output


def _repair_cli_args(tmp_path, out_name):
    return [
        "repair-draft",
        "--spec", str(tmp_path / "missing-gap-spec.json"),
        "--draft", str(tmp_path / "missing-draft"),
        "--out", str(tmp_path / out_name),
        "--generation-request", str(tmp_path / "missing-request.md"),
        "--diagnostic-file", str(tmp_path / "missing-diagnostic.txt"),
        "--path", "op_api/cbrt.cpp",
    ]


def test_repair_draft_llm_gate_failure_prevents_skill_read_and_repair(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli

    monkeypatch.setattr(cli, "_load_llm_or_report", lambda: _configured_status(tmp_path))
    monkeypatch.setattr(
        cli,
        "resolve_skill_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Skill must not be read before repair LLM gate passes")
        ),
    )
    monkeypatch.setattr(
        LlmEngine,
        "_chat_with_receipt",
        staticmethod(
            lambda _system, _user, **_kwargs: (_ for _ in ()).throw(
                LlmHttpError(503)
            )
        ),
    )

    result = cli.main(_repair_cli_args(tmp_path, "repair-gate-one-fail"))

    captured = capsys.readouterr()
    assert result == 4
    assert "修复门禁 1/2] fail" in captured.err
    assert "未读取 Skill" in captured.err
    assert not (tmp_path / "repair-gate-one-fail").exists()
    assert not (tmp_path / "repair-gate-one-fail.preflight").exists()


def test_repair_draft_skill_gate_failure_prevents_model_repair(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli

    calls = []

    def ready(_system, _user, **_kwargs):
        calls.append(1)
        response = "NUMPILOT_LLM_READY."
        return response, _remote_receipt(response)

    monkeypatch.setattr(cli, "_load_llm_or_report", lambda: _configured_status(tmp_path))
    monkeypatch.setattr(LlmEngine, "_chat_with_receipt", staticmethod(ready))
    monkeypatch.setattr(
        cli,
        "resolve_skill_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            cli.CannbotSkillError("missing external Skill")
        ),
    )

    result = cli.main(_repair_cli_args(tmp_path, "repair-gate-two-fail"))

    captured = capsys.readouterr()
    assert result == 2
    assert calls == [1]
    assert "修复门禁 2/2] fail" in captured.err
    assert "actual_remote_repair_call=no" in captured.err
    assert not (tmp_path / "repair-gate-two-fail").exists()
    assert not (tmp_path / "repair-gate-two-fail.preflight").exists()


def test_repair_draft_writes_preflight_only_after_both_gates(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli

    calls = []

    def ready(_system, _user, **_kwargs):
        calls.append(1)
        response = "NUMPILOT_LLM_READY."
        return response, _remote_receipt(response)

    monkeypatch.setattr(cli, "_load_llm_or_report", lambda: _configured_status(tmp_path))
    monkeypatch.setattr(LlmEngine, "_chat_with_receipt", staticmethod(ready))
    monkeypatch.setattr(
        cli, "resolve_skill_bundle", lambda *_args, **_kwargs: _FakeSkillBundle()
    )

    result = cli.main(_repair_cli_args(tmp_path, "repair-after-gates"))

    captured = capsys.readouterr()
    assert result == 2
    assert calls == [1]
    assert "修复门禁 1/2] pass" in captured.out
    assert "修复门禁 2/2] pass" in captured.out
    receipt = json.loads(
        (
            tmp_path
            / "repair-after-gates.preflight"
            / "generation-preflight.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["required_order"] == [
        "llm_readiness", "cannbot_skill_readiness",
    ]
    assert not (tmp_path / "repair-after-gates").exists()


@pytest.mark.parametrize("command", ("generate", "repair"))
def test_asnumpy_cli_failure_after_preflight_is_cleanly_retryable(
    tmp_path, monkeypatch, capsys, command
):
    from numpilot import __main__ as cli

    calls = []

    def ready(_system, _user, **_kwargs):
        calls.append(1)
        response = "NUMPILOT_LLM_READY."
        return response, _remote_receipt(response)

    monkeypatch.setattr(
        cli, "_load_llm_or_report", lambda: _configured_status(tmp_path)
    )
    monkeypatch.setattr(LlmEngine, "_chat_with_receipt", staticmethod(ready))
    monkeypatch.setattr(
        cli, "load_asnumpy_integration_skill", lambda *_args: _FakeSkillBundle()
    )
    monkeypatch.setattr(
        cli,
        "_run_asnumpy_skill_gate",
        lambda *_args, **_kwargs: {
            "gate": "asnumpy_integration_skill_readiness",
            "status": "pass",
        },
    )
    out = tmp_path / ("failed-asnumpy-" + command)
    if command == "generate":
        args = [
            "generate-asnumpy-adapter",
            "--spec", str(tmp_path / "missing-spec.json"),
            "--artifact", str(tmp_path / "missing-artifact"),
            "--asnumpy-reference", str(tmp_path / "missing-reference"),
            "--out", str(out),
        ]
    else:
        args = [
            "repair-asnumpy-adapter",
            "--base-adapter", str(tmp_path / "missing-base"),
            "--asnumpy-reference", str(tmp_path / "missing-reference"),
            "--diagnostic-file", str(tmp_path / "missing-diagnostic.txt"),
            "--path", "csrc/custom_ops/registry/cbrt/op.yaml",
            "--out", str(out),
        ]

    assert cli.main(args) == 2
    assert not out.exists()
    assert cli.main(args) == 2
    assert not out.exists()
    assert calls == [1, 1]
    assert not list(tmp_path.glob(".%s-*-stage-*" % out.name))
    captured = capsys.readouterr()
    assert "门禁 2/2] pass" in captured.out


@pytest.mark.parametrize("repair_profile", ("compile", "precision", "runtime"))
def test_repair_asnumpy_cli_passes_audit_profile_without_real_model(
    tmp_path, monkeypatch, repair_profile
):
    from numpilot import __main__ as cli
    from numpilot.gaps import provider as provider_module

    reference = tmp_path / "asnumpy-reference"
    for relative in (
        "CMakeLists.txt",
        "cmake/CustomOps.cmake",
        "scripts/gen_custom_ops.py",
    ):
        path = reference / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    diagnostic = tmp_path / "diagnostic.txt"
    diagnostic.write_text("local failure\n", encoding="utf-8")
    observed = {}

    class StubProvider:
        def __init__(self, **_kwargs):
            pass

        def repair_asnumpy_adapter_candidate(self, output_root, **kwargs):
            observed.update(kwargs)
            candidate = Path(output_root) / "candidate"
            candidate.mkdir()
            (candidate / "fixture.txt").write_text(
                "generated by stub\n", encoding="utf-8"
            )
            return {
                "status": "repaired_unverified",
                "adapter_artifact_id": "adapter-" + "1" * 20,
                "parent_adapter_artifact_id": "adapter-" + "2" * 20,
                "attempt_count": 1,
                "remote_receipts": [{}],
            }

    readiness = {
        "schema_version": 1,
        "gate": "llm_readiness",
        "status": "pass",
        "receipt": _remote_receipt("NUMPILOT_LLM_READY."),
    }
    monkeypatch.setattr(
        cli, "_load_llm_or_report", lambda: _configured_status(tmp_path)
    )
    monkeypatch.setattr(cli, "_run_llm_readiness_gate", lambda *_a, **_k: readiness)
    monkeypatch.setattr(
        cli, "load_asnumpy_integration_skill", lambda *_args: _FakeSkillBundle()
    )
    monkeypatch.setattr(
        cli,
        "_run_asnumpy_skill_gate",
        lambda *_args, **_kwargs: {
            "schema_version": 1,
            "gate": "asnumpy_integration_skill_readiness",
            "status": "pass",
        },
    )
    monkeypatch.setattr(provider_module, "LlmDraftProvider", StubProvider)
    out = tmp_path / ("profile-" + repair_profile)

    result = cli.main([
        "repair-asnumpy-adapter",
        "--base-adapter", str(tmp_path / "base-adapter"),
        "--asnumpy-reference", str(reference),
        "--diagnostic-file", str(diagnostic),
        "--path", "csrc/custom_ops/registry/cbrt/op.yaml",
        "--repair-profile", repair_profile,
        "--out", str(out),
    ])

    assert result == 0
    assert observed["repair_profile"] == repair_profile
    assert observed["skill_files"] == ASNUMPY_INTEGRATION_SKILL_FILES
    assert out.is_dir()


def test_model_path_traversal_is_rejected_without_escape(tmp_path):
    source = _write_gap_source(tmp_path)
    data = json.loads(_valid_bundle())
    data["files"].append({"path": "../escape.cpp", "content": "bad"})
    provider = LlmDraftProvider(
        chat=lambda _system, _user: json.dumps(data),
        max_attempts=1,
    )

    with pytest.raises(GapCompletionError, match="unsafe draft path"):
        GapCompletionPipeline(provider=provider).run(
            str(source),
            str(tmp_path / "out"),
            target_soc="ascend910b",
            cann_version="9.0.0",
        )

    assert not (tmp_path / "escape.cpp").exists()


def test_model_placeholder_marker_is_rejected_without_publishing_draft(tmp_path):
    source = _write_gap_source(tmp_path)
    data = json.loads(_valid_bundle())
    data["files"][3]["content"] = "// TODO_VERIFY_CANN_API\n"
    provider = LlmDraftProvider(
        chat=lambda _system, _user: json.dumps(data),
        max_attempts=1,
    )

    with pytest.raises(GapCompletionError, match="forbidden placeholder marker"):
        GapCompletionPipeline(provider=provider).run(
            str(source),
            str(tmp_path / "placeholder"),
            target_soc="ascend910b",
            cann_version="9.0.0",
        )

    assert not list((tmp_path / "placeholder").rglob("draft"))


def test_llm_generation_requires_concrete_cann_version(tmp_path):
    source = _write_gap_source(tmp_path)
    pipeline = GapCompletionPipeline(
        provider=LlmDraftProvider(
            chat=lambda _system, _user: _valid_bundle(),
            max_attempts=1,
        )
    )

    with pytest.raises(GapCompletionError, match="concrete CANN version"):
        pipeline.run(
            str(source),
            str(tmp_path / "out"),
            target_soc="ascend910b",
            cann_version="unknown",
        )


def test_existing_draft_hash_tampering_fails_closed(tmp_path):
    source = _write_gap_source(tmp_path)
    pipeline = GapCompletionPipeline(
        provider=LlmDraftProvider(
            chat=lambda _system, _user: _valid_bundle(),
            max_attempts=1,
        )
    )
    first = pipeline.run(
        str(source),
        str(tmp_path / "out"),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    draft = Path(first.drafts[0].workspace)
    (draft / "op_kernel/cbrt.cpp").write_text("tampered", encoding="utf-8")

    with pytest.raises(GapCompletionError, match="hash mismatch"):
        pipeline.run(
            str(source),
            str(tmp_path / "out"),
            target_soc="ascend910b",
            cann_version="9.0.0",
        )


def test_existing_draft_manifest_cannot_claim_activation(tmp_path):
    source = _write_gap_source(tmp_path)
    pipeline = GapCompletionPipeline(
        provider=LlmDraftProvider(
            chat=lambda _system, _user: _valid_bundle(),
            max_attempts=1,
        )
    )
    first = pipeline.run(
        str(source),
        str(tmp_path / "out"),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    draft = Path(first.drafts[0].workspace)
    manifest_path = draft / "operator-artifact.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "activated"
    manifest["gates"]["activation"] = "passed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(GapCompletionError, match="generated_unverified"):
        pipeline.run(
            str(source),
            str(tmp_path / "out"),
            target_soc="ascend910b",
            cann_version="9.0.0",
        )


def test_existing_draft_cannot_claim_ast_fallback(tmp_path):
    source = _write_gap_source(tmp_path)
    pipeline = GapCompletionPipeline(
        provider=LlmDraftProvider(
            chat=lambda _system, _user: _valid_bundle(),
            max_attempts=1,
        )
    )
    first = pipeline.run(
        str(source),
        str(tmp_path / "out"),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    draft = Path(first.drafts[0].workspace)
    manifest_path = draft / "operator-artifact.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation"]["fallback"] = "ast"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(GapCompletionError, match="fallback was disabled"):
        pipeline.run(
            str(source),
            str(tmp_path / "out"),
            target_soc="ascend910b",
            cann_version="9.0.0",
        )


def test_existing_draft_manifest_must_list_every_file(tmp_path):
    source = _write_gap_source(tmp_path)
    pipeline = GapCompletionPipeline(
        provider=LlmDraftProvider(
            chat=lambda _system, _user: _valid_bundle(),
            max_attempts=1,
        )
    )
    first = pipeline.run(
        str(source),
        str(tmp_path / "out"),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )
    draft = Path(first.drafts[0].workspace)
    manifest_path = draft / "operator-artifact.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].pop("op_kernel/cbrt.cpp")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(GapCompletionError, match="file set differs"):
        pipeline.run(
            str(source),
            str(tmp_path / "out"),
            target_soc="ascend910b",
            cann_version="9.0.0",
        )


def test_mixed_generatable_and_signature_gaps_are_partially_blocked(tmp_path):
    source = tmp_path / "mixed.py"
    source.write_text(
        "import numpy as np\n"
        "def forward(x):\n"
        "    return np.cbrt(x) + np.add(x, 1.0)\n",
        encoding="utf-8",
    )

    result = GapCompletionPipeline().run(
        str(source),
        str(tmp_path / "out"),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    assert result.status == "prepared_partially_blocked"
    assert result.specs[0].qualified_api == "numpy.cbrt"
    assert result.blocked
