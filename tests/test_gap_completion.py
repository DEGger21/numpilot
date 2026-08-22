import hashlib
import json
from pathlib import Path
import re

import pytest

from numpilot.agents.analysis import AnalysisAgent
from numpilot.engines.llm import LlmEngine, LlmHttpError, LlmResponseError
from numpilot.gaps.models import GapSpec
from numpilot.gaps.pipeline import GapCompletionError, GapCompletionPipeline
from numpilot.gaps.provider import DraftRepairError, LlmDraftProvider
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


class _FakeSkillBundle:
    root = "/opt/cannbot-skills"
    skill_name = "ascendc-registry-invoke-template"
    commit = "a" * 40
    bundle_sha256 = "b" * 64
    files = {"SKILL.md": "fixture"}
    total_bytes = 7

    def prompt_context(self, _selected_files=None):
        return "## Embedded test Skill\n"

    def manifest(self):
        return {
            "schema_version": 1,
            "repository": "https://gitcode.com/cann/cannbot-skills.git",
            "commit": self.commit,
            "skill_name": self.skill_name,
            "license": "test fixture",
            "bundle_sha256": self.bundle_sha256,
            "total_bytes": self.total_bytes,
            "files": {"SKILL.md": "c" * 64},
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
    bundle = _valid_bundle()
    calls = []

    def remote_chat(_system, _user, **_kwargs):
        calls.append(1)
        return bundle, _remote_receipt(bundle)

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
    assert "actual_remote_call=yes" in first_output
    assert "verified_new_remote_artifact=yes" in first_output
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
    assert "actual_remote_call=no" in second_output
    assert calls == [1]
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
    assert "actual_remote_call=yes" not in captured.out + captured.err


def test_complete_gaps_cli_reports_no_remote_call_when_there_is_no_gap(
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
    monkeypatch.setattr(
        cli,
        "_load_llm_or_report",
        lambda: (_ for _ in ()).throw(
            AssertionError("no gap must not load LLM configuration")
        ),
    )

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
    assert "actual_remote_call=no" in captured.out
    assert "没有可生成的合格缺口" in captured.out


def test_complete_gaps_cli_reuses_existing_draft_without_current_llm_config(
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
    monkeypatch.setattr(
        cli,
        "_load_llm_or_report",
        lambda: (_ for _ in ()).throw(
            AssertionError("reuse must not load current credentials")
        ),
    )

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
    assert "action=reused_existing" in output
    assert "actual_remote_call=no" in output


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
