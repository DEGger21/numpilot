# This file is part of the NumPilot project at Harbin Institute of Technology (HIT).
#
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# SPDX-License-Identifier: Apache-2.0
#
# Authors (accounts):
# - Su Tonghua <@sutonghua>
# - Li Xing <@li-xingHIT>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from types import SimpleNamespace

import pytest

from numpilot.agents.analysis import AnalysisAgent
from numpilot.asnumpy_skill import ASNUMPY_INTEGRATION_SKILL_FILES
from numpilot.cannbot_skills import (
    CANNBOT_GENERATION_SKILL_FILES,
    CANNBOT_REPAIR_SKILL_FILES,
)
from numpilot.engines.llm import LlmEngine, LlmHttpError, LlmResponseError
from numpilot.gaps.models import GapSpec
from numpilot.operator_artifact_audit import audit_store
from numpilot.gaps.pipeline import (
    GapCompletionError,
    GapCompletionPipeline,
    prepare_gap_completion,
)
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
    _render_official_exact_edit_repair_dependencies,
    _should_use_staged_direct,
    _validate_adapter_context_plan,
    _validate_bundle,
    _validate_stage_bundle,
)
from numpilot.gaps.spec import build_gap_specs
from numpilot.llm_config import (
    LlmConfigStatus,
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    model_identity_sha256,
)


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


def _clean_official_receipt(
    tmp_path,
    source,
    monkeypatch,
    *,
    target_soc="Ascend910B4",
    cann_version="9.0.0",
):
    """Create a real, strictly validated two-provider miss for CLI tests."""

    from numpilot.gaps import official_resolution as resolution_module
    from numpilot.gaps.installed_cann import (
        FileEvidence,
        InstalledCannInventory,
        VersionEvidence,
    )
    from numpilot.gaps.official_source import (
        REVIEWED_SOURCE_SEARCH_ALIASES,
        SOURCE_SEARCH_CONTRACT_REVISION,
    )

    preparation = prepare_gap_completion(
        str(source),
        target_soc=target_soc,
        cann_version=cann_version,
    )
    root = str((tmp_path / "cann-9.0.0").absolute())
    inventory = InstalledCannInventory(
        root=root,
        platform="aarch64-linux",
        architecture="aarch64",
        operating_system="linux",
        version=cann_version,
        version_evidence=(VersionEvidence(
            file=FileEvidence(
                relative_path="aarch64-linux/ascend_toolkit_install.info",
                sha256="0" * 64,
                size=1,
            ),
            version=cann_version,
            source_field="version",
        ),),
        headers=(),
        libraries=(),
    )

    class CleanSourceMiss:
        def resolve(self, **lookup):
            aliases = REVIEWED_SOURCE_SEARCH_ALIASES[
                (lookup["cann_version"], lookup["qualified_api"])
            ]
            payload = {
                "schema_version": 1,
                "provider": "ops_nn_source",
                "status": "no_match",
                "reason": "test reviewed direct-operator aliases are absent",
                "requested": dict(lookup),
                "source_ref": "refs/tags/v9.0.0",
                "source_commit": "1" * 40,
                "source_tree": "2" * 40,
                "candidates": [],
                "selected_candidate_id": "",
                "requires_audit": False,
                "resolution_kind": "none",
                "search_contract_revision": SOURCE_SEARCH_CONTRACT_REVISION,
                "searched_aliases": list(aliases),
                "negative_search_complete": True,
                "activation_allowed": False,
            }
            return SimpleNamespace(to_dict=lambda: payload)

    report = resolution_module.resolve_official_operators(
        preparation,
        cann_root=root,
        ops_nn_root=str((tmp_path / "ops-nn").absolute()),
        accept_cann_osl=True,
        inventory_loader=lambda _root: inventory,
        source_resolver_factory=lambda _root: CleanSourceMiss(),
    )
    assert report.model_kernel_generation_allowed is True
    receipt = tmp_path / "official-resolution.json"
    resolution_module.write_official_resolution_report(report, str(receipt))
    monkeypatch.setattr(
        resolution_module,
        "resolve_official_operators",
        lambda _preparation, **_kwargs: report,
    )
    return receipt


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


def test_gap_preparation_rejects_symlink_fifo_and_oversize_sources(tmp_path):
    real = _write_gap_source(tmp_path)
    linked = tmp_path / "linked-source.py"
    linked.symlink_to(real)
    fifo = tmp_path / "source.fifo"
    os.mkfifo(fifo)
    oversize = tmp_path / "oversize.py"
    with open(oversize, "wb") as handle:
        handle.truncate(4 * 1024 * 1024 + 1)

    for unsafe in (linked, fifo, oversize):
        with pytest.raises(GapCompletionError):
            prepare_gap_completion(
                str(unsafe),
                target_soc="ascend910b",
                cann_version="9.0.0",
            )


def test_gap_preparation_rejects_source_changed_during_read(tmp_path, monkeypatch):
    from numpilot.gaps import pipeline as pipeline_module

    source = _write_gap_source(tmp_path)
    real_read = os.read
    changed = []

    def mutate_after_first_read(descriptor, size):
        payload = real_read(descriptor, size)
        if payload and not changed:
            changed.append(True)
            source.write_text(
                GAP_SOURCE.replace("np.cbrt", "np.sqrt"), encoding="utf-8"
            )
        return payload

    monkeypatch.setattr(pipeline_module.os, "read", mutate_after_first_read)

    with pytest.raises(GapCompletionError, match="changed while"):
        prepare_gap_completion(
            str(source),
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
    artifact_store_report = audit_store(out / "operator-artifacts")
    assert artifact_store_report["ok"] is True
    assert artifact_store_report["counts"]["operators"] == 1
    assert artifact_store_report["counts"]["candidates"] == 1
    assert artifact_store_report["errors"] == []
    stored_candidate = next(
        (out / "operator-artifacts").glob("operators/*/candidates/*/candidate.json")
    )
    stored = json.loads(stored_candidate.read_text(encoding="utf-8"))
    assert stored["source_artifact_id"] == artifact["artifact_id"]
    assert (draft.parent / "operator-artifact-store-ref.json").is_file()
    report = json.loads((out / "gap-report.json").read_text(encoding="utf-8"))
    assert report["integrated_count"] == 0
    assert report["qualified_count"] == 0
    assert report["activated_count"] == 0
    assert report["rescanned_count"] == 0
    assert report["claim_boundary"] == {
        "level": "generated_unverified",
        "build_performed": False,
        "unit_test_performed": False,
        "numerical_validation_performed": False,
        "functional_correctness_verified": False,
        "integration_performed": False,
        "qualification_performed": False,
        "activation_performed": False,
        "capability_rescan_performed": False,
        "interface_available": False,
    }
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


@pytest.mark.parametrize(
    ("provider_name", "model"),
    (("openai", "gpt-audited"), ("gateway", "org/model")),
)
def test_remote_provider_accepts_generic_identity_and_binds_model_receipt(
    tmp_path, monkeypatch, provider_name, model
):
    source = _write_gap_source(tmp_path)
    calls = []
    status = LlmConfigStatus(
        path=tmp_path / "llm.json",
        configured=True,
        missing=(),
        source="user-config",
        model=model,
        provider=provider_name,
        token_present=True,
        protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        response_model_aliases=(),
    )
    response = _valid_line_bundle()
    monkeypatch.setattr(
        LlmEngine,
        "_chat_with_receipt",
        staticmethod(lambda *_args, **_kwargs: (
            calls.append(1) and None
        ) or (response, _generic_remote_receipt(
            response, model=model, provider=provider_name
        ))),
    )

    result = GapCompletionPipeline(provider=LlmDraftProvider(
        max_attempts=1,
        configuration_status=status,
    )).run(
        str(source),
        str(tmp_path / (provider_name + "-generic")),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )

    assert result.status == "generated_unverified"
    assert calls == [1]
    generation = result.drafts[0].generation
    assert generation["provider"] == provider_name
    assert generation["protocol"] == OPENAI_CHAT_COMPLETIONS_PROTOCOL
    assert generation["model"] == model


def test_complete_gaps_injected_response_is_local_and_contract_bound(tmp_path, capsys):
    from numpilot import __main__ as cli

    source = _write_gap_source(tmp_path)
    response = tmp_path / "codex-response.txt"
    response.write_text(_valid_line_bundle(), encoding="utf-8")
    out = tmp_path / "injected-out"

    rc = cli.main([
        "complete-gaps",
        str(source),
        "--out", str(out),
        "--provider", "injected",
        "--response-file", str(response),
        "--target-soc", "Ascend910B4",
        "--cann-version", "9.0.0",
        "--max-attempts", "1",
    ])

    assert rc == 0
    terminal = capsys.readouterr()
    assert "actual_remote_generation_call=no" in terminal.out
    report = json.loads((out / "gap-report.json").read_text(encoding="utf-8"))
    assert report["status"] == "generated_unverified"
    draft = report["drafts"][0]
    assert draft["generation"]["mode"] == "injected_test_double"
    assert draft["generation"]["provider"] == "codex-self"
    assert draft["generation"]["model"] == "codex-local"
    assert draft["generation"]["remote_receipts"] == []
    assert draft["generation"]["fallback"] == "disabled"
    assert report["claim_boundary"]["level"] == "generated_unverified"
    assert report["claim_boundary"]["activation_performed"] is False


def test_complete_gaps_injected_requires_response_file_before_output(tmp_path, capsys):
    from numpilot import __main__ as cli

    source = _write_gap_source(tmp_path)
    out = tmp_path / "must-not-exist"
    rc = cli.main([
        "complete-gaps",
        str(source),
        "--out", str(out),
        "--provider", "injected",
        "--target-soc", "Ascend910B4",
        "--cann-version", "9.0.0",
    ])

    assert rc == 2
    assert not out.exists()
    terminal = capsys.readouterr()
    assert "--provider injected" in terminal.err
    assert "不会读取 llm.json" in terminal.err


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


def test_generation_preflight_accepts_official_existing_module_plan(tmp_path):
    from numpilot import __main__ as cli

    plan = _official_adapter_context_plan(tmp_path)

    cli._write_generation_preflight(
        tmp_path,
        {"gate": "llm_readiness", "status": "pass"},
        {"gate": "asnumpy_integration_skill_readiness", "status": "pass"},
        120.0,
        adapter_context_plan=plan,
    )

    receipt = json.loads(
        (tmp_path / "generation-preflight.json").read_text(encoding="utf-8")
    )
    assert receipt["asnumpy_adapter_context_plan"] == plan


def test_generation_preflight_rejects_mismatched_adapter_architecture_mode(
    tmp_path,
):
    from numpilot import __main__ as cli

    plan = _official_adapter_context_plan(tmp_path)
    plan["mode"] = "manual"

    with pytest.raises(OSError, match="context plan is invalid"):
        cli._write_generation_preflight(
            tmp_path,
            {"gate": "llm_readiness", "status": "pass"},
            {
                "gate": "asnumpy_integration_skill_readiness",
                "status": "pass",
            },
            120.0,
            adapter_context_plan=plan,
        )


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


def _generic_remote_receipt(content, *, model, provider, response_model=None):
    response_model = model if response_model is None else response_model
    return {
        "schema_version": 2,
        "protocol": OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        "provider": provider,
        "configured_model": model,
        "utc_start": "2026-08-26T00:00:00Z",
        "utc_end": "2026-08-26T00:00:01Z",
        "http_status": 200,
        "requested_model": model,
        "response_model": response_model,
        "response_model_aliases": [],
        "response_model_identity": (
            "exact" if response_model == model else "mismatch"
        ),
        "model_identity_sha256": model_identity_sha256(
            protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
            provider=provider,
            configured_model=model,
            requested_model=model,
            response_model=response_model,
            response_model_aliases=(),
            response_model_identity=(
                "exact" if response_model == model else "mismatch"
            ),
        ),
        "response_id": "response-generic",
        "response_created": 1787702400,
        "request_body_sha256": "1" * 64,
        "response_body_sha256": "2" * 64,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "request_id": "request-generic",
        "usage": {"total_tokens": 101},
        "tls_verification": "enabled",
    }


_ADAPTER_REFERENCE_CONTEXT = {
    "CMakeLists.txt": "include(cmake/CustomOps.cmake)\n",
    "cmake/CustomOps.cmake": "message(FATAL_ERROR missing vendor)\n",
}
_ADAPTER_OVERLAY_PATHS = (
    "csrc/custom_ops/registry/cbrt/op.yaml",
    "tests/asnumpy_tests/custom_ops_tests/test_cbrt.py",
)


def _adapter_context_plan(tmp_path, context, paths=_ADAPTER_OVERLAY_PATHS):
    hashes = {
        label: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for label, content in sorted(context.items())
    }
    identity = {
        "schema_version": 1,
        "architecture": "manifest_generator",
        "mode": "manifest",
        "context_sha256": hashes,
        "expected_overlay_paths": list(paths),
    }
    plan_sha256 = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        **identity,
        "reference_root": str((tmp_path / "asnumpy-reference-fixture").absolute()),
        "total_bytes": sum(len(item.encode("utf-8")) for item in context.values()),
        "plan_sha256": plan_sha256,
    }


def _official_adapter_context_plan(tmp_path):
    paths = (
        "csrc/linalg/product.cpp",
        "src/asnumpy/linalg/direct.py",
        "tests/asnumpy_tests/linalg_tests/test_product.py",
    )
    context = {
        "include/asnumpy/linalg/product.hpp": "void Matmul();\n",
        "bindings/python/bind_linalg.cpp": "bind matmul\n",
        "src/asnumpy/utils.py": "class ndarray: pass\n",
        "bindings/python/bind_utils.cpp": "bind strides\n",
    }
    hashes = {
        label: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for label, content in context.items()
    }
    dependencies = {
        paths[0]: [
            {
                "source": "reference",
                "path": "include/asnumpy/linalg/product.hpp",
            },
            {
                "source": "signed_official_header",
                "path": "SIGNED_OFFICIAL_HEADER/aclnn_matmul.h",
            },
        ],
        paths[1]: [
            {
                "source": "reference",
                "path": "bindings/python/bind_linalg.cpp",
            },
            {"source": "reference", "path": "src/asnumpy/utils.py"},
            {
                "source": "reference",
                "path": "bindings/python/bind_utils.cpp",
            },
        ],
        paths[2]: [
            {
                "source": "generated_overlay",
                "path": "src/asnumpy/linalg/direct.py",
            },
        ],
    }
    identity = {
        "schema_version": 2,
        "architecture": "official_aclnn_existing_module",
        "mode": "existing_module",
        "context_sha256": hashes,
        "expected_overlay_paths": list(paths),
        "prompt_dependencies": dependencies,
    }
    return {
        **identity,
        "reference_root": str(
            (tmp_path / "official-asnumpy-reference-fixture").absolute()
        ),
        "total_bytes": sum(
            len(content.encode("utf-8")) for content in context.values()
        ),
        "plan_sha256": hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
    }


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_adapter_context_plan_schema_requires_exact_int(tmp_path, schema_version):
    plan = _adapter_context_plan(tmp_path, _ADAPTER_REFERENCE_CONTEXT)
    plan["schema_version"] = schema_version

    with pytest.raises(DraftGenerationError, match="context plan schema"):
        _validate_adapter_context_plan(plan)


def test_adapter_context_read_paths_do_not_expand_writable_overlay(tmp_path):
    context = {
        "scripts/gen_custom_ops.py": "print('fixture')\n",
        "cmake/CustomOps.cmake": "# fixture\n",
        "CMakeLists.txt": "project(asnumpy)\n",
    }
    plan = _adapter_context_plan(
        tmp_path,
        context,
        paths=("src/asnumpy/custom_ops.py",),
    )

    assert _validate_adapter_context_plan(plan)["context_sha256"] == (
        plan["context_sha256"]
    )

    unsafe = json.loads(json.dumps(plan))
    unsafe["expected_overlay_paths"] = ["scripts/generated.py"]
    identity = {
        name: unsafe[name]
        for name in (
            "schema_version",
            "architecture",
            "mode",
            "context_sha256",
            "expected_overlay_paths",
        )
    }
    unsafe["plan_sha256"] = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()

    with pytest.raises(
        DraftGenerationError,
        match="unexpected AsNumpy adapter directory",
    ):
        _validate_adapter_context_plan(unsafe)


def test_adapter_context_plan_accepts_reordered_dependency_keys(tmp_path):
    plan = _official_adapter_context_plan(tmp_path)
    paths = tuple(plan["expected_overlay_paths"])

    # Reports are persisted with ``sort_keys=True``; dependency object key
    # order is not semantic and must not change the rebuilt adapter plan.
    reordered = {
        name: plan["prompt_dependencies"][name]
        for name in reversed(paths)
    }
    assert tuple(reordered) != paths
    plan["prompt_dependencies"] = reordered

    validated = _validate_adapter_context_plan(plan)

    assert tuple(validated["prompt_dependencies"]) == paths


def _write_adapter_base(tmp_path, *, schema_version=2, context=None):
    context = dict(context or _ADAPTER_REFERENCE_CONTEXT)
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
        "schema_version": schema_version,
        "status": "generated_unverified",
        "source_artifact_id": "draft-fixture",
        "model": "glm-audited",
        "provider": "glm",
        "configuration_source": "user-config",
        "attempt_count": 1,
        "files_sha256": hashes,
        "adapter_contract_sha256": "c" * 64,
        "asnumpy_context_sha256": {
            label: hashlib.sha256(content.encode("utf-8")).hexdigest()
            for label, content in sorted(context.items())
        },
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
    if schema_version >= 3:
        report["protocol"] = OPENAI_CHAT_COMPLETIONS_PROTOCOL
        report["response_model_aliases"] = []
    if schema_version == 4:
        report["asnumpy_adapter_context_plan"] = _adapter_context_plan(
            tmp_path, context
        )
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
            path = candidate / ("csrc/extra-%02d.cpp" % index)
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
            path = candidate / ("csrc/large-%02d.cpp" % index)
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


def test_generated_asnumpy_adapter_is_context_bound_schema_four_and_reloadable(
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
    context = {"CMakeLists.txt": "project(asnumpy)\n"}
    paths = (
        "csrc/custom_ops/registry/cbrt/op.yaml",
        "tests/asnumpy_tests/custom_ops_tests/test_cbrt.py",
    )
    report = provider.generate_asnumpy_adapter_candidate(
        out,
        operator_artifact={"artifact_id": "draft-fixture"},
        adapter_contract='{"api_name":"cbrt"}',
        asnumpy_context=context,
        expected_paths=paths,
        adapter_context_plan=_adapter_context_plan(tmp_path, context, paths),
    )

    assert report["schema_version"] == 4
    assert report["protocol"] == OPENAI_CHAT_COMPLETIONS_PROTOCOL
    assert report["response_model_aliases"] == []
    assert report["integration_skill_files"] == list(
        ASNUMPY_INTEGRATION_SKILL_FILES
    )
    assert report["adapter_artifact_id"] == _adapter_artifact_id(report)
    reloaded, _payload, hashes = _load_asnumpy_adapter_report(out)
    assert reloaded == report
    assert hashes == report["files_sha256"]


@pytest.mark.parametrize(
    "path",
    (
        "src/asnumpy/linalg/direct.py",
        "include/asnumpy/linalg/product.hpp",
        "bindings/python/bind_linalg.cpp",
    ),
)
def test_operator_stage_bundle_does_not_inherit_adapter_overlay_paths(path):
    response = "\n".join((
        "NUMPILOT_BUNDLE_V1",
        "SUMMARY: boundary fixture",
        "FILE: " + path,
        "CONTENT:",
        "fixture",
        "END_FILE",
        "END_BUNDLE",
    ))

    with pytest.raises(DraftGenerationError, match="unexpected draft directory"):
        _validate_stage_bundle(response, (path,))


def test_asnumpy_adapter_legacy_paths_generate_reload_and_repair(
    tmp_path, monkeypatch
):
    paths = (
        "src/asnumpy/linalg/direct.py",
        "include/asnumpy/linalg/product.hpp",
        "bindings/python/bind_linalg.cpp",
    )
    context = {
        "src/asnumpy/nn.py": "def softmax():\n    pass\n",
        "include/asnumpy/nn/activation.hpp": "#pragma once\n",
        "bindings/python/module.cpp": "// bindings fixture\n",
    }
    context_hashes = {
        name: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for name, content in sorted(context.items())
    }
    identity = {
        "schema_version": 1,
        "architecture": "legacy_manual",
        "mode": "manual",
        "context_sha256": context_hashes,
        "expected_overlay_paths": list(paths),
    }
    plan = {
        **identity,
        "reference_root": str((tmp_path / "asnumpy-reference").absolute()),
        "total_bytes": sum(
            len(content.encode("utf-8")) for content in context.values()
        ),
        "plan_sha256": hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
    }

    def remote_chat(_system, user, **_kwargs):
        match = re.search(r"(?:Return|Repair) exactly `([^`]+)`", user)
        assert match is not None
        path = match.group(1)
        content = (
            "def matmul():\n    return 'repaired'"
            if user.startswith("Repair exactly")
            else "fixture for " + path
        )
        response = "\n".join((
            "NUMPILOT_BUNDLE_V1",
            "SUMMARY: adapter boundary fixture",
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
    provider = LlmDraftProvider(
        max_attempts=1,
        configuration_status=_configured_status(tmp_path),
        skill_bundle=_FakeSkillBundle(),
        staged_direct=True,
    )
    generated = tmp_path / "generated-legacy-adapter"
    generated.mkdir()
    report = provider.generate_asnumpy_adapter_candidate(
        generated,
        operator_artifact={"artifact_id": "draft-fixture"},
        adapter_contract='{"api_name":"matmul"}',
        asnumpy_context=context,
        expected_paths=paths,
        adapter_context_plan=plan,
    )

    assert set(report["files_sha256"]) == set(paths)
    reloaded, _payload, hashes = _load_asnumpy_adapter_report(generated)
    assert reloaded == report
    assert hashes == report["files_sha256"]

    repaired = tmp_path / "repaired-legacy-adapter"
    repaired.mkdir()
    repair_report = provider.repair_asnumpy_adapter_candidate(
        repaired,
        base_adapter_root=generated,
        diagnostic="local compiler fixture",
        path=paths[0],
        asnumpy_context=context,
        adapter_context_plan=plan,
    )

    assert repair_report["paths"] == [paths[0]]
    assert (repaired / "candidate" / paths[0]).read_text(
        encoding="utf-8"
    ) == "def matmul():\n    return 'repaired'"
    _reloaded, _payload, repaired_hashes = _load_asnumpy_adapter_report(
        repaired
    )
    assert repaired_hashes == repair_report["files_sha256"]


@pytest.mark.parametrize("repair_profile", ("compile", "precision", "runtime"))
def test_remote_model_repairs_hash_bound_asnumpy_adapter_from_raw_log(
    tmp_path, monkeypatch, repair_profile
):
    base = _write_adapter_base(tmp_path, schema_version=4)
    prompts = []
    call_options = []

    def remote_chat(system, user, **kwargs):
        prompts.append((system, user))
        call_options.append(kwargs)
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
        asnumpy_context=dict(_ADAPTER_REFERENCE_CONTEXT),
        adapter_context_plan=_adapter_context_plan(
            tmp_path, _ADAPTER_REFERENCE_CONTEXT
        ),
        repair_profile=repair_profile,
    )

    assert len(prompts) == 1
    assert call_options == [{}]
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


def test_official_exact_edit_repair_binds_absent_signed_header_by_hash():
    reference_path = "include/asnumpy/array/basic.hpp"
    header_path = "SIGNED_OFFICIAL_HEADER/aclnn_cat.h"
    target_path = "csrc/array/basic.cpp"
    reference = (
        "NPUArray Concatenate(const std::vector<NPUArray>& arrays, int axis);\n"
    )
    target = (
        "NPUArray Linspace(const py::object& start) {\n"
        "    return NPUArray();\n"
        "}\n\n"
        "NPUArray Concatenate(const std::vector<NPUArray>& arrays, int axis) {\n"
        "    return NPUArray();\n"
        "}\n"
    )
    header_hash = "b" * 64
    plan = {
        "context_sha256": {
            reference_path: hashlib.sha256(
                reference.encode("utf-8")
            ).hexdigest(),
        },
        "prompt_dependencies": {
            target_path: [
                {"source": "reference", "path": reference_path},
                {"source": "signed_official_header", "path": header_path},
            ],
        },
        "plan_sha256": "a" * 64,
    }

    rendered = _render_official_exact_edit_repair_dependencies(
        target_path=target_path,
        asnumpy_context={reference_path: reference},
        base_context_hashes={header_path: header_hash},
        normalized_plan=plan,
        current_contents={target_path: target},
        current_hashes={
            target_path: hashlib.sha256(target.encode("utf-8")).hexdigest(),
        },
    )

    body = rendered.split("\n\n", 1)[0]
    prefix = "HASH-BOUND REPAIR DEPENDENCY PROJECTION:\n"
    assert body.startswith(prefix)
    projection = json.loads(body[len(prefix):])
    signed = next(
        item for item in projection["dependencies"]
        if item["source"] == "signed_official_header"
    )
    assert signed["sha256"] == header_hash
    assert signed["binding"] == (
        "parent_adapter_report.asnumpy_context_sha256"
    )
    assert signed["content_supplied"] is False
    reference_entry = next(
        item for item in projection["dependencies"]
        if item["source"] == "reference"
    )
    assert reference_entry["content_supplied"] is True
    assert "HASH-BOUND REPAIR DEPENDENCY `%s`" % reference_path in rendered
    assert "HASH-BOUND REPAIR DEPENDENCY `%s`" % header_path not in rendered

    with pytest.raises(
        DraftGenerationError, match="signed-header dependency is missing"
    ):
        _render_official_exact_edit_repair_dependencies(
            target_path=target_path,
            asnumpy_context={reference_path: reference},
            base_context_hashes={},
            normalized_plan=plan,
            current_contents={target_path: target},
            current_hashes={
                target_path: hashlib.sha256(
                    target.encode("utf-8")
                ).hexdigest(),
            },
        )


@pytest.mark.parametrize(
    "mutation",
    ("schema", "artifact_id", "receipt", "content", "skill", "context"),
)
def test_asnumpy_adapter_base_revalidates_all_audit_bindings(
    tmp_path, monkeypatch, mutation
):
    base = _write_adapter_base(tmp_path, schema_version=4)
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
            asnumpy_context=dict(_ADAPTER_REFERENCE_CONTEXT),
            adapter_context_plan=_adapter_context_plan(
                tmp_path, _ADAPTER_REFERENCE_CONTEXT
            ),
            skill_files=CANNBOT_REPAIR_SKILL_FILES,
        )
    assert not isinstance(caught.value, AssertionError)
    assert not (out / "candidate").exists()


def test_asnumpy_adapter_repair_revalidates_exact_parent_report_bytes(
    tmp_path, monkeypatch
):
    base = _write_adapter_base(tmp_path, schema_version=4)

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
        asnumpy_context=dict(_ADAPTER_REFERENCE_CONTEXT),
        adapter_context_plan=_adapter_context_plan(
            tmp_path, _ADAPTER_REFERENCE_CONTEXT
        ),
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
    if signed.generation["mode"] != "remote_llm":
        manifest_path = draft_path / "operator-artifact.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        generation_response = _valid_bundle()
        manifest["generation"].update({
            "mode": "remote_llm",
            "provider": "glm",
            "model": "glm-audited",
            "configuration_source": "test-fixture",
            "origin_action": "fresh_remote",
            "successful_http_response_count": 1,
            "remote_receipts": [_remote_receipt(generation_response)],
        })
        _refresh_artifact_id(manifest)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        signed = _load_existing_draft(draft_path, spec)
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


def test_schema_two_repair_retry_seals_and_reloads_offline(
    tmp_path, monkeypatch
):
    spec, draft_path, signed = _write_simple_signed_draft(tmp_path, monkeypatch)
    repair_response = "\n".join((
        "NUMPILOT_BUNDLE_V1",
        "SUMMARY: repaired after one rejected identity",
        "FILE: op_host/cbrt_def.cpp",
        "CONTENT:",
        "// repaired after retry",
        "END_FILE",
        "END_BUNDLE",
    ))
    calls = []

    def remote_chat(_system, _user, **_kwargs):
        calls.append(1)
        return repair_response, _generic_remote_receipt(
            repair_response,
            model=signed.generation["model"],
            provider=signed.generation["provider"],
            response_model=(
                "gateway/mismatched-model"
                if len(calls) == 1
                else signed.generation["model"]
            ),
        )

    monkeypatch.setattr(
        LlmEngine, "_chat_with_receipt", staticmethod(remote_chat)
    )
    provider = LlmDraftProvider(
        max_attempts=2,
        configuration_status=_configured_status(tmp_path),
        skill_bundle=_FakeSkillBundle(),
    )
    candidate = tmp_path / "schema-two-repair-retry"
    report = provider.repair_to_candidate(
        spec,
        draft_path,
        candidate,
        paths=("op_host/cbrt_def.cpp",),
        diagnostic="compiler requested one bounded correction",
        generation_request="bounded request",
    )

    assert report["schema_version"] == 2
    assert report["attempt_count"] == 2
    assert [
        item["response_model_identity"] for item in report["remote_receipts"]
    ] == ["mismatch", "exact"]

    monkeypatch.setattr(
        LlmEngine,
        "_chat_with_receipt",
        staticmethod(
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("seal/reload must not call a model")
            )
        ),
    )
    sealed_dir = tmp_path / "schema-two-repair-sealed"
    sealed = provider.seal_repair_candidate(
        spec, draft_path, candidate, sealed_dir
    )
    reloaded = _load_existing_draft(sealed_dir, spec)

    assert calls == [1, 1]
    assert reloaded.artifact_id == sealed.artifact_id
    assert reloaded.generation["attempt_count"] == (
        signed.generation["attempt_count"] + 2
    )


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
        "receipt_response_model_mismatch",
        "non_glm_identity",
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
    elif mutation == "receipt_response_model_mismatch":
        generation["remote_receipts"][0]["response_model"] = "glm-other"
    elif mutation == "non_glm_identity":
        generation["provider"] = "openai"
        generation["model"] = "gpt-old"
        generation["remote_receipts"][0]["requested_model"] = "gpt-old"
        generation["remote_receipts"][0]["response_model"] = "gpt-old"
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


def test_complete_gaps_llm_requires_official_receipt_before_model_skill_or_output(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli

    source = _write_gap_source(tmp_path)
    output = tmp_path / "must-not-exist"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("missing official receipt must block locally")

    monkeypatch.setattr(cli, "_load_llm_or_report", forbidden)
    monkeypatch.setattr(cli, "resolve_skill_bundle", forbidden)

    result = cli.main([
        "complete-gaps",
        str(source),
        "--out", str(output),
        "--target-soc", "Ascend910B4",
        "--cann-version", "9.0.0",
        "--provider", "llm",
    ])

    captured = capsys.readouterr()
    assert result == 3
    assert "--official-resolution-report" in captured.err
    assert "actual_remote_generation_call=no" in captured.err
    assert not output.exists()


def test_complete_gaps_cli_reports_fresh_and_reused_actions_without_reloading_config(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli

    source = _write_gap_source(tmp_path)
    out = tmp_path / "cli-out"
    receipt = _clean_official_receipt(tmp_path, source, monkeypatch)
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
        "Ascend910B4",
        "--cann-version",
        "9.0.0",
        "--provider",
        "llm",
        "--official-resolution-report",
        str(receipt),
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
    receipt = _clean_official_receipt(tmp_path, source, monkeypatch)
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
        "Ascend910B4",
        "--cann-version",
        "9.0.0",
        "--provider",
        "llm",
        "--official-resolution-report",
        str(receipt),
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
    receipt = _clean_official_receipt(tmp_path, source, monkeypatch)
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
        "--target-soc", "Ascend910B4", "--cann-version", "9.0.0",
        "--provider", "llm",
        "--official-resolution-report", str(receipt),
    ])

    captured = capsys.readouterr()
    assert result == 4
    assert "生成门禁 1/2] fail" in captured.err
    assert "未读取 Skill" in captured.err
    assert not (tmp_path / "gate-one-fail").exists()


@pytest.mark.parametrize(
    ("provider_name", "model"),
    (("openai", "gpt-audited"), ("gateway", "org/model")),
)
def test_complete_gaps_generic_configuration_reaches_readiness_before_skill(
    tmp_path, monkeypatch, capsys, provider_name, model
):
    from numpilot import __main__ as cli

    source = _write_gap_source(tmp_path)
    receipt = _clean_official_receipt(tmp_path, source, monkeypatch)
    status = LlmConfigStatus(
        path=tmp_path / "llm.json",
        configured=True,
        missing=(),
        source="user-config",
        model=model,
        provider=provider_name,
        token_present=True,
        protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        response_model_aliases=(),
    )
    calls = []

    def fail_readiness(*_args, **_kwargs):
        calls.append("readiness")
        raise LlmHttpError(503)

    def forbidden_skill(*_args, **_kwargs):
        raise AssertionError("Skill must not be read before readiness passes")

    monkeypatch.setattr(cli, "_load_llm_or_report", lambda: status)
    monkeypatch.setattr(cli, "_run_llm_readiness_gate", fail_readiness)
    monkeypatch.setattr(cli, "resolve_skill_bundle", forbidden_skill)

    result = cli.main([
        "complete-gaps", str(source),
        "--out", str(tmp_path / "non-glm-cli"),
        "--target-soc", "Ascend910B4",
        "--cann-version", "9.0.0",
        "--provider", "llm",
        "--official-resolution-report", str(receipt),
    ])

    captured = capsys.readouterr()
    assert result == 4
    assert calls == ["readiness"]
    assert "HTTP status 503" in captured.err
    assert not (tmp_path / "non-glm-cli").exists()


def test_complete_gaps_skill_gate_failure_prevents_operator_generation(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli

    source = _write_gap_source(tmp_path)
    receipt = _clean_official_receipt(tmp_path, source, monkeypatch)
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
        "--target-soc", "Ascend910B4", "--cann-version", "9.0.0",
        "--provider", "llm",
        "--official-resolution-report", str(receipt),
    ])

    captured = capsys.readouterr()
    assert result == 2
    assert calls == [1]
    assert "生成门禁 2/2] fail" in captured.err
    assert "actual_remote_generation_call=no" in captured.err
    assert not (tmp_path / "gate-two-fail").exists()


def test_complete_gaps_cli_skips_remote_gates_when_there_is_no_gap(
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
    assert calls == []
    assert "生成门禁 1/2] pass" not in captured.out
    assert "status=no_gaps" in captured.out


def test_complete_gaps_cli_refuses_injected_existing_draft_in_production(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli

    source = _write_gap_source(tmp_path)
    out = tmp_path / "offline-reuse"
    receipt = _clean_official_receipt(tmp_path, source, monkeypatch)
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
        target_soc="Ascend910B4",
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
        "Ascend910B4",
        "--cann-version",
        "9.0.0",
        "--provider",
        "llm",
        "--official-resolution-report",
        str(receipt),
    ])

    assert result == 2
    terminal = capsys.readouterr()
    output = terminal.out
    assert calls == [1]
    assert "生成门禁 1/2] pass" in output
    assert "生成门禁 2/2] pass" in output
    assert "action=reused_existing" not in output
    assert "receipt-bound configured-model draft" in terminal.err
    assert "verified_new_remote_artifact=no" in terminal.err


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
    assert calls == []
    assert not list(tmp_path.glob(".%s-*-stage-*" % out.name))
    captured = capsys.readouterr()
    assert "本地输入校验失败" in captured.err
    assert "门禁 2/2] pass" not in captured.out


def test_generate_asnumpy_adapter_cli_uses_legacy_context_plan_without_real_model(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli
    from numpilot.gaps import provider as provider_module

    source_path = _write_gap_source(tmp_path)
    source = source_path.read_text(encoding="utf-8")
    spec = build_gap_specs(
        source,
        AnalysisAgent().analyze(source, str(source_path)),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )[0]
    spec_path = tmp_path / "gap-spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    artifact = tmp_path / "operator-artifact"
    adapter = artifact / "asnumpy_adapter/operator.json"
    adapter.parent.mkdir(parents=True)
    adapter_contract = '{"api_name":"cbrt"}\n'
    adapter.write_text(adapter_contract, encoding="utf-8")
    (artifact / "operator-artifact.json").write_text(
        json.dumps({"artifact_id": "draft-" + "a" * 20}),
        encoding="utf-8",
    )
    signed = SimpleNamespace(files={
        "asnumpy_adapter/operator.json": hashlib.sha256(
            adapter_contract.encode("utf-8")
        ).hexdigest(),
    })
    monkeypatch.setattr(
        provider_module, "_load_existing_draft", lambda *_args: signed
    )

    reference = tmp_path / "legacy-asnumpy"
    for relative in (
        "include/asnumpy/nn/activation.hpp",
        "csrc/nn/activation.cpp",
        "bindings/python/bind_nn.cpp",
        "bindings/python/module.cpp",
        "src/asnumpy/nn.py",
        "src/asnumpy/__init__.py",
        "bindings/python/CMakeLists.txt",
        "tests/asnumpy_tests/nn_tests/test_activation.py",
    ):
        target = reference / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("legacy fixture for %s\n" % relative, encoding="utf-8")

    observed = {}

    class StubProvider:
        def __init__(self, **_kwargs):
            pass

        def generate_asnumpy_adapter_candidate(self, output_root, **kwargs):
            observed.update(kwargs)
            candidate = Path(output_root) / "candidate"
            candidate.mkdir()
            (candidate / "stub.txt").write_text("stub\n", encoding="utf-8")
            return {
                "status": "generated_unverified",
                "adapter_artifact_id": "adapter-" + "b" * 20,
                "attempt_count": 8,
                "remote_receipts": [{}],
            }

    monkeypatch.setattr(provider_module, "LlmDraftProvider", StubProvider)
    monkeypatch.setattr(
        cli, "_load_llm_or_report", lambda: _configured_status(tmp_path)
    )
    monkeypatch.setattr(
        cli,
        "_run_llm_readiness_gate",
        lambda *_args, **_kwargs: {
            "schema_version": 1,
            "gate": "llm_readiness",
            "status": "pass",
            "receipt": _remote_receipt("NUMPILOT_LLM_READY."),
        },
    )
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
    output = tmp_path / "legacy-adapter"

    result = cli.main([
        "generate-asnumpy-adapter",
        "--spec", str(spec_path),
        "--artifact", str(artifact),
        "--asnumpy-reference", str(reference),
        "--out", str(output),
    ])

    captured = capsys.readouterr()
    assert result == 0
    assert "architecture=legacy_manual mode=manual" in captured.out
    assert len(observed["asnumpy_context"]) == 8
    assert observed["expected_paths"] == (
        "include/asnumpy/custom_ops/cbrt.hpp",
        "csrc/custom_ops/cbrt.cpp",
        "bindings/python/bind_custom_ops.cpp",
        "bindings/python/module.cpp",
        "src/asnumpy/custom_ops.py",
        "src/asnumpy/__init__.py",
        "bindings/python/CMakeLists.txt",
        "tests/asnumpy_tests/custom_ops_tests/test_cbrt.py",
    )
    preflight = json.loads(
        (output / "generation-preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["asnumpy_adapter_context_plan"]["architecture"] == (
        "legacy_manual"
    )
    assert len(preflight["asnumpy_adapter_context_plan"]["plan_sha256"]) == 64


def test_asnumpy_adapter_manifest_v1_identity_is_accepted():
    from numpilot import __main__ as cli

    spec = SimpleNamespace(
        api_name="softplus",
        qualified_api="operator.softplus",
        asnumpy_api="asnumpy.softplus",
    )
    manifest = {
        "schema": "asnumpy.adapter.manifest.v1",
        "operator": {
            "source_api": "operator.softplus",
            "public_api": "asnumpy.softplus",
        },
        "name_mapping": {
            "python": {
                "public_symbol": "softplus",
                "public_qualified_name": "asnumpy.softplus",
                "private_extension_symbol": "softplus",
            },
        },
    }

    assert cli._resolve_asnumpy_adapter_api_name(manifest, spec) == "softplus"


def test_asnumpy_adapter_manifest_v1_operator_nested_mapping_is_accepted():
    from numpilot import __main__ as cli

    spec = SimpleNamespace(
        api_name="softplus",
        qualified_api="operator.softplus",
        asnumpy_api="asnumpy.softplus",
    )
    manifest = {
        "schema": "asnumpy.adapter.manifest.v1",
        "operator": {
            "source_api": "operator.softplus",
            "public_api": "asnumpy.softplus",
            "name_mapping": {
                "python": {"public_symbol": "softplus"},
            },
        },
    }

    assert cli._resolve_asnumpy_adapter_api_name(manifest, spec) == "softplus"


def test_asnumpy_adapter_manifest_identity_rejects_conflicting_nested_fields():
    from numpilot import __main__ as cli

    spec = SimpleNamespace(
        api_name="softplus",
        qualified_api="operator.softplus",
        asnumpy_api="asnumpy.softplus",
    )
    manifest = {
        "schema": "asnumpy.adapter.manifest.v1",
        "operator": {
            "source_api": "operator.other",
            "public_api": "asnumpy.softplus",
        },
        "name_mapping": {"python": {"public_symbol": "softplus"}},
    }

    assert cli._resolve_asnumpy_adapter_api_name(manifest, spec) is None


def test_asnumpy_adapter_legacy_top_level_identity_remains_accepted():
    from numpilot import __main__ as cli

    spec = SimpleNamespace(
        api_name="cbrt",
        qualified_api="numpy.cbrt",
        asnumpy_api="asnumpy.cbrt",
    )

    assert cli._resolve_asnumpy_adapter_api_name(
        {"api_name": "cbrt", "producer_note": "legacy"}, spec
    ) == "cbrt"


def test_generate_asnumpy_adapter_cli_reports_http_request_without_receipt(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli
    from numpilot.gaps import provider as provider_module

    source_path = _write_gap_source(tmp_path)
    source = source_path.read_text(encoding="utf-8")
    spec = build_gap_specs(
        source,
        AnalysisAgent().analyze(source, str(source_path)),
        target_soc="ascend910b",
        cann_version="9.0.0",
    )[0]
    spec_path = tmp_path / "gap-spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    artifact = tmp_path / "operator-artifact-http"
    adapter = artifact / "asnumpy_adapter/operator.json"
    adapter.parent.mkdir(parents=True)
    adapter_contract = '{"api_name":"cbrt"}\n'
    adapter.write_text(adapter_contract, encoding="utf-8")
    (artifact / "operator-artifact.json").write_text(
        json.dumps({"artifact_id": "draft-" + "a" * 20}),
        encoding="utf-8",
    )
    signed = SimpleNamespace(files={
        "asnumpy_adapter/operator.json": hashlib.sha256(
            adapter_contract.encode("utf-8")
        ).hexdigest(),
    })
    monkeypatch.setattr(
        provider_module, "_load_existing_draft", lambda *_args: signed
    )

    reference = tmp_path / "legacy-asnumpy-http"
    for relative in (
        "include/asnumpy/nn/activation.hpp",
        "csrc/nn/activation.cpp",
        "bindings/python/bind_nn.cpp",
        "bindings/python/module.cpp",
        "src/asnumpy/nn.py",
        "src/asnumpy/__init__.py",
        "bindings/python/CMakeLists.txt",
        "tests/asnumpy_tests/nn_tests/test_activation.py",
    ):
        target = reference / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("legacy fixture for %s\n" % relative, encoding="utf-8")

    class HttpFailureProvider:
        def __init__(self, **_kwargs):
            pass

        def generate_asnumpy_adapter_candidate(self, *_args, **_kwargs):
            raise DraftRepairError(
                "remote adapter call failed: HTTP status 504",
                receipt_count=0,
                request_count=1,
            )

    monkeypatch.setattr(
        provider_module, "LlmDraftProvider", HttpFailureProvider
    )
    monkeypatch.setattr(
        cli, "_load_llm_or_report", lambda: _configured_status(tmp_path)
    )
    monkeypatch.setattr(
        cli,
        "_run_llm_readiness_gate",
        lambda *_args, **_kwargs: {
            "schema_version": 1,
            "gate": "llm_readiness",
            "status": "pass",
            "receipt": _remote_receipt("NUMPILOT_LLM_READY."),
        },
    )
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
    output = tmp_path / "legacy-adapter-http-failure"

    result = cli.main([
        "generate-asnumpy-adapter",
        "--spec", str(spec_path),
        "--artifact", str(artifact),
        "--asnumpy-reference", str(reference),
        "--out", str(output),
        "--max-attempts", "1",
    ])

    captured = capsys.readouterr()
    assert result == 2
    assert "HTTP status 504" in captured.err
    assert (
        "actual_remote_call=yes verified_new_adapter=no receipts=0"
        in captured.err
    )
    assert not output.exists()


@pytest.mark.parametrize("repair_profile", ("compile", "precision", "runtime"))
def test_repair_asnumpy_cli_passes_audit_profile_without_real_model(
    tmp_path, monkeypatch, repair_profile
):
    from numpilot import __main__ as cli
    from numpilot.gaps import provider as provider_module

    reference = tmp_path / "asnumpy-reference"
    for relative in (
        "scripts/gen_custom_ops.py",
        "csrc/custom_ops/registry/angle_v2/op.yaml",
        "tests/asnumpy_tests/custom_ops_tests/test_angle_v2.py",
        "CMakeLists.txt",
        "cmake/CustomOps.cmake",
        "csrc/custom_ops/CMakeLists.txt",
        "include/asnumpy/custom_ops/loader.hpp",
        "csrc/custom_ops/loader.cpp",
    ):
        path = reference / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    diagnostic = tmp_path / "diagnostic.txt"
    diagnostic.write_text("local failure\n", encoding="utf-8")
    observed = {}
    from numpilot.asnumpy_adapter_context import plan_asnumpy_adapter_context

    current_plan = plan_asnumpy_adapter_context(reference, "cbrt").manifest()
    current_hashes = {
        relative: ("0" if index == 0 else "1") * 64
        for index, relative in enumerate(current_plan["expected_overlay_paths"])
    }

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
    monkeypatch.setattr(
        provider_module,
        "_load_asnumpy_adapter_report",
        lambda _root: (
            {
                "schema_version": 4,
                "asnumpy_adapter_context_plan": current_plan,
                "asnumpy_context_sha256": current_plan["context_sha256"],
            },
            b"{}",
            current_hashes,
        ),
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
    assert observed["adapter_context_plan"] == current_plan
    assert out.is_dir()


def test_repair_asnumpy_cli_rejects_context_drift_before_model_or_skill(
    tmp_path, monkeypatch, capsys
):
    from numpilot import __main__ as cli
    from numpilot.asnumpy_adapter_context import plan_asnumpy_adapter_context
    from numpilot.gaps import provider as provider_module

    reference = tmp_path / "asnumpy-reference"
    for relative in (
        "scripts/gen_custom_ops.py",
        "csrc/custom_ops/registry/angle_v2/op.yaml",
        "tests/asnumpy_tests/custom_ops_tests/test_angle_v2.py",
        "CMakeLists.txt",
        "cmake/CustomOps.cmake",
        "csrc/custom_ops/CMakeLists.txt",
        "include/asnumpy/custom_ops/loader.hpp",
        "csrc/custom_ops/loader.cpp",
    ):
        target = reference / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture\n", encoding="utf-8")
    current_plan = plan_asnumpy_adapter_context(reference, "cbrt").manifest()
    stale_plan = json.loads(json.dumps(current_plan))
    first_context = next(iter(stale_plan["context_sha256"]))
    stale_plan["context_sha256"][first_context] = "f" * 64
    diagnostic = tmp_path / "diagnostic.txt"
    diagnostic.write_text("local failure\n", encoding="utf-8")
    hashes = {
        relative: str(index) * 64
        for index, relative in enumerate(
            current_plan["expected_overlay_paths"], start=1
        )
    }
    monkeypatch.setattr(
        provider_module,
        "_load_asnumpy_adapter_report",
        lambda _root: (
            {
                "schema_version": 4,
                "asnumpy_adapter_context_plan": stale_plan,
                "asnumpy_context_sha256": stale_plan["context_sha256"],
            },
            b"{}",
            hashes,
        ),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("context drift must block before model or Skill")

    monkeypatch.setattr(cli, "_load_llm_or_report", forbidden)
    monkeypatch.setattr(cli, "load_asnumpy_integration_skill", forbidden)
    output = tmp_path / "must-not-exist"

    result = cli.main([
        "repair-asnumpy-adapter",
        "--base-adapter", str(tmp_path / "base-adapter"),
        "--asnumpy-reference", str(reference),
        "--diagnostic-file", str(diagnostic),
        "--path", "csrc/custom_ops/registry/cbrt/op.yaml",
        "--out", str(output),
    ])

    captured = capsys.readouterr()
    assert result == 2
    assert "changed" in captured.err
    assert "actual_remote_call=no" in captured.err
    assert not output.exists()


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


def test_continue_on_error_completes_independent_gaps_and_redacts_failure(
    tmp_path,
):
    """An opted-in multi-gap run leaves a useful, non-secret report."""

    source = tmp_path / "two-gaps.py"
    source.write_text(
        "import numpy as np\n"
        "def forward(x):\n"
        "    first = np.cbrt(x)\n"
        "    second = np.spacing(x)\n"
        "    return second\n",
        encoding="utf-8",
    )
    calls = []
    secret = "model-response-secret-that-must-not-be-persisted"

    def chat(_system, _user):
        calls.append(1)
        # The first gap receives an invalid response; the second remains
        # independently generatable.  The provider only raises a sanitized
        # DraftGenerationError after its bounded attempt.
        return secret if len(calls) == 1 else _valid_bundle()

    result = GapCompletionPipeline(
        provider=LlmDraftProvider(chat=chat, max_attempts=1)
    ).run(
        str(source),
        str(tmp_path / "out"),
        target_soc="ascend910b",
        cann_version="9.0.0",
        continue_on_error=True,
    )

    assert result.continue_on_error is True
    assert result.status == "generated_unverified_with_failures"
    assert len(result.specs) == 2
    assert len(result.drafts) == 1
    assert len(result.generation_failures) == 1
    failure = result.generation_failures[0]
    assert failure["qualified_api"] == "numpy.cbrt"
    assert failure["status"] == "generation_failed"
    assert failure["actual_remote_call"] is False
    assert "model-response-secret" not in json.dumps(result.to_dict())
    assert calls == [1, 1]

    report = json.loads(
        (tmp_path / "out" / "gap-report.json").read_text(encoding="utf-8")
    )
    assert report["continue_on_error"] is True
    assert report["generation_failures"] == [failure]
    failed_workspace = tmp_path / "out" / "gaps" / result.specs[0].gap_id
    assert (failed_workspace / "gap-spec.json").is_file()
    assert (failed_workspace / "GENERATION_REQUEST.md").is_file()
    assert not (failed_workspace / "draft").exists()


def test_continue_on_error_remains_opt_in_fail_fast_default(tmp_path):
    source = tmp_path / "two-gaps.py"
    source.write_text(
        "import numpy as np\n"
        "def forward(x):\n"
        "    first = np.cbrt(x)\n"
        "    second = np.spacing(x)\n"
        "    return second\n",
        encoding="utf-8",
    )
    calls = []

    def chat(_system, _user):
        calls.append(1)
        return "invalid bundle"

    with pytest.raises(GapCompletionError, match="draft generation failed"):
        GapCompletionPipeline(
            provider=LlmDraftProvider(chat=chat, max_attempts=1)
        ).run(
            str(source),
            str(tmp_path / "out"),
            target_soc="ascend910b",
            cann_version="9.0.0",
        )
    assert calls == [1]


def test_complete_gaps_cli_exposes_continue_on_error_switch(tmp_path, monkeypatch):
    from numpilot import __main__ as cli

    observed = {}

    def fake_complete(args):
        observed["continue_on_error"] = args.continue_on_error
        return 0

    monkeypatch.setattr(cli, "_cmd_complete_gaps", fake_complete)
    result = cli.main([
        "complete-gaps",
        str(tmp_path / "source.py"),
        "--continue-on-error",
    ])
    assert result == 0
    assert observed == {"continue_on_error": True}


def test_continue_on_error_extracts_only_redacted_http_status(tmp_path):
    source = tmp_path / "two-gaps.py"
    source.write_text(
        "import numpy as np\n"
        "def forward(x):\n"
        "    first = np.cbrt(x)\n"
        "    second = np.spacing(x)\n"
        "    return second\n",
        encoding="utf-8",
    )
    calls = []

    def chat(_system, _user):
        calls.append(1)
        if len(calls) == 1:
            raise LlmHttpError(503)
        return _valid_bundle()

    result = GapCompletionPipeline(
        provider=LlmDraftProvider(chat=chat, max_attempts=1)
    ).run(
        str(source),
        str(tmp_path / "out"),
        target_soc="ascend910b",
        cann_version="9.0.0",
        continue_on_error=True,
    )
    failure = result.generation_failures[0]
    assert failure["error_code"] == "remote_http_error"
    assert failure["http_status"] == 503
    # This test uses an injected callable, so the HTTP-shaped exception is
    # still a local test-double call rather than evidence of a remote request.
    assert failure["actual_remote_call"] is False
    assert "remote LLM HTTP status" not in json.dumps(failure)



def _valid_asc_manifest():
    return json.dumps({
        "schema": "asnumpy.adapter.manifest.v1",
        "name_mapping": {
            "python": {
                "module": "asnumpy",
                "public_symbol": "softplus",
                "public_qualified_name": "asnumpy.softplus",
            },
            "aclnn": {
                "call_style": "two-stage",
                "get_workspace_size": "aclnnSoftplusGetWorkspaceSize",
                "execute": "aclnnSoftplus",
                "symbol_resolution": {
                    "strategy": "explicit-handle-dlsym",
                    "library_soname": "libcust_opapi.so",
                    "global_symbol_call": False,
                },
            },
            "cpp_api": {
                "l2_get_workspace_size": "aclnnSoftplusGetWorkspaceSize",
                "l2_execute": "aclnnSoftplus",
            },
        },
    })


def test_validate_bundle_accepts_asc_two_stage_manifest():
    payload = json.dumps({
        "summary": "ASC implementation with ACLNN two-stage adapter.",
        "files": [
            {"path": "CMakeLists.txt", "content": "cmake_minimum_required(VERSION 3.16)\n"},
            {"path": "build.sh", "content": "#!/usr/bin/env bash\nset -euo pipefail\n"},
            {"path": "op_kernel/softplus.asc", "content": "// ASC source\n"},
            {"path": "tests/test_softplus.py", "content": "# NumPy oracle\n"},
            {"path": "asnumpy_adapter/softplus.json", "content": _valid_asc_manifest()},
        ],
    })
    summary, files = _validate_bundle(payload)
    assert "ASC implementation" in summary
    assert {str(path) for path, _content in files} == {
        "CMakeLists.txt",
        "build.sh",
        "op_kernel/softplus.asc",
        "tests/test_softplus.py",
        "asnumpy_adapter/softplus.json",
    }


def test_validate_bundle_rejects_asc_without_acl_nn_two_stage_contract():
    manifest = json.loads(_valid_asc_manifest())
    manifest["name_mapping"]["aclnn"]["call_style"] = "single-stage"
    payload = json.dumps({
        "summary": "Invalid ASC implementation.",
        "files": [
            {"path": "CMakeLists.txt", "content": "cmake_minimum_required(VERSION 3.16)\n"},
            {"path": "build.sh", "content": "#!/usr/bin/env bash\nset -euo pipefail\n"},
            {"path": "op_kernel/softplus.asc", "content": "// ASC source\n"},
            {"path": "tests/test_softplus.py", "content": "# NumPy oracle\n"},
            {"path": "asnumpy_adapter/softplus.json", "content": json.dumps(manifest)},
        ],
    })
    with pytest.raises(DraftGenerationError, match="ACLNN two-stage ASC interface"):
        _validate_bundle(payload)


def test_validate_bundle_rejects_asc_without_explicit_symbol_resolution():
    manifest = json.loads(_valid_asc_manifest())
    del manifest["name_mapping"]["aclnn"]["symbol_resolution"]
    payload = json.dumps({
        "summary": "ASC implementation without symbol isolation.",
        "files": [
            {"path": "CMakeLists.txt", "content": "cmake_minimum_required(VERSION 3.16)\n"},
            {"path": "build.sh", "content": "#!/usr/bin/env bash\nset -euo pipefail\n"},
            {"path": "op_kernel/softplus.asc", "content": "// ASC source\n"},
            {"path": "tests/test_softplus.py", "content": "# NumPy oracle\n"},
            {"path": "asnumpy_adapter/softplus.json", "content": json.dumps(manifest)},
        ],
    })
    with pytest.raises(
        DraftGenerationError,
        match="no AsNumpy manifest declares a valid ACLNN two-stage ASC interface",
    ):
        _validate_bundle(payload)


def test_validate_bundle_rejects_asc_when_cpp_acl_nn_mapping_drifts():
    manifest = json.loads(_valid_asc_manifest())
    manifest["name_mapping"]["cpp_api"] = {
        "l2_get_workspace_size": "aclnnSoftplusGetWorkspaceSize",
        "l2_execute": "aclnnOther",
    }
    payload = json.dumps({
        "summary": "ASC implementation with a drifted C++ mapping.",
        "files": [
            {"path": "CMakeLists.txt", "content": "cmake_minimum_required(VERSION 3.16)\n"},
            {"path": "build.sh", "content": "#!/usr/bin/env bash\nset -euo pipefail\n"},
            {"path": "op_kernel/softplus.asc", "content": "// ASC source\n"},
            {"path": "tests/test_softplus.py", "content": "# NumPy oracle\n"},
            {"path": "asnumpy_adapter/softplus.json", "content": json.dumps(manifest)},
        ],
    })
    with pytest.raises(
        DraftGenerationError,
        match="no AsNumpy manifest declares a valid ACLNN two-stage ASC interface",
    ):
        _validate_bundle(payload)


def test_validate_bundle_rejects_asc_with_multiple_manifests():
    payload = json.dumps({
        "summary": "ASC implementation with ambiguous manifests.",
        "files": [
            {"path": "CMakeLists.txt", "content": "cmake_minimum_required(VERSION 3.16)\n"},
            {"path": "build.sh", "content": "#!/usr/bin/env bash\nset -euo pipefail\n"},
            {"path": "op_kernel/softplus.asc", "content": "// ASC source\n"},
            {"path": "tests/test_softplus.py", "content": "# NumPy oracle\n"},
            {
                "path": "asnumpy_adapter/softplus.json",
                "content": _valid_asc_manifest(),
            },
            {
                "path": "asnumpy_adapter/other.json",
                "content": _valid_asc_manifest(),
            },
        ],
    })
    with pytest.raises(
        DraftGenerationError,
        match="requires exactly one AsNumpy manifest",
    ):
        _validate_bundle(payload)
