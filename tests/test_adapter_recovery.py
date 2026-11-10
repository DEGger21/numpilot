import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from numpilot.asnumpy_adapter_context import (
    plan_official_aclnn_adapter_context,
)
from numpilot.gaps.adapter_recovery import (
    AdapterRecoveryError,
    _official_adapter_api_name,
    ingest_codex_adapter_candidate,
    ingest_official_adapter_candidate,
)
from numpilot.gaps.official_adapter import (
    OFFICIAL_CANDIDATE_MAPPING_REF,
    OFFICIAL_REQUIRED_SYMBOLS_REF,
    OFFICIAL_SELECTED_HEADER_REF,
    OFFICIAL_SELECTED_LIBRARY_REF,
    OFFICIAL_SEMANTIC_REVIEW_REF,
    bind_official_adapter_context,
    prepare_official_candidate_adapter_input,
)
from numpilot.gaps.official_resolution import load_official_resolution_report
from numpilot.gaps.pipeline import load_script_gap_request
from numpilot.gaps.provider import (
    _adapter_artifact_id,
    _injected_generation_receipt,
)
from numpilot.gaps.recovery import (
    EvidenceRef,
    GapRecoveryState,
    LicenseBinding,
    RecoveryEvent,
    RecoveryOutcome,
    RecoveryPhase,
    RecoveryPolicy,
    RecoveryPolicyProfile,
    RecoveryStore,
    SideEffectTruth,
    StageReceiptKind,
    StageReceiptScope,
    StageReceiptVerdict,
    create_operator_candidate,
    create_stage_receipt,
    recovery_policy_for_profile,
)
from numpilot.gaps.recovery_ingest import ingest_official_recovery
from numpilot.llm_config import OPENAI_CHAT_COMPLETIONS_PROTOCOL
from tests.fixture_support import recovery_fixture_root


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "examples" / "ops_nn_matmul_inference.py"
EVIDENCE = recovery_fixture_root()
MIGRATION = EVIDENCE / "pre-integration-gap-v2" / "script-migration-report.json"
GAP_REQUEST = EVIDENCE / "pre-integration-gap-v2" / "gap-request.json"
OFFICIAL = EVIDENCE / "official-resolution-linkage-v2.json"

OFFICIAL_CONTEXT_PATHS = (
    "include/asnumpy/linalg/product.hpp",
    "csrc/linalg/product.cpp",
    "csrc/linalg/CMakeLists.txt",
    "bindings/python/bind_linalg.cpp",
    "bindings/python/CMakeLists.txt",
    "src/asnumpy/linalg/direct.py",
    "src/asnumpy/linalg/__init__.py",
    "tests/asnumpy_tests/linalg_tests/test_product.py",
    "src/asnumpy/utils.py",
    "bindings/python/bind_utils.cpp",
)


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_ref(name: str, path: Path) -> EvidenceRef:
    return EvidenceRef(
        name,
        str(path.absolute()),
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _find_ref(refs, name):
    return next(item for item in refs if item.name == name)


def _write_private_json(path: Path, value) -> EvidenceRef:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = _canonical_bytes(value) + b"\n"
    path.write_bytes(payload)
    path.chmod(0o600)
    return _file_ref(path.stem.replace("-", "_"), path)


def _remote_receipt(content: str):
    return {
        "schema_version": 1,
        "utc_start": "2026-08-27T12:00:00Z",
        "utc_end": "2026-08-27T12:00:01Z",
        "http_status": 200,
        "requested_model": "glm-audited",
        "response_model": "glm-audited",
        "response_id": "adapter-response",
        "response_created": 1787832000,
        "request_body_sha256": "1" * 64,
        "response_body_sha256": "2" * 64,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "request_id": "adapter-request",
        "usage": {"total_tokens": 101},
        "tls_verification": "enabled",
    }


def _policy(*, network=True, model=True):
    return RecoveryPolicy(
        fallback_policy="reject",
        official_first=True,
        subprocess_allowed=True,
        network_allowed=network,
        system_write_allowed=False,
        model_allowed=model,
        asnumpy_import_allowed=False,
        build_allowed=False,
        execution_allowed=False,
        activation_allowed=False,
    )


def _candidate_ready_fixture(tmp_path: Path, *, policy=None):
    inputs = tmp_path / "inputs"
    inputs.mkdir(mode=0o700)
    migration = inputs / "script-migration-report.json"
    gap_request = inputs / "gap-request.json"
    official = inputs / "official-resolution.json"
    for source, target in (
        (MIGRATION, migration),
        (GAP_REQUEST, gap_request),
        (OFFICIAL, official),
    ):
        shutil.copyfile(source, target)
        target.chmod(0o600)
    state_dir = tmp_path / "recovery"
    ingest_official_recovery(
        SOURCE,
        migration,
        gap_request,
        official,
        state_dir,
        policy=(
            recovery_policy_for_profile(RecoveryPolicyProfile.CANDIDATE_GENERATION)
            if policy is None
            else policy
        ),
    )
    store = RecoveryStore(state_dir)
    status = store.status()
    gap = status.state.gaps[0]
    spec = load_script_gap_request(
        str(gap_request),
        str(SOURCE),
        target_soc=status.state.target.target_soc,
        cann_version=status.state.target.cann_version,
    ).specs[0]
    decision = load_official_resolution_report(str(official)).decisions[0]
    installed = decision.installed_cann

    candidate_root = tmp_path / "operator-candidate"
    candidate_root.mkdir(mode=0o700)
    license_path = candidate_root / "external-terms.json"
    license_ref = _write_private_json(
        license_path,
        {
            "identifier": "LicenseRef-External-Installed-CANN-Terms",
            "accepted": False,
        },
    )
    license_ref = EvidenceRef(
        "external_installed_cann_terms", license_ref.locator, license_ref.sha256
    )
    mapping_ref = EvidenceRef(
        OFFICIAL_CANDIDATE_MAPPING_REF,
        str(official) + "#candidate_mapping",
        installed["candidate_mapping_sha256"],
    )
    header_ref = EvidenceRef(
        OFFICIAL_SELECTED_HEADER_REF,
        str(official) + "#selected_header",
        installed["selected_header"]["sha256"],
    )
    library_ref = EvidenceRef(
        OFFICIAL_SELECTED_LIBRARY_REF,
        str(official) + "#selected_library",
        installed["selected_library"]["sha256"],
    )
    symbols_ref = EvidenceRef(
        OFFICIAL_REQUIRED_SYMBOLS_REF,
        str(official) + "#required_symbols",
        _canonical_sha256({"required_symbols": installed["required_symbols"]}),
    )
    semantic_ref = EvidenceRef(
        OFFICIAL_SEMANTIC_REVIEW_REF,
        str(official) + "#semantic_review",
        _canonical_sha256(
            {"semantic_review": installed["candidate"]["semantic_review"]}
        ),
    )
    candidate = create_operator_candidate(
        origin_kind="official_binary",
        gap_id=gap.gap_id,
        qualified_api=spec.qualified_api,
        gap_spec_ref=_find_ref(gap.refs, "gap_spec"),
        resolution_decision_ref=_find_ref(gap.refs, "resolution_decision"),
        target_soc=status.state.target.target_soc,
        cann_version=status.state.target.cann_version,
        license=LicenseBinding(
            identifier="LicenseRef-External-Installed-CANN-Terms",
            accepted=False,
            ref=license_ref,
        ),
        provenance_refs=(mapping_ref, semantic_ref),
        candidate_refs=(header_ref, library_ref, symbols_ref),
    )
    candidate_path = candidate_root / "operator-candidate.json"
    candidate_payload = _canonical_bytes(candidate.to_dict()) + b"\n"
    candidate_path.write_bytes(candidate_payload)
    candidate_path.chmod(0o600)
    candidate_ref = EvidenceRef(
        "operator_candidate",
        str(candidate_path.absolute()),
        hashlib.sha256(candidate_payload).hexdigest(),
    )
    receipt = create_stage_receipt(
        phase=RecoveryPhase.CANDIDATE_READY,
        kind=StageReceiptKind.OPERATOR_CANDIDATE,
        scope=StageReceiptScope.GAPS,
        covered_gap_ids=(gap.gap_id,),
        binding=status.state.binding,
        parent_state_sha256=status.state_sha256,
        verdict=StageReceiptVerdict.SATISFIED,
        ref=candidate_ref,
    )
    next_gap = GapRecoveryState(
        gap_id=gap.gap_id,
        phase=RecoveryPhase.CANDIDATE_READY,
        outcome=RecoveryOutcome.READY,
        refs=tuple(gap.refs) + (candidate_ref,),
    )
    status = store.append(RecoveryEvent.bound_to(
        status.state,
        phase=RecoveryPhase.CANDIDATE_READY,
        gaps=(next_gap,),
        stage_receipts=(receipt,),
        side_effects=SideEffectTruth(subprocess_performed=True),
        reason="test fixture materialized official candidate",
    ))
    prepared = prepare_official_candidate_adapter_input(candidate, spec, decision)
    return state_dir, status, candidate, candidate_path, prepared


def _adapter_output(tmp_path: Path, candidate, prepared):
    reference = tmp_path / "asnumpy-reference"
    for index, relative in enumerate(OFFICIAL_CONTEXT_PATHS, start=1):
        target = reference / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "reviewed fixture %d for %s\n" % (index, relative),
            encoding="utf-8",
        )
    (reference / "bindings/python/CMakeLists.txt").write_text(
        "target_link_libraries(_core\n"
        "    PUBLIC\n"
        "    ascendcl\n"
        "    runtime\n"
        "    nnopbase\n"
        "    opapi\n"
        "    Python::NumPy\n"
        ")\n"
        "target_link_libraries(_core\n"
        "    PUBLIC\n"
        "    array\n"
        "    linalg\n"
        ")\n",
        encoding="utf-8",
    )
    selected_library = prepared.selected_library
    plan = plan_official_aclnn_adapter_context(
        reference,
        "matmul",
        selected_library_relative_path=selected_library["relative_path"],
        selected_library_sha256=selected_library["sha256"],
    )
    ready = bind_official_adapter_context(prepared, plan, api_name="matmul")
    root = tmp_path / "adapter-output"
    candidate_tree = root / "candidate"
    candidate_tree.mkdir(parents=True)
    for index, relative in enumerate(plan.expected_overlay_paths, start=1):
        target = candidate_tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "generated adapter fixture %d for %s\n" % (index, relative),
            encoding="utf-8",
        )
    file_hashes = {
        relative: hashlib.sha256((candidate_tree / relative).read_bytes()).hexdigest()
        for relative in plan.expected_overlay_paths
    }
    response = "audited official MatMul adapter response"
    report = {
        "schema_version": 4,
        "status": "generated_unverified",
        "source_artifact_id": candidate.candidate_id,
        "model": "glm-audited",
        "provider": "glm",
        "protocol": OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        "response_model_aliases": [],
        "configuration_source": "user-config",
        "attempt_count": 1,
        "files_sha256": file_hashes,
        "adapter_contract_sha256": ready.adapter_contract_sha256,
        "asnumpy_context_sha256": dict(plan.context_sha256),
        "asnumpy_adapter_context_plan": plan.manifest(),
        "skill_bundle_sha256": "a" * 64,
        "integration_skill_files": ["skills/asnumpy-integration.md"],
        "response_sha256": [hashlib.sha256(response.encode("utf-8")).hexdigest()],
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
    report_path = root / "asnumpy-adapter-report.json"
    report_path.write_bytes(_canonical_bytes(report) + b"\n")
    return root, report


def _codex_adapter_output(tmp_path: Path, candidate, prepared):
    root, report = _adapter_output(tmp_path, candidate, prepared)
    response = "codex local window response"
    generation = {
        "mode": "injected_test_double",
        "provider": "codex-self",
        "model": "codex-local",
        "protocol": OPENAI_CHAT_COMPLETIONS_PROTOCOL,
        "response_model_aliases": [],
    }
    report.update({
        "schema_version": 5,
        "generation_mode": "injected_test_double",
        "model": "codex-local",
        "provider": "codex-self",
        "configuration_source": "injected",
        "response_sha256": [hashlib.sha256(response.encode("utf-8")).hexdigest()],
        "remote_receipts": [
            _injected_generation_receipt("system", "user", response, generation)
        ],
        "actual_remote_call": False,
    })
    report["adapter_artifact_id"] = _adapter_artifact_id(report)
    (root / "asnumpy-adapter-report.json").write_bytes(
        _canonical_bytes(report) + b"\n"
    )
    return root, report


def _mutate_report(root: Path, change):
    path = root / "asnumpy-adapter-report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    change(report)
    report["adapter_artifact_id"] = _adapter_artifact_id(report)
    path.write_bytes(_canonical_bytes(report) + b"\n")


def test_official_adapter_api_name_is_derived_from_candidate_api():
    assert _official_adapter_api_name("numpy.concatenate") == "concatenate"
    assert _official_adapter_api_name("numpy.matmul") == "matmul"
    assert _official_adapter_api_name("asnumpy.concatenate") == ""
    assert _official_adapter_api_name("numpy.") == ""
    assert _official_adapter_api_name("numpy.concatenate.extra") == (
        "concatenate.extra"
    )
    assert _official_adapter_api_name(None) == ""
    assert _official_adapter_api_name(7) == ""


def test_generated_official_adapter_advances_with_exact_typed_receipt(tmp_path):
    state_dir, before, candidate, candidate_path, prepared = (
        _candidate_ready_fixture(tmp_path)
    )
    adapter_root, report = _adapter_output(tmp_path, candidate, prepared)

    result = ingest_official_adapter_candidate(
        state_dir, candidate_path, adapter_root
    )

    assert result.appended_revisions == 1
    assert result.adapter_artifact_id == report["adapter_artifact_id"]
    assert result.recovery.history_length == before.history_length + 1
    assert result.recovery.state.phase is RecoveryPhase.ADAPTER_CANDIDATE
    assert result.recovery.next_action == "integrate_candidate"
    gap = result.recovery.state.gaps[0]
    assert gap.phase is RecoveryPhase.ADAPTER_CANDIDATE
    assert _find_ref(gap.refs, "asnumpy_adapter_candidate") == result.adapter_ref
    receipt = next(
        item
        for item in result.recovery.state.stage_receipts
        if item.phase is RecoveryPhase.ADAPTER_CANDIDATE
    )
    assert receipt.kind is StageReceiptKind.ASNUMPY_ADAPTER_CANDIDATE
    assert receipt.scope is StageReceiptScope.GAPS
    assert receipt.covered_gap_ids == (candidate.gap_id,)
    assert receipt.verdict is StageReceiptVerdict.SATISFIED
    assert receipt.ref == result.adapter_ref
    event = result.recovery.state.event_side_effects.to_dict()
    assert event["network_access_performed"] is True
    assert event["model_invocation_performed"] is True
    assert not any(
        value
        for name, value in event.items()
        if name not in {"network_access_performed", "model_invocation_performed"}
    )
    rendered = result.to_dict()
    assert rendered["model_invoked_by_ingest"] is False
    assert rendered["producer_receipt_revalidated"] is True


def test_local_codex_adapter_advances_without_remote_effects(tmp_path):
    state_dir, before, candidate, candidate_path, prepared = (
        _candidate_ready_fixture(tmp_path)
    )
    adapter_root, report = _codex_adapter_output(tmp_path, candidate, prepared)

    result = ingest_codex_adapter_candidate(
        state_dir, candidate_path, adapter_root
    )

    assert result.appended_revisions == 1
    assert result.generation_mode == "injected_test_double"
    assert result.adapter_artifact_id == report["adapter_artifact_id"]
    assert result.recovery.history_length == before.history_length + 1
    assert result.recovery.state.phase is RecoveryPhase.ADAPTER_CANDIDATE
    effects = result.recovery.state.event_side_effects.to_dict()
    assert effects["network_access_performed"] is False
    assert effects["model_invocation_performed"] is False
    rendered = result.to_dict()
    assert rendered["generation_mode"] == "injected_test_double"
    assert rendered["side_effects_performed"]["network_access_performed"] is False
    assert rendered["side_effects_performed"]["model_invocation_performed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("provider", "openai-compatible"),
        ("model", "deepseek-flash"),
        ("configuration_source", "llm.json"),
        ("actual_remote_call", True),
    ),
)
def test_local_codex_adapter_rejects_spoofed_identity(tmp_path, field, value):
    state_dir, _before, candidate, candidate_path, prepared = (
        _candidate_ready_fixture(tmp_path)
    )
    adapter_root, _report = _codex_adapter_output(tmp_path, candidate, prepared)
    before_status = RecoveryStore(state_dir).status()
    _mutate_report(adapter_root, lambda report: report.update({field: value}))

    with pytest.raises(AdapterRecoveryError) as failure:
        ingest_codex_adapter_candidate(state_dir, candidate_path, adapter_root)
    assert failure.value.code == "wrong_adapter_profile"
    after_status = RecoveryStore(state_dir).status()
    assert after_status == before_status


def test_local_codex_adapter_is_idempotent_and_rejects_mode_drift(tmp_path):
    state_dir, _before, candidate, candidate_path, prepared = (
        _candidate_ready_fixture(tmp_path)
    )
    adapter_root, _report = _codex_adapter_output(tmp_path, candidate, prepared)

    first = ingest_codex_adapter_candidate(state_dir, candidate_path, adapter_root)
    second = ingest_codex_adapter_candidate(state_dir, candidate_path, adapter_root)
    assert first.appended_revisions == 1
    assert second.appended_revisions == 0
    assert second.generation_mode == "injected_test_double"

    _mutate_report(
        adapter_root,
        lambda report: report.update({"generation_mode": "remote_llm"}),
    )
    with pytest.raises(AdapterRecoveryError) as failure:
        ingest_codex_adapter_candidate(state_dir, candidate_path, adapter_root)
    assert failure.value.code == "wrong_adapter_profile"


def test_ingest_is_idempotent_and_revalidates_output_each_time(tmp_path):
    state_dir, _status, candidate, candidate_path, prepared = (
        _candidate_ready_fixture(tmp_path)
    )
    adapter_root, _report = _adapter_output(tmp_path, candidate, prepared)

    first = ingest_official_adapter_candidate(state_dir, candidate_path, adapter_root)
    second = ingest_official_adapter_candidate(state_dir, candidate_path, adapter_root)

    assert first.appended_revisions == 1
    assert second.appended_revisions == 0
    assert second.recovery == first.recovery
    assert RecoveryStore(state_dir).status().history_length == 6

    overlay = adapter_root / "candidate/csrc/linalg/product.cpp"
    overlay.write_text("drifted after ingestion\n", encoding="utf-8")
    with pytest.raises(AdapterRecoveryError) as failure:
        ingest_official_adapter_candidate(state_dir, candidate_path, adapter_root)
    assert failure.value.code == "invalid_adapter_output"


def test_concurrent_identical_ingest_appends_once(tmp_path):
    state_dir, _status, candidate, candidate_path, prepared = (
        _candidate_ready_fixture(tmp_path)
    )
    adapter_root, _report = _adapter_output(tmp_path, candidate, prepared)

    call_count = 16
    with concurrent.futures.ThreadPoolExecutor(max_workers=call_count) as pool:
        results = tuple(pool.map(
            lambda _index: ingest_official_adapter_candidate(
                state_dir, candidate_path, adapter_root
            ),
            range(call_count),
        ))

    assert sorted(item.appended_revisions for item in results) == (
        [0] * (call_count - 1) + [1]
    )
    assert len({item.recovery.state_sha256 for item in results}) == 1
    assert RecoveryStore(state_dir).status().history_length == 6


@pytest.mark.parametrize(
    ("network", "model", "code"),
    (
        (False, True, "network_not_authorized"),
        (True, False, "model_not_authorized"),
    ),
)
def test_frozen_policy_must_authorize_reported_model_and_network_effects(
    tmp_path, network, model, code
):
    state_dir, before, candidate, candidate_path, prepared = (
        _candidate_ready_fixture(
            tmp_path, policy=_policy(network=network, model=model)
        )
    )
    adapter_root, _report = _adapter_output(tmp_path, candidate, prepared)

    with pytest.raises(AdapterRecoveryError) as failure:
        ingest_official_adapter_candidate(state_dir, candidate_path, adapter_root)
    assert failure.value.code == code
    assert RecoveryStore(state_dir).status() == before


def test_candidate_must_be_exact_content_bound_gap_reference(tmp_path):
    state_dir, before, candidate, candidate_path, prepared = (
        _candidate_ready_fixture(tmp_path)
    )
    adapter_root, _report = _adapter_output(tmp_path, candidate, prepared)
    candidate_path.write_bytes(candidate_path.read_bytes() + b" \n")

    with pytest.raises(AdapterRecoveryError) as failure:
        ingest_official_adapter_candidate(state_dir, candidate_path, adapter_root)
    assert failure.value.code == "candidate_binding_mismatch"
    assert RecoveryStore(state_dir).status() == before


def test_adapter_root_and_candidate_tree_symlinks_are_rejected(tmp_path):
    state_dir, before, candidate, candidate_path, prepared = (
        _candidate_ready_fixture(tmp_path)
    )
    adapter_root, _report = _adapter_output(tmp_path, candidate, prepared)
    linked = tmp_path / "linked-adapter-output"
    linked.symlink_to(adapter_root, target_is_directory=True)

    with pytest.raises(AdapterRecoveryError) as failure:
        ingest_official_adapter_candidate(state_dir, candidate_path, linked)
    assert failure.value.code == "unsafe_input"

    linked.unlink()
    overlay = adapter_root / "candidate/csrc/linalg/product.cpp"
    outside = tmp_path / "outside.cpp"
    outside.write_text("outside\n", encoding="utf-8")
    overlay.unlink()
    overlay.symlink_to(outside)
    with pytest.raises(AdapterRecoveryError) as failure:
        ingest_official_adapter_candidate(state_dir, candidate_path, adapter_root)
    assert failure.value.code == "invalid_adapter_output"
    assert RecoveryStore(state_dir).status() == before


def test_wrong_schema_profile_and_architecture_fail_closed(tmp_path):
    state_dir, before, candidate, candidate_path, prepared = (
        _candidate_ready_fixture(tmp_path)
    )
    adapter_root, _report = _adapter_output(tmp_path, candidate, prepared)

    def old_profile(report):
        report["schema_version"] = 3
        report.pop("asnumpy_adapter_context_plan")

    _mutate_report(adapter_root, old_profile)
    with pytest.raises(AdapterRecoveryError) as failure:
        ingest_official_adapter_candidate(state_dir, candidate_path, adapter_root)
    assert failure.value.code == "wrong_adapter_profile"

    shutil.rmtree(adapter_root)
    adapter_root, _report = _adapter_output(tmp_path, candidate, prepared)

    def wrong_architecture(report):
        plan = report["asnumpy_adapter_context_plan"]
        plan["architecture"] = "manifest_generator"
        plan["mode"] = "manifest"
        identity = {
            "schema_version": 2,
            "architecture": plan["architecture"],
            "mode": plan["mode"],
            "context_sha256": plan["context_sha256"],
            "expected_overlay_paths": plan["expected_overlay_paths"],
            "prompt_dependencies": plan["prompt_dependencies"],
        }
        plan["plan_sha256"] = _canonical_sha256(identity)

    _mutate_report(adapter_root, wrong_architecture)
    with pytest.raises(AdapterRecoveryError) as failure:
        ingest_official_adapter_candidate(state_dir, candidate_path, adapter_root)
    # A schema-2 prompt-dependency plan is official-route-only, so changing
    # its architecture now fails at the stricter plan-validation boundary.
    assert failure.value.code == "invalid_adapter_output"
    assert RecoveryStore(state_dir).status() == before


def test_source_artifact_and_overlay_hash_drift_are_rejected(tmp_path):
    state_dir, before, candidate, candidate_path, prepared = (
        _candidate_ready_fixture(tmp_path)
    )
    adapter_root, _report = _adapter_output(tmp_path, candidate, prepared)
    _mutate_report(
        adapter_root,
        lambda report: report.__setitem__(
            "source_artifact_id", "candidate-" + "0" * 20
        ),
    )
    with pytest.raises(AdapterRecoveryError) as failure:
        ingest_official_adapter_candidate(state_dir, candidate_path, adapter_root)
    assert failure.value.code == "adapter_source_mismatch"

    shutil.rmtree(adapter_root)
    adapter_root, _report = _adapter_output(tmp_path, candidate, prepared)
    report_path = adapter_root / "asnumpy-adapter-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["files_sha256"]["csrc/linalg/product.cpp"] = "f" * 64
    report["adapter_artifact_id"] = _adapter_artifact_id(report)
    report_path.write_bytes(_canonical_bytes(report) + b"\n")
    with pytest.raises(AdapterRecoveryError) as failure:
        ingest_official_adapter_candidate(state_dir, candidate_path, adapter_root)
    assert failure.value.code == "invalid_adapter_output"
    assert RecoveryStore(state_dir).status() == before


def _two_gap_candidate_store(tmp_path, base_status, candidate_path):
    """Replay a complete typed prefix with one additional waiting gap."""

    primary = base_status.state.gaps[0]
    candidate_ref = _find_ref(primary.refs, "operator_candidate")
    secondary_id = "waiting-gap"
    synthetic = {
        "gap_spec": EvidenceRef(
            "gap_spec", "/synthetic/waiting-gap-spec", "a" * 64
        ),
        "resolution_decision": EvidenceRef(
            "resolution_decision", "/synthetic/waiting-resolution", "b" * 64
        ),
        "operator_candidate": EvidenceRef(
            "operator_candidate", "/synthetic/waiting-candidate", "c" * 64
        ),
    }
    state_dir = tmp_path / "two-gap-recovery"
    store = RecoveryStore(state_dir)
    status = store.create(
        base_status.state.binding,
        gap_ids=(primary.gap_id, secondary_id),
        refs=base_status.state.refs,
    )
    stage_data = (
        (
            RecoveryPhase.ANALYZED,
            StageReceiptKind.ANALYSIS_REPORT,
            _find_ref(base_status.state.refs, "script_migration_report"),
        ),
        (
            RecoveryPhase.GAP_PREPARED,
            StageReceiptKind.GAP_SPEC,
            _find_ref(base_status.state.refs, "gap_request"),
        ),
        (
            RecoveryPhase.RESOLVED,
            StageReceiptKind.OFFICIAL_RESOLUTION_DECISION,
            _find_ref(base_status.state.refs, "official_resolution"),
        ),
    )
    for phase, kind, receipt_ref in stage_data:
        gaps = []
        for item in status.state.gaps:
            refs = tuple(item.refs)
            if phase is RecoveryPhase.GAP_PREPARED:
                refs += (
                    _find_ref(primary.refs, "gap_spec")
                    if item.gap_id == primary.gap_id
                    else synthetic["gap_spec"],
                )
            elif phase is RecoveryPhase.RESOLVED:
                refs += (
                    _find_ref(primary.refs, "resolution_decision")
                    if item.gap_id == primary.gap_id
                    else synthetic["resolution_decision"],
                )
            gaps.append(GapRecoveryState(
                gap_id=item.gap_id,
                phase=phase,
                outcome=RecoveryOutcome.READY,
                refs=refs,
            ))
        receipt = create_stage_receipt(
            phase=phase,
            kind=kind,
            scope=StageReceiptScope.RUN,
            covered_gap_ids=(primary.gap_id, secondary_id),
            binding=status.state.binding,
            parent_state_sha256=status.state_sha256,
            verdict=StageReceiptVerdict.SATISFIED,
            ref=receipt_ref,
        )
        status = store.append(RecoveryEvent.bound_to(
            status.state,
            phase=phase,
            gaps=gaps,
            stage_receipts=(receipt,),
        ))

    candidate_gaps = []
    receipts = []
    for item in status.state.gaps:
        ref = (
            candidate_ref
            if item.gap_id == primary.gap_id
            else synthetic["operator_candidate"]
        )
        candidate_gaps.append(GapRecoveryState(
            gap_id=item.gap_id,
            phase=RecoveryPhase.CANDIDATE_READY,
            outcome=RecoveryOutcome.READY,
            refs=tuple(item.refs) + (ref,),
        ))
        receipts.append(create_stage_receipt(
            phase=RecoveryPhase.CANDIDATE_READY,
            kind=StageReceiptKind.OPERATOR_CANDIDATE,
            scope=StageReceiptScope.GAPS,
            covered_gap_ids=(item.gap_id,),
            binding=status.state.binding,
            parent_state_sha256=status.state_sha256,
            verdict=StageReceiptVerdict.SATISFIED,
            ref=ref,
        ))
    return store.append(RecoveryEvent.bound_to(
        status.state,
        phase=RecoveryPhase.CANDIDATE_READY,
        gaps=candidate_gaps,
        stage_receipts=receipts,
        side_effects=SideEffectTruth(subprocess_performed=True),
    )), state_dir, Path(candidate_ref.locator)


def test_multi_gap_global_phase_remains_at_slowest_gap(tmp_path):
    _single_dir, base_status, candidate, candidate_path, prepared = (
        _candidate_ready_fixture(tmp_path)
    )
    two_status, state_dir, candidate_path = _two_gap_candidate_store(
        tmp_path, base_status, candidate_path
    )
    adapter_root, _report = _adapter_output(tmp_path, candidate, prepared)

    result = ingest_official_adapter_candidate(
        state_dir, candidate_path, adapter_root
    )

    assert result.appended_revisions == 1
    assert result.recovery.history_length == two_status.history_length + 1
    assert result.recovery.state.phase is RecoveryPhase.CANDIDATE_READY
    assert result.recovery.next_action == "prepare_adapter_candidate"
    phases = {item.gap_id: item.phase for item in result.recovery.state.gaps}
    assert phases[candidate.gap_id] is RecoveryPhase.ADAPTER_CANDIDATE
    assert phases["waiting-gap"] is RecoveryPhase.CANDIDATE_READY
