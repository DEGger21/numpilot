import concurrent.futures
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import stat

import pytest

import numpilot.gaps.candidate_materializer as materializer
from numpilot.gaps.candidate_materializer import (
    CandidateMaterializationError,
    EXTERNAL_INSTALLED_CANN_TERMS_ID,
    materialize_official_binary_candidates,
)
from numpilot.gaps.installed_cann import (
    ACLNN_ADAPTER_CANDIDATES,
    CATALOG_REVISION,
)
from numpilot.gaps.official_resolution import (
    INSTALLED_ADAPTER_ROUTE,
    NEW_KERNEL_ROUTE,
    OFFICIAL_SOURCE_ROUTE,
    OfficialGapResolution,
    OfficialOperatorResolutionReport,
    load_official_resolution_report,
    write_official_resolution_report,
)
from numpilot.gaps.official_adapter import (
    OFFICIAL_REQUIRED_SYMBOLS_REF,
    OFFICIAL_SELECTED_HEADER_REF,
    OFFICIAL_SELECTED_LIBRARY_REF,
    prepare_official_candidate_adapter_input,
)
from numpilot.gaps.models import GapSpec
from numpilot.gaps.official_source import SOURCE_SEARCH_CONTRACT_REVISION
from numpilot.gaps.recovery import (
    OperatorCandidateArtifact,
    RecoveryPhase,
    RecoveryPolicyProfile,
    RecoveryStore,
    StageReceiptKind,
    StageReceiptScope,
    StageReceiptVerdict,
    recovery_policy_for_profile,
)
from numpilot.gaps.recovery_ingest import ingest_official_recovery


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "examples" / "ops_nn_matmul_inference.py"
EVIDENCE = REPO / "tests" / "fixtures" / "ops-nn-recovery"
MIGRATION = EVIDENCE / "pre-integration-gap-v2" / "script-migration-report.json"
GAP_REQUEST = EVIDENCE / "pre-integration-gap-v2" / "gap-request.json"
BASE_OFFICIAL = EVIDENCE / "official-resolution-linkage-v2.json"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reviewed_matmul():
    return next(
        item
        for item in ACLNN_ADAPTER_CANDIDATES
        if item.qualified_api == "numpy.matmul"
    )


def _clean_installed_miss(base_decision):
    candidate = _reviewed_matmul()
    return {
        "schema_version": 1,
        "status": "no_match",
        "qualified_api": base_decision.qualified_api,
        "expected_cann_version": "9.0.0",
        "expected_target_soc": "Ascend910B4",
        "reason_code": "candidate_artifacts_absent",
        "reason": "reviewed fixture has no installed artifacts",
        "adapter_candidate": False,
        "semantic_equivalence": "unverified",
        "required_next_gate": None,
        "catalog_revision": CATALOG_REVISION,
        "candidate": candidate.to_dict(),
        "candidate_mapping_sha256": candidate.mapping_sha256,
        "selected_header": None,
        "selected_library": None,
        "required_symbols": list(candidate.required_symbols),
        "missing_symbols": list(candidate.required_symbols),
        "alternatives": [],
        "provenance": "installed_cann_toolkit_read_only_inventory",
    }


def _source_candidate(base_report, base_decision):
    return {
        "schema_version": 1,
        "provider": "ops_nn_source",
        "status": "ambiguous",
        "reason": "fixture source candidate requires audit",
        "requested": {
            "cann_version": base_report.cann_version,
            "target_soc": base_report.target_soc,
            "op_name": "matmul",
            "qualified_api": base_decision.qualified_api,
        },
        "source_ref": "refs/tags/v9.0.0",
        "source_commit": "1" * 40,
        "source_tree": "2" * 40,
        "candidates": [{"candidate_id": "3" * 64}],
        "selected_candidate_id": "",
        "requires_audit": True,
        "resolution_kind": "candidate",
        "search_contract_revision": SOURCE_SEARCH_CONTRACT_REVISION,
        "searched_aliases": ["matmul", "mat_mul", "mat_mul_v3"],
        "negative_search_complete": False,
        "activation_allowed": False,
    }


def _source_clean_miss(base_report, base_decision):
    return {
        "schema_version": 1,
        "provider": "ops_nn_source",
        "status": "no_match",
        "reason": "fixture reviewed source miss",
        "requested": {
            "cann_version": base_report.cann_version,
            "target_soc": base_report.target_soc,
            "op_name": "matmul",
            "qualified_api": base_decision.qualified_api,
        },
        "source_ref": "refs/tags/v9.0.0",
        "source_commit": "1" * 40,
        "source_tree": "2" * 40,
        "candidates": [],
        "selected_candidate_id": "",
        "requires_audit": False,
        "resolution_kind": "none",
        "search_contract_revision": SOURCE_SEARCH_CONTRACT_REVISION,
        "searched_aliases": ["matmul", "mat_mul", "mat_mul_v3"],
        "negative_search_complete": True,
        "activation_allowed": False,
    }


def _make_report(tmp_path: Path, *, route=INSTALLED_ADAPTER_ROUTE):
    base = load_official_resolution_report(str(BASE_OFFICIAL))
    base_decision = base.decisions[0]
    cann_root = tmp_path / "cann-9.0.0"
    header = cann_root / "aarch64-linux" / "include" / "aclnnop" / "aclnn_matmul.h"
    library = cann_root / "aarch64-linux" / "lib64" / "libopapi_nn.so"
    header.parent.mkdir(parents=True)
    library.parent.mkdir(parents=True)
    header.write_bytes(b"fixture aclnn matmul header\n")
    library.write_bytes(b"fixture opapi binary\x00payload\n")
    candidate = _reviewed_matmul()
    symbols_payload = "".join(
        item + "\n" for item in sorted(candidate.required_symbols)
    ).encode("ascii")
    installed = dict(base_decision.installed_cann)
    installed.update(
        {
            "candidate": candidate.to_dict(),
            "candidate_mapping_sha256": candidate.mapping_sha256,
            "catalog_revision": CATALOG_REVISION,
            "selected_header": {
                "relative_path": header.relative_to(cann_root).as_posix(),
                "sha256": _sha256(header.read_bytes()),
                "size": header.stat().st_size,
                "declared_symbols": list(candidate.required_symbols),
            },
            "selected_library": {
                "relative_path": library.relative_to(cann_root).as_posix(),
                "sha256": _sha256(library.read_bytes()),
                "size": library.stat().st_size,
                "probe_tool": "trusted-fixture-nm",
                "aclnn_symbol_count": len(candidate.required_symbols),
                "aclnn_symbols_sha256": _sha256(symbols_payload),
            },
            "required_symbols": list(candidate.required_symbols),
        }
    )
    source = None
    allow_kernel = False
    needs_adapter = True
    if route == OFFICIAL_SOURCE_ROUTE:
        installed = _clean_installed_miss(base_decision)
        source = _source_candidate(base, base_decision)
        needs_adapter = False
    elif route == NEW_KERNEL_ROUTE:
        installed = _clean_installed_miss(base_decision)
        source = _source_clean_miss(base, base_decision)
        allow_kernel = True
        needs_adapter = False
    decision = OfficialGapResolution(
        gap_id=base_decision.gap_id,
        qualified_api=base_decision.qualified_api,
        gap_spec_sha256=base_decision.gap_spec_sha256,
        route=route,
        reason="candidate materializer fixture",
        installed_cann=installed,
        ops_nn_source=source,
        new_kernel_generation_allowed=allow_kernel,
        adapter_generation_required=needs_adapter,
    )
    status = (
        "new_kernel_generation_allowed"
        if route == NEW_KERNEL_ROUTE
        else "official_candidate_found"
    )
    report = OfficialOperatorResolutionReport(
        source_path=base.source_path,
        source_sha256=base.source_sha256,
        target_soc=base.target_soc,
        cann_version=base.cann_version,
        cann_root=str(cann_root),
        ops_nn_root=base.ops_nn_root,
        cann_osl_accepted=True,
        status=status,
        decisions=(decision,),
        installed_inventory=None,
    )
    return report, header, library


def _recovery_fixture(tmp_path, *, policy=None, route=INSTALLED_ADAPTER_ROUTE):
    inputs = tmp_path / "inputs"
    inputs.mkdir(mode=0o700)
    migration = inputs / "script-migration-report.json"
    gap_request = inputs / "gap-request.json"
    for source, target in ((MIGRATION, migration), (GAP_REQUEST, gap_request)):
        shutil.copyfile(source, target)
        target.chmod(0o600)
    report, header, library = _make_report(tmp_path, route=route)
    official = inputs / "official-resolution.json"
    write_official_resolution_report(report, str(official))
    state_dir = tmp_path / "recovery"
    ingest_official_recovery(
        SOURCE,
        migration,
        gap_request,
        official,
        state_dir,
        policy=policy,
    )
    return state_dir, report, header, library


def _authorized_policy():
    return recovery_policy_for_profile(
        RecoveryPolicyProfile.CANDIDATE_GENERATION
    )


def test_official_binary_materialization_rechecks_hashes_and_advances_once(
    tmp_path, monkeypatch
):
    state_dir, report, header, library = _recovery_fixture(
        tmp_path, policy=_authorized_policy()
    )
    calls = []

    def fresh(preparation, **kwargs):
        calls.append((preparation, kwargs))
        return report

    monkeypatch.setattr(materializer, "resolve_official_operators", fresh)
    output = tmp_path / "candidate-output"
    result = materialize_official_binary_candidates(state_dir, output)

    assert result.appended_revisions == 1
    assert result.recovery.state.phase is RecoveryPhase.CANDIDATE_READY
    assert result.recovery.next_action == "prepare_adapter_candidate"
    assert result.recovery.history_length == 5
    assert result.recovery.state.side_effects.subprocess_performed is True
    assert result.recovery.state.side_effects.network_access_performed is False
    assert result.recovery.state.side_effects.model_invocation_performed is False
    assert len(calls) == 1
    assert calls[0][1] == {
        "cann_root": report.cann_root,
        "ops_nn_root": report.ops_nn_root,
        "accept_cann_osl": True,
    }

    item = result.candidates[0]
    candidate_path = Path(item.artifact_ref.locator)
    candidate = OperatorCandidateArtifact.from_dict(
        json.loads(candidate_path.read_text(encoding="utf-8"))
    )
    assert candidate == item.candidate
    assert candidate.origin_kind.value == "official_binary"
    assert candidate.license.identifier == EXTERNAL_INSTALLED_CANN_TERMS_ID
    assert candidate.license.accepted is False
    assert candidate.activation_allowed is False
    refs = {ref.name: ref for ref in candidate.candidate_refs}
    assert refs[OFFICIAL_SELECTED_HEADER_REF].sha256 == _sha256(
        header.read_bytes()
    )
    assert refs[OFFICIAL_SELECTED_LIBRARY_REF].sha256 == _sha256(
        library.read_bytes()
    )
    assert OFFICIAL_REQUIRED_SYMBOLS_REF in refs
    gap_spec = GapSpec.from_dict(
        json.loads(GAP_REQUEST.read_text(encoding="utf-8"))["gap_specs"][0]
    )
    adapter_input = prepare_official_candidate_adapter_input(
        candidate, gap_spec, report.decisions[0]
    )
    assert adapter_input.manifest["origin_kind"] == "official_binary"
    assert adapter_input.model_invocation_allowed is False

    terms = json.loads(
        Path(item.license_ref.locator).read_text(encoding="utf-8")
    )
    assert terms["accepted"] is False
    assert terms["ops_nn_cann_osl_applied_to_installed_binary"] is False
    assert "CANN-Open-Software-License-Agreement" not in candidate.license.identifier
    for path in (
        output,
        output / "gaps",
        candidate_path.parent,
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    for path in output.rglob("*.json"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    receipt = next(
        receipt
        for receipt in result.recovery.state.stage_receipts
        if receipt.phase is RecoveryPhase.CANDIDATE_READY
    )
    assert receipt.kind is StageReceiptKind.OPERATOR_CANDIDATE
    assert receipt.scope is StageReceiptScope.GAPS
    assert receipt.verdict is StageReceiptVerdict.SATISFIED
    assert receipt.ref == item.artifact_ref


def test_receipt_only_policy_blocks_before_resolver_or_output(tmp_path, monkeypatch):
    state_dir, _report, _header, _library = _recovery_fixture(tmp_path)
    output = tmp_path / "must-not-exist"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("resolver must not run without frozen authorization")

    monkeypatch.setattr(materializer, "resolve_official_operators", forbidden)
    with pytest.raises(CandidateMaterializationError) as failure:
        materialize_official_binary_candidates(state_dir, output)
    assert failure.value.code == "subprocess_not_authorized"
    assert not output.exists()
    assert RecoveryStore(state_dir).status().state.phase is RecoveryPhase.RESOLVED


def test_materialization_is_idempotent_and_freshly_rechecks_each_time(
    tmp_path, monkeypatch
):
    state_dir, report, _header, _library = _recovery_fixture(
        tmp_path, policy=_authorized_policy()
    )
    calls = []
    monkeypatch.setattr(
        materializer,
        "resolve_official_operators",
        lambda *_args, **_kwargs: calls.append(True) or report,
    )
    output = tmp_path / "candidate-output"
    first = materialize_official_binary_candidates(state_dir, output)
    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*.json")
    }
    second = materialize_official_binary_candidates(state_dir, output)
    after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*.json")
    }

    assert first.appended_revisions == 1
    assert second.appended_revisions == 0
    assert second.recovery == first.recovery
    assert before == after
    assert len(calls) == 2
    assert RecoveryStore(state_dir).status().history_length == 5


def test_concurrent_identical_materializers_compare_and_append_once(
    tmp_path, monkeypatch
):
    state_dir, report, _header, _library = _recovery_fixture(
        tmp_path, policy=_authorized_policy()
    )
    monkeypatch.setattr(
        materializer, "resolve_official_operators", lambda *_a, **_k: report
    )
    output = tmp_path / "candidate-output"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(
                lambda _index: materialize_official_binary_candidates(
                    state_dir, output
                ),
                range(2),
            )
        )

    assert sorted(item.appended_revisions for item in results) == [0, 1]
    assert results[0].recovery == results[1].recovery
    assert results[0].recovery.state.phase is RecoveryPhase.CANDIDATE_READY
    assert RecoveryStore(state_dir).status().history_length == 5


def test_fresh_resolution_drift_fails_before_output_or_state_advance(
    tmp_path, monkeypatch
):
    state_dir, report, _header, _library = _recovery_fixture(
        tmp_path, policy=_authorized_policy()
    )
    decision = replace(report.decisions[0], reason="fresh evidence changed")
    drifted = replace(report, decisions=(decision,))
    assert drifted.resolution_id != report.resolution_id
    monkeypatch.setattr(
        materializer, "resolve_official_operators", lambda *_a, **_k: drifted
    )
    output = tmp_path / "must-not-exist"

    with pytest.raises(CandidateMaterializationError) as failure:
        materialize_official_binary_candidates(state_dir, output)
    assert failure.value.code == "fresh_resolution_drift"
    assert not output.exists()
    assert RecoveryStore(state_dir).status().state.phase is RecoveryPhase.RESOLVED


def test_selected_file_is_rehashed_independently_after_fresh_resolution(
    tmp_path, monkeypatch
):
    state_dir, report, header, _library = _recovery_fixture(
        tmp_path, policy=_authorized_policy()
    )
    header.write_bytes(b"changed after the official receipt\n")
    monkeypatch.setattr(
        materializer, "resolve_official_operators", lambda *_a, **_k: report
    )
    output = tmp_path / "must-not-exist"

    with pytest.raises(CandidateMaterializationError) as failure:
        materialize_official_binary_candidates(state_dir, output)
    assert failure.value.code == "candidate_file_hash_mismatch"
    assert not output.exists()
    assert RecoveryStore(state_dir).status().state.phase is RecoveryPhase.RESOLVED


@pytest.mark.parametrize(
    "route,expected_code",
    (
        (OFFICIAL_SOURCE_ROUTE, "official_source_not_supported"),
        (NEW_KERNEL_ROUTE, "remote_llm_not_supported"),
    ),
)
def test_unimplemented_origins_fail_closed_without_resolver_output_or_advance(
    tmp_path, monkeypatch, route, expected_code
):
    state_dir, _report, _header, _library = _recovery_fixture(
        tmp_path, policy=_authorized_policy(), route=route
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unsupported route must stop before fresh materialization")

    monkeypatch.setattr(materializer, "resolve_official_operators", forbidden)
    output = tmp_path / "must-not-exist"
    with pytest.raises(CandidateMaterializationError) as failure:
        materialize_official_binary_candidates(state_dir, output)
    assert failure.value.code == expected_code
    assert not output.exists()
    assert RecoveryStore(state_dir).status().state.phase is RecoveryPhase.RESOLVED
