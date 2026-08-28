import concurrent.futures
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import threading

import pytest

import numpilot.gaps as gaps_api
from numpilot.gaps.recovery import (
    EvidenceRef,
    GapRecoveryState,
    LicenseBinding,
    OperatorCandidateArtifact,
    OperatorCandidateOrigin,
    RecoveryBinding,
    RecoveryError,
    RecoveryEvent,
    RecoveryOutcome,
    RecoveryPhase,
    RecoveryPolicy,
    RecoveryPolicyProfile,
    RecoveryRun,
    RecoveryRuntime,
    RecoverySource,
    RecoveryStore,
    RecoveryTarget,
    SideEffectTruth,
    StageReceiptKind,
    StageReceiptRef,
    StageReceiptScope,
    StageReceiptVerdict,
    create_operator_candidate,
    create_stage_receipt,
    load_recovery_status,
    operator_candidate_sha256,
    recovery_binding_sha256,
    recovery_policy_for_profile,
    reduce_recovery,
    stage_receipt_sha256,
    validate_operator_candidate_binding,
)


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ref(name: str, seed: str) -> EvidenceRef:
    return EvidenceRef(
        name=name,
        locator="/evidence/%s.json" % name,
        sha256=hashlib.sha256(seed.encode("utf-8")).hexdigest(),
    )


def _binding(tmp_path: Path, **policy_overrides: bool) -> RecoveryBinding:
    source = tmp_path / "case.py"
    source.write_text("import numpy as np\n", encoding="utf-8")
    policy = RecoveryPolicy(fallback_policy="reject", **policy_overrides)
    return RecoveryBinding(
        run=RecoveryRun("run-test-001"),
        source=RecoverySource(
            str(source), hashlib.sha256(source.read_bytes()).hexdigest()
        ),
        target=RecoveryTarget("Ascend910B3", "8.3.RC1"),
        runtime=RecoveryRuntime(
            backend="npu",
            audit_id="asnumpy-audit-test",
            identity_sha256=hashlib.sha256(b"runtime").hexdigest(),
        ),
        policy=policy,
    )


@pytest.mark.parametrize(
    ("profile", "expected_flags"),
    (
        (
            "receipt-only",
            (False, False, False, False, False, False, False, False),
        ),
        (
            "candidate-generation",
            (True, True, False, True, False, False, False, False),
        ),
        (
            "full-local-recovery",
            (True, True, True, True, True, True, True, True),
        ),
    ),
)
def test_named_recovery_policy_profiles_have_exact_stable_flags(
    profile, expected_flags
):
    policy = recovery_policy_for_profile(profile)
    enum_value = RecoveryPolicyProfile(profile)

    assert policy == recovery_policy_for_profile(enum_value)
    assert policy.fallback_policy == "reject"
    assert policy.official_first is True
    assert (
        policy.subprocess_allowed,
        policy.network_allowed,
        policy.system_write_allowed,
        policy.model_allowed,
        policy.asnumpy_import_allowed,
        policy.build_allowed,
        policy.execution_allowed,
        policy.activation_allowed,
    ) == expected_flags


def test_named_recovery_policy_default_and_invalid_names_fail_closed():
    assert recovery_policy_for_profile() == recovery_policy_for_profile(
        RecoveryPolicyProfile.RECEIPT_ONLY
    )
    for invalid in ("", "candidate_generation", "unknown", False, None):
        with pytest.raises(RecoveryError, match="recovery policy profile"):
            recovery_policy_for_profile(invalid)


def _progress_gap(
    old: GapRecoveryState,
    phase: RecoveryPhase,
    *,
    outcome: RecoveryOutcome = RecoveryOutcome.READY,
    refs=(),
    reason: str = "",
) -> GapRecoveryState:
    return GapRecoveryState(
        gap_id=old.gap_id,
        phase=phase,
        outcome=outcome,
        attempt_count=(
            old.attempt_count + 1
            if outcome is RecoveryOutcome.FAILED_ATTEMPT
            else old.attempt_count
        ),
        refs=tuple(old.refs) + tuple(refs),
        reason=reason,
    )


_RECEIPT_KIND = {
    RecoveryPhase.ANALYZED: StageReceiptKind.ANALYSIS_REPORT,
    RecoveryPhase.GAP_PREPARED: StageReceiptKind.GAP_SPEC,
    RecoveryPhase.RESOLVED: StageReceiptKind.OFFICIAL_RESOLUTION_DECISION,
    RecoveryPhase.CANDIDATE_READY: StageReceiptKind.OPERATOR_CANDIDATE,
    RecoveryPhase.ADAPTER_CANDIDATE: StageReceiptKind.ASNUMPY_ADAPTER_CANDIDATE,
    RecoveryPhase.INTEGRATED_CANDIDATE: StageReceiptKind.INTEGRATION_TRANSACTION,
    RecoveryPhase.QUALIFIED: StageReceiptKind.QUALIFICATION_REPORT,
    RecoveryPhase.ACTIVATED: StageReceiptKind.ACTIVATION_RECEIPT,
    RecoveryPhase.RESCANNED: StageReceiptKind.CAPABILITY_RESCAN_REPORT,
    RecoveryPhase.FINAL_MIGRATED: StageReceiptKind.SCRIPT_MIGRATION_REPORT,
    RecoveryPhase.VERIFIED: StageReceiptKind.SCRIPT_VERIFICATION_RECEIPT,
}


def _stage_receipt(
    state,
    parent_state_sha256: str,
    phase: RecoveryPhase,
    *,
    gap_ids=None,
    scope: StageReceiptScope = StageReceiptScope.RUN,
    verdict: StageReceiptVerdict = StageReceiptVerdict.SATISFIED,
    seed: str | None = None,
) -> StageReceiptRef:
    covered = (
        tuple(item.gap_id for item in state.gaps)
        if gap_ids is None
        else tuple(gap_ids)
    )
    label = seed or "%s-%d" % (phase.value, state.revision)
    return create_stage_receipt(
        phase=phase,
        kind=_RECEIPT_KIND[phase],
        scope=scope,
        covered_gap_ids=covered,
        binding=state.binding,
        parent_state_sha256=parent_state_sha256,
        verdict=verdict,
        ref=_ref("receipt_%s" % phase.value, label),
    )


def _advance_all(
    store: RecoveryStore,
    status,
    phase: RecoveryPhase,
    *,
    side_effects: SideEffectTruth | None = None,
):
    outcome = (
        RecoveryOutcome.COMPLETE
        if phase is RecoveryPhase.VERIFIED
        else RecoveryOutcome.READY
    )
    gaps = tuple(
        _progress_gap(item, phase, outcome=outcome) for item in status.state.gaps
    )
    receipt = _stage_receipt(status.state, status.state_sha256, phase)
    return store.append(
        RecoveryEvent.bound_to(
            status.state,
            phase=phase,
            outcome=outcome,
            gaps=gaps,
            stage_receipts=(receipt,),
            side_effects=side_effects,
        )
    )


def test_write_once_private_hash_chain_and_status(tmp_path):
    store = RecoveryStore(tmp_path / "recovery")
    first = store.create(_binding(tmp_path), gap_ids=("matmul-gap",))
    analysis = _ref("analysis", "analysis")
    analyzed_gap = _progress_gap(
        first.state.gaps[0], RecoveryPhase.ANALYZED, refs=(analysis,)
    )
    receipt = _stage_receipt(first.state, first.state_sha256, RecoveryPhase.ANALYZED)
    second = store.append(
        RecoveryEvent.bound_to(
            first.state,
            phase=RecoveryPhase.ANALYZED,
            refs=(analysis,),
            gaps=(analyzed_gap,),
            stage_receipts=(receipt,),
        )
    )

    root = tmp_path / "recovery"
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "states").stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "states" / "000001.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((root / "states" / "000002.json").stat().st_mode) == 0o600
    assert second.state.parent_state_sha256 == first.state_sha256
    assert second.history_length == 2
    assert second.next_action == "prepare_gap_specs"
    status_value = second.to_dict()
    assert status_value == second.to_dict()
    assert status_value["record_type"] == "numpilot_gap_recovery_status"
    assert status_value["state_sha256"] == second.state_sha256
    assert status_value["state"] == second.state.to_dict()
    assert second.state.stage_receipts == (receipt,)
    assert load_recovery_status(root) == second
    with pytest.raises(RecoveryError, match="state digest is inconsistent"):
        replace(
            second,
            state_sha256=hashlib.sha256(b"not-the-state").hexdigest(),
        )
    with pytest.raises(RecoveryError, match="already initialized"):
        store.create(first.state.binding, gap_ids=("matmul-gap",))


def test_status_waits_for_live_publication_but_rejects_orphaned_temp(
    tmp_path,
    monkeypatch,
):
    """A live writer is serialized; a temp left without a lock fails closed."""

    from numpilot.gaps import recovery as recovery_module

    store = RecoveryStore(tmp_path / "recovery")
    first = store.create(_binding(tmp_path), gap_ids=("gap-1",))
    gap = _progress_gap(first.state.gaps[0], RecoveryPhase.ANALYZED)
    receipt = _stage_receipt(
        first.state,
        first.state_sha256,
        RecoveryPhase.ANALYZED,
    )
    event = RecoveryEvent.bound_to(
        first.state,
        phase=RecoveryPhase.ANALYZED,
        gaps=(gap,),
        stage_receipts=(receipt,),
    )
    publication_open = threading.Event()
    release_publication = threading.Event()
    reader_started = threading.Event()
    original_link = recovery_module.os.link

    def paused_link(*args, **kwargs):
        source = args[0] if args else kwargs.get("src")
        if isinstance(source, str) and source.startswith(".state-"):
            publication_open.set()
            if not release_publication.wait(timeout=5):
                raise AssertionError("test did not release recovery publication")
        return original_link(*args, **kwargs)

    def read_status():
        reader_started.set()
        return store.status()

    monkeypatch.setattr(recovery_module.os, "link", paused_link)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(store.append, event)
        assert publication_open.wait(timeout=5)
        reader = pool.submit(read_status)
        assert reader_started.wait(timeout=5)
        try:
            done, _pending = concurrent.futures.wait((reader,), timeout=0.1)
            assert reader not in done
        finally:
            release_publication.set()
        written = writer.result(timeout=5)
        observed = reader.result(timeout=5)

    assert observed == written
    assert not tuple((tmp_path / "recovery/states").glob(".state-*.tmp"))

    orphan = tmp_path / "recovery/states/.state-orphan.tmp"
    orphan.write_text("interrupted\n", encoding="utf-8")
    orphan.chmod(0o600)
    with pytest.raises(RecoveryError, match="incomplete recovery publication"):
        store.status()


def test_hash_chain_detects_intermediate_tampering(tmp_path):
    store = RecoveryStore(tmp_path / "recovery")
    first = store.create(_binding(tmp_path), gap_ids=("gap-1",))
    gap = _progress_gap(first.state.gaps[0], RecoveryPhase.ANALYZED)
    receipt = _stage_receipt(first.state, first.state_sha256, RecoveryPhase.ANALYZED)
    store.append(
        RecoveryEvent.bound_to(
            first.state,
            phase=RecoveryPhase.ANALYZED,
            gaps=(gap,),
            stage_receipts=(receipt,),
        )
    )
    first_path = tmp_path / "recovery" / "states" / "000001.json"
    value = json.loads(first_path.read_text(encoding="utf-8"))
    value["reason"] = "tampered"
    first_path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.chmod(first_path, 0o600)

    with pytest.raises(RecoveryError, match="parent state digest"):
        store.status()


def test_noncanonical_tail_tampering_is_rejected(tmp_path):
    store = RecoveryStore(tmp_path / "recovery")
    store.create(_binding(tmp_path), gap_ids=("gap-1",))
    state_path = tmp_path / "recovery" / "states" / "000001.json"
    state_path.write_bytes(state_path.read_bytes() + b" \n")
    with pytest.raises(RecoveryError, match="not canonical"):
        store.status()


def test_reducer_rejects_skipped_phase_and_input_drift(tmp_path):
    initial = RecoveryStore(tmp_path / "recovery").create(
        _binding(tmp_path), gap_ids=("gap-1",)
    )
    skipped_gap = _progress_gap(
        initial.state.gaps[0], RecoveryPhase.GAP_PREPARED
    )
    skipped = RecoveryEvent.bound_to(
        initial.state,
        phase=RecoveryPhase.GAP_PREPARED,
        gaps=(skipped_gap,),
    )
    with pytest.raises(RecoveryError, match="advance by exactly one"):
        reduce_recovery(
            initial.state,
            skipped,
            parent_state_sha256=initial.state_sha256,
        )

    changed_binding = replace(
        initial.state.binding,
        target=RecoveryTarget("Ascend310P", "8.3.RC1"),
    )
    drift = RecoveryEvent(
        binding=changed_binding,
        phase=RecoveryPhase.CREATED,
        outcome=RecoveryOutcome.READY,
        gaps=initial.state.gaps,
    )
    with pytest.raises(RecoveryError, match="input drift"):
        reduce_recovery(
            initial.state,
            drift,
            parent_state_sha256=initial.state_sha256,
        )


def test_failed_attempt_increments_attempt_without_phase_advance(tmp_path):
    store = RecoveryStore(tmp_path / "recovery")
    initial = store.create(_binding(tmp_path), gap_ids=("gap-1",))
    failed_gap = _progress_gap(
        initial.state.gaps[0],
        RecoveryPhase.CREATED,
        outcome=RecoveryOutcome.FAILED_ATTEMPT,
        reason="analysis worker returned bounded failure evidence",
    )
    failed = store.append(
        RecoveryEvent.bound_to(
            initial.state,
            phase=RecoveryPhase.CREATED,
            outcome=RecoveryOutcome.FAILED_ATTEMPT,
            gaps=(failed_gap,),
            reason="analysis attempt failed",
        )
    )
    assert failed.state.phase is RecoveryPhase.CREATED
    assert failed.state.gaps[0].attempt_count == 1
    assert failed.next_action == "retry_analyze_source"

    illegal_gap = replace(
        failed_gap,
        phase=RecoveryPhase.ANALYZED,
        attempt_count=2,
    )
    with pytest.raises(RecoveryError, match="may not advance phase"):
        reduce_recovery(
            failed.state,
            RecoveryEvent.bound_to(
                failed.state,
                phase=RecoveryPhase.ANALYZED,
                outcome=RecoveryOutcome.FAILED_ATTEMPT,
                gaps=(illegal_gap,),
                reason="another failure",
            ),
            parent_state_sha256=failed.state_sha256,
        )


def test_side_effect_truth_requires_frozen_policy_authorization(tmp_path):
    initial = RecoveryStore(tmp_path / "recovery").create(
        _binding(tmp_path), gap_ids=("gap-1",)
    )
    with pytest.raises(RecoveryError, match="without frozen policy authorization"):
        RecoveryEvent.bound_to(
            initial.state,
            phase=RecoveryPhase.CREATED,
            gaps=initial.state.gaps,
            side_effects=SideEffectTruth(network_access_performed=True),
        )

    authorized_store = RecoveryStore(tmp_path / "authorized")
    authorized = authorized_store.create(
        _binding(tmp_path, network_allowed=True), gap_ids=("gap-1",)
    )
    failed_gap = _progress_gap(
        authorized.state.gaps[0],
        RecoveryPhase.CREATED,
        outcome=RecoveryOutcome.FAILED_ATTEMPT,
        reason="network lookup failed",
    )
    result = authorized_store.append(
        RecoveryEvent.bound_to(
            authorized.state,
            phase=RecoveryPhase.CREATED,
            outcome=RecoveryOutcome.FAILED_ATTEMPT,
            gaps=(failed_gap,),
            side_effects=SideEffectTruth(network_access_performed=True),
            reason="authorized lookup failed",
        )
    )
    assert result.state.side_effects.network_access_performed is True
    assert result.state.phase is RecoveryPhase.CREATED

    analyzed_gap = _progress_gap(
        result.state.gaps[0], RecoveryPhase.ANALYZED
    )
    receipt = _stage_receipt(
        result.state, result.state_sha256, RecoveryPhase.ANALYZED
    )
    later = authorized_store.append(
        RecoveryEvent.bound_to(
            result.state,
            phase=RecoveryPhase.ANALYZED,
            gaps=(analyzed_gap,),
            stage_receipts=(receipt,),
        )
    )
    assert later.state.side_effects.network_access_performed is True
    assert later.state.to_dict()["side_effects"][
        "network_access_performed"
    ] is True


def test_activation_truth_is_synchronized_with_activated_phase(tmp_path):
    store = RecoveryStore(tmp_path / "recovery")
    status = store.create(
        _binding(tmp_path, activation_allowed=True), gap_ids=("gap-1",)
    )

    early_gap = _progress_gap(status.state.gaps[0], RecoveryPhase.ANALYZED)
    with pytest.raises(RecoveryError, match="only when the resulting phase"):
        reduce_recovery(
            status.state,
            RecoveryEvent.bound_to(
                status.state,
                phase=RecoveryPhase.ANALYZED,
                gaps=(early_gap,),
                side_effects=SideEffectTruth(activation_performed=True),
            ),
            parent_state_sha256=status.state_sha256,
        )

    phases = (
        RecoveryPhase.ANALYZED,
        RecoveryPhase.GAP_PREPARED,
        RecoveryPhase.RESOLVED,
        RecoveryPhase.CANDIDATE_READY,
        RecoveryPhase.ADAPTER_CANDIDATE,
        RecoveryPhase.INTEGRATED_CANDIDATE,
        RecoveryPhase.QUALIFIED,
    )
    for phase in phases:
        status = _advance_all(store, status, phase)

    activated_gap = _progress_gap(
        status.state.gaps[0], RecoveryPhase.ACTIVATED
    )
    activation_receipt = _stage_receipt(
        status.state, status.state_sha256, RecoveryPhase.ACTIVATED
    )
    with pytest.raises(RecoveryError, match="must become true exactly"):
        reduce_recovery(
            status.state,
            RecoveryEvent.bound_to(
                status.state,
                phase=RecoveryPhase.ACTIVATED,
                gaps=(activated_gap,),
                stage_receipts=(activation_receipt,),
            ),
            parent_state_sha256=status.state_sha256,
        )

    status = store.append(
        RecoveryEvent.bound_to(
            status.state,
            phase=RecoveryPhase.ACTIVATED,
            gaps=(activated_gap,),
            stage_receipts=(activation_receipt,),
            side_effects=SideEffectTruth(activation_performed=True),
        )
    )
    assert status.state.side_effects.activation_performed is True

    rescanned = _advance_all(store, status, RecoveryPhase.RESCANNED)
    assert rescanned.state.side_effects.activation_performed is True


def test_stage_receipt_schema_is_strict_deterministic_and_phase_bound(tmp_path):
    binding = _binding(tmp_path)
    parent = hashlib.sha256(b"parent-state").hexdigest()
    receipt = create_stage_receipt(
        phase=RecoveryPhase.ANALYZED,
        kind=StageReceiptKind.ANALYSIS_REPORT,
        scope=StageReceiptScope.RUN,
        covered_gap_ids=("gap-b", "gap-a"),
        binding=binding,
        parent_state_sha256=parent,
        verdict=StageReceiptVerdict.SATISFIED,
        ref=_ref("analysis_receipt", "analysis-receipt"),
    )
    assert receipt.covered_gap_ids == ("gap-a", "gap-b")
    assert receipt.binding_sha256 == recovery_binding_sha256(binding)
    assert len(stage_receipt_sha256(receipt)) == 64
    assert StageReceiptRef.from_dict(receipt.to_dict()) == receipt

    unknown = receipt.to_dict()
    unknown["activation_allowed"] = True
    with pytest.raises(RecoveryError, match="unknown"):
        StageReceiptRef.from_dict(unknown)

    with pytest.raises(RecoveryError, match="kind does not match"):
        create_stage_receipt(
            phase=RecoveryPhase.ANALYZED,
            kind=StageReceiptKind.GAP_SPEC,
            scope=StageReceiptScope.RUN,
            covered_gap_ids=("gap-a", "gap-b"),
            binding=binding,
            parent_state_sha256=parent,
            verdict=StageReceiptVerdict.SATISFIED,
            ref=_ref("wrong_kind", "wrong-kind"),
        )

    with pytest.raises(RecoveryError, match="must not be empty"):
        create_stage_receipt(
            phase=RecoveryPhase.GAP_PREPARED,
            kind=StageReceiptKind.GAP_SPEC,
            scope=StageReceiptScope.GAPS,
            covered_gap_ids=(),
            binding=binding,
            parent_state_sha256=parent,
            verdict=StageReceiptVerdict.SATISFIED,
            ref=_ref("empty_coverage", "empty-coverage"),
        )

    with pytest.raises(RecoveryError, match="id does not match"):
        replace(receipt, receipt_id="stage-000000000000000000000000")


def test_phase_advance_requires_new_satisfied_head_bound_receipt(tmp_path):
    store = RecoveryStore(tmp_path / "recovery")
    initial = store.create(_binding(tmp_path), gap_ids=("gap-1",))
    analyzed_gap = _progress_gap(
        initial.state.gaps[0], RecoveryPhase.ANALYZED
    )

    with pytest.raises(RecoveryError, match="requires a new satisfied"):
        reduce_recovery(
            initial.state,
            RecoveryEvent.bound_to(
                initial.state,
                phase=RecoveryPhase.ANALYZED,
                gaps=(analyzed_gap,),
            ),
            parent_state_sha256=initial.state_sha256,
        )

    wrong_parent = _stage_receipt(
        initial.state,
        hashlib.sha256(b"another-head").hexdigest(),
        RecoveryPhase.ANALYZED,
        seed="wrong-parent",
    )
    with pytest.raises(RecoveryError, match="current head"):
        reduce_recovery(
            initial.state,
            RecoveryEvent.bound_to(
                initial.state,
                phase=RecoveryPhase.ANALYZED,
                gaps=(analyzed_gap,),
                stage_receipts=(wrong_parent,),
            ),
            parent_state_sha256=initial.state_sha256,
        )

    failed = _stage_receipt(
        initial.state,
        initial.state_sha256,
        RecoveryPhase.ANALYZED,
        verdict=StageReceiptVerdict.FAILED,
        seed="failed-analysis",
    )
    with pytest.raises(RecoveryError, match="requires a new satisfied"):
        reduce_recovery(
            initial.state,
            RecoveryEvent.bound_to(
                initial.state,
                phase=RecoveryPhase.ANALYZED,
                gaps=(analyzed_gap,),
                stage_receipts=(failed,),
            ),
            parent_state_sha256=initial.state_sha256,
        )

    satisfied = _stage_receipt(
        initial.state, initial.state_sha256, RecoveryPhase.ANALYZED
    )
    analyzed = store.append(
        RecoveryEvent.bound_to(
            initial.state,
            phase=RecoveryPhase.ANALYZED,
            gaps=(analyzed_gap,),
            stage_receipts=(satisfied,),
        )
    )
    prepared_gap = _progress_gap(
        analyzed.state.gaps[0], RecoveryPhase.GAP_PREPARED
    )
    with pytest.raises(RecoveryError, match="newly added"):
        reduce_recovery(
            analyzed.state,
            RecoveryEvent.bound_to(
                analyzed.state,
                phase=RecoveryPhase.GAP_PREPARED,
                gaps=(prepared_gap,),
                stage_receipts=(satisfied,),
            ),
            parent_state_sha256=analyzed.state_sha256,
        )


def test_multi_gap_receipt_coverage_matches_each_advance(tmp_path):
    store = RecoveryStore(tmp_path / "recovery")
    initial = store.create(
        _binding(tmp_path), gap_ids=("gap-b", "gap-a")
    )
    gap_a, gap_b = initial.state.gaps
    analyzed_a = _progress_gap(gap_a, RecoveryPhase.ANALYZED)
    receipt_a = _stage_receipt(
        initial.state,
        initial.state_sha256,
        RecoveryPhase.ANALYZED,
        scope=StageReceiptScope.GAPS,
        gap_ids=("gap-a",),
        seed="analysis-a",
    )
    partial = store.append(
        RecoveryEvent.bound_to(
            initial.state,
            phase=RecoveryPhase.CREATED,
            gaps=(analyzed_a, gap_b),
            stage_receipts=(receipt_a,),
        )
    )
    assert partial.state.phase is RecoveryPhase.CREATED
    assert partial.state.gaps[0].phase is RecoveryPhase.ANALYZED
    assert partial.state.gaps[1].phase is RecoveryPhase.CREATED

    analyzed_b = _progress_gap(
        partial.state.gaps[1], RecoveryPhase.ANALYZED
    )
    wrong_coverage = _stage_receipt(
        partial.state,
        partial.state_sha256,
        RecoveryPhase.ANALYZED,
        scope=StageReceiptScope.GAPS,
        gap_ids=("gap-a",),
        seed="analysis-wrong-gap",
    )
    with pytest.raises(RecoveryError, match="coverage does not match"):
        reduce_recovery(
            partial.state,
            RecoveryEvent.bound_to(
                partial.state,
                phase=RecoveryPhase.ANALYZED,
                gaps=(partial.state.gaps[0], analyzed_b),
                stage_receipts=(wrong_coverage,),
            ),
            parent_state_sha256=partial.state_sha256,
        )

    receipt_b = _stage_receipt(
        partial.state,
        partial.state_sha256,
        RecoveryPhase.ANALYZED,
        scope=StageReceiptScope.GAPS,
        gap_ids=("gap-b",),
        seed="analysis-b",
    )
    complete = store.append(
        RecoveryEvent.bound_to(
            partial.state,
            phase=RecoveryPhase.ANALYZED,
            gaps=(partial.state.gaps[0], analyzed_b),
            stage_receipts=(receipt_b,),
        )
    )
    assert complete.state.phase is RecoveryPhase.ANALYZED
    assert len(complete.state.stage_receipts) == 2


def test_verified_requires_current_execution_verification_and_receipt(tmp_path):
    store = RecoveryStore(tmp_path / "recovery")
    status = store.create(
        _binding(
            tmp_path,
            activation_allowed=True,
            execution_allowed=True,
        ),
        gap_ids=("gap-1",),
    )
    for phase in (
        RecoveryPhase.ANALYZED,
        RecoveryPhase.GAP_PREPARED,
        RecoveryPhase.RESOLVED,
        RecoveryPhase.CANDIDATE_READY,
        RecoveryPhase.ADAPTER_CANDIDATE,
        RecoveryPhase.INTEGRATED_CANDIDATE,
    ):
        status = _advance_all(store, status, phase)
    status = _advance_all(
        store,
        status,
        RecoveryPhase.QUALIFIED,
        side_effects=SideEffectTruth(execution_performed=True),
    )
    status = _advance_all(
        store,
        status,
        RecoveryPhase.ACTIVATED,
        side_effects=SideEffectTruth(activation_performed=True),
    )
    status = _advance_all(store, status, RecoveryPhase.RESCANNED)
    status = _advance_all(store, status, RecoveryPhase.FINAL_MIGRATED)

    verified_gap = _progress_gap(
        status.state.gaps[0],
        RecoveryPhase.VERIFIED,
        outcome=RecoveryOutcome.COMPLETE,
    )
    receipt = _stage_receipt(
        status.state, status.state_sha256, RecoveryPhase.VERIFIED
    )
    with pytest.raises(RecoveryError, match="current event"):
        reduce_recovery(
            status.state,
            RecoveryEvent.bound_to(
                status.state,
                phase=RecoveryPhase.VERIFIED,
                outcome=RecoveryOutcome.COMPLETE,
                gaps=(verified_gap,),
                stage_receipts=(receipt,),
                side_effects=SideEffectTruth(verification_performed=True),
            ),
            parent_state_sha256=status.state_sha256,
        )
    with pytest.raises(RecoveryError, match="requires a new satisfied"):
        reduce_recovery(
            status.state,
            RecoveryEvent.bound_to(
                status.state,
                phase=RecoveryPhase.VERIFIED,
                outcome=RecoveryOutcome.COMPLETE,
                gaps=(verified_gap,),
                side_effects=SideEffectTruth(
                    execution_performed=True,
                    verification_performed=True,
                ),
            ),
            parent_state_sha256=status.state_sha256,
        )

    verified = store.append(
        RecoveryEvent.bound_to(
            status.state,
            phase=RecoveryPhase.VERIFIED,
            outcome=RecoveryOutcome.COMPLETE,
            gaps=(verified_gap,),
            stage_receipts=(receipt,),
            side_effects=SideEffectTruth(
                execution_performed=True,
                verification_performed=True,
            ),
        )
    )
    assert verified.state.outcome is RecoveryOutcome.COMPLETE
    assert verified.state.event_side_effects.execution_performed is True
    assert verified.state.event_side_effects.verification_performed is True
    assert receipt in verified.state.stage_receipts


def test_initial_state_is_pristine_schema_v2(tmp_path):
    initial = RecoveryStore(tmp_path / "recovery").create(
        _binding(tmp_path, execution_allowed=True), gap_ids=("gap-1",)
    )
    assert initial.state.schema_version == 2
    value = initial.state.to_dict()
    assert value["stage_receipts"] == []
    assert value["event_side_effects"]["verification_performed"] is False

    with pytest.raises(RecoveryError, match="initial recovery state"):
        replace(
            initial.state,
            side_effects=SideEffectTruth(execution_performed=True),
            event_side_effects=SideEffectTruth(execution_performed=True),
        )

    legacy = initial.state.to_dict()
    legacy["schema_version"] = 1
    with pytest.raises(RecoveryError, match="unsupported recovery state schema"):
        type(initial.state).from_dict(legacy)


def test_store_rejects_symlink_and_non_private_modes(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(RecoveryError, match="symlink"):
        RecoveryStore(alias).create(_binding(tmp_path))

    insecure = tmp_path / "insecure"
    insecure.mkdir(mode=0o755)
    with pytest.raises(RecoveryError, match="exactly 0700"):
        RecoveryStore(insecure).create(_binding(tmp_path))

    store = RecoveryStore(tmp_path / "private")
    store.create(_binding(tmp_path))
    state_path = tmp_path / "private" / "states" / "000001.json"
    os.chmod(state_path, 0o644)
    with pytest.raises(RecoveryError, match="private 0600"):
        store.status()


def test_store_rejects_symlinked_state_file(tmp_path):
    store = RecoveryStore(tmp_path / "recovery")
    store.create(_binding(tmp_path))
    state_path = tmp_path / "recovery" / "states" / "000001.json"
    backing = tmp_path / "backing.json"
    state_path.rename(backing)
    state_path.symlink_to(backing)
    with pytest.raises(RecoveryError, match="cannot open recovery state"):
        store.status()


def _candidate(
    origin: OperatorCandidateOrigin,
    *,
    qualified_api: str = "numpy.matmul",
    decision_route: str | None = None,
):
    routes = {
        OperatorCandidateOrigin.OFFICIAL_BINARY:
            "installed_cann_adapter_candidate",
        OperatorCandidateOrigin.OFFICIAL_SOURCE: "official_source_candidate",
        OperatorCandidateOrigin.REMOTE_LLM: "new_kernel_generation_allowed",
    }
    gap_spec = {
        "schema_version": 1,
        "gap_id": "matmul-gap",
        "qualified_api": qualified_api,
        "target_soc": "Ascend910B3",
        "cann_version": "8.3.RC1",
    }
    decision = {
        "gap_id": "matmul-gap",
        "qualified_api": qualified_api,
        "route": decision_route or routes[origin],
        "activation_allowed": False,
    }
    gap_ref = EvidenceRef("gap_spec", "/evidence/gap.json", _digest(gap_spec))
    decision_ref = EvidenceRef(
        "resolution_decision", "/evidence/resolution.json#matmul-gap", _digest(decision)
    )
    license_binding = LicenseBinding(
        identifier="LicenseRef-CANN-OSL",
        accepted=True,
        ref=_ref("license", "license"),
    )
    candidate = create_operator_candidate(
        origin_kind=origin,
        gap_id="matmul-gap",
        qualified_api=qualified_api,
        gap_spec_ref=gap_ref,
        resolution_decision_ref=decision_ref,
        target_soc="Ascend910B3",
        cann_version="8.3.RC1",
        license=license_binding,
        provenance_refs=(_ref("provenance", origin.value),),
        candidate_refs=(_ref("candidate_content", "content-" + origin.value),),
    )
    return candidate, gap_spec, decision


@pytest.mark.parametrize("origin", tuple(OperatorCandidateOrigin))
def test_operator_candidate_three_origins_are_strict_and_route_neutral(origin):
    candidate, gap_spec, decision = _candidate(origin)
    value = candidate.to_dict()
    assert value["origin_kind"] == origin.value
    assert value["activation_allowed"] is False
    assert value["candidate_id"].startswith("candidate-")
    assert len(operator_candidate_sha256(candidate)) == 64
    assert OperatorCandidateArtifact.from_dict(value) == candidate
    validate_operator_candidate_binding(
        candidate, gap_spec=gap_spec, resolution_decision=decision
    )


def test_operator_candidate_rejects_unknown_fields_and_activation_permission():
    candidate, _, _ = _candidate(OperatorCandidateOrigin.OFFICIAL_BINARY)
    unknown = candidate.to_dict()
    unknown["build_allowed"] = True
    with pytest.raises(RecoveryError, match="unknown"):
        OperatorCandidateArtifact.from_dict(unknown)

    nested_unknown = candidate.to_dict()
    nested_unknown["gap_spec_ref"]["size"] = 42
    with pytest.raises(RecoveryError, match="unknown"):
        OperatorCandidateArtifact.from_dict(nested_unknown)

    activation = candidate.to_dict()
    activation["activation_allowed"] = True
    with pytest.raises(RecoveryError, match="never allow activation"):
        OperatorCandidateArtifact.from_dict(activation)


def test_operator_candidate_binding_rejects_gap_target_drift():
    candidate, gap_spec, decision = _candidate(OperatorCandidateOrigin.REMOTE_LLM)
    drifted = dict(gap_spec)
    drifted["target_soc"] = "Ascend310P"
    with pytest.raises(RecoveryError, match="GapSpec binding"):
        validate_operator_candidate_binding(
            candidate, gap_spec=drifted, resolution_decision=decision
        )


def test_operator_candidate_accepts_bounded_multi_segment_api():
    candidate, gap_spec, decision = _candidate(
        OperatorCandidateOrigin.OFFICIAL_SOURCE,
        qualified_api="numpy.linalg.det",
    )
    assert candidate.qualified_api == "numpy.linalg.det"
    validate_operator_candidate_binding(
        candidate, gap_spec=gap_spec, resolution_decision=decision
    )


def test_operator_candidate_origin_must_match_resolution_route():
    candidate, gap_spec, decision = _candidate(
        OperatorCandidateOrigin.OFFICIAL_BINARY,
        decision_route="official_source_candidate",
    )
    with pytest.raises(RecoveryError, match="requires resolution route"):
        validate_operator_candidate_binding(
            candidate, gap_spec=gap_spec, resolution_decision=decision
        )


def test_recovery_public_api_is_exported():
    assert gaps_api.RecoveryStore is RecoveryStore
    assert gaps_api.OperatorCandidateArtifact is OperatorCandidateArtifact
    assert gaps_api.RecoveryPolicyProfile is RecoveryPolicyProfile
    assert gaps_api.recovery_policy_for_profile is recovery_policy_for_profile
    assert gaps_api.reduce_recovery is reduce_recovery
    assert gaps_api.validate_operator_candidate_binding is (
        validate_operator_candidate_binding
    )
