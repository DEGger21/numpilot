import pytest

from numpilot.capability_profile import (
    CapabilityDecision,
    ComputationRoute,
    OPS_NN_MATMUL_ASNUMPY_AUDIT_ID,
    QualificationStatus,
    StaticCallFacts,
    aggregate_decisions,
    build_pinned_route_manifest,
    build_runtime_route_manifest,
    qualify_call,
)


def _facts(**overrides):
    values = {
        "positional_args": 1,
        "keyword_args": (),
        "source_contract_ok": True,
        "dtype": "float32",
        "rank": 1,
        "shape": (8,),
        "axis": "not_applicable",
        "equation": "not_applicable",
        "output_use": "device",
    }
    values.update(overrides)
    return StaticCallFacts(**values)


def test_route_manifest_separates_121_public_apis_from_114_npu_routes():
    manifest = build_pinned_route_manifest()
    top_level = {
        key: entry
        for key, entry in manifest.entries.items()
        if "." not in key
    }

    assert len(top_level) == 121
    assert sum(entry.route.is_npu_computation for entry in top_level.values()) == 114
    assert "angle" not in manifest.entries
    assert manifest.entries["matmul"].route is ComputationRoute.CPU_FALLBACK
    assert manifest.entries["dot"].route is ComputationRoute.CONDITIONAL_CPU_FALLBACK
    assert manifest.entries["floor"].route is ComputationRoute.CONDITIONAL_CPU_FALLBACK
    assert manifest.entries["signbit"].route is ComputationRoute.HOST_ASSISTED


def test_nested_routes_use_canonical_keys_without_inheriting_leaf_names():
    manifest = build_pinned_route_manifest()

    assert manifest.entries["linalg.det"].route is ComputationRoute.COMPOSITE_NPU
    assert manifest.entries["linalg.inv"].route is ComputationRoute.NATIVE_NPU
    assert manifest.entries["linalg.matrix_power"].route is ComputationRoute.CPU_FALLBACK
    assert manifest.entries["random.normal"].route is ComputationRoute.NATIVE_NPU
    assert manifest.entries["random.binomial"].route is ComputationRoute.COMPOSITE_NPU
    assert "det" not in manifest.entries
    assert "normal" not in manifest.entries


def test_ops_nn_matmul_route_override_is_exact_audit_scoped():
    pinned = build_runtime_route_manifest("another-runtime")
    integrated = build_runtime_route_manifest(OPS_NN_MATMUL_ASNUMPY_AUDIT_ID)

    assert pinned.entries["matmul"].route is ComputationRoute.CPU_FALLBACK
    assert integrated.entries["matmul"].route is ComputationRoute.NATIVE_NPU
    assert integrated.entries["full"] == pinned.entries["full"]

    direct = qualify_call(
        "matmul",
        _facts(positional_args=2, rank=2, shape=(8, 4)),
        manifest=integrated,
    )
    assert direct.decision is CapabilityDecision.DIRECT

    for facts, marker in (
        (_facts(positional_args=2, dtype="float64", rank=2, shape=(8, 4)), "float32"),
        (_facts(positional_args=2, rank=1, shape=(8,)), "rank-2"),
        (_facts(positional_args=1, rank=2, shape=(8, 4)), "two positional"),
    ):
        rejected = qualify_call("matmul", facts, manifest=integrated)
        assert rejected.decision is CapabilityDecision.BLOCKED
        assert marker in " ".join(rejected.reasons)


def test_linalg_det_requires_exact_float32_rank_two_square_profile():
    direct = qualify_call(
        "linalg.det",
        _facts(rank=2, shape=(3, 3), output_use="terminal"),
    )

    assert direct.decision is CapabilityDecision.DIRECT
    assert direct.contract.dtype is QualificationStatus.QUALIFIED
    assert direct.contract.rank is QualificationStatus.QUALIFIED
    assert direct.contract.signature is QualificationStatus.QUALIFIED

    for facts, marker in (
        (_facts(dtype="float64", rank=2, shape=(3, 3)), "dtype"),
        (_facts(rank=1, shape=(3,)), "rank"),
        (_facts(rank=2, shape=(2, 3)), "square"),
        (_facts(positional_args=2, rank=2, shape=(3, 3)), "signature"),
    ):
        rejected = qualify_call("linalg.det", facts)
        assert rejected.decision is CapabilityDecision.BLOCKED
        assert marker in " ".join(rejected.reasons)


@pytest.mark.parametrize("key", ["cumsum", "cumprod", "nancumsum", "nancumprod"])
def test_accumulation_axis_none_is_a_contract_rewrite_not_direct(key):
    result = qualify_call(key, _facts(axis=None))

    assert result.decision is CapabilityDecision.BLOCKED
    assert result.contract.axis is QualificationStatus.REJECTED
    assert result.reason_code == "CALL_CONTRACT_REWRITE_REQUIRED"


@pytest.mark.parametrize("key", ["full", "zeros", "ones", "eye"])
def test_creation_dtype_none_or_unknown_fails_closed(key):
    assert qualify_call(key, _facts(dtype=None)).decision is CapabilityDecision.BLOCKED
    assert (
        qualify_call(key, _facts(dtype="unknown")).decision
        is CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
    )


@pytest.mark.parametrize("equation", ["a,b->ab", "abcd,abced->abce"])
def test_einsum_accepts_only_the_two_source_audited_equations(equation):
    accepted = qualify_call(
        "einsum",
        _facts(positional_args=3, equation=equation),
    )
    assert accepted.decision is CapabilityDecision.DIRECT
    assert accepted.contract.equation is QualificationStatus.QUALIFIED


def test_einsum_rejects_unsupported_or_unknown_equations():
    rejected = qualify_call("einsum", _facts(positional_args=3, equation="ij,jk->ik"))
    pending = qualify_call("einsum", _facts(positional_args=3, equation="unknown"))

    assert rejected.decision is CapabilityDecision.BLOCKED
    assert rejected.contract.equation is QualificationStatus.REJECTED
    assert pending.decision is CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
    assert pending.contract.equation is QualificationStatus.PENDING


@pytest.mark.parametrize("key", ["amax", "sum"])
def test_axis_none_is_qualified_only_as_a_terminal_host_scalar_boundary(key):
    terminal = qualify_call(key, _facts(axis=None, output_use="terminal"))
    chained = qualify_call(key, _facts(axis=None, output_use="device"))

    assert terminal.decision is CapabilityDecision.DIRECT
    assert terminal.effective_route is ComputationRoute.HOST_SCALAR_BOUNDARY
    assert terminal.performance_eligible is False
    assert chained.decision is CapabilityDecision.BLOCKED
    assert chained.contract.output is QualificationStatus.REJECTED


@pytest.mark.parametrize("key", ["all", "any"])
def test_logic_reductions_remain_device_arrays_for_axis_none(key):
    terminal = qualify_call(key, _facts(axis=None, output_use="terminal"))
    chained = qualify_call(key, _facts(axis=None, output_use="device"))

    assert terminal.decision is CapabilityDecision.DIRECT
    assert terminal.effective_route is ComputationRoute.NATIVE_NPU
    assert terminal.performance_eligible is True
    assert chained.decision is CapabilityDecision.DIRECT
    assert chained.effective_route is ComputationRoute.NATIVE_NPU


def test_random_normal_route_is_known_but_reproducibility_and_float64_chain_are_pending():
    result = qualify_call(
        "random.normal",
        _facts(
            positional_args=3,
            dtype="float64",
            output_use="device",
            rank=1,
            shape=(4,),
        ),
    )

    assert result.route is ComputationRoute.NATIVE_NPU
    assert result.decision is CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
    assert result.contract.randomness is QualificationStatus.PENDING
    assert result.contract.output is QualificationStatus.PENDING
    assert "float64" in " ".join(result.reasons)


def test_route_and_contract_failures_map_to_the_five_public_decisions():
    assert qualify_call("matmul", _facts(positional_args=2)).decision is (
        CapabilityDecision.FALLBACK_CONSENT
    )
    assert qualify_call("definitely_missing", _facts()).decision is (
        CapabilityDecision.GAP_REQUIRED
    )
    assert qualify_call("linalg.det", _facts(dtype="unknown", rank=None, shape=None)).decision is (
        CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
    )
    assert qualify_call("cumsum", _facts(axis=None)).decision is CapabilityDecision.BLOCKED
    assert qualify_call("sqrt", _facts()).decision is CapabilityDecision.DIRECT


@pytest.mark.parametrize("key", ["matmul", "dot", "floor"])
def test_rejected_signature_cannot_be_hidden_by_a_non_strict_route(key):
    result = qualify_call(
        key,
        _facts(positional_args=0, source_contract_ok=False),
    )

    assert result.contract.signature is QualificationStatus.REJECTED
    assert result.decision is CapabilityDecision.BLOCKED
    assert result.reason_code == "CALL_CONTRACT_REWRITE_REQUIRED"


@pytest.mark.parametrize("key", ["sum", "cumsum", "sort"])
@pytest.mark.parametrize("axis", [1, -2, 5])
def test_static_axis_must_be_inside_the_known_input_rank(key, axis):
    result = qualify_call(key, _facts(axis=axis, rank=1, shape=(8,)))

    assert result.contract.axis is QualificationStatus.REJECTED
    assert result.decision is CapabilityDecision.BLOCKED


def test_file_decision_aggregation_is_fail_closed():
    assert aggregate_decisions([CapabilityDecision.DIRECT]) is CapabilityDecision.DIRECT
    assert aggregate_decisions([
        CapabilityDecision.DIRECT,
        CapabilityDecision.FALLBACK_CONSENT,
    ]) is CapabilityDecision.FALLBACK_CONSENT
    assert aggregate_decisions([
        CapabilityDecision.GAP_REQUIRED,
        CapabilityDecision.CAPABILITY_AUDIT_REQUIRED,
    ]) is CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
    assert aggregate_decisions([
        CapabilityDecision.BLOCKED,
        CapabilityDecision.CAPABILITY_AUDIT_REQUIRED,
    ]) is CapabilityDecision.BLOCKED
