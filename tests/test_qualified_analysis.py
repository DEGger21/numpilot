from dataclasses import dataclass
import hashlib

import pytest

from numpilot.capability_profile import (
    CapabilityDecision,
    ComputationRoute,
    OPS_NN_MATMUL_ASNUMPY_AUDIT_ID,
)
from numpilot.qualified_analysis import analyze_qualified_script


@dataclass(frozen=True)
class _Runtime:
    status: str = "matched"
    identity_sha256: str = "a" * 64
    audit_id: str = "pinned-baseline"

    def to_dict(self):
        return {
            "status": self.status,
            "identity_sha256": self.identity_sha256,
            "audit_id": self.audit_id,
        }


def _analyze(source, *, runtime=None):
    return analyze_qualified_script(
        source,
        "case.py",
        runtime_identity=runtime or _Runtime(),
    )


def test_direct_script_profile_is_inert_hash_bound_and_machine_readable():
    source = """\
import numpy as np
a = np.full((3, 4), 5.0, dtype=np.float32)
b = np.full((3, 4), 3.0, dtype=np.float32)
diff = np.subtract(a, b)
result = np.sqrt(np.clip(diff, 0.0, 2.0))
print(result)
"""
    analysis = _analyze(source)
    payload = analysis.to_dict()

    assert analysis.decision is CapabilityDecision.DIRECT
    assert analysis.source_sha256 == hashlib.sha256(source.encode()).hexdigest()
    assert [call.canonical_key for call in analysis.calls] == [
        "full", "full", "subtract", "clip", "sqrt"
    ]
    assert all(call.decision is CapabilityDecision.DIRECT for call in analysis.calls)
    assert payload["actual_remote_call"] is False
    assert payload["execution_performed"] is False
    assert payload["runtime_identity_sha256"] == "a" * 64


def test_nested_linalg_det_is_direct_only_for_the_exact_static_profile():
    source = """\
import numpy as np
x = np.eye(3, dtype=np.float32)
y = np.tanh(x)
value = np.linalg.det(y)
print(value)
"""
    analysis = _analyze(source)
    det = next(call for call in analysis.calls if call.canonical_key == "linalg.det")

    assert analysis.decision is CapabilityDecision.DIRECT
    assert det.decision is CapabilityDecision.DIRECT
    assert det.facts.dtype == "float32"
    assert det.facts.shape == (3, 3)
    assert det.source_api == "numpy.linalg.det"


def test_runtime_identity_missing_or_drifted_downgrades_direct_calls_to_audit_required():
    source = "import numpy as np\nx = np.full((4,), 2.0, dtype=np.float32)\n"
    analysis = _analyze(
        source,
        runtime=_Runtime(status="audit_required", identity_sha256="b" * 64),
    )

    assert analysis.decision is CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
    assert analysis.calls[0].decision is CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
    assert analysis.calls[0].reason_code == "RUNTIME_IDENTITY_AUDIT_REQUIRED"


def test_axis_none_accumulation_is_blocked_as_a_contract_rewrite():
    source = """\
import numpy as np
x = np.full((5,), 2.0, dtype=np.float32)
y = np.cumsum(x)
z = np.cumprod(y)
print(z)
"""
    analysis = _analyze(source)

    assert analysis.decision is CapabilityDecision.BLOCKED
    assert [call.decision for call in analysis.calls[-2:]] == [
        CapabilityDecision.BLOCKED,
        CapabilityDecision.BLOCKED,
    ]


def test_random_float64_astype_chain_is_blocked_without_becoming_a_gap():
    source = """\
import numpy as np
x = np.random.normal(0, 0.01, (4,)).astype(np.float32)
y = np.expm1(x)
print(y)
"""
    analysis = _analyze(source)
    random_call = next(call for call in analysis.calls if call.canonical_key == "random.normal")

    assert analysis.decision is CapabilityDecision.BLOCKED
    assert random_call.route is ComputationRoute.NATIVE_NPU
    assert random_call.decision is CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
    assert any("astype" in issue for issue in analysis.issues)
    assert not any(call.decision is CapabilityDecision.GAP_REQUIRED for call in analysis.calls)


def test_matmul_and_host_inspection_are_fallback_consent_not_strict_performance():
    source = """\
import numpy as np
a = np.ones((2, 2), dtype=np.float32)
b = np.ones((2, 2), dtype=np.float32)
c = np.matmul(a, b)
print(c)
"""
    analysis = _analyze(source)

    assert analysis.decision is CapabilityDecision.FALLBACK_CONSENT
    assert analysis.performance_valid is False
    assert analysis.speedup is None
    assert next(call for call in analysis.calls if call.canonical_key == "matmul").decision is (
        CapabilityDecision.FALLBACK_CONSENT
    )


def test_ops_nn_matmul_runtime_admits_only_exact_2d_float32_contract():
    source = """\
import numpy as np
a = np.full((2, 3), 0.01, dtype=np.float32)
b = np.full((3, 4), 0.02, dtype=np.float32)
c = np.matmul(a, b)
"""
    runtime = _Runtime(audit_id=OPS_NN_MATMUL_ASNUMPY_AUDIT_ID)
    analysis = _analyze(source, runtime=runtime)
    matmul = next(
        call for call in analysis.calls if call.canonical_key == "matmul"
    )

    assert analysis.decision is CapabilityDecision.DIRECT
    assert matmul.route is ComputationRoute.NATIVE_NPU
    assert matmul.decision is CapabilityDecision.DIRECT

    rejected_sources = (
        source.replace("dtype=np.float32)\nc =", "dtype=np.float64)\nc ="),
        source.replace("(2, 3)", "(3,)", 1),
        source.replace("(2, 3)", "(0, 3)", 1),
        source.replace("(3, 4)", "(5, 4)", 1),
    )
    for rejected_source in rejected_sources:
        rejected = _analyze(rejected_source, runtime=runtime)
        rejected_matmul = next(
            call for call in rejected.calls if call.canonical_key == "matmul"
        )
        assert rejected.decision is CapabilityDecision.BLOCKED
        assert rejected_matmul.decision is CapabilityDecision.BLOCKED


def test_static_keepdims_broadcast_does_not_hide_the_matmul_fallback_decision():
    source = """\
import numpy as np
x = np.full((2, 3), 1.0, dtype=np.float32)
w = np.full((3, 4), 2.0, dtype=np.float32)
logits = np.matmul(x, w)
values = np.exp(logits)
probs = values / np.sum(values, axis=1, keepdims=True)
row_sums = np.sum(probs, axis=1)
print(row_sums)
"""
    analysis = _analyze(source)

    assert analysis.decision is CapabilityDecision.FALLBACK_CONSENT
    assert analysis.required_rewrites == ("array division -> numpy.divide",)
    assert all(call.facts.shape is not None for call in analysis.calls)


def test_terminal_item_calls_are_output_boundaries_but_astype_is_not():
    source = """\
import numpy as np
a = np.ones((5,), dtype=np.float32)
b = np.full((5,), 2.0, dtype=np.float32)
value = np.dot(a, b)
print(value.item())
"""
    analysis = _analyze(source)

    assert analysis.decision is CapabilityDecision.FALLBACK_CONSENT
    assert not analysis.issues


def test_static_numpy_pi_and_terminal_global_reduction_are_qualified_without_execution():
    source = """\
import numpy as np
t = np.linspace(0, 2 * np.pi, 8, dtype=np.float32)
x = np.absolute(np.sin(t))
value = np.amax(x)
print(value)
"""
    analysis = _analyze(source)
    reduction = next(call for call in analysis.calls if call.canonical_key == "amax")

    assert analysis.decision is CapabilityDecision.DIRECT
    assert reduction.effective_route is ComputationRoute.HOST_SCALAR_BOUNDARY
    assert reduction.performance_eligible is False
    assert analysis.required_rewrites == ("numpy.pi -> math.pi",)


def test_unmodelled_third_party_import_and_array_dunders_are_blocked():
    source = """\
import numpy as np
from scipy.special import erf
x = np.arange(-2, 3, dtype=np.float32)
y = 0.5 * x * erf(x / np.sqrt(2.0))
print(y)
"""
    analysis = _analyze(source)

    assert analysis.decision is CapabilityDecision.BLOCKED
    assert any("scipy.special" in issue for issue in analysis.issues)
    assert any("array operator" in issue for issue in analysis.issues)


def test_analysis_never_accepts_a_response_or_secret_field():
    source = "import numpy as np\nx = np.zeros((1,), dtype=np.float32)\n"
    payload = _analyze(source).to_dict()
    forbidden = {"token", "base_url", "response", "response_content", "proxy"}

    assert forbidden.isdisjoint(payload)


def test_numpy_alias_must_come_from_one_exact_import():
    analysis = _analyze("x = np.zeros((1,), dtype=np.float32)\n")

    assert analysis.decision is CapabilityDecision.BLOCKED
    assert any("import numpy as np" in issue for issue in analysis.issues)


def test_terminal_reduction_alias_is_traced_to_a_device_consumer():
    source = """\
import numpy as np
x = np.ones((4,), dtype=np.float32)
scalar = np.sum(x)
alias = scalar
y = np.add(alias, alias)
print(y)
"""
    analysis = _analyze(source)
    reduction = next(call for call in analysis.calls if call.canonical_key == "sum")

    assert analysis.decision is CapabilityDecision.BLOCKED
    assert reduction.facts.output_use == "device"
    assert reduction.decision is CapabilityDecision.BLOCKED


def test_logic_reduction_axis_none_remains_a_device_array_for_consumers():
    source = """\
import numpy as np
x = np.ones((4,), dtype=np.float32)
flags = np.equal(x, x)
value = np.any(flags)
result = np.equal(value, value)
print(result)
"""
    analysis = _analyze(source)
    reduction = next(
        call for call in analysis.calls if call.canonical_key == "any"
    )

    assert analysis.decision is CapabilityDecision.DIRECT
    assert reduction.facts.output_use == "device"
    assert reduction.effective_route is ComputationRoute.NATIVE_NPU
    assert reduction.performance_eligible is True


@pytest.mark.parametrize(
    "division",
    (
        "a / b",
        "a / np.sum(b, axis=1, keepdims=True)",
    ),
)
def test_only_the_source_bound_keepdims_division_pattern_is_rewriteable(division):
    source = """\
import numpy as np
a = np.ones((2, 3), dtype=np.float32)
b = np.ones((4, 5), dtype=np.float32)
value = %s
print(value)
""" % division
    analysis = _analyze(source)

    assert analysis.decision is CapabilityDecision.BLOCKED
    assert any("array operator" in issue for issue in analysis.issues)
    assert "array division -> numpy.divide" not in analysis.required_rewrites


def test_statically_incompatible_binary_shapes_are_rejected():
    source = """\
import numpy as np
a = np.ones((2,), dtype=np.float32)
b = np.ones((3,), dtype=np.float32)
y = np.add(a, b)
print(y)
"""
    analysis = _analyze(source)
    add = next(call for call in analysis.calls if call.canonical_key == "add")

    assert analysis.decision is CapabilityDecision.BLOCKED
    assert add.facts.source_contract_ok is False
    assert add.decision is CapabilityDecision.BLOCKED


def test_item_requires_a_statically_single_element_receiver():
    source = """\
import numpy as np
x = np.ones((4,), dtype=np.float32)
print(x.item())
"""
    analysis = _analyze(source)

    assert analysis.decision is CapabilityDecision.BLOCKED
    assert any("item" in issue for issue in analysis.issues)


def test_item_on_a_host_scalar_reduction_is_rejected():
    source = """\
import numpy as np
x = np.ones((4,), dtype=np.float32)
value = np.amax(x)
host = value.item()
print(host)
"""
    analysis = _analyze(source)

    assert analysis.decision is CapabilityDecision.BLOCKED
    assert any("not a proved device array" in issue for issue in analysis.issues)


@pytest.mark.parametrize(
    "tail",
    (
        "y = x[0]",
        "assert x is not None",
        "print((np := x))",
    ),
)
def test_unqualified_python_surfaces_and_numpy_rebinding_are_blocked(tail):
    source = (
        "import numpy as np\n"
        "x = np.ones((2,), dtype=np.float32)\n"
        + tail
        + "\n"
    )
    analysis = _analyze(source)

    assert analysis.decision is CapabilityDecision.BLOCKED
    assert analysis.issues


def test_pathological_static_allocation_profile_is_blocked():
    source = "import numpy as np\nx = np.zeros((1000000000000,), dtype=np.float32)\n"
    analysis = _analyze(source)

    assert analysis.decision is CapabilityDecision.BLOCKED
    assert any("resource" in issue for issue in analysis.issues)


def test_unimplemented_numpy_constants_do_not_silently_become_asnumpy_attributes():
    source = "import numpy as np\nx = np.full((1,), np.inf, dtype=np.float32)\n"
    analysis = _analyze(source)

    assert analysis.decision is CapabilityDecision.BLOCKED
    assert any("numpy.inf" in issue for issue in analysis.issues)


def test_only_golden_linalg_det_is_currently_call_profile_qualified():
    source = """\
import numpy as np
x = np.eye(2, dtype=np.float32)
y = np.linalg.inv(x, definitely_invalid=True)
print(y)
"""
    analysis = _analyze(source)
    inv = next(call for call in analysis.calls if call.canonical_key == "linalg.inv")

    assert analysis.decision is CapabilityDecision.BLOCKED
    assert inv.facts.source_contract_ok is False
    assert inv.decision is CapabilityDecision.BLOCKED


def test_valid_non_golden_linalg_route_requires_profile_audit():
    source = """\
import numpy as np
x = np.eye(2, dtype=np.float32)
y = np.linalg.inv(x)
print(y)
"""
    analysis = _analyze(source)
    inv = next(call for call in analysis.calls if call.canonical_key == "linalg.inv")

    assert analysis.decision is CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
    assert inv.facts.source_contract_ok is None


@pytest.mark.parametrize(
    "call",
    (
        "np.sum(x, axis=5)",
        "np.cumsum(x, axis=-7)",
        "np.sort(x, axis=3)",
        "np.sum(x, None, None, None, None, None, None)",
    ),
)
def test_axis_range_and_reduction_arity_fail_closed(call):
    source = (
        "import numpy as np\n"
        "x = np.ones((4,), dtype=np.float32)\n"
        "y = " + call + "\n"
        "print(y)\n"
    )
    analysis = _analyze(source)

    assert analysis.decision is CapabilityDecision.BLOCKED


@pytest.mark.parametrize(
    "call",
    (
        "np.sum(x, axis=0, dtype=np.float64)",
        "np.cumsum(x, axis=0, dtype=np.float64)",
    ),
)
def test_explicit_reduction_dtype_override_is_not_inferred_from_input(call):
    source = (
        "import numpy as np\n"
        "x = np.ones((4,), dtype=np.float32)\n"
        "y = " + call + "\n"
        "print(y)\n"
    )
    analysis = _analyze(source)
    target = analysis.calls[-1]

    assert target.facts.dtype == "float64"
    assert target.decision is CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
    assert analysis.decision is CapabilityDecision.CAPABILITY_AUDIT_REQUIRED


def test_einsum_equation_alone_cannot_qualify_wrong_operand_ranks():
    source = """\
import numpy as np
a = np.ones((2, 2), dtype=np.float32)
b = np.ones((3, 3), dtype=np.float32)
y = np.einsum("a,b->ab", a, b)
print(y)
"""
    analysis = _analyze(source)
    einsum = next(call for call in analysis.calls if call.canonical_key == "einsum")

    assert einsum.facts.source_contract_ok is False
    assert einsum.decision is CapabilityDecision.BLOCKED
    assert analysis.decision is CapabilityDecision.BLOCKED


def test_einsum_golden_equation_checks_every_operand_shape():
    source = """\
import numpy as np
a = np.ones((2, 3, 4, 5), dtype=np.float32)
b = np.ones((2, 3, 4, 6, 7), dtype=np.float32)
y = np.einsum("abcd,abced->abce", a, b)
print(y)
"""
    analysis = _analyze(source)
    einsum = next(call for call in analysis.calls if call.canonical_key == "einsum")

    assert einsum.facts.source_contract_ok is False
    assert analysis.decision is CapabilityDecision.BLOCKED


def test_rebinding_print_cannot_make_a_device_value_look_terminal():
    source = """\
import numpy as np
x = np.ones((2,), dtype=np.float32)
print = np.add
value = np.sum(x)
print(value, value)
"""
    analysis = _analyze(source)

    assert analysis.decision is CapabilityDecision.BLOCKED
    assert any("reserved analysis name" in issue for issue in analysis.issues)


def test_reserved_asnumpy_target_alias_cannot_be_prebound():
    source = """\
import numpy as np
ap = 1
x = np.full((2,), 1.0, dtype=np.float32)
print(x)
"""
    analysis = _analyze(source)

    assert analysis.decision is CapabilityDecision.BLOCKED
    assert any("reserved analysis name" in issue for issue in analysis.issues)


def test_annotated_alias_is_traced_to_a_device_consumer():
    source = """\
import numpy as np
a = np.full((2,), 1.0, dtype=np.float32)
x = np.sum(a)
y: object = x
z = np.add(y, y)
print(z)
"""
    analysis = _analyze(source)
    reduction = next(call for call in analysis.calls if call.canonical_key == "sum")

    assert reduction.facts.output_use == "device"
    assert analysis.decision is CapabilityDecision.BLOCKED


@pytest.mark.parametrize("expression", ["a and b", "a or b", "a.T"])
def test_unmodelled_array_boolean_and_attribute_surfaces_are_blocked(expression):
    source = """\
import numpy as np
a = np.ones((2,), dtype=np.float32)
b = np.ones((2,), dtype=np.float32)
c = %s
print(c)
""" % expression
    analysis = _analyze(source)

    assert analysis.decision is CapabilityDecision.BLOCKED
    assert analysis.issues


@pytest.mark.parametrize("key", ["matmul", "dot", "inner", "vdot"])
def test_non_strict_linear_algebra_still_rejects_incompatible_static_shapes(key):
    source = """\
import numpy as np
a = np.ones((2, 3), dtype=np.float32)
b = np.ones((5, 4), dtype=np.float32)
c = np.%s(a, b)
print(c)
""" % key
    analysis = _analyze(source)
    call = next(item for item in analysis.calls if item.canonical_key == key)

    assert call.facts.source_contract_ok is False
    assert call.decision is CapabilityDecision.BLOCKED
    assert analysis.decision is CapabilityDecision.BLOCKED
