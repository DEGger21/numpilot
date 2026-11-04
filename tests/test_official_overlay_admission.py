"""Tests for scoped official-overlay route admission and static analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from numpilot.capabilities import OperatorSignature
from numpilot.capability_profile import (
    CapabilityDecision,
    ComputationRoute,
    OverlayRouteAdmission,
    RouteEntry,
    build_overlay_route_manifest,
    build_pinned_route_manifest,
)
from numpilot.qualified_analysis import (
    QualifiedAnalysisError,
    analyze_qualified_script,
)


QUALIFICATION_ID = "official-aclnn-" + "a" * 20
AUDIT_ID = (
    "unaudited-asnumpy-version+official-qualification-" + "a" * 20
)
ENTRY_CONSTRAINTS = {
    "concatenate": (
        "dtype=float32",
        "rank>=2",
        "rank<=2",
        "contiguous=true",
    ),
    "ones": (
        "dtype=float32",
        "rank>=1",
        "rank<=3",
        "contiguous=true",
    ),
    "zeros": (
        "dtype=float32",
        "rank>=1",
        "rank<=3",
        "contiguous=true",
    ),
}
ENTRY_SIGNATURES = {
    "concatenate": (
        OperatorSignature(positional_args=1, keyword_args=("axis",)),
        OperatorSignature(positional_args=2, keyword_args=()),
        OperatorSignature(positional_args=1, keyword_args=()),
    ),
    "ones": (
        OperatorSignature(positional_args=1, keyword_args=("dtype",)),
        OperatorSignature(positional_args=2, keyword_args=()),
    ),
    "zeros": (
        OperatorSignature(positional_args=1, keyword_args=("dtype",)),
        OperatorSignature(positional_args=2, keyword_args=()),
    ),
}
EXAMPLE = """\
import numpy as np
x = np.ones((3, 2), dtype=np.float32)
y = np.zeros((3, 4), dtype=np.float32)
joined = np.concatenate((x, y), axis=1)
print(joined)
"""


@dataclass(frozen=True)
class _Runtime:
    status: str = "audit_required"
    identity_sha256: str = "1" * 64
    audit_id: str = "unaudited-asnumpy-version"


def _entries() -> dict:
    evidence = "official ACLNN qualification " + QUALIFICATION_ID
    return {
        key: RouteEntry(
            canonical_key=key,
            source_api="numpy." + key,
            asnumpy_api="asnumpy." + key,
            route=ComputationRoute.NATIVE_NPU,
            evidence=evidence,
            constraints=ENTRY_CONSTRAINTS[key],
            signature_variants=ENTRY_SIGNATURES[key],
        )
        for key in ("concatenate", "ones", "zeros")
    }


def _admission(runtime_sha256: str = "1" * 64) -> OverlayRouteAdmission:
    return OverlayRouteAdmission(
        qualification_id=QUALIFICATION_ID,
        overlay_sha256="2" * 64,
        base_manifest_sha256="3" * 64,
        audit_id=AUDIT_ID,
        runtime_identity_sha256=runtime_sha256,
        entries=_entries(),
    )


def _analyze(source: str, *, admission=None, runtime=None):
    return analyze_qualified_script(
        source,
        "examples/concatenate_feature_join.py",
        runtime_identity=_Runtime() if runtime is None else runtime,
        overlay_admission=admission,
    )


def test_admission_fingerprint_is_deterministic_and_scoped() -> None:
    admission = _admission()

    assert admission.entry_keys == ("concatenate", "ones", "zeros")
    assert admission.fingerprint() == {
        "schema_version": admission.schema_version,
        "qualification_id": QUALIFICATION_ID,
        "overlay_sha256": "2" * 64,
        "base_manifest_sha256": "3" * 64,
        "admitted_audit_id": AUDIT_ID,
        "runtime_identity_sha256": "1" * 64,
        "entry_keys": ["concatenate", "ones", "zeros"],
    }
    payload = admission.to_dict()
    assert payload["entries"]["concatenate"]["constraints"] == list(
        ENTRY_CONSTRAINTS["concatenate"]
    )
    assert payload["entries"]["concatenate"]["route"] == "NATIVE_NPU"


def test_overlay_admission_makes_the_example_direct_despite_unaudited_runtime() -> None:
    admission = _admission()

    analysis = _analyze(EXAMPLE, admission=admission)

    assert analysis.decision is CapabilityDecision.DIRECT
    assert [call.canonical_key for call in analysis.calls] == [
        "ones",
        "zeros",
        "concatenate",
    ]
    assert all(
        call.decision is CapabilityDecision.DIRECT for call in analysis.calls
    )
    assert analysis.runtime_identity_status == "audit_required"
    assert analysis.runtime_identity_sha256 == "1" * 64
    assert analysis.overlay_qualification == admission.fingerprint()

    ones, zeros, concatenate = analysis.calls
    assert ones.facts.dtype == "float32"
    assert ones.facts.rank == 2
    assert ones.facts.contiguous is True
    assert zeros.facts.shape == (3, 4)
    assert concatenate.facts.axis == 1
    assert concatenate.facts.shape == (3, 6)
    assert all(
        call.route is ComputationRoute.NATIVE_NPU for call in analysis.calls
    )


def test_without_admission_the_unaudited_runtime_requires_an_audit() -> None:
    analysis = _analyze(EXAMPLE)

    assert analysis.decision is CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
    assert analysis.overlay_qualification is None
    assert analysis.calls
    # ``concatenate`` is not in the pinned catalog, so the base route requires
    # a new gap; the two creation calls exist but the unaudited runtime cannot
    # keep them direct.  Neither path may silently widen to a direct migration.
    concatenate = next(
        call for call in analysis.calls if call.canonical_key == "concatenate"
    )
    assert concatenate.decision is CapabilityDecision.GAP_REQUIRED
    assert concatenate.reason_code == "API_UNAVAILABLE"
    for call in analysis.calls:
        if call.canonical_key == "concatenate":
            continue
        assert call.decision is CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
        assert call.reason_code == "RUNTIME_IDENTITY_AUDIT_REQUIRED"
    assert all(
        call.decision is not CapabilityDecision.DIRECT
        for call in analysis.calls
    )


def test_admission_is_bound_to_the_exact_runtime_identity() -> None:
    with pytest.raises(
        QualifiedAnalysisError, match="another runtime identity"
    ):
        _analyze(EXAMPLE, admission=_admission(runtime_sha256="9" * 64))


def test_admission_type_is_not_duck_typed() -> None:
    with pytest.raises(QualifiedAnalysisError, match="admission type is invalid"):
        _analyze(EXAMPLE, admission={"qualification_id": QUALIFICATION_ID})


def test_out_of_profile_rank_is_audit_required_not_direct() -> None:
    source = EXAMPLE.replace("(3, 2)", "(2, 3, 4)").replace(
        "(3, 4)", "(2, 3, 4)"
    )

    analysis = _analyze(source, admission=_admission())

    assert analysis.decision is CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
    concatenate = next(
        call for call in analysis.calls if call.canonical_key == "concatenate"
    )
    assert concatenate.decision is CapabilityDecision.CAPABILITY_AUDIT_REQUIRED
    assert any(
        "outside the official scoped admission" in reason
        for reason in concatenate.reasons
    )
    assert any("rank=3" in reason for reason in concatenate.reasons)
    assert all(
        call.decision is CapabilityDecision.DIRECT
        for call in analysis.calls
        if call.canonical_key in {"ones", "zeros"}
    )


def test_unadmitted_signature_is_blocked_not_silently_widened() -> None:
    source = """\
import numpy as np
x = np.ones((3, 2), dtype=np.float32)
y = np.ones((3, 2), dtype=np.float32, order="C")
joined = np.concatenate((x, y), axis=1)
print(joined)
"""

    analysis = _analyze(source, admission=_admission())

    assert analysis.decision is CapabilityDecision.BLOCKED
    wide_ones = next(
        call
        for call in analysis.calls
        if call.canonical_key == "ones"
        and call.facts.keyword_args == ("dtype", "order")
    )
    assert wide_ones.decision is CapabilityDecision.BLOCKED


def test_overlay_manifest_replaces_only_the_admitted_entries() -> None:
    pinned = build_pinned_route_manifest()
    admission = _admission()

    manifest = build_overlay_route_manifest(pinned, admission)

    for key in ("concatenate", "ones", "zeros"):
        assert manifest.entries[key] is admission.entries[key]
    assert manifest.entries["matmul"] is pinned.entries["matmul"]
    assert manifest.repository == pinned.repository
    assert manifest.commit == pinned.commit
    assert manifest.tree == pinned.tree


def test_admission_rejects_invalid_ids_empty_entries_and_unscoped_routes() -> None:
    admission = _admission()

    with pytest.raises(ValueError, match="qualification ID"):
        replace(admission, qualification_id="not-an-official-qualification")
    with pytest.raises(ValueError, match="entries are invalid"):
        replace(admission, entries={})
    with pytest.raises(ValueError, match="entries are invalid"):
        replace(
            admission,
            entries={
                "ones": replace(
                    admission.entries["ones"],
                    route=ComputationRoute.CPU_FALLBACK,
                )
            },
        )
    with pytest.raises(ValueError, match="requires profile constraints"):
        RouteEntry(
            canonical_key="ones",
            source_api="numpy.ones",
            asnumpy_api="asnumpy.ones",
            route=ComputationRoute.NATIVE_NPU,
            evidence="unscoped",
            signature_variants=(OperatorSignature(positional_args=1),),
        )
