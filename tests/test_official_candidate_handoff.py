from types import SimpleNamespace

import pytest

from numpilot.gaps.official_candidate_handoff import (
    OfficialCandidateHandoffError,
    _candidate_evidence,
)
from numpilot.gaps.official_resolution import INSTALLED_ADAPTER_ROUTE


def _decision(**installed):
    return SimpleNamespace(
        route=INSTALLED_ADAPTER_ROUTE,
        activation_allowed=False,
        installed_cann={
            "adapter_candidate": True,
            "missing_symbols": [],
            "candidate": {"candidate_id": "numpy.concatenate-aclnn-cat"},
            "selected_header": {
                "relative_path": "aarch64-linux/include/aclnnop/aclnn_cat.h",
                "sha256": "a" * 64,
            },
            "selected_library": {
                "relative_path": "aarch64-linux/lib64/libopapi.so",
                "sha256": "b" * 64,
            },
            "required_symbols": ["aclnnCatGetWorkspaceSize", "aclnnCat"],
            "semantic_equivalence": "unverified",
            **installed,
        },
        gap_id="concat-1",
        qualified_api="numpy.concatenate",
        gap_spec_sha256="c" * 64,
    )


def test_candidate_evidence_binds_both_acl_nn_stages():
    evidence = _candidate_evidence(_decision())
    assert evidence["route"] == INSTALLED_ADAPTER_ROUTE
    assert evidence["required_symbols"] == [
        "aclnnCatGetWorkspaceSize",
        "aclnnCat",
    ]
    assert evidence["header_sha256"] == "a" * 64
    assert evidence["library_sha256"] == "b" * 64


def test_candidate_evidence_rejects_missing_symbol_or_ambiguous_route():
    with pytest.raises(OfficialCandidateHandoffError):
        _candidate_evidence(_decision(missing_symbols=["aclnnCat"]))

    bad_route = _decision()
    bad_route.route = "new_kernel_generation_allowed"
    with pytest.raises(OfficialCandidateHandoffError):
        _candidate_evidence(bad_route)


def test_candidate_evidence_rejects_activation_claim():
    bad = _decision()
    bad.activation_allowed = True
    with pytest.raises(OfficialCandidateHandoffError):
        _candidate_evidence(bad)
