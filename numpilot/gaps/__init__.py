"""Fail-closed operator-gap completion for NumPilot."""

from numpilot.gaps.models import GapCallsite, GapSpec, OperatorDraft
from numpilot.gaps.pipeline import (
    GapCompletionError,
    GapCompletionPipeline,
    GapCompletionResult,
)
from numpilot.gaps.provider import LlmDraftProvider
from numpilot.gaps.qualification import (
    CapabilityOverlay,
    GateEvidence,
    OverlayDurabilityError,
    OverlayPostPublishError,
    QualificationError,
    QualificationEvidence,
    QualificationRef,
    RuntimeBinding,
    ZERO_SHA256,
    apply_overlay,
    capability_manifest_sha256,
    compute_qualification_id,
    load_overlay,
    load_qualification,
    publish_overlay_atomic,
    qualification_constraints_sha256,
    validate_qualification,
)

__all__ = [
    "CapabilityOverlay",
    "GapCallsite",
    "GapCompletionError",
    "GapCompletionPipeline",
    "GapCompletionResult",
    "GapSpec",
    "GateEvidence",
    "LlmDraftProvider",
    "OperatorDraft",
    "OverlayDurabilityError",
    "OverlayPostPublishError",
    "QualificationError",
    "QualificationEvidence",
    "QualificationRef",
    "RuntimeBinding",
    "ZERO_SHA256",
    "apply_overlay",
    "capability_manifest_sha256",
    "compute_qualification_id",
    "load_overlay",
    "load_qualification",
    "publish_overlay_atomic",
    "qualification_constraints_sha256",
    "validate_qualification",
]
