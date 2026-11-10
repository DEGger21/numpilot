"""Overlay-scoped admission for the deterministic benchmark.

The 150-case benchmark executes migrated programs on the shim backend, which
verifies semantic preservation against NumPy.  It does not re-prove AsNumpy
device behaviour: that evidence comes from an official qualification overlay.

This module joins the two evidence sources for the benchmark only:

* the official overlay and its receipt chain are validated against the live
  runtime identity (the real qualification wheel must be importable);
* every source callsite of an admitted API is profiled by executing the
  benchmark program under NumPy and capturing the exact tensor operands;
* the shim backend receives explicit stand-ins for the overlay operators.

An admitted callsite therefore needs three independent facts: a qualified
overlay entry, an exact profile that satisfies that entry's constraints, and a
shim execution that reproduces NumPy semantics.  Missing any one keeps the
callsite unsupported.
"""
from __future__ import annotations

import ast
import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple

import numpy as _np

from numpilot.agents.analysis import (
    numpy_bindings_for_forward,
    resolve_numpy_function,
)
from numpilot.capabilities import (
    AnalysisCallsiteProfile,
    AnalysisProfileSet,
    OperatorCallProfile,
    profile_concatenate_call,
    tensor_profile_from_value,
)

_PROGRAM_FILENAME = "program.py"
_PROFILE_NAMESPACE = "numpilot_benchmark_profile"


@dataclass(frozen=True)
class BenchmarkOverlay:
    """Validated official overlay plus the live resolver that admitted it."""

    qualification_id: str
    overlay_sha256: str
    runtime_identity_sha256: str
    operators: Tuple[str, ...]
    qualified_names: Tuple[str, ...]
    resolver: object = field(repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "qualification_id": self.qualification_id,
            "overlay_sha256": self.overlay_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "operators": list(self.operators),
            "qualified_names": list(self.qualified_names),
            "admission": "official_qualification_overlay",
            "execution": "shim_semantic_verification",
        }


@dataclass(frozen=True)
class OverlayProfiles:
    """Profile build result; ``profiles`` is None when nothing was admitted."""

    profiles: AnalysisProfileSet | None
    matched_callsites: int
    attempted_callsites: int
    notes: Tuple[str, ...] = ()


def load_benchmark_overlay(overlay_path: str, evidence_root: str) -> BenchmarkOverlay:
    """Validate one official overlay against the live overlay runtime.

    Importing ``asnumpy`` must resolve to the qualified wheel.  A different
    installation fails the admission instead of silently degrading to an
    unproven surface.
    """

    from numpilot.backends.npu import NpuBackend, OverlayRuntimeConfig

    resolver = NpuBackend(
        overlay_config=OverlayRuntimeConfig(
            overlay_path=str(overlay_path),
            evidence_root=str(evidence_root),
        )
    )
    identity = resolver.runtime_identity()
    admission = resolver.qualified_overlay_admission(identity)
    if admission is None:
        raise ValueError("capability overlay is not an official qualification overlay")
    entries = admission.entries
    if not entries:
        raise ValueError("official capability overlay admits no operators")
    return BenchmarkOverlay(
        qualification_id=admission.qualification_id,
        overlay_sha256=admission.overlay_sha256,
        runtime_identity_sha256=admission.runtime_identity_sha256,
        operators=tuple(sorted(entries)),
        qualified_names=tuple(sorted(entry.source_api for entry in entries.values())),
        resolver=resolver,
    )


@dataclass
class _Callsite:
    name: str
    qualified_name: str
    lineno: int
    col_offset: int
    positional_args: int
    keyword_args: Tuple[str, ...]
    used: bool = False


def _overlay_callsites(tree: ast.Module, overlay: BenchmarkOverlay) -> list[_Callsite]:
    qualified_names = set(overlay.qualified_names)
    sites = []
    for function in tree.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bindings = numpy_bindings_for_forward(tree, function)
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            resolved = resolve_numpy_function(node.func, bindings)
            if resolved is None or resolved[1] not in qualified_names:
                continue
            sites.append(
                _Callsite(
                    name=resolved[0],
                    qualified_name=resolved[1],
                    lineno=getattr(node, "lineno", 0),
                    col_offset=getattr(node, "col_offset", 0),
                    positional_args=len(node.args),
                    keyword_args=tuple(
                        sorted(keyword.arg or "**" for keyword in node.keywords)
                    ),
                )
            )
    sites.sort(key=lambda site: (site.lineno, site.col_offset))
    return sites


_SCALAR_TYPES = (bool, int, float, complex, str, bytes, type(None))


def _tensor_operands(value):
    """Return ``(tensors, ok)`` for one runtime operand.

    Scalars are signature facts, not tensor operands.  A tuple/list is part of
    the public calling convention (``concatenate`` accepts a sequence), so its
    leaves are flattened.  Any other object is unprofiled and rejects the
    callsite.
    """

    if type(value) is _np.ndarray:
        return [value], True
    if isinstance(value, (tuple, list)):
        tensors = []
        for item in value:
            nested, ok = _tensor_operands(item)
            if not ok:
                return [], False
            tensors.extend(nested)
        return tensors, True
    if isinstance(value, _SCALAR_TYPES):
        return [], True
    return [], False


def _profile_call(args, kwargs):
    tensors = []
    for value in args:
        nested, ok = _tensor_operands(value)
        if not ok:
            return None
        tensors.extend(nested)
    for value in kwargs.values():
        nested, ok = _tensor_operands(value)
        if not ok:
            return None
        tensors.extend(nested)
    if not tensors:
        return None
    profiles = []
    for tensor in tensors:
        profile = tensor_profile_from_value(tensor)
        if profile is None:
            return None
        profiles.append(profile)
    try:
        return OperatorCallProfile(
            inputs=tuple(profiles),
            positional_args=len(args),
            keyword_args=tuple(sorted(kwargs)),
            unprofiled_operands=(),
        )
    except ValueError:
        # The v1 call-profile contract counts one tensor per public operand.
        # A sequence operand (concatenate's first argument) can only be
        # represented when the remaining signature supplies the matching
        # operand count, e.g. ``concatenate((a, b), axis=1)``.  Anything else
        # stays unprofiled rather than claiming a shape the contract cannot
        # express.
        return None


def _record_callsite(site, args, kwargs):
    if len(args) != site.positional_args:
        return None
    if tuple(sorted(kwargs)) != site.keyword_args:
        return None
    if site.name == "concatenate":
        profile = profile_concatenate_call(args, kwargs)
        if profile.sequence_axis is None or profile.unprofiled_operands or not profile.inputs:
            return None
    else:
        profile = _profile_call(args, kwargs)
    if profile is None:
        return None
    site.used = True
    return AnalysisCallsiteProfile(
        qualified_name=site.qualified_name,
        lineno=site.lineno,
        col_offset=site.col_offset,
        profile=profile,
    )


def _run_with_recorders(source, overlay, sites):
    import numpy

    by_line = defaultdict(list)
    for site in sites:
        by_line[site.lineno].append(site)

    recorded = []
    originals = {}
    for name in overlay.operators:
        original = getattr(numpy, name, None)
        if not callable(original):
            raise ValueError(
                "overlay operator %s is absent from the profiling NumPy" % name
            )
        originals[name] = original

        def make_recorder(operator_name, target):
            def recorder(*args, **kwargs):
                frame = sys._getframe(1)
                if (
                    frame.f_code.co_filename == _PROGRAM_FILENAME
                    and frame.f_lineno in by_line
                ):
                    for site in by_line[frame.f_lineno]:
                        if site.used or site.name != operator_name:
                            continue
                        entry = _record_callsite(site, args, kwargs)
                        if entry is not None:
                            recorded.append(entry)
                            break
                return target(*args, **kwargs)

            return recorder

        setattr(numpy, name, make_recorder(name, original))

    namespace = {"__name__": _PROFILE_NAMESPACE}
    try:
        exec(compile(source, _PROGRAM_FILENAME, "exec"), namespace)
        build_inputs = namespace.get("build_inputs")
        forward = namespace.get("forward")
        if not callable(build_inputs) or not callable(forward):
            raise TypeError("benchmark contract functions are missing")
        inputs = build_inputs()
        forward(**inputs)
    finally:
        for name, original in originals.items():
            setattr(numpy, name, original)
    return recorded


def _profile_fingerprint(callsites):
    return tuple(
        (
            callsite.qualified_name,
            callsite.lineno,
            callsite.col_offset,
            callsite.profile.to_dict(),
        )
        for callsite in sorted(
            callsites,
            key=lambda item: (item.lineno, item.col_offset),
        )
    )


def build_analysis_profiles(source: str, overlay: BenchmarkOverlay) -> OverlayProfiles:
    """Profile every overlay callsite reached by ``build_inputs``/``forward``."""

    tree = ast.parse(source)
    sites = _overlay_callsites(tree, overlay)
    if not sites:
        return OverlayProfiles(
            profiles=None,
            matched_callsites=0,
            attempted_callsites=0,
        )
    try:
        recorded = _run_with_recorders(source, overlay, sites)
        # A second execution must observe the same callsites and the same
        # dtype/shape/contiguity on every operand.  Nondeterministic tensor
        # shapes would make the static admission stale at verification time.
        repeated = _run_with_recorders(
            source, overlay, _overlay_callsites(tree, overlay)
        )
    except Exception as exc:  # one case must not abort the sweep
        return OverlayProfiles(
            profiles=None,
            matched_callsites=0,
            attempted_callsites=len(sites),
            notes=("overlay profiling failed: %s" % type(exc).__name__,),
        )
    if _profile_fingerprint(recorded) != _profile_fingerprint(repeated):
        return OverlayProfiles(
            profiles=None,
            matched_callsites=0,
            attempted_callsites=len(sites),
            notes=("overlay profiling is not deterministic",),
        )
    if not recorded:
        return OverlayProfiles(
            profiles=None,
            matched_callsites=0,
            attempted_callsites=len(sites),
            notes=("overlay profiling matched no profileable callsite",),
        )
    profiles = AnalysisProfileSet(
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        callsites=tuple(recorded),
    )
    return OverlayProfiles(
        profiles=profiles,
        matched_callsites=len(recorded),
        attempted_callsites=len(sites),
    )
