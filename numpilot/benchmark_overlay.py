"""Overlay-scoped admission for the deterministic benchmark.

The official operator overlay and optional baseline API certificate remain
independent evidence chains, both bound to the live runtime.  Only migrated
``forward`` calls are profiled; ``build_inputs`` remains the CPU input builder.

The shim consumer uses explicit semantic stand-ins.  The NPU consumer uses the
validated resolver and records separate callsite qualification evidence.  Both
share the same exact argument profiler, including reduction control arguments
and creation metadata.  Neither promotes a whole-runtime provenance audit.
"""
from __future__ import annotations

import ast
import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple

from numpilot.agents.analysis import (
    numpy_bindings_for_forward,
    resolve_numpy_function,
)
from numpilot.capabilities import (
    AnalysisCallsiteProfile,
    AnalysisProfileSet,
    profile_api_call,
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
    overlay_path: str = ""
    evidence_root: str = ""
    baseline_certificate_path: str = ""
    baseline_evidence_root: str = ""
    baseline_admission: dict | None = None

    def to_dict(self) -> dict:
        result = {
            "qualification_id": self.qualification_id,
            "overlay_sha256": self.overlay_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "operators": list(self.operators),
            "qualified_names": list(self.qualified_names),
            "admission": "official_qualification_overlay",
            "execution": "shim_semantic_verification",
        }
        if self.baseline_admission is not None:
            result["baseline_api"] = dict(self.baseline_admission)
        return result


@dataclass(frozen=True)
class OverlayProfiles:
    """Profile build result; ``profiles`` is None when nothing was admitted."""

    profiles: AnalysisProfileSet | None
    matched_callsites: int
    attempted_callsites: int
    notes: Tuple[str, ...] = ()


def load_benchmark_overlay(overlay_path: str, evidence_root: str, *,
                           baseline_api_certificate: str | None = None,
                           baseline_api_evidence_root: str | None = None) -> BenchmarkOverlay:
    """Validate one official overlay against the live overlay runtime.

    Importing ``asnumpy`` must resolve to the qualified wheel.  A different
    installation fails the admission instead of silently degrading to an
    unproven surface.
    """

    from numpilot.backends.npu import NpuBackend, OverlayRuntimeConfig, BaselineApiRuntimeConfig

    if bool(baseline_api_certificate) != bool(baseline_api_evidence_root):
        raise ValueError("baseline API certificate and evidence root must be supplied together")
    baseline_config = (BaselineApiRuntimeConfig(baseline_api_certificate, baseline_api_evidence_root)
                       if baseline_api_certificate else None)
    resolver = NpuBackend(
        overlay_config=OverlayRuntimeConfig(
            overlay_path=str(overlay_path),
            evidence_root=str(evidence_root),
        ),
        baseline_api_config=baseline_config,
    )
    identity = resolver.runtime_identity()
    admission = resolver.qualified_overlay_admission(identity)
    if admission is None:
        raise ValueError("capability overlay is not an official qualification overlay")
    entries = admission.entries
    if not entries:
        raise ValueError("official capability overlay admits no operators")
    baseline = resolver.qualified_baseline_api_admission(identity) if baseline_config else None
    operators = set(entries)
    qualified_names = {entry.source_api for entry in entries.values()}
    if baseline is not None:
        operators.update(baseline["operators"])
        qualified_names.update(baseline["qualified_names"])
    return BenchmarkOverlay(
        qualification_id=admission.qualification_id,
        overlay_sha256=admission.overlay_sha256,
        runtime_identity_sha256=admission.runtime_identity_sha256,
        operators=tuple(sorted(operators)),
        qualified_names=tuple(sorted(qualified_names)),
        resolver=resolver,
        overlay_path=str(overlay_path),
        evidence_root=str(evidence_root),
        baseline_certificate_path=str(baseline_api_certificate or ""),
        baseline_evidence_root=str(baseline_api_evidence_root or ""),
        baseline_admission=baseline,
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
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) or function.name != "forward":
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



def _record_callsite(site, args, kwargs):
    if len(args) != site.positional_args:
        return None
    if tuple(sorted(kwargs)) != site.keyword_args:
        return None
    profile = profile_api_call(site.name, args, kwargs)
    if profile.unprofiled_operands:
        return None
    if not profile.inputs and profile.creation_output is None:
        return None
    if site.name == "concatenate" and profile.sequence_axis is None:
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
