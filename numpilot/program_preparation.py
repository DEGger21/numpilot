"""Whole-program preparation for the ordinary migration mainline.

The analyzer intentionally sees one straight-line ``forward`` body.  Real
programs use helpers, static loops, destructuring and documented equivalent
NumPy spellings.  This module makes those transformations an explicit,
inspectable step *before* analysis instead of a benchmark-only adapter.

The original source is never replaced as the numerical oracle.  Preparation
only produces a second, analysis/migration surface plus a deterministic list
of the transformations that produced it.  If any transformation cannot be
proved safe, the original source is returned unchanged and the normal
fail-closed analyzer reports the unsupported construct.
"""

from __future__ import annotations

from dataclasses import dataclass

from numpilot.benchmark_lowering import BenchmarkLowering, lower_benchmark_program


@dataclass(frozen=True)
class ProgramPreparation:
    original_source: str
    analysis_source: str
    lowering: BenchmarkLowering

    @property
    def applied(self) -> bool:
        return bool(self.lowering.applied) or (
            self.analysis_source != self.original_source
        )

    @property
    def normalized(self) -> bool:
        """True when only the straight-line surface changed."""

        return self.analysis_source != self.original_source and not self.rewrites

    @property
    def rewrites(self) -> tuple:
        return tuple(self.lowering.rewrites)

    @property
    def notes(self) -> tuple:
        notes = tuple(self.lowering.notes)
        if self.normalized and not notes:
            notes = (
                "analysis surface normalized to straight-line SSA temporaries",
            )
        return notes

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "analysis_source_changed": self.analysis_source != self.original_source,
            "normalized": self.normalized,
            "rewrites": [item.to_dict() for item in self.rewrites],
            "notes": list(self.notes),
            "oracle_policy": "original_source_remains_the_numerical_oracle",
        }


def prepare_program(source: str) -> ProgramPreparation:
    """Return the deterministic analysis/migration surface for one program."""

    lowering = lower_benchmark_program(source)
    return ProgramPreparation(
        original_source=source,
        analysis_source=lowering.source,
        lowering=lowering,
    )
