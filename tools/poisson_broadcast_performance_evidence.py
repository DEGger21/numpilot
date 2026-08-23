#!/usr/bin/env python3
"""Run the fail-closed Poisson benchmark with broadcast-optimized inputs.

The evidence protocol is inherited unchanged from
``poisson_performance_evidence.py``.  Only the deterministic input contract is
switched to the optimization variant.  Keeping this as a separate runner
preserves the hash identity of the published baseline evidence.
"""

from pathlib import Path

import numpy as np

from examples.poisson_ritz_iteration_broadcast import build_scaled_inputs
from tools import poisson_performance_evidence as _baseline


def _build_scaled_inputs(
    interior_size: int,
    boundary_size: int,
    centre_size: int,
) -> list:
    inputs = list(build_scaled_inputs(interior_size, boundary_size, centre_size))
    if len(inputs) != 28:
        raise ValueError(
            f"broadcast Poisson iteration contract requires 28 inputs, got {len(inputs)}"
        )
    if any(not isinstance(item, np.ndarray) for item in inputs):
        raise TypeError("every broadcast Poisson input must be a NumPy ndarray")
    return inputs


def main(argv=None) -> int:
    """Install the optimized contract, then run the unchanged protocol."""

    # The baseline implementation resolves ``__file__`` at call time for
    # worker subprocesses and evidence identity.  Pointing that module global
    # at this wrapper makes every worker re-enter this runner and binds
    # receipts to this file, while leaving the published baseline file
    # byte-for-byte unchanged.  Keep the patch inside main so importing this
    # helper cannot mutate another test or evidence reader's module state.
    _baseline._build_scaled_inputs = _build_scaled_inputs
    _baseline.__file__ = str(Path(__file__).resolve())
    return _baseline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
