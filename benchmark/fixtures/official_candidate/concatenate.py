"""Minimal official-candidate hand-off fixture.

The fixture intentionally contains one NumPy compute call only.  Inputs are
plain Python lists so the official resolver can audit ``numpy.concatenate``
without an unrelated random/default_rng capability decision.
"""
import numpy as np


def build_inputs():
    return {
        "x": [[1.0, 2.0], [3.0, 4.0]],
        "y": [[5.0], [6.0]],
    }


def forward(x, y):
    return {"out": np.concatenate((x, y), axis=1)}
