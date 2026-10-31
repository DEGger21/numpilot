"""F005_max_initial: Maximum includes the initial candidate.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Witness NumPy output: [0, 0].
Pinned 1929a3ba wrapper max rejects initial: TypeError before its body.
This signature mismatch is checkable without a CANN runtime.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([[-5, -2], [-3, -4]], dtype=np.float32),
    }


def forward(x):
    out = np.max(x, axis=1, initial=0.0)
    return {
        'out': out,
        'shifted': np.add(out, 1.0),
    }
