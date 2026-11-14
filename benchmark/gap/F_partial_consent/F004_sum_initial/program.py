"""F004_sum_initial: Reduction initial value is included.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Witness NumPy output: [13, 17].
Pinned 1929a3ba wrapper sum rejects initial: TypeError before its body.
This signature mismatch is checkable without a CANN runtime.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([[1, 2], [3, 4]], dtype=np.float32),
    }


def forward(x):
    out = np.sum(x, axis=1, initial=10.0)
    return {
        'out': out,
        'squared': np.square(out),
    }
