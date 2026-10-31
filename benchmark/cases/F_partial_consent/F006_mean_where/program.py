"""F006_mean_where: Masked row means exclude unselected entries.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Witness NumPy output: [2, 7].
Pinned 1929a3ba wrapper mean rejects where: TypeError before its body.
This signature mismatch is checkable without a CANN runtime.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([[1, 99, 3], [6, 8, 100]], dtype=np.float32),
        'mask': np.array([[True, False, True], [True, True, False]]),
    }


def forward(x, mask):
    out = np.mean(x, axis=1, where=mask)
    return {
        'out': out,
        'root': np.sqrt(out),
    }
