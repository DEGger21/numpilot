"""D2005_range_composition: Range-normalized row differences.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Equivalent: ptp(x, axis=1, keepdims=True) = max(x,1,True)-min(x,1,True).
Only the named alias/composition is absent from the pinned export catalog.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(455)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "y": rng.standard_normal((16384, 256), dtype=np.float32),
    }


def forward(x, y):
    residual = np.subtract(x, y)
    span = np.ptp(residual, axis=1, keepdims=True)
    minimum = np.min(residual, axis=1, keepdims=True)
    shifted = np.subtract(residual, minimum)
    maximum_1 = np.maximum(span, 1e-05)
    normalized = np.divide(shifted, maximum_1)
    out = np.clip(normalized, 0.0, 1.0)
    return {
        'out': out,
        'span': span,
        'minimum': minimum,
    }
