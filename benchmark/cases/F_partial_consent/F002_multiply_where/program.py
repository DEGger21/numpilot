"""F002_multiply_where: Masked multiplication preserves sentinels.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Witness NumPy output: [6, -7, 18].
Pinned 1929a3ba wrapper multiply rejects where/out: TypeError before its body.
This signature mismatch is checkable without a CANN runtime.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([2, 4, 6], dtype=np.float32),
        'mask': np.array([True, False, True]),
        'buffer': np.full(3, -7, dtype=np.float32),
    }


def forward(x, mask, buffer):
    np.multiply(x, 3.0, out=buffer, where=mask)
    return {
        'out': buffer,
        'magnitude': np.absolute(buffer),
    }
