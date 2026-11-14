"""B002_where_preserves: Masked update preserves unselected sentinels.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Traps: where=, out=. Pinned multiply rejects these keywords.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([2, 4, 8, 16], dtype=np.float32),
        'mask': np.array([True, False, True, False]),
        'buffer': np.array([10, 20, 30, 40], dtype=np.float32),
    }


def forward(x, mask, buffer):
    np.multiply(x, 2.0, out=buffer, where=mask)
    return {
        'out': buffer,
    }
