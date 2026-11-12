"""B006_transposed_input: Noncontiguous transposed matrix.

Inputs are constructed on the CPU; forward returns named AsNumpy arrays.
Trap: noncontiguous input; x is F-contiguous, not C-contiguous.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    return {'x': base.T, 'y': np.full((6, 4), 0.5, dtype=np.float32)}


def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.multiply(x, y)
    return {'out': __numpilot_v_1}