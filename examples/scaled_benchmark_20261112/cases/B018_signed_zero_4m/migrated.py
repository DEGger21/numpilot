"""B018_signed_zero: Signs of IEEE zero survive copy-sign.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: IEEE signed zero, not distinguishable by ordinary equality.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    return {
        "x": np.resize(np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32), (4_000_000,)),
        "y": np.ones(4_000_000, dtype=np.float32),
    }


def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.signbit(x)
    __numpilot_v_2 = ap.copysign(y, x)
    return {'sign': __numpilot_v_1, 'out': __numpilot_v_2}