"""B004_mixed_int_float: Promotion of int32 and float32.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: dtype promotion; 16777217 detects premature float32 conversion.
"""
import numpy as np
import asnumpy as ap

def build_inputs():
    rng = np.random.default_rng(204)
    x = rng.integers(-1000, 1000, size=(4_000_000,), dtype=np.int32)
    x[0] = 16777217
    return {
        "x": x,
        "y": rng.standard_normal((4_000_000,), dtype=np.float32),
    }

def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.add(x, y)
    return {'out': __numpilot_v_1}