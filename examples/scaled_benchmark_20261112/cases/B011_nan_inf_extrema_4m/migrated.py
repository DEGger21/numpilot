"""B011_nan_inf_extrema: Propagating and ignoring NaN differ.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: NaN/Inf propagation is different for maximum and fmax.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    return {
        "x": np.resize(np.array([np.nan, np.inf, -np.inf, 1], dtype=np.float32), (4_000_000,)),
        "y": np.resize(np.array([2, 3, -4, np.nan], dtype=np.float32), (4_000_000,)),
    }


def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.maximum(x, y)
    __numpilot_v_2 = ap.fmax(x, y)
    return {'maximum': __numpilot_v_1, 'fmax': __numpilot_v_2}