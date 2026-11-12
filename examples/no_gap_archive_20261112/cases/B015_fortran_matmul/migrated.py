"""B015_fortran_matmul: Matrix product with Fortran-ordered inputs.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: both inputs use column-major storage.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(215)
    x = np.asfortranarray(rng.standard_normal((6, 5)).astype(np.float32))
    y = np.asfortranarray(rng.standard_normal((5, 3)).astype(np.float32))
    return {'x': x, 'y': y}


def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.matmul(x, y)
    __numpilot_v_2 = ap.square(__numpilot_v_1)
    __numpilot_v_3 = ap.sum(__numpilot_v_2, axis=1)
    return {'out': __numpilot_v_1, 'row_power': __numpilot_v_3}