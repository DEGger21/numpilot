"""H004_asin_alias: Inverse sine for a symmetric angular grid.

Inputs are constructed on the CPU; forward returns named device arrays.
Alias reference; NumPy >= 2.0.
np.asin(x) = np.arcsin(x).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    x = np.linspace(-0.9, 0.9, 25, dtype=np.float64)
    return {'x': x, 'offset': np.full(25, 0.025, dtype=np.float64)}


def forward(x, offset):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(offset, np.ndarray):
        offset = ap.ndarray.from_numpy(offset)
    __numpilot_v_1 = ap.arcsin(x)
    __numpilot_v_2 = ap.add(x, offset)
    __numpilot_v_3 = ap.arcsin(__numpilot_v_2)
    __numpilot_v_4 = ap.subtract(__numpilot_v_3, __numpilot_v_1)
    __numpilot_v_5 = ap.divide(__numpilot_v_4, offset)
    return {'base': __numpilot_v_1, 'sensitivity': __numpilot_v_5}