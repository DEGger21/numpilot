"""H005_atan_alias: Inverse tangent compression at three scales.

Inputs are constructed on the CPU; forward returns named AsNumpy arrays.
Alias reference; NumPy >= 2.0.
np.atan(x) = np.arctan(x).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np
import asnumpy as ap

def build_inputs():
    x = np.array([-100, -10, -1, 0, 1, 10, 100], dtype=np.float32)
    return {'x': x, 'scale': np.full(7, 0.25, dtype=np.float32)}

def forward(x, scale):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(scale, np.ndarray):
        scale = ap.ndarray.from_numpy(scale)
    __numpilot_v_1 = ap.multiply(x, scale)
    __numpilot_v_2 = ap.arctan(__numpilot_v_1)
    __numpilot_v_3 = ap.arctan(x)
    __numpilot_v_4 = ap.subtract(__numpilot_v_3, __numpilot_v_2)
    return {'fine': __numpilot_v_2, 'coarse': __numpilot_v_3, 'delta': __numpilot_v_4}