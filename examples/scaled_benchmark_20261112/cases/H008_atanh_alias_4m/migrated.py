"""H008_atanh_alias: Inverse hyperbolic tangent inside the open domain.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Alias reference; NumPy >= 2.0.
np.atanh(x) = np.arctanh(x).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np
import asnumpy as ap

def build_inputs():
    x = np.linspace(-0.95, 0.95, 4_000_000, dtype=np.float64)
    return {
        "x": x,
        "shrink": np.full(4_000_000, 0.5, dtype=np.float64),
    }

def forward(x, shrink):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(shrink, np.ndarray):
        shrink = ap.ndarray.from_numpy(shrink)
    __numpilot_v_1 = ap.arctanh(x)
    __numpilot_v_2 = ap.multiply(x, shrink)
    __numpilot_v_3 = ap.arctanh(__numpilot_v_2)
    __numpilot_v_4 = ap.subtract(__numpilot_v_1, __numpilot_v_3)
    return {'full': __numpilot_v_1, 'half': __numpilot_v_3, 'difference': __numpilot_v_4}