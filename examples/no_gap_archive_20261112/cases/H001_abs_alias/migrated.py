"""H001_abs_alias: Absolute-value alias on signed samples.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Alias reference; NumPy >= 2.0.
np.abs(x) = np.absolute(x).
CPU input construction and scalar output boxing are outside compute coverage.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(801)
    x = rng.standard_normal((5, 9), dtype=np.float32)
    x[0, 0] = -0.0
    return {'x': x, 'y': x[:, ::-1].copy()}


def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.absolute(x)
    __numpilot_v_2 = ap.absolute(y)
    __numpilot_v_3 = ap.subtract(__numpilot_v_1, __numpilot_v_2)
    return {'left': __numpilot_v_1, 'difference': __numpilot_v_3}