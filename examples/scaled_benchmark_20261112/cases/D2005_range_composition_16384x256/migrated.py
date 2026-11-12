"""D2005_range_composition: Range-normalized row differences.

Inputs are constructed on the CPU; forward returns named device arrays.
Equivalent: ptp(x, axis=1, keepdims=True) = max(x,1,True)-min(x,1,True).
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(455)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "y": rng.standard_normal((16384, 256), dtype=np.float32),
    }


def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.subtract(x, y)
    __numpilot_v_2 = ap.subtract(ap.max(__numpilot_v_1, axis=1, keepdims=True), ap.min(__numpilot_v_1, axis=1, keepdims=True))
    __numpilot_v_3 = ap.min(__numpilot_v_1, axis=1, keepdims=True)
    __numpilot_v_4 = ap.subtract(__numpilot_v_1, __numpilot_v_3)
    __numpilot_v_5 = ap.maximum(__numpilot_v_2, ap.full(__numpilot_v_2.shape, 1e-05, dtype=__numpilot_v_2.dtype))
    __numpilot_v_6 = ap.divide(__numpilot_v_4, __numpilot_v_5)
    __numpilot_v_7 = ap.clip(__numpilot_v_6, 0.0, 1.0)
    return {'out': __numpilot_v_7, 'span': __numpilot_v_2, 'minimum': __numpilot_v_3}