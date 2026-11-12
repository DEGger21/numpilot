"""D1001_signed_compression: Two signed cube-root feature branches.

Inputs are constructed on the CPU; forward returns named device arrays.
"""
import numpy as np
import asnumpy as ap

def build_inputs():
    rng = np.random.default_rng(401)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "y": rng.standard_normal((16384, 256), dtype=np.float32),
    }

def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.mean(x, axis=1, keepdims=True)
    __numpilot_v_2 = ap.subtract(x, __numpilot_v_1)
    __numpilot_v_3 = ap.mean(y, axis=1, keepdims=True)
    __numpilot_v_4 = ap.subtract(y, __numpilot_v_3)
    __numpilot_v_5 = ap.absolute(__numpilot_v_2)
    __numpilot_v_6 = ap.maximum(__numpilot_v_5, ap.full(__numpilot_v_5.shape, 0.25, dtype=__numpilot_v_5.dtype))
    __numpilot_v_7 = ap.divide(__numpilot_v_2, __numpilot_v_6)
    __numpilot_v_8 = ap.absolute(__numpilot_v_4)
    __numpilot_v_9 = ap.maximum(__numpilot_v_8, ap.full(__numpilot_v_8.shape, 0.25, dtype=__numpilot_v_8.dtype))
    __numpilot_v_10 = ap.divide(__numpilot_v_4, __numpilot_v_9)
    __numpilot_v_11 = ap.copysign(ap.power(ap.absolute(__numpilot_v_7), ap.full_like(__numpilot_v_7, 0.3333333333333333, dtype=__numpilot_v_7.dtype)), __numpilot_v_7)
    __numpilot_v_12 = ap.copysign(ap.power(ap.absolute(__numpilot_v_10), ap.full_like(__numpilot_v_10, 0.3333333333333333, dtype=__numpilot_v_10.dtype)), __numpilot_v_10)
    __numpilot_v_13 = ap.add(__numpilot_v_11, __numpilot_v_12)
    __numpilot_v_14 = ap.tanh(__numpilot_v_13)
    __numpilot_v_15 = ap.square(__numpilot_v_14)
    __numpilot_v_16 = ap.mean(__numpilot_v_15, axis=1)
    return {'out': __numpilot_v_14, 'energy': __numpilot_v_16}