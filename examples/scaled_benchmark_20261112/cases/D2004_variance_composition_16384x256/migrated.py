"""D2004_variance_composition: Variance-driven residual normalization.

Inputs are constructed on the CPU; forward returns named AsNumpy arrays.
Equivalent: var(x, axis=1, keepdims=True) = mean(square(x-mean(x,1,True)),1,True).
Only the named alias/composition is absent from the pinned export catalog.
"""
import numpy as np
import asnumpy as ap

def build_inputs():
    rng = np.random.default_rng(454)
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
    __numpilot_v_2 = ap.mean(ap.square(ap.subtract(__numpilot_v_1, ap.mean(__numpilot_v_1, axis=1, keepdims=True))), axis=1, keepdims=True)
    __numpilot_v_3 = ap.mean(__numpilot_v_1, axis=1, keepdims=True)
    __numpilot_v_4 = ap.subtract(__numpilot_v_1, __numpilot_v_3)
    __numpilot_v_5 = ap.add(__numpilot_v_2, ap.full(__numpilot_v_2.shape, 0.0001, dtype=__numpilot_v_2.dtype))
    __numpilot_v_6 = ap.sqrt(__numpilot_v_5)
    __numpilot_v_7 = ap.divide(__numpilot_v_4, __numpilot_v_6)
    __numpilot_v_8 = ap.tanh(__numpilot_v_7)
    return {'out': __numpilot_v_8, 'variance': __numpilot_v_2, 'scale': __numpilot_v_6}