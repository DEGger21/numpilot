"""C004_cosine_rows: Cosine similarities with zero-row safeguards.

Inputs are constructed on the CPU; forward returns named AsNumpy arrays.
Source: ../../../../examples/custom_operator_cosine_similarity.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(304)
    x = rng.standard_normal((16384, 256), dtype=np.float32)
    y = rng.standard_normal((16384, 256), dtype=np.float32)
    x[0] = 0
    y[1] = 0
    return {
        "x": x,
        "y": y,
    }


def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.square(x)
    __numpilot_v_2 = ap.sum(__numpilot_v_1, axis=1)
    __numpilot_v_3 = ap.sqrt(__numpilot_v_2)
    __numpilot_v_4 = ap.square(y)
    __numpilot_v_5 = ap.sum(__numpilot_v_4, axis=1)
    __numpilot_v_6 = ap.sqrt(__numpilot_v_5)
    __numpilot_v_7 = ap.multiply(x, y)
    __numpilot_v_8 = ap.sum(__numpilot_v_7, axis=1)
    __numpilot_v_9 = ap.multiply(__numpilot_v_3, __numpilot_v_6)
    __numpilot_v_10 = ap.maximum(__numpilot_v_9, ap.full(__numpilot_v_9.shape, 1e-06, dtype=__numpilot_v_9.dtype))
    __numpilot_v_11 = ap.divide(__numpilot_v_8, __numpilot_v_10)
    __numpilot_v_12 = ap.subtract(ap.full(__numpilot_v_11.shape, 1.0, dtype=__numpilot_v_11.dtype), __numpilot_v_11)
    __numpilot_v_13 = ap.clip(__numpilot_v_12, 0.0, 2.0)
    __numpilot_v_14 = ap.sqrt(__numpilot_v_13)
    __numpilot_v_15 = ap.sqrt(__numpilot_v_14)
    return {'cosine': __numpilot_v_11, 'distance': __numpilot_v_12, 'score': __numpilot_v_15}