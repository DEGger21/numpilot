"""D2003_acos_alias: Angular distance from bounded correlations.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Equivalent: np.acos(cosine) -> np.arccos(cosine); NumPy >= 2.0.
Only the named alias/composition is absent from the pinned export catalog.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(453)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "y": rng.standard_normal((16384, 256), dtype=np.float32),
    }


def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    __numpilot_v_1 = ap.multiply(x, y)
    __numpilot_v_2 = ap.sum(__numpilot_v_1, axis=1)
    __numpilot_v_3 = ap.square(x)
    __numpilot_v_4 = ap.sum(__numpilot_v_3, axis=1)
    __numpilot_v_5 = ap.sqrt(__numpilot_v_4)
    __numpilot_v_6 = ap.square(y)
    __numpilot_v_7 = ap.sum(__numpilot_v_6, axis=1)
    __numpilot_v_8 = ap.sqrt(__numpilot_v_7)
    __numpilot_v_9 = ap.multiply(__numpilot_v_5, __numpilot_v_8)
    __numpilot_v_10 = ap.maximum(__numpilot_v_9, ap.full(__numpilot_v_9.shape, 1e-06, dtype=__numpilot_v_9.dtype))
    __numpilot_v_11 = ap.divide(__numpilot_v_2, __numpilot_v_10)
    __numpilot_v_12 = ap.clip(__numpilot_v_11, -1.0, 1.0)
    __numpilot_v_13 = ap.arccos(__numpilot_v_12)
    return {'angle': __numpilot_v_13, 'cosine': __numpilot_v_12}