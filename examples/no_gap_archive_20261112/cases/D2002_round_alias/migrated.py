import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(452)
    return {
        'x': rng.standard_normal((8, 13), dtype=np.float32),
        'y': rng.standard_normal((8, 13), dtype=np.float32),
    }


def forward(x, y):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)

    __numpilot_v_1 = ap.mean(x, axis=1, keepdims=True)
    __numpilot_v_2 = ap.subtract(x, __numpilot_v_1)
    __numpilot_v_3 = ap.square(__numpilot_v_2)
    __numpilot_v_4 = ap.mean(__numpilot_v_3, axis=1, keepdims=True)
    __numpilot_v_5 = ap.full(__numpilot_v_4.shape, 1e-05, dtype=__numpilot_v_4.dtype)
    __numpilot_v_6 = ap.add(__numpilot_v_4, __numpilot_v_5)
    __numpilot_v_7 = ap.sqrt(__numpilot_v_6)
    __numpilot_v_8 = ap.divide(__numpilot_v_2, __numpilot_v_7)
    __numpilot_v_9 = ap.around(__numpilot_v_8, decimals=2)
    __numpilot_v_10 = ap.subtract(__numpilot_v_9, __numpilot_v_8)
    return {'out': __numpilot_v_9, 'error': __numpilot_v_10, 'reference': y}