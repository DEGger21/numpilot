"""C003_layer_normalization: Shared row moments feed normalization and diagnostics.

Inputs are constructed on the CPU; forward returns named device arrays.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(303)
    x = rng.standard_normal((16384, 256), dtype=np.float32)
    x[0] = 4.0
    return {
        "x": x,
        "gain": np.linspace(0.5, 1.5, 256, dtype=np.float32),
        "bias": np.linspace(-0.2, 0.2, 256, dtype=np.float32),
    }


def forward(x, gain, bias):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(gain, np.ndarray):
        gain = ap.ndarray.from_numpy(gain)
    if isinstance(bias, np.ndarray):
        bias = ap.ndarray.from_numpy(bias)
    __numpilot_v_1 = ap.mean(x, axis=1, keepdims=True)
    __numpilot_v_2 = ap.subtract(x, __numpilot_v_1)
    __numpilot_v_3 = ap.multiply(__numpilot_v_2, __numpilot_v_2)
    __numpilot_v_4 = ap.mean(__numpilot_v_3, axis=1, keepdims=True)
    __numpilot_v_5 = ap.add(__numpilot_v_4, ap.full(__numpilot_v_4.shape, 1e-05, dtype=__numpilot_v_4.dtype))
    __numpilot_v_6 = ap.sqrt(__numpilot_v_5)
    __numpilot_v_7 = ap.divide(__numpilot_v_2, __numpilot_v_6)
    __numpilot_v_8 = ap.multiply(__numpilot_v_7, gain)
    __numpilot_v_9 = ap.add(__numpilot_v_8, bias)
    __numpilot_v_10 = ap.clip(__numpilot_v_9, -3.0, 3.0)
    __numpilot_v_11 = ap.tanh(__numpilot_v_10)
    __numpilot_v_12 = ap.mean(__numpilot_v_7, axis=1)
    __numpilot_v_13 = ap.square(__numpilot_v_7)
    __numpilot_v_14 = ap.mean(__numpilot_v_13, axis=1)
    return {'out': __numpilot_v_11, 'mean': __numpilot_v_12, 'power': __numpilot_v_14}