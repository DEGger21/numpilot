"""C002_mlp_inference: Two layer classifier with stable probabilities.

Inputs are constructed on the CPU; forward returns named AsNumpy arrays.
Source: ../../../../examples/mlp_forward.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(302)
    return {'x': rng.standard_normal((9, 12), dtype=np.float32), 'w1': rng.standard_normal((12, 8), dtype=np.float32) * 0.1, 'b1': np.linspace(-0.1, 0.1, 8, dtype=np.float32), 'w2': rng.standard_normal((8, 4), dtype=np.float32) * 0.2, 'b2': np.zeros(4, dtype=np.float32)}


def forward(x, w1, b1, w2, b2):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(w1, np.ndarray):
        w1 = ap.ndarray.from_numpy(w1)
    if isinstance(b1, np.ndarray):
        b1 = ap.ndarray.from_numpy(b1)
    if isinstance(w2, np.ndarray):
        w2 = ap.ndarray.from_numpy(w2)
    if isinstance(b2, np.ndarray):
        b2 = ap.ndarray.from_numpy(b2)
    __numpilot_v_1 = ap.matmul(x, w1)
    __numpilot_v_2 = ap.add(__numpilot_v_1, b1)
    __numpilot_v_3 = ap.maximum(__numpilot_v_2, ap.full(__numpilot_v_2.shape, 0.0, dtype=__numpilot_v_2.dtype))
    __numpilot_v_4 = ap.minimum(__numpilot_v_3, ap.full(__numpilot_v_3.shape, 6.0, dtype=__numpilot_v_3.dtype))
    __numpilot_v_5 = ap.matmul(__numpilot_v_4, w2)
    __numpilot_v_6 = ap.add(__numpilot_v_5, b2)
    __numpilot_v_7 = ap.max(__numpilot_v_6, axis=1, keepdims=True)
    __numpilot_v_8 = ap.subtract(__numpilot_v_6, __numpilot_v_7)
    __numpilot_v_9 = ap.exp(__numpilot_v_8)
    __numpilot_v_10 = ap.sum(__numpilot_v_9, axis=1, keepdims=True)
    __numpilot_v_11 = ap.divide(__numpilot_v_9, __numpilot_v_10)
    __numpilot_v_12 = ap.max(__numpilot_v_11, axis=1)
    __numpilot_v_13 = ap.sum(__numpilot_v_11, axis=1)
    return {'probabilities': __numpilot_v_11, 'confidence': __numpilot_v_12, 'mass': __numpilot_v_13}