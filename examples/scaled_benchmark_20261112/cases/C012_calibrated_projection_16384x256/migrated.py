"""C012_calibrated_projection: Matrix inference with iterative calibration.

Inputs are constructed on the CPU; forward returns named AsNumpy arrays.
Source: ../../../../examples/ops_nn_matmul_inference.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(312)
    return {
        "features": rng.standard_normal((16384, 256), dtype=np.float32) * 0.01,
        "weights": rng.standard_normal((256, 64), dtype=np.float32) * 0.02,
        "target": np.full((16384, 64), 0.25, dtype=np.float32),
    }


def forward(features, weights, target):
    if isinstance(features, np.ndarray):
        features = ap.ndarray.from_numpy(features)
    if isinstance(weights, np.ndarray):
        weights = ap.ndarray.from_numpy(weights)
    if isinstance(target, np.ndarray):
        target = ap.ndarray.from_numpy(target)
    __numpilot_v_1 = ap.matmul(features, weights)
    __numpilot_v_2 = ap.clip(__numpilot_v_1, -4.0, 4.0)
    __numpilot_v_3 = ap.max(__numpilot_v_2, axis=1, keepdims=True)
    __numpilot_v_4 = ap.subtract(__numpilot_v_2, __numpilot_v_3)
    __numpilot_v_5 = ap.exp(__numpilot_v_4)
    __numpilot_v_6 = ap.sum(__numpilot_v_5, axis=1, keepdims=True)
    __numpilot_v_7 = ap.divide(__numpilot_v_5, __numpilot_v_6)
    __numpilot_v_8 = ap.subtract(__numpilot_v_7, target)
    __numpilot_v_9 = ap.clip(__numpilot_v_8, -1.0, 1.0)
    __numpilot_v_10 = ap.multiply(__numpilot_v_9, ap.full(__numpilot_v_9.shape, 0.5, dtype=__numpilot_v_9.dtype))
    __numpilot_v_11 = ap.subtract(__numpilot_v_2, __numpilot_v_10)
    __numpilot_v_12 = ap.max(__numpilot_v_11, axis=1, keepdims=True)
    __numpilot_v_13 = ap.subtract(__numpilot_v_11, __numpilot_v_12)
    __numpilot_v_14 = ap.exp(__numpilot_v_13)
    __numpilot_v_15 = ap.sum(__numpilot_v_14, axis=1, keepdims=True)
    __numpilot_v_16 = ap.divide(__numpilot_v_14, __numpilot_v_15)
    __numpilot_v_17 = ap.subtract(__numpilot_v_16, target)
    __numpilot_v_18 = ap.clip(__numpilot_v_17, -1.0, 1.0)
    __numpilot_v_19 = ap.multiply(__numpilot_v_18, ap.full(__numpilot_v_18.shape, 0.5, dtype=__numpilot_v_18.dtype))
    __numpilot_v_20 = ap.subtract(__numpilot_v_11, __numpilot_v_19)
    __numpilot_v_21 = ap.max(__numpilot_v_20, axis=1, keepdims=True)
    __numpilot_v_22 = ap.subtract(__numpilot_v_20, __numpilot_v_21)
    __numpilot_v_23 = ap.exp(__numpilot_v_22)
    __numpilot_v_24 = ap.sum(__numpilot_v_23, axis=1, keepdims=True)
    __numpilot_v_25 = ap.divide(__numpilot_v_23, __numpilot_v_24)
    __numpilot_v_26 = ap.subtract(__numpilot_v_25, target)
    __numpilot_v_27 = ap.clip(__numpilot_v_26, -1.0, 1.0)
    __numpilot_v_28 = ap.multiply(__numpilot_v_27, ap.full(__numpilot_v_27.shape, 0.1, dtype=__numpilot_v_27.dtype))
    __numpilot_v_29 = ap.subtract(__numpilot_v_20, __numpilot_v_28)
    __numpilot_v_30 = ap.max(__numpilot_v_29, axis=1, keepdims=True)
    __numpilot_v_31 = ap.subtract(__numpilot_v_29, __numpilot_v_30)
    __numpilot_v_32 = ap.exp(__numpilot_v_31)
    __numpilot_v_33 = ap.sum(__numpilot_v_32, axis=1, keepdims=True)
    __numpilot_v_34 = ap.divide(__numpilot_v_32, __numpilot_v_33)
    __numpilot_v_35 = ap.subtract(__numpilot_v_34, target)
    __numpilot_v_36 = ap.clip(__numpilot_v_35, -1.0, 1.0)
    __numpilot_v_37 = ap.multiply(__numpilot_v_36, ap.full(__numpilot_v_36.shape, 0.1, dtype=__numpilot_v_36.dtype))
    __numpilot_v_38 = ap.subtract(__numpilot_v_29, __numpilot_v_37)
    __numpilot_v_39 = ap.max(__numpilot_v_38, axis=1, keepdims=True)
    __numpilot_v_40 = ap.subtract(__numpilot_v_38, __numpilot_v_39)
    __numpilot_v_41 = ap.exp(__numpilot_v_40)
    __numpilot_v_42 = ap.sum(__numpilot_v_41, axis=1, keepdims=True)
    __numpilot_v_43 = ap.divide(__numpilot_v_41, __numpilot_v_42)
    __numpilot_v_44 = ap.subtract(__numpilot_v_43, target)
    return {'logits': __numpilot_v_2, 'out': __numpilot_v_43, 'residual': __numpilot_v_44}