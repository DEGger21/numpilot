"""C006_softplus_network: Stable softplus and logistic gating.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Source: ../../../../examples/custom_operator_softplus.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    x = np.linspace(-80, 80, 4_194_304, dtype=np.float32).reshape(4096, 1024)
    return {
        "x": x,
        "bias": np.linspace(-1, 1, 1024, dtype=np.float32),
        "residual": np.full((4096, 1024), 0.1, dtype=np.float32),
    }


def softplus(x):
    magnitude = np.absolute(x)
    tail = np.exp(np.negative(magnitude))
    correction = np.log1p(tail)
    positive = np.maximum(x, 0.0)
    return np.add(correction, positive)


def sigmoid(x):
    bounded = np.clip(x, -40.0, 40.0)
    denominator = np.add(1.0, np.exp(np.negative(bounded)))
    return np.reciprocal(denominator)


def row_peak(x):
    return np.max(x, axis=1)


def forward(x, bias, residual):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(bias, np.ndarray):
        bias = ap.ndarray.from_numpy(bias)
    if isinstance(residual, np.ndarray):
        residual = ap.ndarray.from_numpy(residual)

    # stage 1: stable softplus on the raw input
    __numpilot_v_1 = ap.absolute(x)
    __numpilot_v_2 = ap.negative(__numpilot_v_1)
    __numpilot_v_3 = ap.exp(__numpilot_v_2)
    __numpilot_v_4 = ap.log1p(__numpilot_v_3)
    __numpilot_v_5 = ap.maximum(x, ap.full(x.shape, 0.0, dtype=x.dtype))
    __numpilot_v_6 = ap.add(__numpilot_v_4, __numpilot_v_5)

    # stage 2: logistic gate
    __numpilot_v_7 = ap.clip(x, -40.0, 40.0)
    __numpilot_v_8 = ap.negative(__numpilot_v_7)
    __numpilot_v_9 = ap.exp(__numpilot_v_8)
    __numpilot_v_10 = ap.add(ap.full(__numpilot_v_9.shape, 1.0, dtype=__numpilot_v_9.dtype), __numpilot_v_9)
    __numpilot_v_11 = ap.reciprocal(__numpilot_v_10)

    # stage 3: gated propagation
    __numpilot_v_12 = ap.multiply(__numpilot_v_6, __numpilot_v_11)
    __numpilot_v_13 = ap.add(__numpilot_v_12, bias)

    # stage 4: second softplus block
    __numpilot_v_14 = ap.absolute(__numpilot_v_13)
    __numpilot_v_15 = ap.negative(__numpilot_v_14)
    __numpilot_v_16 = ap.exp(__numpilot_v_15)
    __numpilot_v_17 = ap.log1p(__numpilot_v_16)
    __numpilot_v_18 = ap.maximum(__numpilot_v_13, ap.full(__numpilot_v_13.shape, 0.0, dtype=__numpilot_v_13.dtype))
    __numpilot_v_19 = ap.add(__numpilot_v_17, __numpilot_v_18)

    # stage 5: residual + bias accumulation
    __numpilot_v_20 = ap.add(__numpilot_v_19, residual)
    __numpilot_v_21 = ap.add(__numpilot_v_20, bias)

    # stage 6: third softplus block
    __numpilot_v_22 = ap.absolute(__numpilot_v_21)
    __numpilot_v_23 = ap.negative(__numpilot_v_22)
    __numpilot_v_24 = ap.exp(__numpilot_v_23)
    __numpilot_v_25 = ap.log1p(__numpilot_v_24)
    __numpilot_v_26 = ap.maximum(__numpilot_v_21, ap.full(__numpilot_v_21.shape, 0.0, dtype=__numpilot_v_21.dtype))
    __numpilot_v_27 = ap.add(__numpilot_v_25, __numpilot_v_26)

    # stage 7: final residual + bias + activation
    __numpilot_v_28 = ap.add(__numpilot_v_27, residual)
    __numpilot_v_29 = ap.add(__numpilot_v_28, bias)
    __numpilot_v_30 = ap.tanh(__numpilot_v_29)
    __numpilot_v_31 = ap.add(__numpilot_v_30, residual)

    # stage 8: row statistics
    __numpilot_v_32 = ap.mean(__numpilot_v_31, axis=1)
    __numpilot_v_33 = ap.max(__numpilot_v_31, axis=1)

    return {'out': __numpilot_v_31, 'softplus': __numpilot_v_6, 'gate': __numpilot_v_11, 'activity': __numpilot_v_32, 'peak': __numpilot_v_33}