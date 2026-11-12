"""C010_energy_feedback: Energy network with damped feedback.

Inputs are constructed on the CPU; forward returns named device arrays.
Source: ../../../../examples/elementwise_forward.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    return {
        "x": np.linspace(-2, 2, 4_194_304, dtype=np.float32).reshape(2048, 2048),
        "y": np.linspace(2, -2, 4_194_304, dtype=np.float32).reshape(2048, 2048),
        "drive": np.full((2048, 2048), 0.05, dtype=np.float32),
    }


def amplitude(x, y):
    x_energy = np.square(x)
    y_energy = np.square(y)
    energy = np.add(x_energy, y_energy)
    magnitude = np.sqrt(energy)
    return np.tanh(magnitude)


def feedback(state, shared, drive):
    damping = np.multiply(state, np.full(state.shape, 0.8, dtype=state.dtype))
    forcing = np.multiply(shared, drive)
    updated = np.add(damping, forcing)
    return updated


def compare_rows(current, previous):
    delta = np.subtract(current, previous)
    return np.mean(np.square(delta), axis=1)


def forward(x, y, drive):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    if isinstance(drive, np.ndarray):
        drive = ap.ndarray.from_numpy(drive)
    __numpilot_v_1 = ap.square(x)
    __numpilot_v_2 = ap.square(y)
    __numpilot_v_3 = ap.add(__numpilot_v_1, __numpilot_v_2)
    __numpilot_v_4 = ap.sqrt(__numpilot_v_3)
    __numpilot_v_5 = ap.tanh(__numpilot_v_4)
    __numpilot_v_6 = ap.square(x)
    __numpilot_v_7 = ap.square(y)
    __numpilot_v_8 = ap.add(__numpilot_v_6, __numpilot_v_7)
    __numpilot_v_9 = ap.sqrt(__numpilot_v_8)
    __numpilot_v_10 = ap.tanh(__numpilot_v_9)
    __numpilot_v_11 = ap.multiply(x, ap.full(x.shape, 0.8, dtype=x.dtype))
    __numpilot_v_12 = ap.multiply(__numpilot_v_10, drive)
    __numpilot_v_13 = ap.add(__numpilot_v_11, __numpilot_v_12)
    __numpilot_v_14 = ap.square(__numpilot_v_13)
    __numpilot_v_15 = ap.square(y)
    __numpilot_v_16 = ap.add(__numpilot_v_14, __numpilot_v_15)
    __numpilot_v_17 = ap.sqrt(__numpilot_v_16)
    __numpilot_v_18 = ap.tanh(__numpilot_v_17)
    __numpilot_v_19 = ap.multiply(y, ap.full(y.shape, 0.8, dtype=y.dtype))
    __numpilot_v_20 = ap.multiply(__numpilot_v_18, drive)
    __numpilot_v_21 = ap.add(__numpilot_v_19, __numpilot_v_20)
    __numpilot_v_22 = ap.square(__numpilot_v_13)
    __numpilot_v_23 = ap.square(__numpilot_v_21)
    __numpilot_v_24 = ap.add(__numpilot_v_22, __numpilot_v_23)
    __numpilot_v_25 = ap.sqrt(__numpilot_v_24)
    __numpilot_v_26 = ap.tanh(__numpilot_v_25)
    __numpilot_v_27 = ap.multiply(__numpilot_v_13, ap.full(__numpilot_v_13.shape, 0.8, dtype=__numpilot_v_13.dtype))
    __numpilot_v_28 = ap.multiply(__numpilot_v_26, drive)
    __numpilot_v_29 = ap.add(__numpilot_v_27, __numpilot_v_28)
    __numpilot_v_30 = ap.square(__numpilot_v_29)
    __numpilot_v_31 = ap.square(__numpilot_v_21)
    __numpilot_v_32 = ap.add(__numpilot_v_30, __numpilot_v_31)
    __numpilot_v_33 = ap.sqrt(__numpilot_v_32)
    __numpilot_v_34 = ap.tanh(__numpilot_v_33)
    __numpilot_v_35 = ap.multiply(__numpilot_v_21, ap.full(__numpilot_v_21.shape, 0.8, dtype=__numpilot_v_21.dtype))
    __numpilot_v_36 = ap.multiply(__numpilot_v_34, drive)
    __numpilot_v_37 = ap.add(__numpilot_v_35, __numpilot_v_36)
    __numpilot_v_38 = ap.square(__numpilot_v_29)
    __numpilot_v_39 = ap.square(__numpilot_v_37)
    __numpilot_v_40 = ap.add(__numpilot_v_38, __numpilot_v_39)
    __numpilot_v_41 = ap.sqrt(__numpilot_v_40)
    __numpilot_v_42 = ap.tanh(__numpilot_v_41)
    __numpilot_v_43 = ap.subtract(__numpilot_v_42, __numpilot_v_5)
    __numpilot_v_44 = ap.square(__numpilot_v_43)
    __numpilot_v_45 = ap.mean(__numpilot_v_44, axis=1)
    return {'out': __numpilot_v_42, 'left': __numpilot_v_29, 'right': __numpilot_v_37, 'change': __numpilot_v_45}