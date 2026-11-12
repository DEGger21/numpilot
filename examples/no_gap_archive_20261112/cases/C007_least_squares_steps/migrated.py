"""C007_least_squares_steps: Bounded gradient descent on squared error.

Inputs are constructed on the CPU; forward returns named device arrays.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(307)
    return {'predict': rng.standard_normal((8, 11), dtype=np.float32), 'label': rng.standard_normal((8, 11), dtype=np.float32), 'penalty': np.full((8, 11), 0.05, dtype=np.float32)}


def forward(predict, label, penalty):
    if isinstance(predict, np.ndarray):
        predict = ap.ndarray.from_numpy(predict)
    if isinstance(label, np.ndarray):
        label = ap.ndarray.from_numpy(label)
    if isinstance(penalty, np.ndarray):
        penalty = ap.ndarray.from_numpy(penalty)
    __numpilot_v_1 = ap.subtract(predict, label)
    __numpilot_v_2 = ap.multiply(__numpilot_v_1, __numpilot_v_1)
    __numpilot_v_3 = ap.mean(__numpilot_v_2, axis=1)
    __numpilot_v_4 = ap.subtract(predict, label)
    __numpilot_c_1 = ap.full(__numpilot_v_4.shape, 2.0, dtype=__numpilot_v_4.dtype)
    __numpilot_v_5 = ap.multiply(__numpilot_v_4, __numpilot_c_1)
    __numpilot_v_6 = ap.multiply(predict, penalty)
    __numpilot_v_7 = ap.add(__numpilot_v_5, __numpilot_v_6)
    __numpilot_v_8 = ap.clip(__numpilot_v_7, -4.0, 4.0)
    __numpilot_c_2 = ap.full(__numpilot_v_8.shape, 0.1, dtype=__numpilot_v_8.dtype)
    __numpilot_v_9 = ap.multiply(__numpilot_v_8, __numpilot_c_2)
    __numpilot_v_10 = ap.subtract(predict, __numpilot_v_9)
    __numpilot_v_11 = ap.subtract(__numpilot_v_10, label)
    __numpilot_c_3 = ap.full(__numpilot_v_11.shape, 2.0, dtype=__numpilot_v_11.dtype)
    __numpilot_v_12 = ap.multiply(__numpilot_v_11, __numpilot_c_3)
    __numpilot_v_13 = ap.multiply(__numpilot_v_10, penalty)
    __numpilot_v_14 = ap.add(__numpilot_v_12, __numpilot_v_13)
    __numpilot_v_15 = ap.clip(__numpilot_v_14, -4.0, 4.0)
    __numpilot_c_4 = ap.full(__numpilot_v_15.shape, 0.1, dtype=__numpilot_v_15.dtype)
    __numpilot_v_16 = ap.multiply(__numpilot_v_15, __numpilot_c_4)
    __numpilot_v_17 = ap.subtract(__numpilot_v_10, __numpilot_v_16)
    __numpilot_v_18 = ap.subtract(__numpilot_v_17, label)
    __numpilot_c_5 = ap.full(__numpilot_v_18.shape, 2.0, dtype=__numpilot_v_18.dtype)
    __numpilot_v_19 = ap.multiply(__numpilot_v_18, __numpilot_c_5)
    __numpilot_v_20 = ap.multiply(__numpilot_v_17, penalty)
    __numpilot_v_21 = ap.add(__numpilot_v_19, __numpilot_v_20)
    __numpilot_v_22 = ap.clip(__numpilot_v_21, -4.0, 4.0)
    __numpilot_c_6 = ap.full(__numpilot_v_22.shape, 0.1, dtype=__numpilot_v_22.dtype)
    __numpilot_v_23 = ap.multiply(__numpilot_v_22, __numpilot_c_6)
    __numpilot_v_24 = ap.subtract(__numpilot_v_17, __numpilot_v_23)
    __numpilot_v_25 = ap.subtract(__numpilot_v_24, label)
    __numpilot_c_7 = ap.full(__numpilot_v_25.shape, 2.0, dtype=__numpilot_v_25.dtype)
    __numpilot_v_26 = ap.multiply(__numpilot_v_25, __numpilot_c_7)
    __numpilot_v_27 = ap.multiply(__numpilot_v_24, penalty)
    __numpilot_v_28 = ap.add(__numpilot_v_26, __numpilot_v_27)
    __numpilot_v_29 = ap.clip(__numpilot_v_28, -4.0, 4.0)
    __numpilot_c_8 = ap.full(__numpilot_v_29.shape, 0.1, dtype=__numpilot_v_29.dtype)
    __numpilot_v_30 = ap.multiply(__numpilot_v_29, __numpilot_c_8)
    __numpilot_v_31 = ap.subtract(__numpilot_v_24, __numpilot_v_30)
    __numpilot_v_32 = ap.subtract(__numpilot_v_31, label)
    __numpilot_c_9 = ap.full(__numpilot_v_32.shape, 2.0, dtype=__numpilot_v_32.dtype)
    __numpilot_v_33 = ap.multiply(__numpilot_v_32, __numpilot_c_9)
    __numpilot_v_34 = ap.multiply(__numpilot_v_31, penalty)
    __numpilot_v_35 = ap.add(__numpilot_v_33, __numpilot_v_34)
    __numpilot_v_36 = ap.clip(__numpilot_v_35, -4.0, 4.0)
    __numpilot_c_10 = ap.full(__numpilot_v_36.shape, 0.025, dtype=__numpilot_v_36.dtype)
    __numpilot_v_37 = ap.multiply(__numpilot_v_36, __numpilot_c_10)
    __numpilot_v_38 = ap.subtract(__numpilot_v_31, __numpilot_v_37)
    __numpilot_v_39 = ap.subtract(__numpilot_v_38, label)
    __numpilot_c_11 = ap.full(__numpilot_v_39.shape, 2.0, dtype=__numpilot_v_39.dtype)
    __numpilot_v_40 = ap.multiply(__numpilot_v_39, __numpilot_c_11)
    __numpilot_v_41 = ap.multiply(__numpilot_v_38, penalty)
    __numpilot_v_42 = ap.add(__numpilot_v_40, __numpilot_v_41)
    __numpilot_v_43 = ap.clip(__numpilot_v_42, -4.0, 4.0)
    __numpilot_c_12 = ap.full(__numpilot_v_43.shape, 0.025, dtype=__numpilot_v_43.dtype)
    __numpilot_v_44 = ap.multiply(__numpilot_v_43, __numpilot_c_12)
    __numpilot_v_45 = ap.subtract(__numpilot_v_38, __numpilot_v_44)
    __numpilot_v_46 = ap.subtract(__numpilot_v_45, label)
    __numpilot_v_47 = ap.multiply(__numpilot_v_46, __numpilot_v_46)
    __numpilot_v_48 = ap.mean(__numpilot_v_47, axis=1)
    return {'out': __numpilot_v_45, 'initial_loss': __numpilot_v_3, 'final_loss': __numpilot_v_48}