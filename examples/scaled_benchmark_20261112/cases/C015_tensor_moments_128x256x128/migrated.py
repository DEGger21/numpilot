"""C015_tensor_moments: Three-dimensional channel moment feedback.

Inputs are constructed on the CPU; forward returns named device arrays.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(315)
    return {
        "x": rng.standard_normal((128, 256, 128), dtype=np.float32),
        "drive": rng.standard_normal((128, 1, 128), dtype=np.float32) * 0.1,
        "gain": np.linspace(0.8, 1.2, 128, dtype=np.float32),
    }


def forward(x, drive, gain):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(drive, np.ndarray):
        drive = ap.ndarray.from_numpy(drive)
    if isinstance(gain, np.ndarray):
        gain = ap.ndarray.from_numpy(gain)

    __numpilot_v_1 = ap.mean(x, axis=1, keepdims=True)
    __numpilot_v_2 = ap.subtract(x, __numpilot_v_1)
    __numpilot_v_3 = ap.square(__numpilot_v_2)
    __numpilot_v_4 = ap.mean(__numpilot_v_3, axis=1, keepdims=True)
    __numpilot_v_5 = ap.mean(x, axis=1, keepdims=True)
    __numpilot_v_6 = ap.subtract(x, __numpilot_v_5)
    __numpilot_v_7 = ap.square(__numpilot_v_6)
    __numpilot_v_8 = ap.mean(__numpilot_v_7, axis=1, keepdims=True)
    __numpilot_v_9 = ap.subtract(x, __numpilot_v_5)
    __numpilot_v_10 = ap.add(__numpilot_v_8, ap.full(__numpilot_v_8.shape, 0.0001, dtype=__numpilot_v_8.dtype))
    __numpilot_v_11 = ap.sqrt(__numpilot_v_10)
    __numpilot_v_12 = ap.divide(__numpilot_v_9, __numpilot_v_11)
    __numpilot_v_13 = ap.multiply(__numpilot_v_12, gain)
    __numpilot_v_14 = ap.add(__numpilot_v_13, drive)
    __numpilot_v_15 = ap.clip(__numpilot_v_14, -2.0, 2.0)
    __numpilot_v_16 = ap.mean(__numpilot_v_15, axis=1, keepdims=True)
    __numpilot_v_17 = ap.subtract(__numpilot_v_15, __numpilot_v_16)
    __numpilot_v_18 = ap.square(__numpilot_v_17)
    __numpilot_v_19 = ap.mean(__numpilot_v_18, axis=1, keepdims=True)
    __numpilot_v_20 = ap.subtract(__numpilot_v_15, __numpilot_v_16)
    __numpilot_v_21 = ap.add(__numpilot_v_19, ap.full(__numpilot_v_19.shape, 0.0001, dtype=__numpilot_v_19.dtype))
    __numpilot_v_22 = ap.sqrt(__numpilot_v_21)
    __numpilot_v_23 = ap.divide(__numpilot_v_20, __numpilot_v_22)
    __numpilot_v_24 = ap.tanh(__numpilot_v_23)
    __numpilot_v_25 = ap.mean(__numpilot_v_24, axis=1, keepdims=True)
    __numpilot_v_26 = ap.subtract(__numpilot_v_24, __numpilot_v_25)
    __numpilot_v_27 = ap.square(__numpilot_v_26)
    __numpilot_v_28 = ap.mean(__numpilot_v_27, axis=1, keepdims=True)
    __numpilot_v_29 = ap.subtract(__numpilot_v_24, __numpilot_v_25)
    __numpilot_v_30 = ap.add(__numpilot_v_28, ap.full(__numpilot_v_28.shape, 0.0001, dtype=__numpilot_v_28.dtype))
    __numpilot_v_31 = ap.sqrt(__numpilot_v_30)
    __numpilot_v_32 = ap.divide(__numpilot_v_29, __numpilot_v_31)
    __numpilot_v_33 = ap.multiply(__numpilot_v_32, gain)
    __numpilot_v_34 = ap.add(__numpilot_v_33, drive)
    __numpilot_v_35 = ap.clip(__numpilot_v_34, -2.0, 2.0)
    __numpilot_v_36 = ap.mean(__numpilot_v_35, axis=1, keepdims=True)
    __numpilot_v_37 = ap.subtract(__numpilot_v_35, __numpilot_v_36)
    __numpilot_v_38 = ap.square(__numpilot_v_37)
    __numpilot_v_39 = ap.mean(__numpilot_v_38, axis=1, keepdims=True)
    __numpilot_v_40 = ap.subtract(__numpilot_v_35, __numpilot_v_36)
    __numpilot_v_41 = ap.add(__numpilot_v_39, ap.full(__numpilot_v_39.shape, 0.0001, dtype=__numpilot_v_39.dtype))
    __numpilot_v_42 = ap.sqrt(__numpilot_v_41)
    __numpilot_v_43 = ap.divide(__numpilot_v_40, __numpilot_v_42)
    __numpilot_v_44 = ap.tanh(__numpilot_v_43)
    __numpilot_v_45 = ap.mean(__numpilot_v_44, axis=1, keepdims=True)
    __numpilot_v_46 = ap.subtract(__numpilot_v_44, __numpilot_v_45)
    __numpilot_v_47 = ap.square(__numpilot_v_46)
    __numpilot_v_48 = ap.mean(__numpilot_v_47, axis=1, keepdims=True)
    __numpilot_v_49 = ap.subtract(__numpilot_v_44, __numpilot_v_45)
    __numpilot_v_50 = ap.add(__numpilot_v_48, ap.full(__numpilot_v_48.shape, 0.0001, dtype=__numpilot_v_48.dtype))
    __numpilot_v_51 = ap.sqrt(__numpilot_v_50)
    __numpilot_v_52 = ap.divide(__numpilot_v_49, __numpilot_v_51)
    __numpilot_v_53 = ap.multiply(__numpilot_v_52, gain)
    __numpilot_v_54 = ap.add(__numpilot_v_53, drive)
    __numpilot_v_55 = ap.clip(__numpilot_v_54, -2.0, 2.0)
    __numpilot_v_56 = ap.mean(__numpilot_v_55, axis=1, keepdims=True)
    __numpilot_v_57 = ap.subtract(__numpilot_v_55, __numpilot_v_56)
    __numpilot_v_58 = ap.square(__numpilot_v_57)
    __numpilot_v_59 = ap.mean(__numpilot_v_58, axis=1, keepdims=True)
    __numpilot_v_60 = ap.subtract(__numpilot_v_56, __numpilot_v_1)
    __numpilot_v_61 = ap.subtract(__numpilot_v_59, __numpilot_v_4)
    return {'out': __numpilot_v_55, 'mean_change': __numpilot_v_60, 'variance_change': __numpilot_v_61}