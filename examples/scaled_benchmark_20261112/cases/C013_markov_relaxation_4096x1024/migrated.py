"""C013_markov_relaxation: Finite-state mass propagation and relaxation.

Inputs are constructed on the CPU; forward returns named device arrays.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(313)
    matrix = rng.uniform(0.1, 1, (1024, 1024)).astype(np.float32)
    matrix /= matrix.sum(axis=1, keepdims=True)
    state = np.zeros((4096, 1024), dtype=np.float32)
    state[:, 0] = 1.0
    return {
        "transition": matrix,
        "state": state,
        "prior": np.full((1, 1024), 1 / 1024, dtype=np.float32),
    }


def forward(transition, state, prior):
    if isinstance(transition, np.ndarray):
        transition = ap.ndarray.from_numpy(transition)
    if isinstance(state, np.ndarray):
        state = ap.ndarray.from_numpy(state)
    if isinstance(prior, np.ndarray):
        prior = ap.ndarray.from_numpy(prior)
    __numpilot_v_1 = ap.matmul(state, transition)
    __numpilot_v_2 = ap.maximum(__numpilot_v_1, ap.full(__numpilot_v_1.shape, 0.0, dtype=__numpilot_v_1.dtype))
    __numpilot_v_3 = ap.multiply(__numpilot_v_2, ap.full(__numpilot_v_2.shape, 0.9, dtype=__numpilot_v_2.dtype))
    __numpilot_v_4 = ap.multiply(prior, ap.full(prior.shape, 0.1, dtype=prior.dtype))
    __numpilot_v_5 = ap.add(__numpilot_v_3, __numpilot_v_4)
    __numpilot_v_6 = ap.sum(__numpilot_v_5, axis=1, keepdims=True)
    __numpilot_v_7 = ap.maximum(__numpilot_v_6, ap.full(__numpilot_v_6.shape, 1e-07, dtype=__numpilot_v_6.dtype))
    __numpilot_v_8 = ap.divide(__numpilot_v_5, __numpilot_v_7)
    __numpilot_v_9 = ap.matmul(__numpilot_v_8, transition)
    __numpilot_v_10 = ap.maximum(__numpilot_v_9, ap.full(__numpilot_v_9.shape, 0.0, dtype=__numpilot_v_9.dtype))
    __numpilot_v_11 = ap.multiply(__numpilot_v_10, ap.full(__numpilot_v_10.shape, 0.8, dtype=__numpilot_v_10.dtype))
    __numpilot_v_12 = ap.multiply(prior, ap.full(prior.shape, 0.2, dtype=prior.dtype))
    __numpilot_v_13 = ap.add(__numpilot_v_11, __numpilot_v_12)
    __numpilot_v_14 = ap.sum(__numpilot_v_13, axis=1, keepdims=True)
    __numpilot_v_15 = ap.maximum(__numpilot_v_14, ap.full(__numpilot_v_14.shape, 1e-07, dtype=__numpilot_v_14.dtype))
    __numpilot_v_16 = ap.divide(__numpilot_v_13, __numpilot_v_15)
    __numpilot_v_17 = ap.matmul(__numpilot_v_16, transition)
    __numpilot_v_18 = ap.maximum(__numpilot_v_17, ap.full(__numpilot_v_17.shape, 0.0, dtype=__numpilot_v_17.dtype))
    __numpilot_v_19 = ap.multiply(__numpilot_v_18, ap.full(__numpilot_v_18.shape, 0.9, dtype=__numpilot_v_18.dtype))
    __numpilot_v_20 = ap.multiply(prior, ap.full(prior.shape, 0.1, dtype=prior.dtype))
    __numpilot_v_21 = ap.add(__numpilot_v_19, __numpilot_v_20)
    __numpilot_v_22 = ap.sum(__numpilot_v_21, axis=1, keepdims=True)
    __numpilot_v_23 = ap.maximum(__numpilot_v_22, ap.full(__numpilot_v_22.shape, 1e-07, dtype=__numpilot_v_22.dtype))
    __numpilot_v_24 = ap.divide(__numpilot_v_21, __numpilot_v_23)
    __numpilot_v_25 = ap.matmul(__numpilot_v_24, transition)
    __numpilot_v_26 = ap.maximum(__numpilot_v_25, ap.full(__numpilot_v_25.shape, 0.0, dtype=__numpilot_v_25.dtype))
    __numpilot_v_27 = ap.multiply(__numpilot_v_26, ap.full(__numpilot_v_26.shape, 0.8, dtype=__numpilot_v_26.dtype))
    __numpilot_v_28 = ap.multiply(prior, ap.full(prior.shape, 0.2, dtype=prior.dtype))
    __numpilot_v_29 = ap.add(__numpilot_v_27, __numpilot_v_28)
    __numpilot_v_30 = ap.sum(__numpilot_v_29, axis=1, keepdims=True)
    __numpilot_v_31 = ap.maximum(__numpilot_v_30, ap.full(__numpilot_v_30.shape, 1e-07, dtype=__numpilot_v_30.dtype))
    __numpilot_v_32 = ap.divide(__numpilot_v_29, __numpilot_v_31)
    __numpilot_v_33 = ap.matmul(__numpilot_v_32, transition)
    __numpilot_v_34 = ap.maximum(__numpilot_v_33, ap.full(__numpilot_v_33.shape, 0.0, dtype=__numpilot_v_33.dtype))
    __numpilot_v_35 = ap.multiply(__numpilot_v_34, ap.full(__numpilot_v_34.shape, 0.9, dtype=__numpilot_v_34.dtype))
    __numpilot_v_36 = ap.multiply(prior, ap.full(prior.shape, 0.1, dtype=prior.dtype))
    __numpilot_v_37 = ap.add(__numpilot_v_35, __numpilot_v_36)
    __numpilot_v_38 = ap.sum(__numpilot_v_37, axis=1, keepdims=True)
    __numpilot_v_39 = ap.maximum(__numpilot_v_38, ap.full(__numpilot_v_38.shape, 1e-07, dtype=__numpilot_v_38.dtype))
    __numpilot_v_40 = ap.divide(__numpilot_v_37, __numpilot_v_39)
    __numpilot_v_41 = ap.matmul(__numpilot_v_40, transition)
    __numpilot_v_42 = ap.maximum(__numpilot_v_41, ap.full(__numpilot_v_41.shape, 0.0, dtype=__numpilot_v_41.dtype))
    __numpilot_v_43 = ap.multiply(__numpilot_v_42, ap.full(__numpilot_v_42.shape, 0.8, dtype=__numpilot_v_42.dtype))
    __numpilot_v_44 = ap.multiply(prior, ap.full(prior.shape, 0.2, dtype=prior.dtype))
    __numpilot_v_45 = ap.add(__numpilot_v_43, __numpilot_v_44)
    __numpilot_v_46 = ap.sum(__numpilot_v_45, axis=1, keepdims=True)
    __numpilot_v_47 = ap.maximum(__numpilot_v_46, ap.full(__numpilot_v_46.shape, 1e-07, dtype=__numpilot_v_46.dtype))
    __numpilot_v_48 = ap.divide(__numpilot_v_45, __numpilot_v_47)
    __numpilot_v_49 = ap.matmul(__numpilot_v_48, transition)
    __numpilot_v_50 = ap.maximum(__numpilot_v_49, ap.full(__numpilot_v_49.shape, 0.0, dtype=__numpilot_v_49.dtype))
    __numpilot_v_51 = ap.multiply(__numpilot_v_50, ap.full(__numpilot_v_50.shape, 0.9, dtype=__numpilot_v_50.dtype))
    __numpilot_v_52 = ap.multiply(prior, ap.full(prior.shape, 0.1, dtype=prior.dtype))
    __numpilot_v_53 = ap.add(__numpilot_v_51, __numpilot_v_52)
    __numpilot_v_54 = ap.sum(__numpilot_v_53, axis=1, keepdims=True)
    __numpilot_v_55 = ap.maximum(__numpilot_v_54, ap.full(__numpilot_v_54.shape, 1e-07, dtype=__numpilot_v_54.dtype))
    __numpilot_v_56 = ap.divide(__numpilot_v_53, __numpilot_v_55)
    __numpilot_v_57 = ap.matmul(__numpilot_v_56, transition)
    __numpilot_v_58 = ap.maximum(__numpilot_v_57, ap.full(__numpilot_v_57.shape, 0.0, dtype=__numpilot_v_57.dtype))
    __numpilot_v_59 = ap.multiply(__numpilot_v_58, ap.full(__numpilot_v_58.shape, 0.8, dtype=__numpilot_v_58.dtype))
    __numpilot_v_60 = ap.multiply(prior, ap.full(prior.shape, 0.2, dtype=prior.dtype))
    __numpilot_v_61 = ap.add(__numpilot_v_59, __numpilot_v_60)
    __numpilot_v_62 = ap.sum(__numpilot_v_61, axis=1, keepdims=True)
    __numpilot_v_63 = ap.maximum(__numpilot_v_62, ap.full(__numpilot_v_62.shape, 1e-07, dtype=__numpilot_v_62.dtype))
    __numpilot_v_64 = ap.divide(__numpilot_v_61, __numpilot_v_63)
    __numpilot_v_65 = ap.sum(__numpilot_v_64, axis=1)
    __numpilot_v_66 = ap.subtract(__numpilot_v_64, prior)
    __numpilot_v_67 = ap.absolute(__numpilot_v_66)
    __numpilot_v_68 = ap.sum(__numpilot_v_67, axis=1)
    return {'out': __numpilot_v_64, 'mass': __numpilot_v_65, 'distance': __numpilot_v_68}