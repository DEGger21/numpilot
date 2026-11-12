"""C011_residual_gemm: Residual matrix network with shared projection.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Source: ../../../../examples/big_gemm.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(311)
    return {'x': rng.standard_normal((512, 512), dtype=np.float32), 'w1': rng.standard_normal((512, 512), dtype=np.float32) * 0.1, 'w2': rng.standard_normal((512, 512), dtype=np.float32) * 0.1, 'w3': rng.standard_normal((512, 64), dtype=np.float32) * 0.1}


def forward(x, w1, w2, w3):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(w1, np.ndarray):
        w1 = ap.ndarray.from_numpy(w1)
    if isinstance(w2, np.ndarray):
        w2 = ap.ndarray.from_numpy(w2)
    if isinstance(w3, np.ndarray):
        w3 = ap.ndarray.from_numpy(w3)
    # stage 1: shared projection + activation
    __numpilot_v_1 = ap.matmul(x, w1)
    __numpilot_zero_1 = ap.full(__numpilot_v_1.shape, 0.0, dtype=__numpilot_v_1.dtype)
    __numpilot_v_2 = ap.maximum(__numpilot_v_1, __numpilot_zero_1)
    # stage 2: residual block 1
    __numpilot_v_3 = ap.matmul(__numpilot_v_2, w2)
    __numpilot_zero_2 = ap.full(__numpilot_v_3.shape, 0.0, dtype=__numpilot_v_3.dtype)
    __numpilot_v_4 = ap.maximum(__numpilot_v_3, __numpilot_zero_2)
    __numpilot_quarter_1 = ap.full(__numpilot_v_4.shape, 0.25, dtype=__numpilot_v_4.dtype)
    __numpilot_v_5 = ap.multiply(__numpilot_v_4, __numpilot_quarter_1)
    __numpilot_v_6 = ap.add(__numpilot_v_2, __numpilot_v_5)
    __numpilot_zero_3 = ap.full(__numpilot_v_6.shape, 0.0, dtype=__numpilot_v_6.dtype)
    __numpilot_v_7 = ap.maximum(__numpilot_v_6, __numpilot_zero_3)
    # stage 3: residual block 2
    __numpilot_v_8 = ap.matmul(__numpilot_v_7, w2)
    __numpilot_zero_4 = ap.full(__numpilot_v_8.shape, 0.0, dtype=__numpilot_v_8.dtype)
    __numpilot_v_9 = ap.maximum(__numpilot_v_8, __numpilot_zero_4)
    __numpilot_quarter_2 = ap.full(__numpilot_v_9.shape, 0.25, dtype=__numpilot_v_9.dtype)
    __numpilot_v_10 = ap.multiply(__numpilot_v_9, __numpilot_quarter_2)
    __numpilot_v_11 = ap.add(__numpilot_v_7, __numpilot_v_10)
    __numpilot_zero_5 = ap.full(__numpilot_v_11.shape, 0.0, dtype=__numpilot_v_11.dtype)
    __numpilot_v_12 = ap.maximum(__numpilot_v_11, __numpilot_zero_5)
    # stage 4: residual block 3
    __numpilot_v_13 = ap.matmul(__numpilot_v_12, w2)
    __numpilot_zero_6 = ap.full(__numpilot_v_13.shape, 0.0, dtype=__numpilot_v_13.dtype)
    __numpilot_v_14 = ap.maximum(__numpilot_v_13, __numpilot_zero_6)
    __numpilot_quarter_3 = ap.full(__numpilot_v_14.shape, 0.25, dtype=__numpilot_v_14.dtype)
    __numpilot_v_15 = ap.multiply(__numpilot_v_14, __numpilot_quarter_3)
    __numpilot_v_16 = ap.add(__numpilot_v_12, __numpilot_v_15)
    # stage 5: head
    __numpilot_v_17 = ap.tanh(__numpilot_v_16)
    __numpilot_v_18 = ap.matmul(__numpilot_v_17, w3)
    __numpilot_v_19 = ap.square(__numpilot_v_17)
    __numpilot_v_20 = ap.mean(__numpilot_v_19, axis=1)
    __numpilot_v_21 = ap.sqrt(__numpilot_v_20)
    __numpilot_v_22 = ap.sum(__numpilot_v_18, axis=1)
    return {'out': __numpilot_v_18, 'hidden': __numpilot_v_17, 'summary': __numpilot_v_21, 'row_logits': __numpilot_v_22}