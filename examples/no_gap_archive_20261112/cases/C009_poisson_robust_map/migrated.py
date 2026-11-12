"""C009_poisson_robust_map: Poisson source and robust residual penalties.

Inputs are constructed on the CPU; forward returns named AsNumpy arrays.
Source: ../../../../examples/poisson_source_smooth_l1_map.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    return {
        'x': np.linspace(0.04, 0.96, 81, dtype=np.float32).reshape(9, 9),
        'y': np.linspace(0.96, 0.04, 81, dtype=np.float32).reshape(9, 9),
        'scale': np.full((9, 9), 0.92, dtype=np.float32),
    }


def forward(x, y, scale):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = ap.ndarray.from_numpy(y)
    if isinstance(scale, np.ndarray):
        scale = ap.ndarray.from_numpy(scale)

    # Poisson forcing term: sin(pi * x) * sin(pi * y) * 2 * pi^2
    __numpilot_v_1 = ap.multiply(x, ap.full(x.shape, 3.1415927, dtype=x.dtype))
    __numpilot_v_2 = ap.multiply(y, ap.full(y.shape, 3.1415927, dtype=y.dtype))
    __numpilot_v_3 = ap.sin(__numpilot_v_1)
    __numpilot_v_4 = ap.sin(__numpilot_v_2)
    __numpilot_v_5 = ap.multiply(__numpilot_v_3, __numpilot_v_4)
    __numpilot_v_6 = ap.multiply(
        __numpilot_v_5,
        ap.full(__numpilot_v_5.shape, 19.739208, dtype=__numpilot_v_5.dtype),
    )

    __numpilot_v_7 = ap.multiply(__numpilot_v_6, scale)

    # Robust residual penalty (smooth-L1) before the iterations
    __numpilot_v_8 = ap.subtract(__numpilot_v_7, __numpilot_v_6)
    __numpilot_v_9 = ap.absolute(__numpilot_v_8)
    __numpilot_v_10 = ap.minimum(
        __numpilot_v_9,
        ap.full(__numpilot_v_9.shape, 1.0, dtype=__numpilot_v_9.dtype),
    )
    __numpilot_v_11 = ap.square(__numpilot_v_10)
    __numpilot_v_12 = ap.multiply(
        __numpilot_v_11,
        ap.full(__numpilot_v_11.shape, 0.5, dtype=__numpilot_v_11.dtype),
    )
    __numpilot_v_13 = ap.subtract(__numpilot_v_9, __numpilot_v_10)
    __numpilot_v_14 = ap.add(__numpilot_v_12, __numpilot_v_13)

    # Iteration 0 (clipped step)
    __numpilot_v_15 = ap.subtract(__numpilot_v_7, __numpilot_v_6)
    __numpilot_v_16 = ap.clip(__numpilot_v_15, -1.0, 1.0)
    __numpilot_v_17 = ap.multiply(
        __numpilot_v_16,
        ap.full(__numpilot_v_16.shape, 0.1, dtype=__numpilot_v_16.dtype),
    )
    __numpilot_v_18 = ap.subtract(__numpilot_v_7, __numpilot_v_17)

    # Iteration 1 (clipped step)
    __numpilot_v_19 = ap.subtract(__numpilot_v_18, __numpilot_v_6)
    __numpilot_v_20 = ap.clip(__numpilot_v_19, -1.0, 1.0)
    __numpilot_v_21 = ap.multiply(
        __numpilot_v_20,
        ap.full(__numpilot_v_20.shape, 0.1, dtype=__numpilot_v_20.dtype),
    )
    __numpilot_v_22 = ap.subtract(__numpilot_v_18, __numpilot_v_21)

    # Iteration 2 (tanh step)
    __numpilot_v_23 = ap.subtract(__numpilot_v_22, __numpilot_v_6)
    __numpilot_v_24 = ap.tanh(__numpilot_v_23)
    __numpilot_v_25 = ap.multiply(
        __numpilot_v_24,
        ap.full(__numpilot_v_24.shape, 0.1, dtype=__numpilot_v_24.dtype),
    )
    __numpilot_v_26 = ap.subtract(__numpilot_v_22, __numpilot_v_25)

    # Iteration 3 (tanh step)
    __numpilot_v_27 = ap.subtract(__numpilot_v_26, __numpilot_v_6)
    __numpilot_v_28 = ap.tanh(__numpilot_v_27)
    __numpilot_v_29 = ap.multiply(
        __numpilot_v_28,
        ap.full(__numpilot_v_28.shape, 0.1, dtype=__numpilot_v_28.dtype),
    )
    __numpilot_v_30 = ap.subtract(__numpilot_v_26, __numpilot_v_29)

    # Robust residual penalty (smooth-L1) after the iterations
    __numpilot_v_31 = ap.subtract(__numpilot_v_30, __numpilot_v_6)
    __numpilot_v_32 = ap.absolute(__numpilot_v_31)
    __numpilot_v_33 = ap.minimum(
        __numpilot_v_32,
        ap.full(__numpilot_v_32.shape, 1.0, dtype=__numpilot_v_32.dtype),
    )
    __numpilot_v_34 = ap.square(__numpilot_v_33)
    __numpilot_v_35 = ap.multiply(
        __numpilot_v_34,
        ap.full(__numpilot_v_34.shape, 0.5, dtype=__numpilot_v_34.dtype),
    )
    __numpilot_v_36 = ap.subtract(__numpilot_v_32, __numpilot_v_33)
    __numpilot_v_37 = ap.add(__numpilot_v_35, __numpilot_v_36)

    return {
        'source': __numpilot_v_6,
        'loss_before': __numpilot_v_14,
        'loss_after': __numpilot_v_37,
        'out': __numpilot_v_30,
    }