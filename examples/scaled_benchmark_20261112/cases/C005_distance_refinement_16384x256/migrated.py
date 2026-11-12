"""C005_distance_refinement: Iterative pointwise residual contraction.

Inputs are constructed on the CPU; forward returns named NPU arrays.
Source: ../../../../examples/custom_operator_euclidean_distance.py
Adaptation: bounded inputs, named outputs, explicit computational stages.
"""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(305)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "target": rng.standard_normal((16384, 256), dtype=np.float32),
        "weight": np.linspace(0.5, 1.0, 256, dtype=np.float32),
    }


def distance(x, y):
    delta = ap.subtract(x, y)
    squared = ap.square(delta)
    return ap.sqrt(squared)


def contract(current, target, rate):
    error = ap.subtract(current, target)
    correction = ap.multiply(error, rate)
    updated = ap.subtract(current, correction)
    return updated


def weighted_rows(error, weight):
    weighted = ap.multiply(error, weight)
    total = ap.sum(weighted, axis=1)
    return total


def forward(x, target, weight):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(target, np.ndarray):
        target = ap.ndarray.from_numpy(target)
    if isinstance(weight, np.ndarray):
        weight = ap.ndarray.from_numpy(weight)

    # stage 1: initial pointwise euclidean distance
    __numpilot_v_1 = ap.subtract(x, target)
    __numpilot_v_2 = ap.square(__numpilot_v_1)
    __numpilot_v_3 = ap.sqrt(__numpilot_v_2)

    # stage 2: iterative residual contraction (5 unrolled iterations)
    __numpilot_v_4 = ap.subtract(x, target)
    __numpilot_v_5 = ap.multiply(__numpilot_v_4, ap.full(__numpilot_v_4.shape, 0.2, dtype=__numpilot_v_4.dtype))
    __numpilot_v_6 = ap.subtract(x, __numpilot_v_5)
    __numpilot_v_7 = ap.subtract(__numpilot_v_6, target)
    __numpilot_v_8 = ap.multiply(__numpilot_v_7, ap.full(__numpilot_v_7.shape, 0.2, dtype=__numpilot_v_7.dtype))
    __numpilot_v_9 = ap.subtract(__numpilot_v_6, __numpilot_v_8)
    __numpilot_v_10 = ap.subtract(__numpilot_v_9, target)
    __numpilot_v_11 = ap.multiply(__numpilot_v_10, ap.full(__numpilot_v_10.shape, 0.2, dtype=__numpilot_v_10.dtype))
    __numpilot_v_12 = ap.subtract(__numpilot_v_9, __numpilot_v_11)
    __numpilot_v_13 = ap.subtract(__numpilot_v_12, target)
    __numpilot_v_14 = ap.multiply(__numpilot_v_13, ap.full(__numpilot_v_13.shape, 0.1, dtype=__numpilot_v_13.dtype))
    __numpilot_v_15 = ap.subtract(__numpilot_v_12, __numpilot_v_14)
    __numpilot_v_16 = ap.subtract(__numpilot_v_15, target)
    __numpilot_v_17 = ap.multiply(__numpilot_v_16, ap.full(__numpilot_v_16.shape, 0.1, dtype=__numpilot_v_16.dtype))
    __numpilot_v_18 = ap.subtract(__numpilot_v_15, __numpilot_v_17)

    # stage 3: final distance and weighted-row improvement
    __numpilot_v_19 = ap.subtract(__numpilot_v_18, target)
    __numpilot_v_20 = ap.square(__numpilot_v_19)
    __numpilot_v_21 = ap.sqrt(__numpilot_v_20)
    __numpilot_v_22 = ap.multiply(__numpilot_v_3, weight)
    __numpilot_v_23 = ap.sum(__numpilot_v_22, axis=1)
    __numpilot_v_24 = ap.multiply(__numpilot_v_21, weight)
    __numpilot_v_25 = ap.sum(__numpilot_v_24, axis=1)
    __numpilot_v_26 = ap.subtract(__numpilot_v_23, __numpilot_v_25)

    return {'out': __numpilot_v_18, 'distance': __numpilot_v_21, 'improvement': __numpilot_v_26}