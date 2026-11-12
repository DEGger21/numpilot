"""D2003_acos_alias: Angular distance from bounded correlations.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Equivalent: np.acos(cosine) -> np.arccos(cosine); NumPy >= 2.0.
Only the named alias/composition is absent from the pinned export catalog.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(453)
    return {
        "x": rng.standard_normal((16384, 256), dtype=np.float32),
        "y": rng.standard_normal((16384, 256), dtype=np.float32),
    }


def forward(x, y):
    multiply_1 = np.multiply(x, y)
    numerator = np.sum(multiply_1, axis=1)
    square_2 = np.square(x)
    sum_3 = np.sum(square_2, axis=1)
    nx = np.sqrt(sum_3)
    square_4 = np.square(y)
    sum_5 = np.sum(square_4, axis=1)
    ny = np.sqrt(sum_5)
    multiply_6 = np.multiply(nx, ny)
    denominator = np.maximum(multiply_6, 1e-06)
    divide_7 = np.divide(numerator, denominator)
    cosine = np.clip(divide_7, -1.0, 1.0)
    angle = np.acos(cosine)
    return {
        'angle': angle,
        'cosine': cosine,
    }
