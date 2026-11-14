"""D1010_median_center: Robust center with quadratic residual diagnostics.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
median is applied to odd-length float32 rows.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(410)
    return {
        'x': rng.standard_normal((9, 17), dtype=np.float32),
    }


def forward(x):
    bounded = np.clip(x, -5.0, 5.0)
    center = np.median(bounded, axis=1, keepdims=True)
    residual = np.subtract(bounded, center)
    square_1 = np.square(residual)
    power = np.mean(square_1, axis=1, keepdims=True)
    add_2 = np.add(power, 1e-05)
    scale = np.sqrt(add_2)
    standard = np.divide(residual, scale)
    output = np.tanh(standard)
    return {
        'out': output,
        'center': center,
        'power': power,
    }
