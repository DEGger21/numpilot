"""I010_quantile_axis: Several quantiles over the middle tensor axis.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I2: data plus quantile levels and an explicit axis.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(910)
    return {
        'x': rng.standard_normal((4, 13, 6), dtype=np.float32),
        'q': np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float64),
    }


def forward(x, q):
    return {
        'out': np.quantile(x, q, axis=1, keepdims=True),
    }
