"""D2004_variance_composition: Variance-driven residual normalization.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Equivalent: var(x, axis=1, keepdims=True) = mean(square(x-mean(x,1,True)),1,True).
Only the named alias/composition is absent from the pinned export catalog.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(454)
    return {
        'x': rng.standard_normal((8, 13), dtype=np.float32),
        'y': rng.standard_normal((8, 13), dtype=np.float32),
    }


def forward(x, y):
    residual = np.subtract(x, y)
    variance = np.var(residual, axis=1, keepdims=True)
    mean_1 = np.mean(residual, axis=1, keepdims=True)
    center = np.subtract(residual, mean_1)
    add_2 = np.add(variance, 0.0001)
    scale = np.sqrt(add_2)
    normalized = np.divide(center, scale)
    out = np.tanh(normalized)
    return {
        'out': out,
        'variance': variance,
        'scale': scale,
    }
