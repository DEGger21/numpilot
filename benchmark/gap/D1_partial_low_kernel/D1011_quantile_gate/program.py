"""D1011_quantile_gate: Upper-quantile clipping of row activations.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
quantile selects the linear 75th percentile.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(411)
    return {
        'x': rng.standard_normal((10, 19), dtype=np.float32),
    }


def forward(x):
    positive = np.maximum(x, 0.0)
    threshold = np.quantile(positive, 0.75, axis=1, keepdims=True)
    clipped = np.minimum(positive, threshold)
    total = np.sum(clipped, axis=1, keepdims=True)
    maximum_1 = np.maximum(total, 1e-05)
    normalized = np.divide(clipped, maximum_1)
    square_2 = np.square(normalized)
    energy = np.sum(square_2, axis=1)
    mask = np.greater(positive, threshold)
    return {
        'out': normalized,
        'clipped_mask': mask,
        'energy': energy,
    }
