import asnumpy as ap
"""D1007_repeat_samples: Upsample a bounded nonlinear response.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
repeat uses a fixed factor of three on axis=1.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(407)
    return {
        'x': rng.standard_normal((1024, 1024), dtype=np.float32),
    }

def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    mean_1 = ap.mean(x, axis=1, keepdims=True)
    center = ap.subtract(x, mean_1)
    positive = ap.maximum(center, ap.full_like(center, 0.0))
    negative = ap.minimum(center, ap.full_like(center, 0.0))
    multiply_2 = ap.multiply(negative, ap.full_like(negative, 0.1))
    response = ap.add(positive, multiply_2)
    bounded = ap.tanh(response)
    upsampled = ap.repeat(bounded, 3, axis=1)
    square_3 = ap.square(upsampled)
    energy = ap.mean(square_3, axis=1)
    return {
        'out': upsampled,
        'energy': energy,
    }
