"""D1007_repeat_samples: Upsample a bounded nonlinear response.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
repeat uses a fixed factor of three on axis=1.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(407)
    return {
        'x': rng.standard_normal((7, 11), dtype=np.float32),
    }


def forward(x):
    mean_1 = np.mean(x, axis=1, keepdims=True)
    center = np.subtract(x, mean_1)
    positive = np.maximum(center, 0.0)
    negative = np.minimum(center, 0.0)
    multiply_2 = np.multiply(negative, 0.1)
    response = np.add(positive, multiply_2)
    bounded = np.tanh(response)
    upsampled = np.repeat(bounded, 3, axis=1)
    square_3 = np.square(upsampled)
    energy = np.mean(square_3, axis=1)
    return {
        'out': upsampled,
        'energy': energy,
    }
