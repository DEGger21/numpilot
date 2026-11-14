"""D1013_gather_channels: Gather selected normalized channels.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
take preserves repeated and out-of-order indices.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(413)
    return {
        'x': rng.standard_normal((7, 12), dtype=np.float32),
        'indices': np.array([8, 2, 8, 0, 11], dtype=np.int64),
    }


def forward(x, indices):
    mean = np.mean(x, axis=1, keepdims=True)
    centered = np.subtract(x, mean)
    square_1 = np.square(centered)
    variance = np.mean(square_1, axis=1, keepdims=True)
    add_2 = np.add(variance, 1e-05)
    scale = np.sqrt(add_2)
    normalized = np.divide(centered, scale)
    selected = np.take(normalized, indices, axis=1)
    selected_power = np.square(selected)
    return {
        "selected_power": selected_power,
        'out': np.tanh(selected),
        'scale': scale,
    }
