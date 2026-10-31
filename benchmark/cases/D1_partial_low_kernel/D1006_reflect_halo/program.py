"""D1006_reflect_halo: Reflect a preconditioned signal at the boundary.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
pad supplies a two-cell reflected halo.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(406)
    return {
        'x': rng.standard_normal((12, 9), dtype=np.float32),
    }


def forward(x):
    mean = np.mean(x, axis=1, keepdims=True)
    centered = np.subtract(x, mean)
    square_1 = np.square(centered)
    power = np.mean(square_1, axis=1, keepdims=True)
    add_2 = np.add(power, 0.0001)
    scale = np.sqrt(add_2)
    normalized = np.divide(centered, scale)
    bounded = np.clip(normalized, -3.0, 3.0)
    halo = np.pad(bounded, ((0, 0), (2, 2)), mode='reflect')
    add_3 = np.add(halo[:, :-4], halo[:, 2:-2])
    add_4 = np.add(add_3, halo[:, 4:])
    smoothed = np.multiply(add_4, 1 / 3)
    return {
        'out': smoothed,
        'normalized': normalized,
    }
