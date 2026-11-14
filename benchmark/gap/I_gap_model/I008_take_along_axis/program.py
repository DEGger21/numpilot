"""I008_take_along_axis: Per-row indexed gathering with duplicates.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I2: data and integer indices share a batch axis.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(908)
    return {
        'x': rng.standard_normal((6, 9), dtype=np.float32),
        'indices': rng.integers(0, 9, (6, 4), dtype=np.int64),
    }


def forward(x, indices):
    return {
        'out': np.take_along_axis(x, indices, axis=1),
    }
