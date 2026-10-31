"""I007_stack_axis: Insert a new axis between batch and features.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I2: multiple inputs with an explicit new axis.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(907)
    return {
        'x': rng.standard_normal((5, 7), dtype=np.float32),
        'y': rng.standard_normal((5, 7), dtype=np.float32),
    }


def forward(x, y):
    return {
        'out': np.stack((x, y), axis=1),
    }
