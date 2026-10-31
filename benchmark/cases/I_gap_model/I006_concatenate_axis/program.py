"""I006_concatenate_axis: Join two unequal-width feature tensors.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I2: multiple inputs with an explicit axis.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(906)
    return {
        'x': rng.standard_normal((3, 4, 5), dtype=np.float32),
        'y': rng.standard_normal((3, 2, 5), dtype=np.float32),
    }


def forward(x, y):
    return {
        'out': np.concatenate((x, y), axis=1),
    }
