"""I015_partition_boundaries: Order statistics over a large irregular row extent.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I3: non-power-of-two rows and endpoint order statistics.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(915)
    x = rng.standard_normal((17, 2053), dtype=np.float32)
    x[:, 0] = -np.inf
    x[:, -1] = np.inf
    return {
        'x': x,
    }


def forward(x):
    return {
        'out': np.partition(x, kth=(0, 1026, 2052), axis=1),
    }
