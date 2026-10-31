"""I014_weighted_bincount: Weighted integer bins with unused tail bins.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I3: int32 indices, promoted weighted sums, and empty bins.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(914)
    return {
        'labels': rng.integers(0, 31, 8193, dtype=np.int32),
        'weights': rng.uniform(-2, 2, 8193).astype(np.float32),
    }


def forward(labels, weights):
    return {
        'out': np.bincount(labels, weights=weights, minlength=37),
    }
