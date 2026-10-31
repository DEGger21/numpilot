"""I009_weighted_histogram: Weighted two-dimensional histogram.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
I2: multiple inputs and distinct bin axes.
The target API is absent from the pinned public compute exports.
Inputs include the contract boundary rather than only nominal values.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(909)
    return {
        'x': rng.uniform(-1, 1, 127),
        'y': rng.uniform(-1, 1, 127),
        'weights': rng.uniform(0, 2, 127),
    }


def forward(x, y, weights):
    counts, xe, ye = np.histogram2d(x, y, bins=(7, 5), range=((-1,1),(-1,1)), weights=weights)
    return {
        'out': counts,
        'x_edges': xe,
        'y_edges': ye,
    }
