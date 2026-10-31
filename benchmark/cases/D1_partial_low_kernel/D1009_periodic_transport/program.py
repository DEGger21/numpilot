"""D1009_periodic_transport: One periodic upwind update.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
roll is the sole missing API; the transport coefficient is bounded.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(409)
    return {
        'x': rng.standard_normal((8, 15), dtype=np.float32),
    }


def forward(x):
    bounded = np.clip(x, -2.0, 2.0)
    upstream = np.roll(bounded, 1, axis=1)
    gradient = np.subtract(bounded, upstream)
    flux = np.multiply(gradient, 0.2)
    updated = np.subtract(bounded, flux)
    square_1 = np.square(updated)
    energy = np.sum(square_1, axis=1)
    subtract_2 = np.subtract(updated, bounded)
    absolute_3 = np.absolute(subtract_2)
    change = np.sum(absolute_3, axis=1)
    return {
        'out': updated,
        'energy': energy,
        'change': change,
    }
