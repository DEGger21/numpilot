"""G001_concat_three_views: Three independent equal-contract feature joins.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Three static call sites share exactly one contract:
concatenate|axis=1|float32|two_(4,8)_arrays
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(701)
    return {
        'a': rng.standard_normal((4, 8), dtype=np.float32),
        'b': rng.standard_normal((4, 8), dtype=np.float32),
        'c': rng.standard_normal((4, 8), dtype=np.float32),
    }


def forward(a, b, c):
    ab = np.concatenate((a, b), axis=1)
    bc = np.concatenate((b, c), axis=1)
    ca = np.concatenate((c, a), axis=1)
    left = np.tanh(ab)
    middle = np.square(bc)
    right = np.absolute(ca)
    combined = np.add(np.add(left, middle), right)
    energy = np.mean(combined, axis=1)
    return {
        'out': combined,
        'energy': energy,
        'joined': ab,
    }
