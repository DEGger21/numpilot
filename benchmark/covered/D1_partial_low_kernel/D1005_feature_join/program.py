"""D1005_feature_join: Fuse independently normalized feature blocks.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Two concatenate sites share axis=1, float32, and operand shapes.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(405)
    return {
        'a': rng.standard_normal((4, 8), dtype=np.float32),
        'b': rng.standard_normal((4, 8), dtype=np.float32),
        'c': rng.standard_normal((4, 8), dtype=np.float32),
        'd': rng.standard_normal((4, 8), dtype=np.float32),
    }


def forward(a, b, c, d):
    multiply_1 = np.multiply(a, 0.5)
    left_a = np.tanh(multiply_1)
    multiply_2 = np.multiply(b, 0.25)
    left_b = np.tanh(multiply_2)
    right_c = np.maximum(c, 0.0)
    right_d = np.minimum(d, 1.0)
    left = np.concatenate((left_a, left_b), axis=1)
    right = np.concatenate((right_c, right_d), axis=1)
    fused = np.add(left, right)
    square_3 = np.square(fused)
    energy = np.mean(square_3, axis=1)
    absolute_4 = np.absolute(fused)
    add_5 = np.add(1.0, absolute_4)
    normalized = np.divide(fused, add_5)
    return {
        'out': normalized,
        'energy': energy,
    }
