"""D1004_two_track_velocity: Compare finite differences of calibrated tracks.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
diff appears twice with n=1, axis=1, float32.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(404)
    return {
        'x': rng.standard_normal((6, 18), dtype=np.float32),
        'y': rng.standard_normal((6, 18), dtype=np.float32),
    }


def forward(x, y):
    multiply_1 = np.multiply(x, 0.5)
    left = np.add(multiply_1, 0.25)
    multiply_2 = np.multiply(y, 0.25)
    right = np.subtract(multiply_2, 0.1)
    left_velocity = np.diff(left, axis=1)
    right_velocity = np.diff(right, axis=1)
    residual = np.subtract(left_velocity, right_velocity)
    squared = np.square(residual)
    mean_3 = np.mean(squared, axis=1)
    rmse = np.sqrt(mean_3)
    bounded = np.clip(residual, -1.0, 1.0)
    return {
        'residual': residual,
        'bounded': bounded,
        'rmse': rmse,
    }
