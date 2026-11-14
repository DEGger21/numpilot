"""D1003_rounding_envelope: Outward rounding of a bounded transform.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
nextafter appears twice with equal-shaped float32 operands.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(403)
    return {
        'x': rng.standard_normal((1024, 1024), dtype=np.float32),
        'target': rng.standard_normal((1024, 1024), dtype=np.float32),
    }


def forward(x, target):
    bounded = np.clip(x, -2.0, 2.0)
    square = np.square(bounded)
    shifted = np.add(square, np.full_like(square, 1.0))
    estimate = np.sqrt(shifted)
    absolute_1 = np.absolute(target)
    lower_target = np.negative(absolute_1)
    absolute_2 = np.absolute(target)
    upper_target = np.add(absolute_2, np.full_like(absolute_2, 4.0))
    lower = np.nextafter(estimate, lower_target)
    upper = np.nextafter(estimate, upper_target)
    width = np.subtract(upper, lower)
    maximum_3 = np.maximum(estimate, np.full_like(estimate, 1.0))
    relative = np.divide(width, maximum_3)
    return {
        'lower': lower,
        'upper': upper,
        'relative_width': relative,
    }
