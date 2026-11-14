import asnumpy as ap
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
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    if isinstance(target, np.ndarray):
        target = ap.ndarray.from_numpy(target)
    bounded = ap.clip(x, -2.0, 2.0)
    square = ap.square(bounded)
    shifted = ap.add(square, ap.full_like(square, 1.0))
    estimate = ap.sqrt(shifted)
    absolute_1 = ap.absolute(target)
    lower_target = ap.negative(absolute_1)
    absolute_2 = ap.absolute(target)
    upper_target = ap.add(absolute_2, ap.full_like(absolute_2, 4.0))
    lower = ap.nextafter(estimate, lower_target)
    upper = ap.nextafter(estimate, upper_target)
    width = ap.subtract(upper, lower)
    maximum_3 = ap.maximum(estimate, ap.full_like(estimate, 1.0))
    relative = ap.divide(width, maximum_3)
    return {
        'lower': lower,
        'upper': upper,
        'relative_width': relative,
    }
