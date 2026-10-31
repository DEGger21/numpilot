"""E009_resampled_curve: Sorted coordinate interpolation and integration.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
The compute path combines at least three missing APIs with a supported anchor.
Named intermediate outputs expose indexing, shape and numerical behavior.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(509)
    x = rng.standard_normal(96, dtype=np.float32)
    x[:12] = 0.0
    x[-1] = 4.0
    x[0] = -4.0
    return {
        'x': x,
    }


def forward(x):
    coordinates = np.sort(x)
    knots = np.unique(coordinates)
    values = np.tanh(knots)
    query = np.linspace(knots[0], knots[-1], 53)
    interpolated = np.interp(query, knots, values)
    slopes = np.gradient(interpolated, query)
    areas = np.trapezoid(interpolated, query)
    curvature = np.diff(slopes)
    signbit_1 = np.signbit(curvature)
    diff_2 = np.diff(signbit_1)
    turns = np.flatnonzero(diff_2)
    knot_spacing = np.diff(knots)
    return {
        "knot_spacing": knot_spacing,
        'query': query,
        'out': interpolated,
        'slopes': slopes,
        'area': np.asarray(areas),
        'turns': turns,
    }
