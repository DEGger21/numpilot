"""B019_boolean_keepdims: Boolean reductions keep axes.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Traps: negative axis, keepdims, bool output.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([[[True, False], [True, True]], [[False, False], [True, False]]]),
    }


def forward(x):
    return {
        'all': np.all(x, axis=-1, keepdims=True),
        'any': np.any(x, axis=1, keepdims=True),
    }
