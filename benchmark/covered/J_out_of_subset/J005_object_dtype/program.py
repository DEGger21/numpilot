"""J005_object_dtype: Arbitrary precision object arithmetic.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Analysis-only negative case: harness_runnable=False.
Do not execute as part of the deterministic NumPy case sweep.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([10 ** 40, 10 ** 50, -10 ** 60], dtype=object),
    }


def forward(x):
    return {
        'out': np.multiply(x, x),
    }
