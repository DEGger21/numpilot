"""J013_array_protocol: Custom array-function dispatch.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Analysis-only negative case: harness_runnable=False.
Do not execute as part of the deterministic NumPy case sweep.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.arange(12, dtype=np.float64).reshape(3, 4),
    }


class ShiftedSum:
    def __init__(self, values):
        self.values = values

    def __array_function__(self, func, types, args, kwargs):
        if func is np.sum:
            return np.sum(self.values, **kwargs) + 10.0
        return NotImplemented


def forward(x):
    wrapped = ShiftedSum(x)
    return {
        'out': np.asarray(np.sum(wrapped, axis=1)),
    }
