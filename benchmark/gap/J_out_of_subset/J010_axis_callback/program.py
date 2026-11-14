"""J010_axis_callback: Python callback supplied to apply_along_axis.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Analysis-only negative case: harness_runnable=False.
Do not execute as part of the deterministic NumPy case sweep.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.arange(24, dtype=np.float32).reshape(4, 6),
    }


def row_score(row):
    return row[-1] - row[0] + np.median(row)


def forward(x):
    return {
        'out': np.apply_along_axis(row_score, 1, x),
    }
