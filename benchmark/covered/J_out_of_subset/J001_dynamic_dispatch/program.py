"""J001_dynamic_dispatch: Runtime attribute dispatch.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Analysis-only negative case: harness_runnable=False.
Do not execute as part of the deterministic NumPy case sweep.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([-1, 0, 2], dtype=np.float32),
        'name': np.array('square'),
    }


def forward(x, name):
    operation = getattr(np, name.item())
    return {
        'out': operation(x),
    }
