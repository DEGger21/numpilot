"""J014_dynamic_import: Module imported from an input string.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Analysis-only negative case: harness_runnable=False.
Do not execute as part of the deterministic NumPy case sweep.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([-1, 0, 1], dtype=np.float32),
        'module': np.array('numpy'),
    }


def forward(x, module):
    import importlib
    library = importlib.import_module(module.item())
    return {
        'out': library.exp(x),
    }
