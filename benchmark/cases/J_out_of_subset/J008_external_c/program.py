"""J008_external_c: C library calls through ctypes.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Analysis-only negative case: harness_runnable=False.
Do not execute as part of the deterministic NumPy case sweep.
"""
import numpy as np


def build_inputs():
    return {
        'x': np.array([-3.5, -0.25, 2.0], dtype=np.float64),
    }


def forward(x):
    import ctypes
    import ctypes.util
    library = ctypes.util.find_library("m") or ctypes.util.find_library("c") or "msvcrt"
    handle = ctypes.CDLL(library)
    absolute = handle.fabs
    absolute.argtypes = [ctypes.c_double]
    absolute.restype = ctypes.c_double
    return {
        'out': np.array([absolute(value) for value in x]),
    }
