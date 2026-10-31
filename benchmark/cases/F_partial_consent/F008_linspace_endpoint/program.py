"""F008_linspace_endpoint: Open-interval sampling excludes the endpoint.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Witness NumPy output: [0, 0.25, 0.5, 0.75].
Pinned 1929a3ba wrapper linspace rejects endpoint: TypeError before its body.
This signature mismatch is checkable without a CANN runtime.
"""
import numpy as np


def build_inputs():
    return {
        'offset': np.array([0.0], dtype=np.float32),
    }


def forward(offset):
    grid = np.linspace(0.0, 1.0, 4, endpoint=False, dtype=np.float32)
    return {
        'out': np.add(grid, offset),
    }
