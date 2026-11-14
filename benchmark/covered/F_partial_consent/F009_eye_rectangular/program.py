"""F009_eye_rectangular: Rectangular shifted identity matrix.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Witness NumPy output: [[0,2,0,0],[0,0,2,0]].
Pinned 1929a3ba wrapper eye rejects M/k: TypeError before its body.
This signature mismatch is checkable without a CANN runtime.
"""
import numpy as np


def build_inputs():
    return {
        'gain': np.array([2.0], dtype=np.float32),
    }


def forward(gain):
    matrix = np.eye(2, M=4, k=1, dtype=np.float32)
    return {
        'out': np.multiply(matrix, gain),
    }
