"""B015_fortran_matmul: Matrix product with Fortran-ordered inputs.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Trap: both inputs use column-major storage.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(215)
    x = np.asfortranarray(rng.standard_normal((2048, 2048)).astype(np.float32))
    y = np.asfortranarray(rng.standard_normal((2048, 2048)).astype(np.float32))
    return {
        "x": x,
        "y": y,
    }


def forward(x, y):
    product = np.matmul(x, y)
    return {
        'out': product,
        'row_power': np.sum(np.square(product), axis=1),
    }
